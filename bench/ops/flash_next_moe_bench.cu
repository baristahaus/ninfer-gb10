// Flash-Next 512-way MoE at decode-verify rows with production NVFP4 expert banks.
//
// Question: is the routed W4A4 path at the memory roofline for the experts a round actually
// reads? Routing is controlled exactly: token t's input is the unit vector e_t, and the router
// row of each expert in token t's chosen set scores 8 on dimension t, so top-10 selects that set.
// Sweeping the number of distinct experts per call and fitting time against it separates the
// per-expert streaming cost (slope -> effective GB/s) from the fixed cost of the call (router,
// shared expert, quantize, reduce; intercept).
//
// Each call uses its own layer bank (--layers, about 1.42 GB each), as consecutive model layers
// do, so no expert weight is reused from L2 between calls. Expert contents are arbitrary valid
// codes: the timing does not depend on values.
//
// --profile runs one eager pass over the layers between cudaProfilerStart/Stop for
// `ncu --profile-from-start off`. The process holds only the banks (a few GB): see
// tools/gb10/moe_microbench.sh for the memory guard that must precede any ncu run on GB10.
//
// --probe answers where the routed kernels lose bandwidth (ncu's DRAM counters are unusable on
// GB10). For each --distinct value it selects that many experts per layer bank (8 rows' worth of
// assignments, spread over the bank like production routing) and times, per layer:
// - contig: a plain 16-byte streaming read of exactly those experts' code and scale bytes, the
//   ceiling for this footprint;
// - rows: the same code bytes read the way the W4A4 kernel walks them (128-row items, S bytes
//   of every row per step), isolating the cost of the row-strided access pattern;
// - gate/down: the production W4A4 kernel on the same experts at its former decode schedule and at
//   alternative BlockN/BlockK/stage/grid choices, each phase alone, and the former and production
//   pairs back to back.

#include "core/device.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include "ninfer/ops/flash_next_moe.h"
#include "ninfer_bench_common.h"
#include "ops/linear/nvfp4/nvfp4_geometry.h"
#include "ops/sparse_moe/flash_next/flash_next_nvfp4_w4a4.cuh"
#include "quantized_weight.cuh"

#include <cuda_bf16.h>

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

using namespace ninfer;

namespace {

constexpr int kHidden       = 2560;
constexpr int kExperts      = 512;
constexpr int kIntermediate = 640;
constexpr int kTop          = 10;
constexpr double kDramGBs   = 246.0; // tools/bench/hardware/gb10.json, measured streaming read

struct Options {
    int tokens = 8;
    std::vector<int> distinct{10, 20, 40, 64, 80};
    int layers   = 4;
    int warmup   = 5;
    int repeat   = 30;
    bool profile = false;
    bool probe   = false;
};

std::vector<int> parse_list(std::string_view raw) {
    std::vector<int> out;
    std::size_t begin = 0;
    while (begin <= raw.size()) {
        const std::size_t end = raw.find(',', begin);
        out.push_back(std::stoi(std::string(raw.substr(begin, end - begin))));
        if (end == std::string_view::npos) { break; }
        begin = end + 1;
    }
    return out;
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const auto value           = [&]() -> std::string_view {
            if (i + 1 >= argc) { throw std::invalid_argument(std::string(arg) + " needs a value"); }
            return argv[++i];
        };
        if (arg == "--tokens") {
            options.tokens = std::stoi(std::string(value()));
        } else if (arg == "--distinct") {
            options.distinct = parse_list(value());
        } else if (arg == "--layers") {
            options.layers = std::stoi(std::string(value()));
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(std::string(value()));
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(std::string(value()));
        } else if (arg == "--profile") {
            options.profile = true;
        } else if (arg == "--probe") {
            options.probe = true;
        } else {
            throw std::invalid_argument("unknown argument " + std::string(arg));
        }
    }
    if (options.tokens < 1 || options.tokens > 16 || options.layers < 1) {
        throw std::invalid_argument("--tokens must be 1..16 (the decode route) and --layers >= 1");
    }
    for (const int d : options.distinct) {
        if (d < kTop || d > kExperts) {
            throw std::invalid_argument("--distinct values in 10..512");
        }
    }
    return options;
}

std::uint16_t bf16(float value) { return bench::f32_to_bf16(value); }

Weight bf16_weight(const DeviceBuffer& storage, int rows, int columns) {
    Weight out{};
    out.payload = out.qdata = storage.p;
    out.payload_bytes       = storage.bytes;
    out.qtype               = QType::BF16;
    out.layout              = QuantLayout::Contiguous;
    out.n = out.shape[0] = out.padded_shape[0] = rows;
    out.k = out.shape[1] = out.padded_shape[1] = columns;
    out.ndim                                   = 2;
    return out;
}

// One expert-major NVFP4 bank in the production layout, filled with valid codes (FP4 1.0),
// unit E4M3 block scales and unit divisors.
struct Bank {
    DeviceBuffer storage;
    WeightGeometry geometry;
    std::uint64_t streamed_bytes_per_expert = 0;

    Bank(int rows, int columns) {
        const std::array<std::uint64_t, 3> shape{kExperts, static_cast<std::uint64_t>(rows),
                                                 static_cast<std::uint64_t>(columns)};
        geometry = weight_geometry(QType::NVFP4, QuantLayout::ExpertBlockScaleK16M128x4, shape);
        storage  = DeviceBuffer(geometry.bytes);
        CUDA_CHECK(cudaMemset(storage.p, 0x22, geometry.code_bytes));
        CUDA_CHECK(cudaMemset(static_cast<std::uint8_t*>(storage.p) + geometry.scale_offset, 0x38,
                              geometry.scale_bytes));
        const std::uint64_t divisors = (geometry.bytes - geometry.divisor_offset) / sizeof(float);
        const std::vector<float> ones(divisors, 1.0F);
        storage.copy_from_host(ones.data(), ones.size() * sizeof(float), geometry.divisor_offset);
        streamed_bytes_per_expert = (geometry.code_bytes + geometry.scale_bytes) / kExperts;
    }

    ops::FlashNextExpertBank view(const float* input_divisors, int rows, int columns) const {
        const auto* base = static_cast<const std::uint8_t*>(storage.p);
        return {
            .codes                 = base,
            .scales                = base + geometry.scale_offset,
            .weight_scale_divisors = reinterpret_cast<const float*>(base + geometry.divisor_offset),
            .input_scale_divisors  = input_divisors,
            .qtype                 = QType::NVFP4,
            .experts               = kExperts,
            .rows                  = rows,
            .columns               = columns,
        };
    }
};

// Token t selects 10 experts out of a pool of `distinct`, offset so consecutive tokens overlap
// as little as the pool allows. Returns the router matrix and the realized distinct count.
std::pair<std::vector<std::uint16_t>, int> router_for(int tokens, int distinct) {
    std::vector<std::uint16_t> router(static_cast<std::size_t>(kExperts) * kHidden, bf16(0.0F));
    std::set<int> used;
    for (int t = 0; t < tokens; ++t) {
        for (int j = 0; j < kTop; ++j) {
            const int expert                                       = (t * kTop + j) % distinct;
            router[static_cast<std::size_t>(expert) * kHidden + t] = bf16(8.0F);
            used.insert(expert);
        }
    }
    return {router, static_cast<int>(used.size())};
}

namespace probe {

namespace fn        = ops::detail::flash_next;
using GateGeometry  = ops::detail::Nvfp4Geometry<2 * kIntermediate, kHidden>;
using DownGeometry  = ops::detail::Nvfp4Geometry<kHidden, kIntermediate>;
constexpr int kTile = 128; // weight rows per scale tile and per W4A4 work item

// Copies of the production MoE's grouped work, row and output policies (flash_next_moe.cu keeps
// them private). The kernel template is the production one.
struct GateRows {
    static constexpr bool kContiguous = false;
    int expert                        = 0;
    int rows_per_branch               = 0;

    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        const int branch = local_row >= rows_per_branch ? 1 : 0;
        const int logical =
            row_begin + local_row - branch * rows_per_branch + branch * kIntermediate;
        return expert * (2 * kIntermediate) + logical;
    }
};

struct DownRows {
    static constexpr bool kContiguous = false;
    int expert                        = 0;

    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        return expert * kHidden + row_begin + local_row;
    }
};

template <class RowPolicy>
struct Work {
    static constexpr bool kPersistent = true;
    const int* job_count;
    const int* job_experts;
    const int* job_columns;
    const int* offsets;
    const float* weight_divisors;
    const float* input_divisors;
    int output_rows;

    __device__ __forceinline__ int work_count(int rows_per_block) const {
        return *job_count * (output_rows / rows_per_block);
    }

    __device__ __forceinline__ void configure(int work, int rows_per_block, int& token_begin,
                                              int& active_tokens, int& row_begin, float& alpha,
                                              RowPolicy& rows) const {
        const int row_blocks = output_rows / rows_per_block;
        const int job        = work / row_blocks;
        const int expert     = job_experts[job];
        token_begin          = offsets[expert] + job_columns[job];
        active_tokens        = offsets[expert + 1];
        row_begin            = (work - job * row_blocks) * rows_per_block;
        alpha                = 1.0F / (weight_divisors[expert] * input_divisors[expert]);
        rows.expert          = expert;
    }
};

struct SiluOutput {
    __nv_bfloat16* data;

    __device__ __forceinline__ void store_pair_vector(int row, int token, uint4 gate_raw,
                                                      uint4 up_raw) const {
        const auto* gate  = reinterpret_cast<const __nv_bfloat162*>(&gate_raw);
        const auto* up    = reinterpret_cast<const __nv_bfloat162*>(&up_raw);
        auto* destination = reinterpret_cast<__nv_bfloat162*>(
            data + static_cast<std::int64_t>(token) * kIntermediate + row);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 g    = __bfloat1622float2(gate[pair]);
            const float2 u    = __bfloat1622float2(up[pair]);
            destination[pair] = __floats2bfloat162_rn((g.x / (1.0F + expf(-g.x))) * u.x,
                                                      (g.y / (1.0F + expf(-g.y))) * u.y);
        }
    }
};

struct PlainOutput {
    __nv_bfloat16* data;
    int rows;

    __device__ __forceinline__ void store_vector(int row, int token, uint4 values) const {
        *reinterpret_cast<uint4*>(data + static_cast<std::int64_t>(token) * rows + row) = values;
    }
};

struct Chunk {
    const uint4* data;
    unsigned vectors;
};

// Plain streaming read: each CTA takes 64 KB chunks round-robin, 8 independent 16-byte loads in
// flight per thread.
__global__ void __launch_bounds__(256)
    contiguous_read_kernel(const Chunk* chunks, int count, unsigned* sink) {
    unsigned acc = 0;
    for (int c = static_cast<int>(blockIdx.x); c < count; c += static_cast<int>(gridDim.x)) {
        const Chunk chunk = chunks[c];
        unsigned v        = threadIdx.x;
        for (; v + 7U * 256U < chunk.vectors; v += 8U * 256U) {
            uint4 x[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) { x[u] = __ldcg(chunk.data + v + u * 256U); }
#pragma unroll
            for (int u = 0; u < 8; ++u) { acc ^= x[u].x ^ x[u].y ^ x[u].z ^ x[u].w; }
        }
        for (; v < chunk.vectors; v += 256U) {
            const uint4 x = __ldcg(chunk.data + v);
            acc ^= x.x ^ x.y ^ x.z ^ x.w;
        }
    }
    if (acc == 0x9e3779b9U) { sink[blockIdx.x] = acc; }
}

// The W4A4 kernel's traversal without its pipeline or MMA: per 128-row item, step through K
// taking Segment bytes of every row per step.
template <int RowBytes, int Segment>
__global__ void __launch_bounds__(256)
    row_segment_read_kernel(const std::uint8_t* const* items, int count, unsigned* sink) {
    static_assert(RowBytes % Segment == 0 && Segment % 16 == 0);
    constexpr int kPieces = Segment / 16;
    constexpr int kTasks  = kTile * kPieces;
    constexpr int kSteps  = RowBytes / Segment;
    unsigned acc          = 0;
    for (int i = static_cast<int>(blockIdx.x); i < count; i += static_cast<int>(gridDim.x)) {
        const std::uint8_t* base = items[i];
#pragma unroll 4
        for (int step = 0; step < kSteps; ++step) {
#pragma unroll
            for (int task = static_cast<int>(threadIdx.x); task < kTasks; task += 256) {
                const int row   = task / kPieces;
                const int piece = task - row * kPieces;
                const uint4 x   = __ldcg(reinterpret_cast<const uint4*>(
                    base + static_cast<std::int64_t>(row) * RowBytes + step * Segment + piece * 16));
                acc ^= x.x ^ x.y ^ x.z ^ x.w;
            }
        }
    }
    if (acc == 0x9e3779b9U) { sink[blockIdx.x] = acc; }
}

// The selected experts and their grouped job list for one --distinct value, plus zeroed
// activations and outputs (timing does not depend on values).
struct Jobs {
    int experts     = 0;
    int assignments = 0;
    std::vector<int> selected; // physical expert ids, ascending
    DeviceBuffer job_count, job_experts, job_columns, offsets;
    DeviceBuffer gate_codes, gate_scales, down_codes, down_scales, activation, output;

    Jobs(int tokens, int requested) {
        // Same overlap pattern as router_for, with logical expert i placed at bank position
        // 37 i mod 512 so the selection spans the bank like production routing.
        std::vector<int> count(kExperts, 0);
        for (int t = 0; t < tokens; ++t) {
            for (int j = 0; j < kTop; ++j) {
                ++count[static_cast<std::size_t>(((t * kTop + j) % requested) * 37 % kExperts)];
            }
        }
        std::vector<int> offset(kExperts + 1, 0);
        for (int e = 0; e < kExperts; ++e) {
            offset[e + 1] = offset[e] + count[e];
            if (count[e] > 0) { selected.push_back(e); }
        }
        experts     = static_cast<int>(selected.size());
        assignments = tokens * kTop;
        job_count   = bench_upload(std::vector<int>{experts});
        job_experts = bench_upload(selected);
        job_columns = bench_upload(std::vector<int>(selected.size(), 0));
        offsets     = bench_upload(offset);
        gate_codes  = zeroed(static_cast<std::size_t>(assignments) * kHidden / 2);
        gate_scales = zeroed(static_cast<std::size_t>(assignments) * kHidden / 16);
        down_codes  = zeroed(static_cast<std::size_t>(assignments) * kIntermediate / 2);
        down_scales = zeroed(static_cast<std::size_t>(assignments) * kIntermediate / 16);
        activation  = zeroed(static_cast<std::size_t>(assignments) * kIntermediate * 2);
        output      = zeroed(static_cast<std::size_t>(assignments) * kHidden * 2);
    }

    static DeviceBuffer bench_upload(const std::vector<int>& values) {
        DeviceBuffer out(values.size() * sizeof(int));
        out.copy_from_host(values.data(), out.bytes);
        return out;
    }

    static DeviceBuffer zeroed(std::size_t bytes) {
        DeviceBuffer out(bytes);
        out.fill(0);
        return out;
    }
};

struct BankView {
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    const float* divisors;
    std::uint64_t code_bytes_per_expert;
    std::uint64_t scale_bytes_per_expert;
};

BankView view_of(const Bank& bank) {
    const auto* base = static_cast<const std::uint8_t*>(bank.storage.p);
    return {base, base + bank.geometry.scale_offset,
            reinterpret_cast<const float*>(base + bank.geometry.divisor_offset),
            bank.geometry.code_bytes / kExperts, bank.geometry.scale_bytes / kExperts};
}

template <class Schedule>
void launch_gate(const Jobs& jobs, const BankView& bank, const float* input_divisors, int grid,
                 cudaStream_t stream) {
    constexpr int kRows = Schedule::kBlockN / 2;
    const int items     = jobs.experts * (kIntermediate / kRows);
    fn::nvfp4_w4a4_mma_kernel<GateGeometry, Schedule, fn::Nvfp4IdentityEpilogue, SiluOutput,
                              GateRows, true, Work<GateRows>>
        <<<std::min(grid, items), Schedule::kThreads, 0, stream>>>(
            fn::Nvfp4W4a4MaterializedActivation{
                static_cast<const std::uint8_t*>(jobs.gate_codes.p),
                static_cast<const std::uint8_t*>(jobs.gate_scales.p)},
            bank.codes, bank.scales, jobs.assignments, 1.0F, fn::Nvfp4IdentityEpilogue{},
            SiluOutput{static_cast<__nv_bfloat16*>(jobs.activation.p)}, GateRows{0, kRows},
            Work<GateRows>{static_cast<const int*>(jobs.job_count.p),
                           static_cast<const int*>(jobs.job_experts.p),
                           static_cast<const int*>(jobs.job_columns.p),
                           static_cast<const int*>(jobs.offsets.p), bank.divisors,
                           input_divisors, kIntermediate});
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_down(const Jobs& jobs, const BankView& bank, const float* input_divisors, int grid,
                 cudaStream_t stream) {
    const int items = jobs.experts * (kHidden / Schedule::kBlockN);
    fn::nvfp4_w4a4_mma_kernel<DownGeometry, Schedule, fn::Nvfp4IdentityEpilogue, PlainOutput,
                              DownRows, false, Work<DownRows>>
        <<<std::min(grid, items), Schedule::kThreads, 0, stream>>>(
            fn::Nvfp4W4a4MaterializedActivation{
                static_cast<const std::uint8_t*>(jobs.down_codes.p),
                static_cast<const std::uint8_t*>(jobs.down_scales.p)},
            bank.codes, bank.scales, jobs.assignments, 1.0F, fn::Nvfp4IdentityEpilogue{},
            PlainOutput{static_cast<__nv_bfloat16*>(jobs.output.p), kHidden}, DownRows{},
            Work<DownRows>{static_cast<const int*>(jobs.job_count.p),
                           static_cast<const int*>(jobs.job_experts.p),
                           static_cast<const int*>(jobs.job_columns.p),
                           static_cast<const int*>(jobs.offsets.p), bank.divisors,
                           input_divisors, kHidden});
    CUDA_CHECK(cudaGetLastError());
}

using Launch = void (*)(const Jobs&, const BankView&, const float*, int, cudaStream_t);

struct Variant {
    const char* name;
    Launch launch;
};

// <BlockM tokens, BlockN rows, BlockK, WarpsM, WarpsN, Stages, MinBlocksPerSm>. The first entry
// of each list is the schedule production used until 2026-10-02 (kept as the baseline); the
// production decode schedules are now BN64 BK512 S2 (gate/up) and BN64 BK128 S4 (down).
template <int N, int K, int S>
using Decode = fn::Nvfp4W4a4MmaSchedule<16, N, K, 1, 8, S, 1>;

const std::array<Variant, 8> kGateVariants{{
    {"BN128 BK128 S2 (former)", &launch_gate<Decode<128, 128, 2>>},
    {"BN128 BK128 S3", &launch_gate<Decode<128, 128, 3>>},
    {"BN128 BK128 S4", &launch_gate<Decode<128, 128, 4>>},
    {"BN128 BK256 S2", &launch_gate<Decode<128, 256, 2>>},
    {"BN64 BK256 S3", &launch_gate<Decode<64, 256, 3>>},
    {"BN64 BK256 S4", &launch_gate<Decode<64, 256, 4>>},
    {"BN64 BK512 S2", &launch_gate<Decode<64, 512, 2>>},
    {"BN256 BK128 S2 (wide)", &launch_gate<Decode<256, 128, 2>>},
}};

// The down rows hold 640 inputs (320 bytes), so BlockK is 128 at most for this kernel.
const std::array<Variant, 5> kDownVariants{{
    {"BN128 BK128 S2 (former)", &launch_down<Decode<128, 128, 2>>},
    {"BN128 BK128 S3", &launch_down<Decode<128, 128, 3>>},
    {"BN128 BK128 S4", &launch_down<Decode<128, 128, 4>>},
    {"BN64 BK128 S4", &launch_down<Decode<64, 128, 4>>},
    {"BN256 BK128 S2", &launch_down<Decode<256, 128, 2>>},
}};

void run(const Options& options, const std::vector<Bank>& gate_up, const std::vector<Bank>& down,
         const float* input_divisors, cudaStream_t stream) {
    const int sms = [] {
        int device = 0;
        int count  = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
        return count;
    }();
    DeviceBuffer sink(4096 * sizeof(unsigned));
    std::vector<BankView> gate_views;
    std::vector<BankView> down_views;
    for (std::size_t layer = 0; layer < gate_up.size(); ++layer) {
        gate_views.push_back(view_of(gate_up[layer]));
        down_views.push_back(view_of(down[layer]));
    }
    const auto layers = static_cast<int>(gate_views.size());

    for (const int requested : options.distinct) {
        const Jobs jobs(options.tokens, requested);
        const double gate_codes =
            static_cast<double>(jobs.experts) * gate_views[0].code_bytes_per_expert;
        const double gate_bytes =
            gate_codes + static_cast<double>(jobs.experts) * gate_views[0].scale_bytes_per_expert;
        const double down_codes =
            static_cast<double>(jobs.experts) * down_views[0].code_bytes_per_expert;
        const double down_bytes =
            down_codes + static_cast<double>(jobs.experts) * down_views[0].scale_bytes_per_expert;
        std::printf("\nprobe: T=%d rows, %d distinct experts, %d layer banks, %.1f MB gate/up + "
                    "%.1f MB down per layer, %d SMs\n",
                    options.tokens, jobs.experts, layers, gate_bytes / 1e6, down_bytes / 1e6, sms);
        std::printf("%-40s %6s %10s %10s %8s\n", "case", "grid", "us/layer", "GB/s", "% 246");
        const auto report = [&](const std::string& name, int grid, double bytes,
                                const auto& per_layer) {
            bench::TimedGraph graph;
            graph.capture(stream, [&](cudaStream_t s) {
                for (int layer = 0; layer < layers; ++layer) { per_layer(layer, s); }
            });
            const auto timing = bench::measure_graph(graph, stream, options.warmup, options.repeat);
            const double us   = timing.median_us / layers;
            const double gbs  = bytes / (us * 1e3);
            std::printf("%-40s %6d %10.1f %10.1f %8.1f\n", name.c_str(), grid, us, gbs,
                        100.0 * gbs / kDramGBs);
        };

        // Chunk lists per layer: contiguous code and scale blocks of every selected expert.
        const auto chunk_list = [&](int layer, bool scales) {
            std::vector<Chunk> chunks;
            const auto add = [&](const std::uint8_t* begin, std::uint64_t bytes) {
                constexpr std::uint64_t kChunk = 64 * 1024;
                for (std::uint64_t at = 0; at < bytes; at += kChunk) {
                    chunks.push_back(
                        {reinterpret_cast<const uint4*>(begin + at),
                         static_cast<unsigned>(std::min(kChunk, bytes - at) / sizeof(uint4))});
                }
            };
            for (const int e : jobs.selected) {
                for (const BankView* bank : {&gate_views[layer], &down_views[layer]}) {
                    add(bank->codes + e * bank->code_bytes_per_expert, bank->code_bytes_per_expert);
                    if (scales) {
                        add(bank->scales + e * bank->scale_bytes_per_expert,
                            bank->scale_bytes_per_expert);
                    }
                }
            }
            DeviceBuffer out(chunks.size() * sizeof(Chunk));
            out.copy_from_host(chunks.data(), out.bytes);
            return std::make_pair(std::move(out), static_cast<int>(chunks.size()));
        };
        for (const bool scales : {true, false}) {
            std::vector<std::pair<DeviceBuffer, int>> lists;
            for (int layer = 0; layer < layers; ++layer) {
                lists.push_back(chunk_list(layer, scales));
            }
            for (const int per_sm : {1, 2, 3, 4}) {
                if (!scales && per_sm != 2) { continue; }
                const int grid = per_sm * sms;
                report(scales ? "contig codes+scales" : "contig codes only", grid,
                       scales ? gate_bytes + down_bytes : gate_codes + down_codes,
                       [&](int layer, cudaStream_t s) {
                           contiguous_read_kernel<<<grid, 256, 0, s>>>(
                               static_cast<const Chunk*>(lists[layer].first.p),
                               lists[layer].second, static_cast<unsigned*>(sink.p));
                       });
            }
        }

        // Row-strided traversal of the code bytes, 128-row items, production grid (3 per SM).
        const auto item_list = [&](int layer, bool gate) {
            const BankView& bank = gate ? gate_views[layer] : down_views[layer];
            const int rows       = gate ? 2 * kIntermediate : kHidden;
            const int row_bytes  = gate ? kHidden / 2 : kIntermediate / 2;
            std::vector<const std::uint8_t*> items;
            for (const int e : jobs.selected) {
                for (int tile = 0; tile < rows / kTile; ++tile) {
                    items.push_back(bank.codes + (static_cast<std::uint64_t>(e) * rows +
                                                  static_cast<std::uint64_t>(tile) * kTile) *
                                                     row_bytes);
                }
            }
            DeviceBuffer out(items.size() * sizeof(const std::uint8_t*));
            out.copy_from_host(items.data(), out.bytes);
            return std::make_pair(std::move(out), static_cast<int>(items.size()));
        };
        {
            const int grid = 3 * sms;
            std::vector<std::pair<DeviceBuffer, int>> gate_items;
            std::vector<std::pair<DeviceBuffer, int>> down_items;
            for (int layer = 0; layer < layers; ++layer) {
                gate_items.push_back(item_list(layer, true));
                down_items.push_back(item_list(layer, false));
            }
            const auto rows = [&](const char* name, double bytes, auto kernel, bool gate) {
                auto& lists = gate ? gate_items : down_items;
                report(name, grid, bytes, [&](int layer, cudaStream_t s) {
                    kernel<<<grid, 256, 0, s>>>(
                        static_cast<const std::uint8_t* const*>(lists[layer].first.p),
                        lists[layer].second, static_cast<unsigned*>(sink.p));
                });
            };
            rows("rows gate codes S=64B (BK128)", gate_codes,
                 row_segment_read_kernel<kHidden / 2, 64>, true);
            rows("rows gate codes S=128B (BK256)", gate_codes,
                 row_segment_read_kernel<kHidden / 2, 128>, true);
            rows("rows gate codes S=256B (BK512)", gate_codes,
                 row_segment_read_kernel<kHidden / 2, 256>, true);
            rows("rows down codes S=64B (BK128)", down_codes,
                 row_segment_read_kernel<kIntermediate / 2, 64>, false);
            rows("rows down codes S=320B (whole row)", down_codes,
                 row_segment_read_kernel<kIntermediate / 2, 320>, false);
        }

        // The production W4A4 kernel on the same experts.
        for (const int per_sm : {2, 3}) {
            const int grid = per_sm * sms;
            for (const Variant& variant : kGateVariants) {
                report(std::string("gate ") + variant.name, grid, gate_bytes,
                       [&](int layer, cudaStream_t s) {
                           variant.launch(jobs, gate_views[layer], input_divisors, grid, s);
                       });
            }
            for (const Variant& variant : kDownVariants) {
                report(std::string("down ") + variant.name, grid, down_bytes,
                       [&](int layer, cudaStream_t s) {
                           variant.launch(jobs, down_views[layer], input_divisors, grid, s);
                       });
            }
        }
        for (const auto& [name, gate, down] :
             {std::tuple{"gate+down former, back to back", 0, 0},
              std::tuple{"gate+down production, back to back", 6, 3}}) {
            report(name, 3 * sms, gate_bytes + down_bytes, [&](int layer, cudaStream_t s) {
                kGateVariants[gate].launch(jobs, gate_views[layer], input_divisors, 3 * sms, s);
                kDownVariants[down].launch(jobs, down_views[layer], input_divisors, 3 * sms, s);
            });
        }
    }
}

} // namespace probe

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        cudaStream_t stream   = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        std::vector<Bank> gate_up;
        std::vector<Bank> down;
        for (int layer = 0; layer < options.layers; ++layer) {
            gate_up.emplace_back(2 * kIntermediate, kHidden);
            down.emplace_back(kHidden, kIntermediate);
        }
        DeviceBuffer input_divisors(kExperts * sizeof(float));
        {
            const std::vector<float> ones(kExperts, 1.0F);
            input_divisors.copy_from_host(ones.data(), ones.size() * sizeof(float));
        }
        auto shared_gate_up = bench::make_fp8_weight(2 * kIntermediate, kHidden);
        auto shared_down    = bench::make_fp8_weight(kHidden, kIntermediate);
        DeviceBuffer shared_scale(kHidden * sizeof(std::uint16_t));
        shared_scale.fill();
        DeviceBuffer router(static_cast<std::size_t>(kExperts) * kHidden * sizeof(std::uint16_t));

        DeviceBuffer input(static_cast<std::size_t>(kHidden) * options.tokens *
                           sizeof(std::uint16_t));
        {
            std::vector<std::uint16_t> host(static_cast<std::size_t>(kHidden) * options.tokens,
                                            bf16(0.0F));
            for (int t = 0; t < options.tokens; ++t) {
                host[static_cast<std::size_t>(t) * kHidden + t] = bf16(1.0F);
            }
            input.copy_from_host(host.data(), host.size() * sizeof(std::uint16_t));
        }
        DeviceBuffer output(static_cast<std::size_t>(kHidden) * options.tokens *
                            sizeof(std::uint16_t));
        Tensor input_tensor(input.p, DType::BF16, {kHidden, options.tokens});
        Tensor output_tensor(output.p, DType::BF16, {kHidden, options.tokens});
        WorkspaceArena workspace(ops::flash_next_moe_workspace_capacity_bytes(options.tokens));

        std::vector<ops::FlashNextMoeWeights> layers;
        for (int layer = 0; layer < options.layers; ++layer) {
            layers.push_back({
                .router         = bf16_weight(router, kExperts, kHidden),
                .shared_gate_up = shared_gate_up.weight,
                .shared_down    = shared_down.weight,
                .shared_scale   = bf16_weight(shared_scale, 1, kHidden),
                .routed_gate_up = gate_up[layer].view(static_cast<const float*>(input_divisors.p),
                                                      2 * kIntermediate, kHidden),
                .routed_down    = down[layer].view(static_cast<const float*>(input_divisors.p),
                                                   kHidden, kIntermediate),
            });
        }
        if (options.probe) {
            probe::run(options, gate_up, down, static_cast<const float*>(input_divisors.p),
                       stream);
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }
        const double expert_bytes = static_cast<double>(gate_up[0].streamed_bytes_per_expert +
                                                        down[0].streamed_bytes_per_expert);
        std::printf("flash_next_moe NVFP4 decode route: T=%d rows, %d layer banks of %.2f GB, "
                    "%.3f MB streamed per selected expert\n",
                    options.tokens, options.layers,
                    static_cast<double>(gate_up[0].geometry.bytes + down[0].geometry.bytes) / 1e9,
                    expert_bytes / 1e6);
        std::printf("%-9s %12s %12s %14s\n", "distinct", "us/layer", "min us", "routed GB/s");

        std::vector<std::pair<double, double>> points;
        for (const int requested : options.distinct) {
            const auto [host_router, distinct] = router_for(options.tokens, requested);
            router.copy_from_host(host_router.data(), host_router.size() * sizeof(std::uint16_t));
            bench::TimedGraph graph;
            graph.capture(stream, [&](cudaStream_t s) {
                for (const auto& weights : layers) {
                    ops::flash_next_moe(input_tensor, weights, output_tensor, workspace, s);
                }
            });
            const auto timing = bench::measure_graph(graph, stream, options.warmup, options.repeat);
            const double per_layer = timing.median_us / options.layers;
            const double gbs       = distinct * expert_bytes / (per_layer * 1e3);
            points.emplace_back(distinct, per_layer);
            std::printf("%-9d %12.1f %12.1f %14.1f\n", distinct, per_layer,
                        timing.min_us / options.layers, gbs);
        }
        if (points.size() >= 2) {
            double sx = 0, sy = 0, sxx = 0, sxy = 0;
            for (const auto& [x, y] : points) {
                sx += x;
                sy += y;
                sxx += x * x;
                sxy += x * y;
            }
            const double n         = static_cast<double>(points.size());
            const double slope     = (n * sxy - sx * sy) / (n * sxx - sx * sx);
            const double intercept = (sy - slope * sx) / n;
            const double slope_gbs = expert_bytes / (slope * 1e3);
            std::printf(
                "fit: %.2f us per distinct expert (%.0f GB/s, %.0f%% of %.0f GB/s), %.1f us "
                "fixed per layer\n",
                slope, slope_gbs, 100.0 * slope_gbs / kDramGBs, kDramGBs, intercept);
        }

        if (options.profile) {
            const auto [host_router, distinct] =
                router_for(options.tokens, options.distinct.back());
            router.copy_from_host(host_router.data(), host_router.size() * sizeof(std::uint16_t));
            for (const auto& weights : layers) {
                ops::flash_next_moe(input_tensor, weights, output_tensor, workspace, stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaProfilerStart());
            for (const auto& weights : layers) {
                ops::flash_next_moe(input_tensor, weights, output_tensor, workspace, stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaProfilerStop());
            std::printf("profiled one eager pass over %d layers at %d distinct experts\n",
                        options.layers, distinct);
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "flash_next_moe_bench: %s\n", error.what());
        return 1;
    }
}
