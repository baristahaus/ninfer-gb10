// Flash-Next dense FP8 projections at decode-verify widths against their weight bytes.
//
// Question: how far below a plain read of its bytes does each production FP8 small-T projection
// run, and which sliced-K schedule recovers the difference without changing any output bit? For
// every Flash-Next FP8 shape on the T = 2..16 route, this times the production schedule
// (Fp8SlicedInstance<T, W, 2>, W = the shape's K-warp count), schedules that keep W and therefore
// every output's accumulation order (one stage, the other occupancy bound, activations in L1 or
// not, two or four row tiles per CTA, the K-warp pairs split over two or four CTAs), and a 16-byte
// read of the same bytes. Each variant's output is compared bitwise with the production
// schedule's. Every call reads its own copy of the weight (--budget-mb of copies per shape), so
// nothing is reused from L2. Outputs use the plain BF16 store: the production consumers' fused
// epilogues (HyperConnection SiLU and gate mix, shared-expert SwiGLU) are not included.

#include "core/device.h"
#include "core/weight.h"
#include "ninfer_bench_common.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/fp8/fp8_a16_sliced_k_mma.cuh"
#include "ops/linear/fp8/fp8_flash_next_route.cuh"
#include "ops/linear/fp8/fp8_instances.cuh"
#include "ops/linear/fp8/fp8_operands.h"
#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "quantized_weight.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops::detail;

namespace {

constexpr double kDramGBs = 246.0; // tools/bench/hardware/gb10.json, measured streaming read

struct Options {
    std::vector<int> tokens{8, 16};
    double budget_mb = 384.0;
    int warmup       = 5;
    int repeat       = 30;
};

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const auto value           = [&]() -> std::string {
            if (i + 1 >= argc) { throw std::invalid_argument(std::string(arg) + " needs a value"); }
            return argv[++i];
        };
        if (arg == "--tokens") {
            options.tokens.clear();
            const std::string list = value();
            std::size_t begin      = 0;
            while (begin <= list.size()) {
                const std::size_t end = list.find(',', begin);
                options.tokens.push_back(std::stoi(list.substr(begin, end - begin)));
                if (end == std::string::npos) { break; }
                begin = end + 1;
            }
        } else if (arg == "--budget-mb") {
            options.budget_mb = std::stod(value());
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(value());
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(value());
        } else {
            throw std::invalid_argument("unknown argument " + std::string(arg));
        }
    }
    for (const int t : options.tokens) {
        if (t != 8 && t != 16) { throw std::invalid_argument("--tokens values are 8 and 16"); }
    }
    return options;
}

struct Chunk {
    const uint4* data;
    unsigned vectors;
};

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

__device__ __forceinline__ unsigned mix(unsigned v) {
    v ^= v >> 16;
    v *= 0x7feb352dU;
    v ^= v >> 15;
    v *= 0x846ca68bU;
    return v ^ (v >> 16);
}

// Finite E4M3 codes (|code| <= 0x7e) so every output's bits depend on its summation order.
__global__ void fill_codes_kernel(std::uint8_t* codes, std::uint64_t bytes, unsigned seed) {
    for (std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         i < bytes; i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const unsigned h = mix(static_cast<unsigned>(i) ^ seed);
        codes[i]         = static_cast<std::uint8_t>((h & 0x80U) | (h % 0x7fU));
    }
}

__global__ void fill_activations_kernel(__nv_bfloat16* x, std::uint64_t count) {
    for (std::uint64_t i = blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         i < count; i += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const unsigned h = mix(static_cast<unsigned>(i) * 2654435761U + 17U);
        x[i] = __float2bfloat16_rn((static_cast<float>(h & 0xffffU) / 32768.0f - 1.0f) * 2.0f);
    }
}

struct Shape {
    const char* name;
    int rows;
    int k;
};

struct Context {
    const Options& options;
    cudaStream_t stream;
    int sms;
};

template <int Rows, int K>
struct Bank {
    std::vector<bench::PackedQuantizedWeight> copies;
    double bytes_per_call = 0.0;
    std::vector<Chunk> chunks;
    DeviceBuffer chunk_list;
    DeviceBuffer sink{4096 * sizeof(unsigned)};

    explicit Bank(double budget_mb) {
        const int count = std::max(
            2, static_cast<int>(budget_mb * 1e6 / (static_cast<double>(Rows) * K) + 0.5));
        for (int copy = 0; copy < count; ++copy) {
            copies.push_back(bench::make_fp8_weight(Rows, K));
            fill_codes_kernel<<<1024, 256>>>(static_cast<std::uint8_t*>(copies.back().storage.p),
                                             copies.back().low_bytes, 0x5bd1e995U * (copy + 1));
            CUDA_CHECK(cudaGetLastError());
        }
        bytes_per_call = static_cast<double>(copies[0].low_bytes + copies[0].scale_bytes);
        for (const auto& copy : copies) {
            const auto* base = static_cast<const std::uint8_t*>(copy.storage.p);
            for (const auto [offset, bytes] :
                 {std::pair{std::uint64_t{0}, copy.low_bytes},
                  std::pair{copy.scale_offset, copy.scale_bytes / 16 * 16}}) {
                for (std::uint64_t at = 0; at < bytes; at += 65536) {
                    chunks.push_back(
                        {reinterpret_cast<const uint4*>(base + offset + at),
                         static_cast<unsigned>(std::min<std::uint64_t>(65536, bytes - at) / 16)});
                }
            }
        }
        chunk_list = DeviceBuffer(chunks.size() * sizeof(Chunk));
        chunk_list.copy_from_host(chunks.data(), chunk_list.bytes);
    }
};

template <int Rows, int K, int T>
void probe_shape(const Context& context, const char* name) {
    constexpr int W = flash_next::sliced_k_warps<K, 8>();
    Bank<Rows, K> bank(context.options.budget_mb);
    const int calls = static_cast<int>(bank.copies.size());

    DeviceBuffer x(static_cast<std::size_t>(K) * T * 2);
    fill_activations_kernel<<<256, 256, 0, context.stream>>>(static_cast<__nv_bfloat16*>(x.p),
                                                            static_cast<std::uint64_t>(K) * T);
    CUDA_CHECK(cudaGetLastError());
    DeviceBuffer reference(static_cast<std::size_t>(Rows) * T * 2);
    DeviceBuffer out(static_cast<std::size_t>(Rows) * T * 2);
    // Split-pair scratch, sized for the widest split (W / 2 pairs per row tile) and reused by
    // every call: launches are stream ordered and each tile's last CTA re-zeroes its counter.
    DeviceBuffer pairs(static_cast<std::size_t>(Rows / 16) * (W / 2) * (T / 8) * 32 * 4 *
                       sizeof(float));
    DeviceBuffer counters(static_cast<std::size_t>(Rows / 16) * sizeof(unsigned));
    CUDA_CHECK(cudaMemsetAsync(counters.p, 0, counters.bytes, context.stream));
    const Fp8SlicedSplitScratch scratch{static_cast<float*>(pairs.p),
                                        static_cast<unsigned*>(counters.p)};

    const auto operands = [&](int copy) {
        auto p   = fp8_a16_operands(Tensor(x.p, DType::BF16, {K, T}), bank.copies[copy].weight);
        p.tokens = T;
        return p;
    };
    const auto report = [&](double bytes, const auto& body) {
        bench::TimedGraph graph;
        graph.capture(context.stream, [&](cudaStream_t s) { body(s); });
        const auto timing =
            bench::measure_graph(graph, context.stream, context.options.warmup,
                                 context.options.repeat);
        const double us  = timing.median_us / calls;
        const double gbs = bytes / (us * 1e3);
        return std::pair{us, gbs};
    };
    const auto print = [&](const std::string& label, double us, double gbs, const char* bits) {
        std::printf("%-12s %-3d %-44s %9.2f %9.1f %7.1f  %s\n", name, T, label.c_str(), us, gbs,
                    100.0 * gbs / kDramGBs, bits);
    };

    double best_read = 0.0;
    for (const int requested : {2, 3, 4}) {
        const int grid    = requested * context.sms;
        const auto [us, gbs] = report(bank.bytes_per_call, [&](cudaStream_t s) {
            contiguous_read_kernel<<<grid, 256, 0, s>>>(
                static_cast<const Chunk*>(bank.chunk_list.p),
                static_cast<int>(bank.chunks.size()),
                static_cast<unsigned*>(bank.sink.p));
        });
        best_read = std::max(best_read, gbs);
        print("contiguous read, grid " + std::to_string(grid), us, gbs, "");
    }

    bool have_reference = false;
    const auto run = [&]<class Schedule>(const char* label) {
        using S = Fp8ScheduleInstance<Schedule, K, T>;
        const auto launch = [&](int copy, void* destination, cudaStream_t s) {
            launch_fp8_a16_sliced_k_mma<S>(
                operands(copy), LinearBf16Output{static_cast<__nv_bfloat16*>(destination), Rows},
                LinearIdentityEpilogue{}, s, Fp8IdentityRows{}, scratch);
        };
        void* destination = have_reference ? out.p : reference.p;
        launch(0, destination, context.stream);
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
        const char* bits = "reference";
        if (have_reference) {
            std::vector<std::uint8_t> a(reference.bytes);
            std::vector<std::uint8_t> b(out.bytes);
            reference.copy_to_host(a.data(), a.size());
            out.copy_to_host(b.data(), b.size());
            bits = std::memcmp(a.data(), b.data(), a.size()) == 0 ? "bitwise" : "DIFFERS";
        }
        have_reference    = true;
        const auto [us, gbs] = report(bank.bytes_per_call, [&](cudaStream_t s) {
            for (int copy = 0; copy < calls; ++copy) launch(copy, out.p, s);
        });
        std::string full = std::string(label) + " (" + std::to_string(S::kThreads) + " thr, " +
                           std::to_string(S::kSharedBytes / 1024) + " KB)";
        print(full, us, gbs, bits);
    };

    using Production = Fp8SlicedInstance<T, W, 2>;
    constexpr int kOtherMin = Production::kMinBlocksPerSm == 1 ? 2 : 1;
    run.template operator()<Production>("production S2");
    run.template operator()<Fp8SlicedInstance<T, W, 1>>("stages 1");
    run.template operator()<Fp8A16SlicedKMmaSchedule<W, T, kOtherMin, ops::Cache::ca, ops::Cache::cg,
                                                     Fp8ActivationStage::PaddedZero, 2>>(
        kOtherMin == 1 ? "S2 min blocks 1" : "S2 min blocks 2");
    run.template operator()<Fp8A16SlicedKMmaSchedule<W, T, Production::kMinBlocksPerSm, ops::Cache::cg,
                                                     ops::Cache::cg, Fp8ActivationStage::PaddedZero,
                                                     2>>("S2 activations cg");
    run.template operator()<Fp8A16SlicedKMmaSchedule<W, T, 1, ops::Cache::ca, ops::Cache::cg,
                                                     Fp8ActivationStage::PaddedZero, 2, 2>>(
        "S2 row tiles 2");
    run.template operator()<Fp8A16SlicedKMmaSchedule<W, T, 1, ops::Cache::ca, ops::Cache::cg,
                                                     Fp8ActivationStage::PaddedZero, 1, 2>>(
        "S1 row tiles 2");
    if constexpr (Fp8A16SlicedKMmaSchedule<W, T, 1, ops::Cache::ca, ops::Cache::cg,
                                           Fp8ActivationStage::PaddedZero, 1, 4>::kSharedBytes <=
                  99 * 1024) {
        run.template operator()<Fp8A16SlicedKMmaSchedule<W, T, 1, ops::Cache::ca, ops::Cache::cg,
                                                         Fp8ActivationStage::PaddedZero, 1, 4>>(
            "S1 row tiles 4");
    }
    if constexpr (W % 4 == 0) {
        run.template operator()<Fp8A16SlicedKMmaSchedule<W, T, 2, ops::Cache::ca, ops::Cache::cg,
                                                         Fp8ActivationStage::PaddedZero, 2, 1,
                                                         2>>("S2 pair split 2");
    }
    if constexpr (W % 8 == 0) {
        run.template operator()<Fp8A16SlicedKMmaSchedule<W, T, 2, ops::Cache::ca, ops::Cache::cg,
                                                         Fp8ActivationStage::PaddedZero, 2, 1,
                                                         4>>("S2 pair split 4");
    }
    std::printf("%-12s %-3d %-44s best read %.1f GB/s, %d calls of %.2f MB\n", name, T, "summary",
                best_read, calls, bank.bytes_per_call / 1e6);
}

template <int T>
void probe_all(const Context& context) {
    probe_shape<10240, 2560, T>(context, "gdn.qkv");
    probe_shape<6144, 2560, T>(context, "gdn.z");
    probe_shape<12288, 2560, T>(context, "qsa.q_gate");
    probe_shape<2560, 6144, T>(context, "out[gdn,qsa]");
    probe_shape<320, 10240, T>(context, "hc.down");
    probe_shape<10240, 320, T>(context, "hc.up");
    probe_shape<1280, 2560, T>(context, "shared.gu");
    probe_shape<2560, 640, T>(context, "shared.down");
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        cudaStream_t stream   = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        int device = 0;
        int sms    = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        const Context context{options, stream, sms};
        std::printf("flash_next FP8 small-T probe: %d SMs, %.0f MB of weight copies per shape\n",
                    sms, options.budget_mb);
        std::printf("%-12s %-3s %-44s %9s %9s %7s  %s\n", "shape", "T", "schedule", "us/call",
                    "GB/s", "% 246", "vs production");
        for (const int tokens : options.tokens) {
            if (tokens == 8) probe_all<8>(context);
            if (tokens == 16) probe_all<16>(context);
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "flash_next_fp8_small_t_bench: %s\n", error.what());
        return 1;
    }
}
