// Flash-Next HyperConnection combine+mix at decode-verify widths with FP8 down/up projections.
//
// Question: where does hyper.combine_mix lose time against its weight bytes? Per call it runs the
// combine + grouped RMSNorm (with the injection partials), the fused FP8 down+SiLU [320,10240]
// and the fused FP8 up+gate-mix [10240,320] (6.6 MB of FP8 codes and row scales). This times, per
// layer, the whole op, each projection launch alone, and a plain 16-byte read of exactly the two
// projections' bytes; the normalization's share is the op minus its two projections. Each layer
// has its own weights (--layers), so nothing is reused from L2 between calls.

#include "core/device.h"
#include "core/weight.h"
#include "ninfer/ops/hyperconnection.h"
#include "ninfer_bench_common.h"
#include "ops/linear/fp8/fp8_flash_next.h"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr int kHidden  = 2560;
constexpr int kStreams = 4;
constexpr int kHyper   = kStreams * kHidden;
constexpr int kRank    = 320;
constexpr double kDramGBs = 246.0; // tools/bench/hardware/gb10.json, measured streaming read

struct Options {
    std::vector<int> tokens{2, 4, 8, 16};
    int layers = 48;
    int warmup = 5;
    int repeat = 30;
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
        } else if (arg == "--layers") {
            options.layers = std::stoi(value());
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(value());
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(value());
        } else {
            throw std::invalid_argument("unknown argument " + std::string(arg));
        }
    }
    for (const int t : options.tokens) {
        if (t < 2 || t > 64) { throw std::invalid_argument("--tokens values in 2..64"); }
    }
    return options;
}

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

        std::vector<bench::PackedQuantizedWeight> downs;
        std::vector<bench::PackedQuantizedWeight> ups;
        for (int layer = 0; layer < options.layers; ++layer) {
            downs.push_back(bench::make_fp8_weight(kRank, kHyper));
            ups.push_back(bench::make_fp8_weight(kHyper, kRank));
        }
        DeviceBuffer norm(static_cast<std::size_t>(kHyper) * 2);
        DeviceBuffer injection_weight(static_cast<std::size_t>(kStreams) * kHyper * 2);
        norm.fill();
        injection_weight.fill();
        std::vector<ops::HyperConnectionWeights> weights;
        for (int layer = 0; layer < options.layers; ++layer) {
            weights.push_back({
                .norm      = Tensor(norm.p, DType::BF16, {kHyper}),
                .down      = downs[layer].weight,
                .up        = ups[layer].weight,
                .injection = bf16_weight(injection_weight, kStreams, kHyper),
            });
        }
        const double projection_bytes =
            static_cast<double>(downs[0].weight.payload_bytes + ups[0].weight.payload_bytes);

        // Chunk lists: the two projections' payloads of every layer, 64 KB pieces.
        std::vector<Chunk> chunks;
        for (int layer = 0; layer < options.layers; ++layer) {
            for (const Weight* w : {&downs[layer].weight, &ups[layer].weight}) {
                const auto* base = static_cast<const std::uint8_t*>(w->payload);
                const std::uint64_t bytes = w->payload_bytes / 16 * 16;
                for (std::uint64_t at = 0; at < bytes; at += 65536) {
                    chunks.push_back({reinterpret_cast<const uint4*>(base + at),
                                      static_cast<unsigned>(std::min<std::uint64_t>(65536, bytes - at) /
                                                            16)});
                }
            }
        }
        DeviceBuffer chunk_list(chunks.size() * sizeof(Chunk));
        chunk_list.copy_from_host(chunks.data(), chunk_list.bytes);
        DeviceBuffer sink(4096 * sizeof(unsigned));

        std::printf("flash_next HyperConnection combine+mix, FP8 down/up: %d layers, %.2f MB of "
                    "projection bytes per layer, %d SMs\n",
                    options.layers, projection_bytes / 1e6, sms);
        std::printf("%-4s %-34s %10s %10s %8s\n", "T", "case", "us/layer", "GB/s", "% 246");
        const auto report = [&](int tokens, const char* name, double bytes, const auto& body) {
            bench::TimedGraph graph;
            graph.capture(stream, [&](cudaStream_t s) { body(s); });
            const auto timing = bench::measure_graph(graph, stream, options.warmup, options.repeat);
            const double us   = timing.median_us / options.layers;
            const double gbs  = bytes > 0 ? bytes / (us * 1e3) : 0.0;
            std::printf("%-4d %-34s %10.2f %10.1f %8.1f\n", tokens, name, us, gbs,
                        100.0 * gbs / kDramGBs);
            return us;
        };

        for (const int requested : {2, 3, 4}) {
            const int grid = requested * sms;
            report(0, ("contig projection bytes, grid " + std::to_string(grid)).c_str(),
                   projection_bytes, [&](cudaStream_t s) {
                       contiguous_read_kernel<<<grid, 256, 0, s>>>(
                           static_cast<const Chunk*>(chunk_list.p),
                           static_cast<int>(chunks.size()), static_cast<unsigned*>(sink.p));
                   });
        }

        for (const int tokens : options.tokens) {
            DeviceBuffer hyper(static_cast<std::size_t>(kHyper) * tokens * 2);
            DeviceBuffer block_output(static_cast<std::size_t>(kHidden) * tokens * 2);
            DeviceBuffer previous_injection(static_cast<std::size_t>(kStreams) * tokens * 2);
            DeviceBuffer block_input(static_cast<std::size_t>(kHidden) * tokens * 2);
            DeviceBuffer injection(static_cast<std::size_t>(kStreams) * tokens * 2);
            DeviceBuffer normalized(static_cast<std::size_t>(kHyper) * tokens * 2);
            DeviceBuffer low_rank(static_cast<std::size_t>(kRank) * tokens * 2);
            DeviceBuffer partials(static_cast<std::size_t>(kStreams) * kStreams * tokens * 4);
            for (DeviceBuffer* buffer : {&hyper, &block_output, &previous_injection, &block_input,
                                         &injection, &normalized, &low_rank, &partials}) {
                buffer->fill();
            }
            Tensor hyper_t(hyper.p, DType::BF16, {kHyper, tokens});
            Tensor block_output_t(block_output.p, DType::BF16, {kHidden, tokens});
            Tensor previous_injection_t(previous_injection.p, DType::BF16, {kStreams, tokens});
            Tensor block_input_t(block_input.p, DType::BF16, {kHidden, tokens});
            Tensor injection_t(injection.p, DType::BF16, {kStreams, tokens});
            Tensor normalized_t(normalized.p, DType::BF16, {kHyper, tokens});
            Tensor low_rank_t(low_rank.p, DType::BF16, {kRank, tokens});
            Tensor partials_t(partials.p, DType::FP32, {kStreams * kStreams, tokens});
            WorkspaceArena workspace(ops::hyperconnection_mix_workspace_capacity_bytes(tokens, true));

            const double op = report(tokens, "combine_mix (whole op)", projection_bytes,
                                     [&](cudaStream_t s) {
                                         for (const auto& w : weights) {
                                             ops::hyperconnection_combine_mix(
                                                 hyper_t, block_output_t, previous_injection_t, w,
                                                 block_input_t, &injection_t, workspace, s);
                                         }
                                     });
            const double down = report(tokens, "fp8 down+SiLU alone",
                                       static_cast<double>(downs[0].weight.payload_bytes),
                                       [&](cudaStream_t s) {
                                           for (const auto& w : weights) {
                                               ops::detail::flash_next::launch_fp8_hc_down_silu(
                                                   normalized_t, w.down, low_rank_t, s);
                                           }
                                       });
            const double up = report(tokens, "fp8 up+mix alone",
                                     static_cast<double>(ups[0].weight.payload_bytes),
                                     [&](cudaStream_t s) {
                                         for (const auto& w : weights) {
                                             ops::detail::flash_next::launch_fp8_hc_up_mix(
                                                 low_rank_t, w.up, normalized_t, &partials_t,
                                                 block_input_t, &injection_t, s);
                                         }
                                     });
            std::printf("%-4d %-34s %10.2f\n", tokens, "remainder (norm + boundaries)",
                        op - down - up);
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "flash_next_hc_bench: %s\n", error.what());
        return 1;
    }
}
