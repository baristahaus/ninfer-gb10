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

#include "core/device.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include "ninfer/ops/flash_next_moe.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

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
        const double expert_bytes = static_cast<double>(gate_up[0].streamed_bytes_per_expert +
                                                        down[0].streamed_bytes_per_expert);
        std::printf("flash_next_moe NVFP4 decode route: T=%d rows, %d layer banks of %.2f GB, "
                    "%.3f MB streamed per selected expert\n",
                    options.tokens, options.layers,
                    static_cast<double>(gate_up[0].geometry.bytes + down[0].geometry.bytes) / 1e9,
                    expert_bytes / 1e6);
        std::printf("%-10s %-9s %12s %12s %14s\n", "wide_gate", "distinct", "us/layer", "min us",
                    "routed GB/s");

        for (const bool wide : {false, true}) {
            std::vector<std::pair<double, double>> points;
            for (const int requested : options.distinct) {
                const auto [host_router, distinct] = router_for(options.tokens, requested);
                router.copy_from_host(host_router.data(),
                                      host_router.size() * sizeof(std::uint16_t));
                bench::TimedGraph graph;
                graph.capture(stream, [&](cudaStream_t s) {
                    for (const auto& weights : layers) {
                        ops::flash_next_moe(input_tensor, weights, output_tensor, workspace, s,
                                            nullptr, wide);
                    }
                });
                const auto timing =
                    bench::measure_graph(graph, stream, options.warmup, options.repeat);
                const double per_layer = timing.median_us / options.layers;
                const double gbs       = distinct * expert_bytes / (per_layer * 1e3);
                points.emplace_back(distinct, per_layer);
                std::printf("%-10s %-9d %12.1f %12.1f %14.1f\n", wide ? "on" : "off", distinct,
                            per_layer, timing.min_us / options.layers, gbs);
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
                std::printf("fit wide_gate=%s: %.2f us per distinct expert (%.0f GB/s, %.0f%% of "
                            "%.0f GB/s), %.1f us fixed per layer\n",
                            wide ? "on" : "off", slope, slope_gbs, 100.0 * slope_gbs / kDramGBs,
                            kDramGBs, intercept);
            }
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
