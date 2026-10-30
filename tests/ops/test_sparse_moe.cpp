#include "core/weight.h"
#include "ninfer/ops/sparse_moe.h"

#include "ops/exl3_ref.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// SparseMoe is the exact qwen3_6_35b_a3b post-mixer Op. qwen3_6_27b has a dense SwiGLU
// post-mixer and therefore contributes no second SparseMoe geometry.
constexpr std::int32_t kHidden         = 2048;
constexpr std::int32_t kExperts        = 256;
constexpr std::int32_t kTopK           = 8;
constexpr std::int32_t kIntermediate   = 512;
constexpr std::int32_t kExpertGateRows = 2 * kIntermediate;
constexpr std::int32_t kRoutedGateRows = kExperts * kExpertGateRows;
constexpr std::int32_t kRoutedDownRows = kExperts * kHidden;
constexpr std::int32_t kSharedGateRows = 2 * kIntermediate;

// All registered variants consume represented BF16 activations and expose one BF16 destination.
// The bound belongs to that A16 compute profile, not to a codec, T, private route, or schedule.
// The limits retain modest headroom over the measured maxima from the complete case matrix:
// rel-L2 1.117e-2 and pointwise 3.687e-3.
constexpr ReductionCriterion kSparseMoeA16Tolerance{
    /*relative_l2*/ 1.2e-2,
    /*gross_absolute*/ 4.0e-3,
    /*gross_relative_to_max_reference*/ 0.0,
};

// The EXL3 fixture's experts decode to FP16 codebook values under near-unit FP16 transform
// vectors, so its destination magnitudes (~10^2) sit far above the groupwise fixture's
// (~10^-1) and one BF16 destination quantum is ~0.25 there. This criterion keeps the A16
// profile's relative-L2 and absolute-floor bounds and adds the max-reference-relative term
// that bounds the destination's own quantization (2^-8 of the maximum reference). Measured
// maxima: rel-L2 1.69e-3, maximum error 0.249 at a maximum reference of 93.6.
constexpr ReductionCriterion kSparseMoeExl3A16Tolerance{
    /*relative_l2*/ 1.2e-2,
    /*gross_absolute*/ 4.0e-3,
    /*gross_relative_to_max_reference*/ 4.0e-3,
};

constexpr std::size_t kOutputGuardBytes = 256;
constexpr std::uint8_t kOutputGuardByte = 0xa5;

struct QuantGeometry {
    std::int32_t group;
    std::size_t code_bytes_per_group;
    std::size_t high_bytes_per_group;
};

QuantGeometry quant_geometry(QType qtype) {
    switch (qtype) {
    case QType::Q4_G64_FP16:
        return {64, 32, 0};
    case QType::Q5_G64_FP16:
        return {64, 32, 8};
    case QType::Q6_G64_FP16:
        return {64, 32, 16};
    case QType::Q8_G32_FP16:
        return {32, 32, 0};
    default:
        throw std::invalid_argument("sparse_moe test: unsupported codec");
    }
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

int compare_output(const std::string& label, const std::vector<double>& actual,
                   const std::vector<double>& reference, const ReductionCriterion& criterion) {
    return verify_reduction(label, actual, reference, criterion);
}

class GuardedBf16Output {
public:
    explicit GuardedBf16Output(std::size_t words)
        : storage_(words * sizeof(std::uint16_t), kOutputGuardBytes, kOutputGuardByte),
          words_(words) {}

    void* data() noexcept { return storage_.data(); }

    const void* data() const noexcept { return storage_.data(); }

    std::vector<double> values() const { return from_device_bf16(data(), words_); }

    int verify_guards(const std::string& label) const { return storage_.verify_guards(label); }

private:
    GuardedDeviceBuffer storage_;
    std::size_t words_;
};

class DeviceRowSplit {
public:
    DeviceRowSplit(QType qtype, std::int32_t rows, std::int32_t columns)
        : qtype_(qtype), rows_(rows), columns_(columns), geometry_(quant_geometry(qtype)),
          groups_per_row_(columns / geometry_.group),
          code_row_bytes_(static_cast<std::size_t>(groups_per_row_) *
                          geometry_.code_bytes_per_group),
          high_row_bytes_(static_cast<std::size_t>(groups_per_row_) *
                          geometry_.high_bytes_per_group),
          scale_row_bytes_(static_cast<std::size_t>(groups_per_row_) * sizeof(std::uint16_t)),
          codes_(static_cast<std::size_t>(rows) * code_row_bytes_),
          scales_(static_cast<std::size_t>(rows) * scale_row_bytes_) {
        codes_.fill(0);
        scales_.fill(0);
        if (high_row_bytes_ != 0) {
            high_ =
                std::make_unique<DeviceBuffer>(static_cast<std::size_t>(rows) * high_row_bytes_);
            high_->fill(0);
        }
    }

    void copy_rows(const quantized_weight::PackedWeight& source, std::int32_t destination_row) {
        const std::int32_t source_rows = source.weight.n;
        if (source.weight.qtype != qtype_ || source.weight.k != columns_ || source_rows < 1 ||
            destination_row < 0 || destination_row > rows_ - source_rows) {
            throw std::invalid_argument("sparse_moe test: invalid packed row copy");
        }
        codes_.copy_from_host(source.payload.data(),
                              static_cast<std::size_t>(source_rows) * code_row_bytes_,
                              static_cast<std::size_t>(destination_row) * code_row_bytes_);
        if (high_row_bytes_ != 0) {
            high_->copy_from_host(source.payload.data() + source.high_plane_offset,
                                  static_cast<std::size_t>(source_rows) * high_row_bytes_,
                                  static_cast<std::size_t>(destination_row) * high_row_bytes_);
        }
        scales_.copy_from_host(source.payload.data() + source.scale_plane_offset,
                               static_cast<std::size_t>(source_rows) * scale_row_bytes_,
                               static_cast<std::size_t>(destination_row) * scale_row_bytes_);
    }

    Weight weight() const {
        Weight result{};
        result.payload          = codes_.p;
        result.payload_bytes    = codes_.bytes + (high_ ? high_->bytes : 0) + scales_.bytes;
        result.high_plane_bytes = high_ ? high_->bytes : 0;
        result.qtype            = qtype_;
        result.group_size       = static_cast<std::uint32_t>(geometry_.group);
        result.qdata            = codes_.p;
        result.qhigh            = high_ ? high_->p : nullptr;
        result.scales           = scales_.p;
        result.n                = rows_;
        result.k                = columns_;
        result.group            = geometry_.group;
        result.layout           = QuantLayout::RowSplit;
        result.scale_dtype      = DType::FP16;
        result.ndim             = 2;
        result.shape[0]         = rows_;
        result.shape[1]         = columns_;
        result.padded_shape[0]  = rows_;
        result.padded_shape[1]  = columns_;
        return result;
    }

    int verify_rows(const std::string& label, const quantized_weight::PackedWeight& source,
                    std::int32_t destination_row) const {
        const std::int32_t source_rows = source.weight.n;
        int failures                   = 0;
        failures += verify_plane(label + " code", codes_, source.payload.data(),
                                 static_cast<std::size_t>(destination_row) * code_row_bytes_,
                                 static_cast<std::size_t>(source_rows) * code_row_bytes_);
        if (high_row_bytes_ != 0) {
            failures += verify_plane(label + " high", *high_,
                                     source.payload.data() + source.high_plane_offset,
                                     static_cast<std::size_t>(destination_row) * high_row_bytes_,
                                     static_cast<std::size_t>(source_rows) * high_row_bytes_);
        }
        failures += verify_plane(label + " scale", scales_,
                                 source.payload.data() + source.scale_plane_offset,
                                 static_cast<std::size_t>(destination_row) * scale_row_bytes_,
                                 static_cast<std::size_t>(source_rows) * scale_row_bytes_);
        return failures;
    }

private:
    static int verify_plane(const std::string& label, const DeviceBuffer& device,
                            const std::uint8_t* expected, std::size_t offset, std::size_t bytes) {
        std::vector<std::uint8_t> actual(bytes);
        device.copy_to_host(actual.data(), bytes, offset);
        if (std::memcmp(actual.data(), expected, bytes) == 0) { return 0; }
        std::cerr << label << ": persistent weight was modified\n";
        return 1;
    }

    QType qtype_;
    std::int32_t rows_;
    std::int32_t columns_;
    QuantGeometry geometry_;
    std::int32_t groups_per_row_;
    std::size_t code_row_bytes_;
    std::size_t high_row_bytes_;
    std::size_t scale_row_bytes_;
    DeviceBuffer codes_;
    std::unique_ptr<DeviceBuffer> high_;
    DeviceBuffer scales_;
};

Weight dense_bf16_weight(void* data, std::int32_t rows, std::int32_t columns) {
    Weight result{};
    result.payload         = data;
    result.payload_bytes   = static_cast<std::uint64_t>(rows) * columns * sizeof(std::uint16_t);
    result.qtype           = QType::BF16;
    result.qdata           = data;
    result.n               = rows;
    result.k               = columns;
    result.layout          = QuantLayout::Contiguous;
    result.ndim            = 2;
    result.shape[0]        = rows;
    result.shape[1]        = columns;
    result.padded_shape[0] = rows;
    result.padded_shape[1] = columns;
    return result;
}

template <std::int32_t TopK>
struct RoutePatternT {
    std::array<int, TopK> selected;
    int tied_excluded = -1;
};

using RoutePattern = RoutePatternT<kTopK>;

constexpr std::array<RoutePattern, 3> kRoutePatterns{{
    {{{255, 0, 17, 31, 63, 127, 191, 223}}, -1},
    {{{0, 17, 31, 63, 127, 191, 223, 254}}, 255},
    {{{223, 191, 127, 63, 31, 17, 0, 255}}, -1},
}};

// The second admitted geometry's route patterns: the same three scenarios (the intended route,
// the tie boundary with the higher id excluded, and the reverse ordering) over 512 experts and
// top-10, with the two extra ids drawn from the upper half of the expert range.
constexpr std::array<int, 11> kExl3RouteExpertIds125B{
    {0, 17, 31, 63, 127, 191, 223, 254, 255, 383, 511}};
constexpr std::array<RoutePatternT<10>, 3> kRoutePatterns125B{{
    {{{511, 0, 17, 31, 63, 127, 191, 223, 254, 383}}, -1},
    {{{0, 17, 31, 63, 127, 191, 223, 254, 255, 383}}, 511},
    {{{223, 191, 127, 63, 31, 17, 0, 511, 383, 255}}, -1},
}};

template <std::int32_t Hidden, std::size_t PatternCount>
std::vector<float> make_input_rows(int pattern) {
    std::vector<float> input(Hidden);
    input[0] = 1.0f;
    for (int column = 1; column < Hidden; ++column) {
        input[column] = 0.025f + static_cast<float>((column * 7 + pattern * 11) % 19) * 0.002f;
    }
    for (std::size_t marker = 0; marker < PatternCount; ++marker) {
        input[Hidden - static_cast<int>(PatternCount) + static_cast<int>(marker)] = 0.0f;
    }
    input[Hidden - static_cast<int>(PatternCount) + pattern] = 1.0f;
    round_to_bf16(input);
    return input;
}

std::vector<float> make_input(int pattern) {
    return make_input_rows<kHidden, kRoutePatterns.size()>(pattern);
}

template <std::int32_t Hidden>
std::vector<float> make_residual_rows(int pattern) {
    std::vector<float> residual(Hidden);
    for (int row = 0; row < Hidden; ++row) {
        residual[row] = 0.125f + static_cast<float>((row * 5 + pattern * 13) % 23) * 0.003f;
    }
    round_to_bf16(residual);
    return residual;
}

std::vector<float> make_residual(int pattern) { return make_residual_rows<kHidden>(pattern); }

template <std::int32_t Hidden, std::int32_t Experts, std::int32_t TopK, std::size_t PatternCount>
std::vector<float> make_router_rows(
    const std::array<RoutePatternT<TopK>, PatternCount>& patterns) {
    std::vector<float> router(static_cast<std::size_t>(Experts + 1) * Hidden);
    for (int row = 0; row < Experts + 1; ++row) {
        for (int column = 0; column < Hidden; ++column) {
            const int pattern = (column * 13 + 5) % 17 - 8;
            router[static_cast<std::size_t>(row) * Hidden + column] =
                static_cast<float>(pattern) * 0.001f;
        }
    }
    for (int expert = 0; expert < Experts; ++expert) {
        router[static_cast<std::size_t>(expert) * Hidden] -= 8.0f;
    }
    for (std::size_t pattern = 0; pattern < PatternCount; ++pattern) {
        const int marker = Hidden - static_cast<int>(PatternCount) + static_cast<int>(pattern);
        const RoutePatternT<TopK>& route = patterns[pattern];
        for (int rank = 0; rank < TopK; ++rank) {
            const float score =
                rank == TopK - 1 && route.tied_excluded >= 0 ? 2.0f : 4.0f - 0.25f * rank;
            router[static_cast<std::size_t>(route.selected[rank]) * Hidden + marker] +=
                score + 8.0f;
        }
        if (route.tied_excluded >= 0) {
            router[static_cast<std::size_t>(route.tied_excluded) * Hidden + marker] += 10.0f;
        }
    }
    router[static_cast<std::size_t>(Experts) * Hidden] += 0.375f;
    round_to_bf16(router);
    return router;
}

std::vector<float> make_router() { return make_router_rows<kHidden, kExperts, kTopK>(kRoutePatterns); }

std::vector<float> make_gate_up(std::int32_t rows, std::int32_t columns, std::uint32_t seed,
                                float expert_factor) {
    std::vector<float> source(static_cast<std::size_t>(rows) * columns);
    const std::int32_t split = rows / 2;
    for (std::int32_t row = 0; row < rows; ++row) {
        const float bias       = row < split ? 0.75f : 1.15f;
        const float row_factor = 1.0f + static_cast<float>((row + seed) % 7) * 0.025f;
        for (std::int32_t column = 0; column < columns; ++column) {
            const int pattern = static_cast<int>((row * 11LL + column * 5LL + seed) % 15) - 7;
            source[static_cast<std::size_t>(row) * columns + column] =
                0.008f * expert_factor * row_factor * (static_cast<float>(pattern) + bias);
        }
    }
    return source;
}

std::vector<float> make_down(std::int32_t rows, std::int32_t columns, std::uint32_t seed,
                             float expert_factor) {
    std::vector<float> source(static_cast<std::size_t>(rows) * columns);
    for (std::int32_t row = 0; row < rows; ++row) {
        const float bias = 0.45f + static_cast<float>((row + seed) % 5) * 0.08f;
        for (std::int32_t column = 0; column < columns; ++column) {
            const int pattern = static_cast<int>((row * 7LL + column * 13LL + seed) % 17) - 8;
            source[static_cast<std::size_t>(row) * columns + column] =
                0.007f * expert_factor * (static_cast<float>(pattern) + bias);
        }
    }
    return source;
}

struct HostExpert {
    int id;
    quantized_weight::PackedWeight gate_up;
    quantized_weight::PackedWeight down;
};

template <class Expert>
const Expert& find_expert(const std::vector<Expert>& experts, int id) {
    const auto found = std::find_if(experts.begin(), experts.end(),
                                    [id](const Expert& expert) { return expert.id == id; });
    if (found == experts.end()) {
        throw std::logic_error("sparse_moe oracle selected an unpopulated expert");
    }
    return *found;
}

template <class Function>
void parallel_rows(std::int32_t rows, Function&& function) {
    const unsigned available   = std::max(1U, std::thread::hardware_concurrency());
    const std::int32_t threads = std::min(rows, static_cast<std::int32_t>(available));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));
    for (std::int32_t thread = 0; thread < threads; ++thread) {
        const std::int32_t begin =
            static_cast<std::int32_t>((static_cast<std::int64_t>(rows) * thread) / threads);
        const std::int32_t end =
            static_cast<std::int32_t>((static_cast<std::int64_t>(rows) * (thread + 1)) / threads);
        workers.emplace_back([&, begin, end] {
            for (std::int32_t row = begin; row < end; ++row) { function(row); }
        });
    }
    for (std::thread& worker : workers) { worker.join(); }
}

double dot_fp64(const std::vector<float>& matrix, std::int32_t row, std::int32_t columns,
                const std::vector<double>& input) {
    const float* weights = matrix.data() + static_cast<std::size_t>(row) * columns;
    double result        = 0.0;
    for (std::int32_t column = 0; column < columns; ++column) {
        result += static_cast<double>(weights[column]) * input[column];
    }
    return result;
}

// The public routing, shared by every SparseMoe oracle: FP64 router scores, the ordered top-k
// with softmax weights, and the shared-expert sigmoid scale. Throws when the router does not
// produce the test's intended ordered top-k.
template <std::int32_t TopK>
struct SparseMoeRoutingT {
    std::array<int, TopK> selected{};
    std::array<double, TopK> route_weight{};
    double shared_scale = 0.0;
};

using SparseMoeRouting = SparseMoeRoutingT<kTopK>;

template <std::int32_t Hidden, std::int32_t Experts, std::int32_t TopK>
SparseMoeRoutingT<TopK> route_sparse_moe_rows(const std::vector<float>& input,
                                              const std::vector<float>& router,
                                              const RoutePatternT<TopK>& intended_route) {
    const std::vector<double> x(input.begin(), input.end());
    std::vector<double> scores(Experts + 1);
    parallel_rows(Experts + 1,
                  [&](std::int32_t row) { scores[row] = dot_fp64(router, row, Hidden, x); });

    std::vector<int> ranked(Experts);
    std::iota(ranked.begin(), ranked.end(), 0);
    std::sort(ranked.begin(), ranked.end(), [&](int left, int right) {
        return scores[left] > scores[right] || (scores[left] == scores[right] && left < right);
    });

    SparseMoeRoutingT<TopK> routing;
    double route_denominator = 0.0;
    for (int route = 0; route < TopK; ++route) {
        routing.selected[route] = ranked[route];
        routing.route_weight[route] =
            std::exp(scores[routing.selected[route]] - scores[routing.selected[0]]);
        route_denominator += routing.route_weight[route];
    }
    for (double& weight : routing.route_weight) { weight /= route_denominator; }
    if (routing.selected != intended_route.selected) {
        throw std::logic_error("sparse_moe test router did not create its intended ordered top-k");
    }
    routing.shared_scale = 1.0 / (1.0 + std::exp(-scores[Experts]));
    return routing;
}

SparseMoeRouting route_sparse_moe(const std::vector<float>& input, const std::vector<float>& router,
                                  const RoutePattern& intended_route) {
    return route_sparse_moe_rows<kHidden, kExperts, kTopK>(input, router, intended_route);
}

// The groupwise SparseMoe oracle. It directly evaluates the complete public formula from
// represented BF16 values and independently decoded logical weights, with FP64 accumulation
// throughout. It has no production route, staging cast, workspace dtype, reduction tree, or
// output rounding.
std::vector<double> sparse_moe_oracle(const std::vector<float>& input,
                                      const std::vector<float>& residual,
                                      const std::vector<float>& router,
                                      const std::vector<HostExpert>& experts,
                                      const quantized_weight::PackedWeight& shared_gate_up,
                                      const quantized_weight::PackedWeight& shared_down,
                                      const RoutePattern& intended_route) {
    const SparseMoeRouting routing = route_sparse_moe(input, router, intended_route);
    const std::vector<double> x(input.begin(), input.end());

    std::array<std::vector<double>, kTopK> routed_activation;
    for (int route = 0; route < kTopK; ++route) {
        routed_activation[route].resize(kIntermediate);
        const HostExpert& expert = find_expert(experts, routing.selected[route]);
        parallel_rows(kIntermediate, [&](std::int32_t row) {
            const double gate = dot_fp64(expert.gate_up.dequant, row, kHidden, x);
            const double up   = dot_fp64(expert.gate_up.dequant, kIntermediate + row, kHidden, x);
            routed_activation[route][row] = (gate / (1.0 + std::exp(-gate))) * up;
        });
    }

    std::vector<double> shared_activation(kIntermediate);
    parallel_rows(kIntermediate, [&](std::int32_t row) {
        const double gate      = dot_fp64(shared_gate_up.dequant, row, kHidden, x);
        const double up        = dot_fp64(shared_gate_up.dequant, kIntermediate + row, kHidden, x);
        shared_activation[row] = (gate / (1.0 + std::exp(-gate))) * up;
    });

    std::vector<double> output(kHidden);
    parallel_rows(kHidden, [&](std::int32_t row) {
        double value = static_cast<double>(residual[row]);
        for (int route = 0; route < kTopK; ++route) {
            const HostExpert& expert = find_expert(experts, routing.selected[route]);
            value += routing.route_weight[route] *
                     dot_fp64(expert.down.dequant, row, kIntermediate, routed_activation[route]);
        }
        value += routing.shared_scale *
                 dot_fp64(shared_down.dequant, row, kIntermediate, shared_activation);
        output[row] = value;
    });
    return output;
}

struct CodecProfile {
    const char* name;
    QType routed_gate_up;
    QType routed_down;
    std::span<const std::int32_t> token_cases;
    bool verify_graph_replay;
};

class SparseMoeFixture {
public:
    explicit SparseMoeFixture(const CodecProfile& profile)
        : profile_(profile), router_(make_router()), router_bits_(bf16_bits(router_)),
          device_router_(to_device(router_bits_)),
          routed_gate_(profile.routed_gate_up, kRoutedGateRows, kHidden),
          routed_down_(profile.routed_down, kRoutedDownRows, kIntermediate),
          shared_gate_(QType::Q8_G32_FP16, kSharedGateRows, kHidden),
          shared_down_device_(QType::Q8_G32_FP16, kHidden, kIntermediate) {
        for (int pattern = 0; pattern < static_cast<int>(kRoutePatterns.size()); ++pattern) {
            inputs_.push_back(make_input(pattern));
            residuals_.push_back(make_residual(pattern));
        }

        std::vector<int> expert_ids;
        for (const RoutePattern& route : kRoutePatterns) {
            for (int expert : route.selected) {
                if (std::find(expert_ids.begin(), expert_ids.end(), expert) == expert_ids.end()) {
                    expert_ids.push_back(expert);
                }
            }
        }
        std::sort(expert_ids.begin(), expert_ids.end());
        for (int expert : expert_ids) {
            const float factor = 0.8f + static_cast<float>((expert * 3) % 11) * 0.045f;
            auto gate_up       = quantized_weight::pack_row_split_lowbit(
                make_gate_up(kExpertGateRows, kHidden, 100U + static_cast<std::uint32_t>(expert),
                                   factor),
                kExpertGateRows, kHidden, profile.routed_gate_up);
            auto down = quantized_weight::pack_row_split_lowbit(
                make_down(kHidden, kIntermediate, 300U + static_cast<std::uint32_t>(expert),
                          factor),
                kHidden, kIntermediate, profile.routed_down);
            routed_gate_.copy_rows(gate_up, expert * kExpertGateRows);
            routed_down_.copy_rows(down, expert * kHidden);
            experts_.push_back({expert, std::move(gate_up), std::move(down)});
        }

        shared_gate_host_ = quantized_weight::pack_q8_g32_row_split(
            make_gate_up(kSharedGateRows, kHidden, 0x512U, 0.93f), kSharedGateRows, kHidden);
        shared_down_host_ = quantized_weight::pack_q8_g32_row_split(
            make_down(kHidden, kIntermediate, 0x731U, 0.87f), kHidden, kIntermediate);
        shared_gate_.copy_rows(shared_gate_host_, 0);
        shared_down_device_.copy_rows(shared_down_host_, 0);

        for (int pattern = 0; pattern < static_cast<int>(kRoutePatterns.size()); ++pattern) {
            references_.push_back(sparse_moe_oracle(inputs_[pattern], residuals_[pattern], router_,
                                                    experts_, shared_gate_host_, shared_down_host_,
                                                    kRoutePatterns[pattern]));
        }
    }

    int run(std::int32_t tokens, int first_pattern, bool graph_replay) {
        const std::string label = std::string(profile_.name) + " T=" + std::to_string(tokens);
        std::vector<float> input(static_cast<std::size_t>(kHidden) * tokens);
        std::vector<float> residual(static_cast<std::size_t>(kHidden) * tokens);
        std::vector<double> reference(static_cast<std::size_t>(kHidden) * tokens);
        for (std::int32_t token = 0; token < tokens; ++token) {
            const int pattern = (first_pattern + token) % static_cast<int>(kRoutePatterns.size());
            std::copy(inputs_[pattern].begin(), inputs_[pattern].end(),
                      input.begin() + static_cast<std::size_t>(token) * kHidden);
            std::copy(residuals_[pattern].begin(), residuals_[pattern].end(),
                      residual.begin() + static_cast<std::size_t>(token) * kHidden);
            std::copy(references_[pattern].begin(), references_[pattern].end(),
                      reference.begin() + static_cast<std::size_t>(token) * kHidden);
        }

        const std::vector<std::uint16_t> input_bits    = bf16_bits(input);
        const std::vector<std::uint16_t> residual_bits = bf16_bits(residual);
        DeviceBuffer device_input                      = to_device(input_bits);
        DeviceBuffer residual_seed                     = to_device(residual_bits);
        GuardedBf16Output destination_storage(residual.size());
        cuda_check(cudaMemcpy(destination_storage.data(), residual_seed.p, residual_seed.bytes,
                              cudaMemcpyDeviceToDevice),
                   "seed SparseMoe destination");

        const ops::SparseMoeWeights weights{
            dense_bf16_weight(device_router_.p, kExperts + 1, kHidden),
            routed_gate_.weight(),
            routed_down_.weight(),
            shared_gate_.weight(),
            shared_down_device_.weight(),
        };
        Tensor x(device_input.p, DType::BF16, {kHidden, tokens});
        Tensor destination(destination_storage.data(), DType::BF16, {kHidden, tokens});
        const std::size_t workspace_bytes = ops::sparse_moe_workspace_capacity_bytes(
            weights.routed_gate_up.qtype, weights.routed_down.qtype, tokens, tokens);
        WorkspaceArena workspace(workspace_bytes);

        if (graph_replay) {
            cudaStream_t stream  = nullptr;
            cudaGraph_t graph    = nullptr;
            cudaGraphExec_t exec = nullptr;
            cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                       "create SparseMoe graph stream");
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                       "begin SparseMoe graph capture");
            cuda_check(cudaMemcpyAsync(destination.data, residual_seed.p, destination.bytes(),
                                       cudaMemcpyDeviceToDevice, stream),
                       "capture SparseMoe residual seed");
            ops::sparse_moe(x, weights, ops::SparseMoeEpilogue::AddResidual, destination, workspace,
                            stream);
            cuda_check(cudaStreamEndCapture(stream, &graph), "end SparseMoe graph capture");
            cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
                       "instantiate SparseMoe graph");
            cuda_check(cudaGraphLaunch(exec, stream), "launch SparseMoe graph");
            cuda_check(cudaGraphLaunch(exec, stream), "replay SparseMoe graph");
            cuda_check(cudaStreamSynchronize(stream), "synchronize SparseMoe graph");
            cuda_check(cudaGraphExecDestroy(exec), "destroy SparseMoe graph executable");
            cuda_check(cudaGraphDestroy(graph), "destroy SparseMoe graph");
            cuda_check(cudaStreamDestroy(stream), "destroy SparseMoe graph stream");
        } else {
            ops::sparse_moe(x, weights, ops::SparseMoeEpilogue::AddResidual, destination, workspace,
                            nullptr);
            cuda_synchronize();
        }

        int failures =
            compare_output(label, destination_storage.values(), reference, kSparseMoeA16Tolerance);
        failures += destination_storage.verify_guards(label);
        failures +=
            verify_exact((label + " input preservation").c_str(),
                         from_device<std::uint16_t>(device_input, input_bits.size()), input_bits);
        if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
            std::cerr << label << ": workspace query/execution high-water mismatch\n";
            ++failures;
        }
        return failures;
    }

    int verify_persistent_inputs() const {
        int failures = 0;
        failures += verify_exact((std::string(profile_.name) + " router preservation").c_str(),
                                 from_device<std::uint16_t>(device_router_, router_bits_.size()),
                                 router_bits_);
        for (const HostExpert& expert : experts_) {
            failures += routed_gate_.verify_rows(
                std::string(profile_.name) + " routed gate expert " + std::to_string(expert.id),
                expert.gate_up, expert.id * kExpertGateRows);
            failures += routed_down_.verify_rows(
                std::string(profile_.name) + " routed down expert " + std::to_string(expert.id),
                expert.down, expert.id * kHidden);
        }
        failures += shared_gate_.verify_rows(std::string(profile_.name) + " shared gate",
                                             shared_gate_host_, 0);
        failures += shared_down_device_.verify_rows(std::string(profile_.name) + " shared down",
                                                    shared_down_host_, 0);
        return failures;
    }

private:
    const CodecProfile& profile_;
    std::vector<float> router_;
    std::vector<std::uint16_t> router_bits_;
    DeviceBuffer device_router_;
    DeviceRowSplit routed_gate_;
    DeviceRowSplit routed_down_;
    DeviceRowSplit shared_gate_;
    DeviceRowSplit shared_down_device_;
    std::vector<std::vector<float>> inputs_;
    std::vector<std::vector<float>> residuals_;
    std::vector<HostExpert> experts_;
    quantized_weight::PackedWeight shared_gate_host_;
    quantized_weight::PackedWeight shared_down_host_;
    std::vector<std::vector<double>> references_;
};

int run_profile(const CodecProfile& profile) {
    SparseMoeFixture fixture(profile);
    int failures        = 0;
    std::size_t witness = 0;
    for (std::size_t index = 0; index < profile.token_cases.size(); ++index) {
        const std::int32_t tokens = profile.token_cases[index];
        witness =
            std::max(witness, ops::sparse_moe_workspace_capacity_bytes(
                                  profile.routed_gate_up, profile.routed_down, tokens, tokens));
        // Decode starts with the exact top-8 boundary tie; multi-token cases cycle the tie,
        // high/low expert ids, and a different ordering of the same experts.
        failures +=
            fixture.run(tokens, index == 0 ? 1 : 0, profile.verify_graph_replay && index == 1);
    }
    const std::size_t interval = ops::sparse_moe_workspace_capacity_bytes(
        profile.routed_gate_up, profile.routed_down, 1, profile.token_cases.back());
    if (interval != witness) {
        std::cerr << profile.name << ": interval workspace capacity missed a route witness\n";
        ++failures;
    }
    failures += fixture.verify_persistent_inputs();
    return failures;
}

// ---------------------------------------------------------------------------
// SP-7 EXL3 (single-token decode route)
// ---------------------------------------------------------------------------

// The route patterns select these nine experts; each resolves its own bitrate and trellis
// extent through the runtime directory, and id % 4 covers every rate of both patterns below.
constexpr std::array<int, 9> kExl3RouteExpertIds{{0, 17, 31, 63, 127, 191, 223, 254, 255}};
constexpr std::array<int, 4> kExl3McgRates{{2, 4, 8, 16}};
constexpr std::array<int, 4> kExl3Mul1Rates{{3, 5, 7, 2}};  // the half-integer rates are MUL1-only

std::uint16_t exl3_ring_word(std::uint32_t seed, std::size_t index) {
    std::uint32_t state = seed * 2654435761u + static_cast<std::uint32_t>(index) * 40503u;
    state ^= state >> 13;
    state *= 0x5bd1e995u;
    state ^= state >> 15;
    return static_cast<std::uint16_t>(state >> 16);
}

// One expert projection's ring words: (k/16) * (n/16) tiles of 8 * 2K int16 values.
std::vector<std::uint16_t> make_exl3_trellis(std::int32_t k, std::int32_t n, int k2,
                                             std::uint32_t seed) {
    const std::size_t words = static_cast<std::size_t>(k / 16) * static_cast<std::size_t>(n / 16) *
                              static_cast<std::size_t>(8 * k2);
    std::vector<std::uint16_t> trellis(words);
    for (std::size_t index = 0; index < words; ++index) {
        trellis[index] = exl3_ring_word(seed, index);
    }
    return trellis;
}

// FP16 transform vectors on values that are exact in binary64 (multiples of 1/16), so the
// oracle and the kernel read bit-identical suh and svh.
std::vector<std::uint16_t> make_exl3_fp16_plane(int rows, int columns, std::uint32_t seed) {
    std::vector<std::uint16_t> plane(static_cast<std::size_t>(rows) * columns);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            const double magnitude =
                0.25 + static_cast<double>((row * 131 + column * 17 + seed) % 16) * 0.0625;
            const double sign = ((row * 7 + column * 3 + seed) % 2) == 0 ? 1.0 : -1.0;
            plane[static_cast<std::size_t>(row) * columns + column] =
                exl3_ref::round_binary16(sign * magnitude);
        }
    }
    return plane;
}

struct Exl3BankLayout {
    std::size_t payload_bytes = 0;
    std::uint64_t suh_offset  = 0;
    std::uint64_t svh_offset  = 0;
};

struct Exl3ExpertSource {
    int k2 = 0;
    const std::vector<std::uint16_t>* trellis = nullptr;
};

// One exl3_expert_bank payload (contract section 4): the 16-byte-per-expert directory, the
// per-expert trellis extents (256-byte aligned, increasing in expert id), then the FP16
// suh [experts, k] and svh [experts, n] planes. Experts without trellis data keep a
// zero-filled extent at their computed offset: the layout stays valid and neither the
// kernel nor the oracle reads unselected experts.
Exl3BankLayout append_exl3_bank(std::vector<std::uint8_t>& payload, std::int32_t experts,
                                std::int32_t k, std::int32_t n,
                                const std::vector<Exl3ExpertSource>& experts_sources,
                                const std::vector<std::uint16_t>& suh,
                                const std::vector<std::uint16_t>& svh) {
    if (static_cast<int>(experts_sources.size()) != experts || k % 16 != 0 || n % 16 != 0) {
        throw std::logic_error("sparse_moe test: invalid EXL3 bank geometry");
    }
    const auto align256 = [](std::size_t value) { return (value + 255) / 256 * 256; };
    std::vector<std::uint64_t> offsets(experts);
    std::vector<std::uint32_t> extents(experts);
    std::size_t cursor = static_cast<std::size_t>(experts) * 16;
    for (int expert = 0; expert < experts; ++expert) {
        const int k2 = experts_sources[expert].k2;
        if (k2 < 2 || k2 > 16) { throw std::logic_error("sparse_moe test: invalid EXL3 2K"); }
        const std::size_t extent = static_cast<std::size_t>(k) * n * k2 / 16;
        offsets[expert]          = cursor;
        extents[expert]          = static_cast<std::uint32_t>(extent);
        cursor                   = align256(cursor + extent);
    }
    const std::uint64_t suh_offset = cursor;
    cursor = align256(cursor + static_cast<std::size_t>(experts) * k * 2);
    const std::uint64_t svh_offset = cursor;
    cursor = align256(cursor + static_cast<std::size_t>(experts) * n * 2);
    payload.assign(cursor, 0);
    for (int expert = 0; expert < experts; ++expert) {
        std::uint8_t* entry = payload.data() + static_cast<std::size_t>(expert) * 16;
        std::memcpy(entry, &offsets[expert], 8);
        std::memcpy(entry + 8, &extents[expert], 4);
        entry[12] = static_cast<std::uint8_t>(experts_sources[expert].k2);
    }
    for (int expert = 0; expert < experts; ++expert) {
        const std::vector<std::uint16_t>* trellis = experts_sources[expert].trellis;
        if (trellis == nullptr) { continue; }
        const std::size_t expected = static_cast<std::size_t>(k) / 16 *
                                     static_cast<std::size_t>(n / 16) *
                                     static_cast<std::size_t>(8 * experts_sources[expert].k2);
        if (trellis->size() != expected) {
            throw std::logic_error("sparse_moe test: EXL3 trellis size mismatch");
        }
        std::memcpy(payload.data() + offsets[expert], trellis->data(), expected * 2);
    }
    if (suh.size() != static_cast<std::size_t>(experts) * k ||
        svh.size() != static_cast<std::size_t>(experts) * n) {
        throw std::logic_error("sparse_moe test: EXL3 plane size mismatch");
    }
    std::memcpy(payload.data() + suh_offset, suh.data(), suh.size() * 2);
    std::memcpy(payload.data() + svh_offset, svh.data(), svh.size() * 2);
    return {cursor, suh_offset, svh_offset};
}

Weight exl3_bank_weight(const DeviceBuffer& bank, const Exl3BankLayout& layout,
                        std::int32_t experts, std::int32_t rows_per_expert, std::int32_t k,
                        std::uint32_t codebook) {
    Weight result{};
    result.payload         = bank.p;
    result.payload_bytes   = layout.payload_bytes;
    result.qtype           = QType::EXL3;
    result.layout          = QuantLayout::Exl3ExpertBank;
    result.ndim            = 2;
    result.n               = experts * rows_per_expert;
    result.k               = k;
    result.shape[0]        = result.n;
    result.shape[1]        = k;
    result.padded_shape[0] = result.n;
    result.padded_shape[1] = k;
    result.exl3_trellis_offset = 0;
    result.exl3_suh_offset      = layout.suh_offset;
    result.exl3_svh_offset      = layout.svh_offset;
    result.exl3_codebook        = codebook;
    result.exl3_experts         = static_cast<std::uint32_t>(experts);
    result.exl3_rows_per_expert = static_cast<std::uint32_t>(rows_per_expert);
    return result;
}

// The oracle's expert weights: decode_inner gives the exact FP16 inner matrix, and
// linear_weight composes W = diag(suh) Hn W_inner Hn diag(svh) in binary64 (the contract's
// 2.1 semantic weight), so one projection is a plain x^T W in FP64 with no production
// staging, Hadamard schedule, or workspace cast copied.
struct Exl3DecodedExpert {
    int id = -1;
    std::vector<double> w_gate;  // [kHidden, kIntermediate]
    std::vector<double> w_up;    // [kHidden, kIntermediate]
    std::vector<double> w_down;  // [kIntermediate, kHidden]
};

Exl3DecodedExpert decode_exl3_expert(int id, int k2, exl3_ref::Codebook codebook, int hidden,
                                      int intermediate,
                                      const std::vector<std::uint16_t>& gate_trellis,
                                      const std::vector<std::uint16_t>& up_trellis,
                                      const std::vector<std::uint16_t>& down_trellis,
                                      const std::vector<std::uint16_t>& gate_suh,
                                      const std::vector<std::uint16_t>& gate_svh,
                                      const std::vector<std::uint16_t>& up_suh,
                                      const std::vector<std::uint16_t>& up_svh,
                                      const std::vector<std::uint16_t>& down_suh,
                                      const std::vector<std::uint16_t>& down_svh) {
    Exl3DecodedExpert expert;
    expert.id     = id;
    expert.w_gate = exl3_ref::linear_weight(
        exl3_ref::decode_inner(gate_trellis.data(), hidden, intermediate, k2, codebook),
        gate_suh, gate_svh, hidden, intermediate);
    expert.w_up = exl3_ref::linear_weight(
        exl3_ref::decode_inner(up_trellis.data(), hidden, intermediate, k2, codebook), up_suh,
        up_svh, hidden, intermediate);
    expert.w_down = exl3_ref::linear_weight(
        exl3_ref::decode_inner(down_trellis.data(), intermediate, hidden, k2, codebook),
        down_suh, down_svh, intermediate, hidden);
    return expert;
}

// The EXL3 SparseMoe oracle: FP64 routing (shared with the groupwise oracle), the expert
// projections as x^T W over the composed semantic weights, the SiLU product, the shared expert
// unchanged (groupwise Q8 or row-scaled FP8), and FP64 accumulation into the residual.
template <std::int32_t Hidden, std::int32_t Experts, std::int32_t TopK, std::int32_t Intermediate,
          std::size_t PatternCount>
std::vector<double> sparse_moe_exl3_oracle_rows(
    const std::vector<float>& input, const std::vector<float>& residual,
    const std::vector<float>& router, const std::vector<Exl3DecodedExpert>& experts,
    const quantized_weight::PackedWeight& shared_gate_up,
    const quantized_weight::PackedWeight& shared_down,
    const std::array<RoutePatternT<TopK>, PatternCount>& intended_routes, int pattern) {
    const SparseMoeRoutingT<TopK> routing =
        route_sparse_moe_rows<Hidden, Experts, TopK>(input, router, intended_routes[pattern]);
    const std::vector<double> x(input.begin(), input.end());

    std::array<std::vector<double>, TopK> routed_activation;
    for (int route = 0; route < TopK; ++route) {
        const Exl3DecodedExpert& expert = find_expert(experts, routing.selected[route]);
        routed_activation[route].resize(Intermediate);
        parallel_rows(Intermediate, [&](std::int32_t row) {
            const double* gate_column = expert.w_gate.data() + row;
            const double* up_column   = expert.w_up.data() + row;
            double gate               = 0.0;
            double up                 = 0.0;
            for (std::int32_t i = 0; i < Hidden; ++i) {
                gate += x[i] * gate_column[static_cast<std::size_t>(i) * Intermediate];
                up += x[i] * up_column[static_cast<std::size_t>(i) * Intermediate];
            }
            routed_activation[route][row] = (gate / (1.0 + std::exp(-gate))) * up;
        });
    }

    std::vector<double> shared_activation(Intermediate);
    parallel_rows(Intermediate, [&](std::int32_t row) {
        const double gate = dot_fp64(shared_gate_up.dequant, row, Hidden, x);
        const double up   = dot_fp64(shared_gate_up.dequant, Intermediate + row, Hidden, x);
        shared_activation[row] = (gate / (1.0 + std::exp(-gate))) * up;
    });

    std::vector<double> output(Hidden);
    parallel_rows(Hidden, [&](std::int32_t row) {
        double value = static_cast<double>(residual[row]);
        for (int route = 0; route < TopK; ++route) {
            const Exl3DecodedExpert& expert = find_expert(experts, routing.selected[route]);
            const double* down_column       = expert.w_down.data() + row;
            double y                        = 0.0;
            for (std::int32_t c = 0; c < Intermediate; ++c) {
                y += routed_activation[route][c] *
                     down_column[static_cast<std::size_t>(c) * Hidden];
            }
            value += routing.route_weight[route] * y;
        }
        value += routing.shared_scale *
                 dot_fp64(shared_down.dequant, row, Intermediate, shared_activation);
        output[row] = value;
    });
    return output;
}

// One admission expectation: the call must reject the route with std::invalid_argument.
int expect_exl3_rejected(const ops::SparseMoeWeights& weights, std::int32_t tokens,
                          std::int32_t hidden, const std::string& label) {
    try {
        std::vector<float> input(static_cast<std::size_t>(tokens) * hidden, 0.5f);
        round_to_bf16(input);
        const std::vector<std::uint16_t> bits = bf16_bits(input);
        DeviceBuffer device_input              = to_device(bits);
        GuardedBf16Output destination_storage(input.size());
        Tensor x(device_input.p, DType::BF16, {hidden, tokens});
        Tensor destination(destination_storage.data(), DType::BF16, {hidden, tokens});
        const std::size_t workspace_bytes =
            ops::sparse_moe_workspace_capacity_bytes(QType::EXL3, QType::EXL3, 1, 1);
        WorkspaceArena workspace(workspace_bytes);
        ops::sparse_moe(x, weights, ops::SparseMoeEpilogue::AddResidual, destination, workspace,
                        nullptr);
        cuda_synchronize();
    } catch (const std::invalid_argument&) {
        std::cout << "  rejected as expected: " << label << "\n";
        return 0;
    }
    std::cerr << label << ": EXL3 admission accepted an invalid route\n";
    return 1;
}

// Row-scaled FP8 shared experts live on the device as their packed payload (codes plane then
// the 256-aligned BF16 scale plane), so the fixture keeps one device copy and repoints the
// registered weight at it.
class DevicePackedWeight {
public:
    explicit DevicePackedWeight(const quantized_weight::PackedWeight& packed)
        : device_(to_device(packed.payload)), weight_(packed.weight) {
        weight_.payload = device_.p;
        weight_.qdata   = device_.p;
        weight_.scales  = static_cast<const std::uint8_t*>(device_.p) + packed.scale_plane_offset;
    }

    const Weight& weight() const noexcept { return weight_; }

    const DeviceBuffer& device() const noexcept { return device_; }

private:
    DeviceBuffer device_;
    Weight weight_{};
};

// Host decodes for the row-scaled FP8 shared-expert oracle: every finite FP8 E4M3 value is
// exact in binary32, and the BF16 row scales are exact too, so the oracle's dequantized
// matrix is bit-identical to the kernel's decoded weights.
float fp8_e4m3_value(std::uint8_t code) {
    const bool negative = (code & 0x80U) != 0U;
    const int exponent  = (code >> 3) & 0xFU;
    const int mantissa  = code & 0x7U;
    double value        = 0.0;
    if (exponent == 0) {
        value = std::ldexp(static_cast<double>(mantissa), -9);
    } else if (exponent == 15 && mantissa == 7) {
        value = std::numeric_limits<double>::quiet_NaN();
    } else {
        value = std::ldexp(1.0 + static_cast<double>(mantissa) / 8.0, exponent - 7);
    }
    return static_cast<float>(negative ? -value : value);
}

float bf16_value(std::uint16_t bits) {
    const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16;
    float value                 = 0.0F;
    std::memcpy(&value, &widened, sizeof(value));
    return value;
}

void decode_fp8_row_scale(const quantized_weight::PackedWeight& packed, std::int32_t n,
                          std::int32_t k, std::vector<float>& dequant) {
    dequant.assign(static_cast<std::size_t>(n) * k, 0.0F);
    for (std::int32_t row = 0; row < n; ++row) {
        const std::size_t scale_offset = packed.scale_plane_offset + static_cast<std::size_t>(row) * 2;
        const std::uint16_t scale_bits = static_cast<std::uint16_t>(
            packed.payload[scale_offset] |
            (static_cast<std::uint16_t>(packed.payload[scale_offset + 1]) << 8));
        const float scale = bf16_value(scale_bits);
        for (std::int32_t column = 0; column < k; ++column) {
            dequant[static_cast<std::size_t>(row) * k + column] =
                fp8_e4m3_value(packed.payload[static_cast<std::size_t>(row) * k + column]) * scale;
        }
    }
}

template <std::int32_t TopK, std::size_t PatternCount>
std::array<RoutePatternT<TopK>, PatternCount>
patterns_span(std::span<const RoutePatternT<TopK>, PatternCount> patterns) {
    std::array<RoutePatternT<TopK>, PatternCount> out;
    std::copy_n(patterns.begin(), PatternCount, out.begin());
    return out;
}

template <std::int32_t Hidden, std::int32_t Experts, std::int32_t TopK, std::int32_t Intermediate>
class SparseMoeExl3Fixture {
    static constexpr std::int32_t kSharedGateRows = 2 * Intermediate;
    static constexpr int kFixturePatterns = 3;

public:
    SparseMoeExl3Fixture(exl3_ref::Codebook codebook, const char* name, QType shared_qtype,
                         std::span<const int> route_ids,
                         std::span<const RoutePatternT<TopK>, kFixturePatterns> patterns)
        : name_(name), codebook_(codebook), shared_qtype_(shared_qtype),
          route_ids_(route_ids.begin(), route_ids.end()),
          rates_(codebook == exl3_ref::Codebook::Mcg ? kExl3McgRates : kExl3Mul1Rates),
          router_(make_router_rows<Hidden, Experts, TopK>(patterns_span(patterns))),
          router_bits_(bf16_bits(router_)),
          device_router_(to_device(router_bits_)),
          shared_gate_(QType::Q8_G32_FP16, kSharedGateRows, Hidden),
          shared_down_device_(QType::Q8_G32_FP16, Hidden, Intermediate) {
        for (int pattern = 0; pattern < kFixturePatterns; ++pattern) {
            inputs_.push_back(make_input_rows<Hidden, kFixturePatterns>(pattern));
            residuals_.push_back(make_residual_rows<Hidden>(pattern));
        }
        if (shared_qtype_ == QType::Q8_G32_FP16) {
            shared_gate_host_ = quantized_weight::pack_q8_g32_row_split(
                make_gate_up(kSharedGateRows, Hidden, 0x9a1U, 0.91f), kSharedGateRows, Hidden);
            shared_down_host_ = quantized_weight::pack_q8_g32_row_split(
                make_down(Hidden, Intermediate, 0x9c3U, 0.88f), Hidden, Intermediate);
            shared_gate_.copy_rows(shared_gate_host_, 0);
            shared_down_device_.copy_rows(shared_down_host_, 0);
        } else {
            shared_gate_host_ = quantized_weight::make_patterned_weight(
                QType::FP8_E4M3FN_ROW_BF16, kSharedGateRows, Hidden, 0x9a1U);
            decode_fp8_row_scale(shared_gate_host_, kSharedGateRows, Hidden,
                                 shared_gate_host_.dequant);
            shared_down_host_ = quantized_weight::make_patterned_weight(
                QType::FP8_E4M3FN_ROW_BF16, Hidden, Intermediate, 0x9c3U);
            decode_fp8_row_scale(shared_down_host_, Hidden, Intermediate,
                                 shared_down_host_.dequant);
            shared_gate_fp8_.emplace(shared_gate_host_);
            shared_down_fp8_.emplace(shared_down_host_);
        }
        gate_trellis_.resize(Experts);
        up_trellis_.resize(Experts);
        down_trellis_.resize(Experts);
        build_banks();
        decode_experts();
        for (int pattern = 0; pattern < kFixturePatterns; ++pattern) {
            references_.push_back(sparse_moe_exl3_oracle_rows<Hidden, Experts, TopK, Intermediate,
                                                              kFixturePatterns>(
                inputs_[pattern], residuals_[pattern], router_, experts_, shared_gate_host_,
                shared_down_host_, patterns_span(patterns), pattern));
        }
    }

    // One decode per route pattern (the tie boundary, the tie-excluded reorder,
    // and the reverse ordering of the same experts), pattern 1 captured and replayed as a graph.
    int run(int pattern, bool graph_replay) {
        const std::string label         = std::string(name_) + " pattern " + std::to_string(pattern);
        const std::vector<float>& input = inputs_[pattern];
        const std::vector<float>& residual     = residuals_[pattern];
        const std::vector<double>& reference    = references_[pattern];

        const std::vector<std::uint16_t> input_bits    = bf16_bits(input);
        const std::vector<std::uint16_t> residual_bits = bf16_bits(residual);
        DeviceBuffer device_input                      = to_device(input_bits);
        DeviceBuffer residual_seed                      = to_device(residual_bits);
        GuardedBf16Output destination_storage(residual.size());
        cuda_check(cudaMemcpy(destination_storage.data(), residual_seed.p, residual_seed.bytes,
                              cudaMemcpyDeviceToDevice),
                   "seed SparseMoe destination");

        const ops::SparseMoeWeights weights = make_weights();
        Tensor x(device_input.p, DType::BF16, {Hidden, 1});
        Tensor destination(destination_storage.data(), DType::BF16, {Hidden, 1});
        const ops::SparseMoeShape shape{Hidden, Experts, TopK, Intermediate};
        const std::size_t workspace_bytes =
            ops::sparse_moe_workspace_capacity_bytes(QType::EXL3, QType::EXL3, 1, 1, shape);
        WorkspaceArena workspace(workspace_bytes);

        if (graph_replay) {
            cudaStream_t stream  = nullptr;
            cudaGraph_t graph    = nullptr;
            cudaGraphExec_t exec = nullptr;
            cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                       "create SparseMoe graph stream");
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                       "begin SparseMoe graph capture");
            cuda_check(cudaMemcpyAsync(destination.data, residual_seed.p, destination.bytes(),
                                       cudaMemcpyDeviceToDevice, stream),
                       "capture SparseMoe residual seed");
            ops::sparse_moe(x, weights, ops::SparseMoeEpilogue::AddResidual, destination, workspace,
                            stream);
            cuda_check(cudaStreamEndCapture(stream, &graph), "end SparseMoe graph capture");
            cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
                       "instantiate SparseMoe graph");
            cuda_check(cudaGraphLaunch(exec, stream), "launch SparseMoe graph");
            cuda_check(cudaGraphLaunch(exec, stream), "replay SparseMoe graph");
            cuda_check(cudaStreamSynchronize(stream), "synchronize SparseMoe graph");
            cuda_check(cudaGraphExecDestroy(exec), "destroy SparseMoe graph executable");
            cuda_check(cudaGraphDestroy(graph), "destroy SparseMoe graph");
            cuda_check(cudaStreamDestroy(stream), "destroy SparseMoe graph stream");
        } else {
            ops::sparse_moe(x, weights, ops::SparseMoeEpilogue::AddResidual, destination, workspace,
                            nullptr);
            cuda_synchronize();
        }

        int failures =
            compare_output(label, destination_storage.values(), reference, kSparseMoeExl3A16Tolerance);
        failures += destination_storage.verify_guards(label);
        failures +=
            verify_exact((label + " input preservation").c_str(),
                         from_device<std::uint16_t>(device_input, input_bits.size()), input_bits);
        if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
            std::cerr << label << ": workspace query/execution high-water mismatch\n";
            ++failures;
        }
        return failures;
    }

    int verify_persistent_inputs() const {
        int failures = 0;
        const std::string prefix(name_);
        failures += verify_exact((prefix + " router preservation").c_str(),
                                 from_device<std::uint16_t>(device_router_, router_bits_.size()),
                                 router_bits_);
        if (shared_qtype_ == QType::Q8_G32_FP16) {
            failures += shared_gate_.verify_rows(prefix + " shared gate", shared_gate_host_, 0);
            failures +=
                shared_down_device_.verify_rows(prefix + " shared down", shared_down_host_, 0);
        } else {
            failures += verify_exact((prefix + " shared gate fp8 preservation").c_str(),
                                     from_device<std::uint8_t>(shared_gate_fp8_->device(),
                                                               shared_gate_host_.payload.size()),
                                     shared_gate_host_.payload);
            failures += verify_exact((prefix + " shared down fp8 preservation").c_str(),
                                     from_device<std::uint8_t>(shared_down_fp8_->device(),
                                                               shared_down_host_.payload.size()),
                                     shared_down_host_.payload);
        }
        failures += verify_exact((prefix + " exl3 gate bank preservation").c_str(),
                                 from_device<std::uint8_t>(gate_bank_, gate_payload_.size()),
                                 gate_payload_);
        failures += verify_exact((prefix + " exl3 up bank preservation").c_str(),
                                 from_device<std::uint8_t>(up_bank_, up_payload_.size()),
                                 up_payload_);
        failures += verify_exact((prefix + " exl3 down bank preservation").c_str(),
                                 from_device<std::uint8_t>(down_bank_, down_payload_.size()),
                                 down_payload_);
        return failures;
    }

    // The route's admission contract: 3INST banks are refused, the fused and warm banks stay
    // null, the route stays untiered, bank shapes are hard
    // checked, and the workspace query refuses unpaired EXL3 banks.
    int run_admission() {
        int failures = 0;
        const ops::SparseMoeWeights base = make_weights();
        {
            ops::SparseMoeWeights mutated  = base;
            mutated.exl3_routed_gate.exl3_codebook = 0;
            failures += expect_exl3_rejected(mutated, 1, Hidden,
                                             std::string(name_) + " 3inst codebook");
        }
        {
            ops::SparseMoeWeights mutated = base;
            mutated.routed_gate_up        = shared_gate_.weight();
            failures +=
                expect_exl3_rejected(mutated, 1, Hidden,
                                     std::string(name_) + " fused bank conflict");
        }
        {
            ops::SparseMoeWeights mutated = base;
            mutated.tiered_hot_experts    = 8;
            failures +=
                expect_exl3_rejected(mutated, 1, Hidden,
                                     std::string(name_) + " tiered conflict");
        }
        {
            ops::SparseMoeWeights mutated           = base;
            mutated.exl3_routed_up.exl3_rows_per_expert = Intermediate - 1;
            failures += expect_exl3_rejected(mutated, 1, Hidden,
                                             std::string(name_) + " wrong rows per expert");
        }
        try {
            (void)ops::sparse_moe_workspace_capacity_bytes(QType::EXL3, QType::Q4_G64_FP16, 1, 1,
                                                           ops::SparseMoeShape{Hidden, Experts, TopK, Intermediate});
        } catch (const std::invalid_argument&) {
            std::cout << "  rejected as expected: " << name_ << " unpaired EXL3 workspace\n";
            return failures;
        }
        std::cerr << name_ << ": EXL3 workspace accepted an unpaired bank\n";
        return failures + 1;
    }

private:
    void build_banks() {
        const std::uint32_t codebook = codebook_ == exl3_ref::Codebook::Mcg ? 1u : 2u;
        std::vector<Exl3ExpertSource> gate_sources(Experts), up_sources(Experts),
            down_sources(Experts);
        for (int expert = 0; expert < Experts; ++expert) {
            gate_sources[expert].k2 = rates_[expert % 4];
            up_sources[expert].k2   = rates_[expert % 4];
            down_sources[expert].k2 = rates_[expert % 4];
        }
        for (const int id : route_ids_) {
            const int k2       = rates_[id % 4];
            gate_trellis_[id] = make_exl3_trellis(Hidden, Intermediate, k2, 1000u + id);
            up_trellis_[id]   = make_exl3_trellis(Hidden, Intermediate, k2, 2000u + id);
            down_trellis_[id] = make_exl3_trellis(Intermediate, Hidden, k2, 3000u + id);
            gate_sources[id].trellis = &gate_trellis_[id];
            up_sources[id].trellis   = &up_trellis_[id];
            down_sources[id].trellis = &down_trellis_[id];
        }
        gate_suh_ = make_exl3_fp16_plane(Experts, Hidden, 41);
        gate_svh_ = make_exl3_fp16_plane(Experts, Intermediate, 42);
        up_suh_   = make_exl3_fp16_plane(Experts, Hidden, 43);
        up_svh_   = make_exl3_fp16_plane(Experts, Intermediate, 44);
        down_suh_ = make_exl3_fp16_plane(Experts, Intermediate, 45);
        down_svh_ = make_exl3_fp16_plane(Experts, Hidden, 46);

        gate_layout_ = append_exl3_bank(gate_payload_, Experts, Hidden, Intermediate,
                                        gate_sources, gate_suh_, gate_svh_);
        up_layout_   = append_exl3_bank(up_payload_, Experts, Hidden, Intermediate, up_sources,
                                        up_suh_, up_svh_);
        down_layout_ = append_exl3_bank(down_payload_, Experts, Intermediate, Hidden,
                                        down_sources, down_suh_, down_svh_);
        gate_bank_    = to_device(gate_payload_);
        up_bank_      = to_device(up_payload_);
        down_bank_    = to_device(down_payload_);
        exl3_gate_    = exl3_bank_weight(gate_bank_, gate_layout_, Experts, Intermediate, Hidden, codebook);
        exl3_up_      = exl3_bank_weight(up_bank_, up_layout_, Experts, Intermediate, Hidden, codebook);
        exl3_down_    = exl3_bank_weight(down_bank_, down_layout_, Experts, Hidden, Intermediate, codebook);
    }

    void decode_experts() {
        experts_.resize(route_ids_.size());
        const auto plane_row = [](const std::vector<std::uint16_t>& plane, int id, int columns) {
            return std::vector<std::uint16_t>(
                plane.begin() + static_cast<std::size_t>(id) * columns,
                plane.begin() + static_cast<std::size_t>(id + 1) * columns);
        };
        parallel_rows(static_cast<std::int32_t>(route_ids_.size()),
                      [&](std::int32_t index) {
                          const int id = route_ids_[index];
                          const int k2 = rates_[id % 4];
                          experts_[index] = decode_exl3_expert(
                              id, k2, codebook_, Hidden, Intermediate, gate_trellis_[id],
                              up_trellis_[id], down_trellis_[id],
                              plane_row(gate_suh_, id, Hidden),
                              plane_row(gate_svh_, id, Intermediate),
                              plane_row(up_suh_, id, Hidden),
                              plane_row(up_svh_, id, Intermediate),
                              plane_row(down_suh_, id, Intermediate),
                              plane_row(down_svh_, id, Hidden));
                      });
    }

    ops::SparseMoeWeights make_weights() const {
        ops::SparseMoeWeights weights;
        weights.router_shared_gate = dense_bf16_weight(device_router_.p, Experts + 1, Hidden);
        weights.routed_down       = exl3_down_;
        if (shared_qtype_ == QType::Q8_G32_FP16) {
            weights.shared_gate_up    = shared_gate_.weight();
            weights.shared_down       = shared_down_device_.weight();
        } else {
            weights.shared_gate_up    = shared_gate_fp8_->weight();
            weights.shared_down       = shared_down_fp8_->weight();
        }
        weights.exl3_routed_gate  = exl3_gate_;
        weights.exl3_routed_up    = exl3_up_;
        return weights;
    }

    const char* name_;
    exl3_ref::Codebook codebook_;
    QType shared_qtype_;
    std::vector<int> route_ids_;
    std::array<int, 4> rates_;
    std::vector<float> router_;
    std::vector<std::uint16_t> router_bits_;
    DeviceBuffer device_router_;
    DeviceRowSplit shared_gate_;
    DeviceRowSplit shared_down_device_;
    std::optional<DevicePackedWeight> shared_gate_fp8_;
    std::optional<DevicePackedWeight> shared_down_fp8_;
    quantized_weight::PackedWeight shared_gate_host_;
    quantized_weight::PackedWeight shared_down_host_;
    std::vector<std::vector<float>> inputs_;
    std::vector<std::vector<float>> residuals_;
    std::vector<std::vector<double>> references_;
    std::vector<std::vector<std::uint16_t>> gate_trellis_, up_trellis_, down_trellis_;
    std::vector<std::uint16_t> gate_suh_, gate_svh_, up_suh_, up_svh_, down_suh_, down_svh_;
    std::vector<std::uint8_t> gate_payload_, up_payload_, down_payload_;
    Exl3BankLayout gate_layout_, up_layout_, down_layout_;
    DeviceBuffer gate_bank_, up_bank_, down_bank_;
    Weight exl3_gate_, exl3_up_, exl3_down_;
    std::vector<Exl3DecodedExpert> experts_;
};

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    // These are public-behavior cases, not route assertions. They exercise decode (T=1), the
    // Small-T supported-domain edges, each profile's first prefill T, the wide-prefill boundary,
    // and one call crossing the 4096-token internal slice without observing any private plan.
    constexpr std::array<std::int32_t, 6> kQ4Q5Tokens{{1, 2, 46, 47, 768, 4097}};
    constexpr std::array<std::int32_t, 5> kQ4Q6Tokens{{1, 2, 46, 47, 768}};
    constexpr std::array<std::int32_t, 5> kQ8Q8Tokens{{1, 2, 19, 20, 768}};
    const std::array<CodecProfile, 3> profiles{{
        {"sparse_moe q4+q5 a16", QType::Q4_G64_FP16, QType::Q5_G64_FP16, kQ4Q5Tokens, true},
        {"sparse_moe q4+q6 a16", QType::Q4_G64_FP16, QType::Q6_G64_FP16, kQ4Q6Tokens, false},
        {"sparse_moe q8+q8 a16", QType::Q8_G32_FP16, QType::Q8_G32_FP16, kQ8Q8Tokens, false},
    }};

    int failures = 0;
    for (const CodecProfile& profile : profiles) { failures += run_profile(profile); }

    // SP-7 EXL3 (single-token decode route): both codebooks with mixed per-expert bitrates
    // (the mul1 fixture covers the MUL1-only half-integer rates), graph replay on pattern 1,
    // persistent-input preservation, and the route's admission contract. The 125B geometry
    // runs the same three route patterns over 512 experts with the row-scaled FP8 shared
    // expert the Flash-Next artifact carries.
    {
        SparseMoeExl3Fixture<kHidden, kExperts, kTopK, kIntermediate>
            mcg(exl3_ref::Codebook::Mcg, "sparse_moe exl3 mcg a16", QType::Q8_G32_FP16,
                kExl3RouteExpertIds, kRoutePatterns);
        for (int pattern = 0; pattern < static_cast<int>(kRoutePatterns.size()); ++pattern) {
            failures += mcg.run(pattern, pattern == 1);
        }
        failures += mcg.verify_persistent_inputs();
        failures += mcg.run_admission();

        SparseMoeExl3Fixture<kHidden, kExperts, kTopK, kIntermediate>
            mul1(exl3_ref::Codebook::Mul1, "sparse_moe exl3 mul1 a16", QType::Q8_G32_FP16,
                 kExl3RouteExpertIds, kRoutePatterns);
        for (int pattern = 0; pattern < static_cast<int>(kRoutePatterns.size()); ++pattern) {
            failures += mul1.run(pattern, pattern == 1);
        }
        failures += mul1.verify_persistent_inputs();
    }

    {
        constexpr std::int32_t kHidden125B = ops::kSparseMoe125B.hidden;
        constexpr std::int32_t kExperts125B = ops::kSparseMoe125B.experts;
        constexpr std::int32_t kTopK125B = ops::kSparseMoe125B.top_k;
        constexpr std::int32_t kIntermediate125B = ops::kSparseMoe125B.intermediate;
        SparseMoeExl3Fixture<kHidden125B, kExperts125B, kTopK125B, kIntermediate125B>
            flash_next(exl3_ref::Codebook::Mcg, "sparse_moe exl3 125b fp8 shared a16",
                       QType::FP8_E4M3FN_ROW_BF16, kExl3RouteExpertIds125B, kRoutePatterns125B);
        for (int pattern = 0; pattern < static_cast<int>(kRoutePatterns125B.size()); ++pattern) {
            failures += flash_next.run(pattern, pattern == 1);
        }
        failures += flash_next.verify_persistent_inputs();
    }

    std::cout << (failures == 0 ? "OK" : "FAIL") << " sparse_moe correctness\n";
    return failures == 0 ? 0 : 1;
}
