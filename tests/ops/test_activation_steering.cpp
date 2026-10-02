#include "ninfer/ops/activation_steering.h"
#include "ninfer/ops/hyperconnection.h"
#include "ops/op_tester.h"
#include "core/arena.h"
#include "core/device.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

// Steering is fused into the attention-input hyperconnection mix. Both entry points are checked
// against an independent FP64 simultaneous-projection oracle applied to the represented BF16
// input (the hyper state for hyperconnection_mix, the represented combine for
// hyperconnection_combine_mix), and the normalized output must come from the steered values.

using namespace ninfer;
using namespace ninfer::test;

namespace {
constexpr int kWidth = 2560, kLanes = 4, kTokens = 2, kHyper = kWidth * kLanes, kRank = 320;

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

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

std::vector<std::uint16_t> read_bits(const DeviceBuffer& buffer, std::size_t count) {
    std::vector<std::uint16_t> bits(count);
    buffer.copy_to_host(bits.data(), count * 2);
    return bits;
}

// FP64 oracle for one represented input; throws on mismatch.
void check_steered(const std::vector<std::uint16_t>& input, const std::vector<std::uint16_t>& got,
                   const std::vector<float>& directions, int mask, int rank, float strength,
                   bool preserve) {
    for (int token = 0; token < kTokens; ++token)
        for (int lane = 0; lane < kLanes; ++lane) {
            const int base = (token * kLanes + lane) * kWidth;
            std::vector<double> reference(kWidth), delta(kWidth);
            double before = 0, after = 0;
            for (int d = 0; d < kWidth; ++d) {
                reference[d] = bf16_to_f32(input[base + d]);
                before += reference[d] * reference[d];
            }
            const bool steered = (mask & (1 << lane)) != 0;
            if (steered)
                for (int r = 0; r < rank; ++r) {
                    double dot = 0;
                    for (int d = 0; d < kWidth; ++d)
                        dot += double(bf16_to_f32(input[base + d])) *
                               directions[(r * kLanes + lane) * kWidth + d];
                    for (int d = 0; d < kWidth; ++d)
                        delta[d] += dot * directions[(r * kLanes + lane) * kWidth + d];
                }
            for (int d = 0; d < kWidth; ++d) {
                reference[d] -= strength * delta[d];
                after += reference[d] * reference[d];
            }
            const double scale = preserve && after > 0 ? std::sqrt(before / after) : 1;
            for (int d = 0; d < kWidth; ++d) {
                const double want = reference[d] * scale;
                const double have = bf16_to_f32(got[base + d]);
                if (std::abs(want - have) > 0.009 + 0.006 * std::abs(want))
                    throw std::runtime_error("FP64 simultaneous projection oracle mismatch");
                if (!steered && got[base + d] != input[base + d])
                    throw std::runtime_error("unmasked lane changed");
            }
        }
}
} // namespace

int main() {
    try {
        DeviceContext ctx;
        std::vector<float> hyper_values(kHyper * kTokens), block_values(kWidth * kTokens);
        std::vector<float> injection_values(kLanes * kTokens), norm_values(kHyper);
        std::vector<float> down_values(std::size_t(kRank) * kHyper), up_values(std::size_t(kHyper) * kRank);
        fill_uniform(hyper_values, 1337, -1, 1);
        fill_uniform(block_values, 7331, -1, 1);
        fill_uniform(injection_values, 4242, -2, 2);
        fill_uniform(norm_values, 99, -0.2F, 0.2F);
        fill_uniform(down_values, 17, -0.02F, 0.02F);
        fill_uniform(up_values, 23, -0.05F, 0.05F);
        const auto hyper_bits = bf16_bits(hyper_values);

        std::vector<float> directions(48ULL * ops::kSteeringMaxRank * kLanes * kWidth);
        for (int r = 0; r < ops::kSteeringMaxRank; ++r)
            for (int lane = 0; lane < kLanes; ++lane) {
                double square = 0;
                for (int d = 0; d < kWidth; ++d) {
                    const float value = std::sin(float(d * 3 + r * 7 + lane * 11) * 0.017F);
                    directions[(r * kLanes + lane) * kWidth + d] = value;
                    square += double(value) * value;
                }
                for (int d = 0; d < kWidth; ++d)
                    directions[(r * kLanes + lane) * kWidth + d] /= std::sqrt(square);
            }

        auto d_hyper = to_device(hyper_bits), d_scratch = to_device(hyper_bits);
        auto d_block = to_device(bf16_bits(block_values));
        auto d_inj   = to_device(bf16_bits(injection_values));
        auto d_norm  = to_device(bf16_bits(norm_values));
        auto d_down = to_device(bf16_bits(down_values)), d_up = to_device(bf16_bits(up_values));
        DeviceBuffer d_input(std::size_t(kWidth) * kTokens * 2), d_plain(d_input.bytes);
        auto d_directions = to_device(directions);
        std::vector<int> ranks(48), masks(48);
        masks[0]    = 5;
        auto d_rank = to_device(ranks), d_mask = to_device(masks);
        ops::ActivationDevice control;
        control.directions = static_cast<float*>(d_directions.p);
        control.ranks      = static_cast<int*>(d_rank.p);
        control.masks      = static_cast<int*>(d_mask.p);
        auto d_control     = to_device(std::vector{control});

        ops::HyperConnectionWeights weights{Tensor(d_norm.p, DType::BF16, {kHyper}),
                                            bf16_weight(d_down, kRank, kHyper),
                                            bf16_weight(d_up, kHyper, kRank), Weight{}};
        Tensor hyper(d_hyper.p, DType::BF16, {kHyper, kTokens});
        Tensor scratch(d_scratch.p, DType::BF16, {kHyper, kTokens});
        Tensor block(d_block.p, DType::BF16, {kWidth, kTokens});
        Tensor injection(d_inj.p, DType::BF16, {kLanes, kTokens});
        Tensor input(d_input.p, DType::BF16, {kWidth, kTokens});
        Tensor plain(d_plain.p, DType::BF16, {kWidth, kTokens});
        WorkspaceArena workspace(ops::hyperconnection_mix_workspace_capacity_bytes(kTokens, false));
        const ops::HyperConnectionActivation steer{
            .steering = static_cast<ops::ActivationDevice*>(d_control.p), .layer = 0, .width = 1};

        // The represented combine every combine_mix check starts from.
        d_scratch.copy_from_host(hyper_bits.data(), hyper_bits.size() * 2);
        ops::hyperconnection_combine(scratch, block, injection, ctx.stream);
        ctx.synchronize();
        const auto combined_bits = read_bits(d_scratch, hyper_bits.size());

        // Both routes are captured once; rank, strength and policy reload through fixed addresses.
        cudaGraph_t graphs[2]{};
        cudaGraphExec_t executables[2]{};
        for (int route = 0; route < 2; ++route) {
            CUDA_CHECK(cudaStreamBeginCapture(ctx.stream, cudaStreamCaptureModeThreadLocal));
            if (route == 0) {
                ops::hyperconnection_mix(hyper, weights, input, nullptr, workspace, ctx.stream,
                                         nullptr, &steer);
            } else {
                ops::hyperconnection_combine_mix(hyper, block, injection, weights, input, nullptr,
                                                 workspace, ctx.stream, nullptr, &steer);
            }
            CUDA_CHECK(cudaStreamEndCapture(ctx.stream, &graphs[route]));
            CUDA_CHECK(cudaGraphInstantiate(&executables[route], graphs[route], nullptr, nullptr, 0));
        }

        for (int route = 0; route < 2; ++route) {
            const auto& represented = route == 0 ? hyper_bits : combined_bits;
            for (int rank : {0, 1, 3, ops::kSteeringMaxRank})
                for (float strength : {0.0F, 0.35F, 1.0F})
                    for (int preserve : {0, 1}) {
                        ranks[0] = rank;
                        d_rank.copy_from_host(ranks.data(), ranks.size() * sizeof(int));
                        control.norm_preserve = preserve;
                        for (int row = 0; row < kTokens; ++row) control.rows[row].strength = strength;
                        d_control.copy_from_host(&control, sizeof(control));
                        d_hyper.copy_from_host(hyper_bits.data(), hyper_bits.size() * 2);
                        CUDA_CHECK(cudaGraphLaunch(executables[route], ctx.stream));
                        ctx.synchronize();
                        const auto result = read_bits(d_hyper, hyper_bits.size());
                        const auto mixed  = read_bits(d_input, std::size_t(kWidth) * kTokens);
                        if ((rank == 0 || strength == 0) && result != represented)
                            throw std::runtime_error("inactive steering is not bitwise identity");
                        check_steered(represented, result, directions, masks[0], rank, strength,
                                      preserve);

                        // The mix must normalize the steered state: an unsteered mix of the
                        // steered hyper reproduces the block input bit for bit.
                        d_scratch.copy_from_host(result.data(), result.size() * 2);
                        ops::hyperconnection_mix(scratch, weights, plain, nullptr, workspace,
                                                 ctx.stream);
                        ctx.synchronize();
                        if (read_bits(d_plain, mixed.size()) != mixed)
                            throw std::runtime_error("mix did not consume the steered state");

                        d_hyper.copy_from_host(hyper_bits.data(), hyper_bits.size() * 2);
                        CUDA_CHECK(cudaGraphLaunch(executables[route], ctx.stream));
                        ctx.synchronize();
                        if (read_bits(d_hyper, hyper_bits.size()) != result)
                            throw std::runtime_error("graph repeated execution is not bitwise exact");
                    }
        }
        for (int route = 0; route < 2; ++route) {
            CUDA_CHECK(cudaGraphExecDestroy(executables[route]));
            CUDA_CHECK(cudaGraphDestroy(graphs[route]));
        }
        std::cout << "fused mix and combine_mix: FP64 oracle at ranks 0,1,3,32; strengths 0,.35,1;"
                     " masks; norm preservation PASS\n"
                     "steered normalization, inactive bitwise identity, graph reload and"
                     " repeatability PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
