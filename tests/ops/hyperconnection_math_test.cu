// The HyperConnection elementwise mathematics, checked against an independent FP64 oracle.
//
// WHY THIS EXISTS: the two fp8 kernel trees (ops/linear/fp8 and ops/linear/kernels/fp8) differ in
// their epilogue contract, and the stream-quad path is the piece only the flat tree has. Merging
// them means moving gate_mix_position and finish_injection into the kernels tree, which changes the
// accumulation those two helpers perform. Before that port can be judged, the helpers need an
// oracle of their own: until now nothing checked them except the 125B model, and its artifact is
// not present on either lane host.
//
// THE ORACLE evaluates the documented formula in FP64 from the represented inputs. It does not copy
// the kernel's association, its bf16 rounding points, or its use of fmaf; it recomputes
//   block_input[d,t] = bf16(0.25 * sum_s sigmoid(bf16(logit_s)) * normalized[s,d,t])
//   injection[e,t]   = bf16(sum_s partials[t*16 + s*4 + e])
// from the same public values, so a change that keeps the route self-consistent but wrong against
// the formula fails here.
//
// Compiled as a .cu because the helpers are __device__ functions, and it launches a kernel, so it
// needs a device and skips without one.

#include "cuda_fixture.h"
#include "core/weight.h"
#include "ops/launcher/hyperconnection_math.cuh"
#include "ops/linear/projection/fp8/fp8_flash_next.h"
#include "ops/op_tester.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;
namespace hc = ninfer::ops::detail::hyperconnection;

int failures = 0;

float bf16_round(float value) { return __bfloat162float(__float2bfloat16_rn(value)); }

std::uint16_t to_bf16_bits(float value) {
    const __nv_bfloat16 represented = __float2bfloat16_rn(value);
    std::uint16_t bits              = 0;
    std::memcpy(&bits, &represented, sizeof(bits));
    return bits;
}

// Each helper gets its own kernel: the mix writes a destination buffer and the injection uses a
// different index layout, so one kernel for both would hide which helper failed.
__global__ void gate_mix_kernel(const float* logits, const std::uint16_t* normalized,
                                std::uint16_t* block_input, int tokens) {
    const int position =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (position >= hc::kHidden) { return; }
    for (int token = 0; token < tokens; ++token) {
        float stream_logits[hc::kStreams];
        for (int stream = 0; stream < hc::kStreams; ++stream) {
            stream_logits[stream] = logits[static_cast<std::int64_t>(token) * hc::kHyper +
                                           stream * hc::kHidden + position];
        }
        hc::gate_mix_position(stream_logits, reinterpret_cast<const __nv_bfloat16*>(normalized),
                              reinterpret_cast<__nv_bfloat16*>(block_input), position, token);
    }
}

__global__ void injection_kernel(const float* partials, std::uint16_t* injection, int tokens) {
    const int destination = static_cast<int>(blockIdx.x);
    if (destination >= hc::kStreams) { return; }
    for (int token = 0; token < tokens; ++token) {
        hc::finish_injection(partials, reinterpret_cast<__nv_bfloat16*>(injection), destination,
                             token);
    }
}

// Deterministic values that exercise the sigmoid's interesting range and both signs, and that are
// exactly representable so the oracle and the kernel start from the same inputs.
std::vector<float> make_logits(int tokens) {
    std::vector<float> logits(static_cast<std::size_t>(tokens) * hc::kHyper);
    for (int token = 0; token < tokens; ++token) {
        for (int stream = 0; stream < hc::kStreams; ++stream) {
            for (int position = 0; position < hc::kHidden; ++position) {
                const int step   = (position * 7 + stream * 13 + token * 29) % 33;
                logits[static_cast<std::size_t>(token) * hc::kHyper + stream * hc::kHidden +
                       position] = static_cast<float>(step) * 0.25F - 4.0F;
            }
        }
    }
    return logits;
}

std::vector<std::uint16_t> make_normalized(int tokens) {
    std::vector<std::uint16_t> normalized(static_cast<std::size_t>(hc::kHidden) * hc::kStreams *
                                          tokens);
    for (int token = 0; token < tokens; ++token) {
        for (int stream = 0; stream < hc::kStreams; ++stream) {
            for (int position = 0; position < hc::kHidden; ++position) {
                const int step = (position * 5 + stream * 3 + token * 11) % 17;
                normalized[static_cast<std::size_t>(position) +
                           static_cast<std::size_t>(hc::kHidden) *
                               (stream + hc::kStreams * token)] =
                    to_bf16_bits(static_cast<float>(step) * 0.5F - 4.0F);
            }
        }
    }
    return normalized;
}

std::vector<float> make_partials(int tokens) {
    std::vector<float> partials(static_cast<std::size_t>(tokens) * hc::kInjectionPartialsPerToken);
    for (int token = 0; token < tokens; ++token) {
        for (int index = 0; index < hc::kInjectionPartialsPerToken; ++index) {
            partials[static_cast<std::size_t>(token) * hc::kInjectionPartialsPerToken + index] =
                static_cast<float>((token * 31 + index * 17) % 23) * 0.125F - 1.0F;
        }
    }
    return partials;
}

int run_gate_mix(int tokens) {
    const auto logits     = make_logits(tokens);
    const auto normalized = make_normalized(tokens);

    // Oracle: FP64, from the represented bf16 inputs, no fmaf and no intermediate rounding except
    // the two the contract names (the gate logit's bf16 boundary and the result's bf16 store).
    std::vector<float> expected(static_cast<std::size_t>(hc::kHidden) * tokens);
    for (int token = 0; token < tokens; ++token) {
        for (int position = 0; position < hc::kHidden; ++position) {
            double sum = 0.0;
            for (int stream = 0; stream < hc::kStreams; ++stream) {
                const double logit = static_cast<double>(
                    bf16_round(logits[static_cast<std::size_t>(token) * hc::kHyper +
                                      stream * hc::kHidden + position]));
                const double gate        = 1.0 / (1.0 + std::exp(-logit));
                const std::uint16_t bits = normalized[static_cast<std::size_t>(position) +
                                                      static_cast<std::size_t>(hc::kHidden) *
                                                          (stream + hc::kStreams * token)];
                __nv_bfloat16 value;
                std::memcpy(&value, &bits, sizeof(value));
                sum += gate * static_cast<double>(__bfloat162float(value));
            }
            expected[static_cast<std::size_t>(position) +
                     static_cast<std::size_t>(hc::kHidden) * token] =
                static_cast<float>(sum * 0.25);
        }
    }

    GuardedDeviceBuffer device_logits(logits.size() * sizeof(float));
    GuardedDeviceBuffer device_normalized(normalized.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_block_input(expected.size() * sizeof(std::uint16_t));
    device_logits.copy_from_host(logits.data(), device_logits.bytes());
    device_normalized.copy_from_host(normalized.data(), device_normalized.bytes());
    device_block_input.fill(0xcd);

    constexpr int kBlock = 256;
    gate_mix_kernel<<<(hc::kHidden + kBlock - 1) / kBlock, kBlock, 0, nullptr>>>(
        static_cast<const float*>(device_logits.data()),
        static_cast<const std::uint16_t*>(device_normalized.data()),
        static_cast<std::uint16_t*>(device_block_input.data()), tokens);
    cuda_synchronize();

    const auto got = from_device<std::uint16_t>(device_block_input.data(), expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const __nv_bfloat16 want = __float2bfloat16_rn(expected[index]);
        std::uint16_t want_bits  = 0;
        std::memcpy(&want_bits, &want, sizeof(want_bits));
        if (got[index] != want_bits) {
            std::cerr << "FAIL: gate mix differs at " << index << ": got " << got[index] << " want "
                      << want_bits << "\n";
            ++failures;
            break;
        }
    }
    return failures;
}

int run_injection(int tokens) {
    const auto partials = make_partials(tokens);
    std::vector<float> expected(static_cast<std::size_t>(hc::kStreams) * tokens);
    for (int token = 0; token < tokens; ++token) {
        for (int destination = 0; destination < hc::kStreams; ++destination) {
            double sum = 0.0;
            for (int source = 0; source < hc::kStreams; ++source) {
                sum += static_cast<double>(
                    partials[static_cast<std::size_t>(token) * hc::kInjectionPartialsPerToken +
                             source * hc::kStreams + destination]);
            }
            expected[static_cast<std::size_t>(destination) + hc::kStreams * token] =
                static_cast<float>(sum);
        }
    }

    GuardedDeviceBuffer device_partials(partials.size() * sizeof(float));
    GuardedDeviceBuffer device_injection(expected.size() * sizeof(std::uint16_t));
    device_partials.copy_from_host(partials.data(), device_partials.bytes());
    device_injection.fill(0xcd);

    injection_kernel<<<hc::kStreams, 32, 0, nullptr>>>(
        static_cast<const float*>(device_partials.data()),
        static_cast<std::uint16_t*>(device_injection.data()), tokens);
    cuda_synchronize();

    const auto got = from_device<std::uint16_t>(device_injection.data(), expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const __nv_bfloat16 want = __float2bfloat16_rn(expected[index]);
        std::uint16_t want_bits  = 0;
        std::memcpy(&want_bits, &want, sizeof(want_bits));
        if (got[index] != want_bits) {
            std::cerr << "FAIL: injection differs at " << index << ": got " << got[index]
                      << " want " << want_bits << "\n";
            ++failures;
            break;
        }
    }
    return failures;
}

// ---------------------------------------------------------------------------------------------
// The up-mix ROUTE, not just its helpers. The oracle above proves gate_mix_position and
// finish_injection; this case proves that the route feeds them the right logits: it builds an FP8
// row-scaled weight by hand, runs launch_fp8_hc_up_mix, and compares block_input and injection
// against the same formula extended over the matmul.
//
// THE CRITERION IS NUMERICAL, NOT EXACT. The kernel accumulates the dot product in FP32 and
// evaluates the sigmoid with expf; the oracle accumulates in FP64 and evaluates with std::exp. A
// bit-equal comparison would therefore be a test of the association, which the contract does not
// fix. The criterion is relative: every element must agree within 1e-2 of its magnitude, which is
// about one BF16 ulp at these values.
constexpr double kRouteRelativeTolerance = 1e-2;

std::uint8_t encode_e4m3(float value) {
    return __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
}

float decode_e4m3(std::uint8_t bits) {
    __nv_fp8_e4m3 code;
    code.__x = bits;
    return static_cast<float>(code);
}

std::vector<std::uint8_t> make_codes(std::int32_t n, std::int32_t k) {
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(n) * k);
    for (std::int32_t row = 0; row < n; ++row) {
        for (std::int32_t column = 0; column < k; ++column) {
            // Small signed values, exactly representable in E4M3, repeating so the dot product has
            // both signs and a non-trivial sum.
            const int step = static_cast<int>((row * 3 + column * 7) % 15) - 7;
            codes[static_cast<std::size_t>(row) * k + column] =
                encode_e4m3(static_cast<float>(step) * 0.25F);
        }
    }
    return codes;
}

int run_up_mix_route(int tokens) {
    constexpr std::int32_t kStreams = hc::kStreams;
    constexpr std::int32_t kHidden  = hc::kHidden;
    constexpr std::int32_t kRank    = hc::kRank;
    constexpr std::int32_t kRows    = kStreams * kHidden;

    const auto codes = make_codes(kRows, kRank);

    // One BF16 scale per output row, all powers of two so the scale itself is exact.
    std::vector<std::uint16_t> scale_bits(kRows);
    std::vector<float> scales(kRows);
    for (std::int32_t row = 0; row < kRows; ++row) {
        scales[row]     = (row % 3 == 0) ? 0.5F : ((row % 3 == 1) ? 0.25F : 1.0F);
        scale_bits[row] = to_bf16_bits(scales[row]);
    }

    std::vector<std::uint16_t> low_rank(static_cast<std::size_t>(kRank) * tokens);
    for (int token = 0; token < tokens; ++token) {
        for (std::int32_t column = 0; column < kRank; ++column) {
            const int step = (column * 11 + token * 5) % 9;
            low_rank[static_cast<std::size_t>(column) + static_cast<std::size_t>(kRank) * token] =
                to_bf16_bits(static_cast<float>(step) * 0.5F - 2.0F);
        }
    }

    const auto normalized = make_normalized(tokens);
    const auto partials   = make_partials(tokens);

    // Oracle: decode the E4M3 code, scale it, dot with the BF16 low rank in FP64, then the same
    // gate mix and injection the helper oracle states.
    std::vector<float> want_block_input(static_cast<std::size_t>(kHidden) * tokens);
    std::vector<float> want_injection(static_cast<std::size_t>(kStreams) * tokens);
    std::vector<double> logits(static_cast<std::size_t>(kRows) * tokens);
    for (std::int32_t row = 0; row < kRows; ++row) {
        for (int token = 0; token < tokens; ++token) {
            double sum = 0.0;
            for (std::int32_t column = 0; column < kRank; ++column) {
                const float weight =
                    decode_e4m3(codes[static_cast<std::size_t>(row) * kRank + column]) *
                    scales[row];
                __nv_bfloat16 activation;
                std::memcpy(&activation,
                            &low_rank[static_cast<std::size_t>(column) +
                                      static_cast<std::size_t>(kRank) * token],
                            sizeof(activation));
                sum +=
                    static_cast<double>(weight) * static_cast<double>(__bfloat162float(activation));
            }
            logits[static_cast<std::size_t>(row) * tokens + token] = sum;
        }
    }
    for (int token = 0; token < tokens; ++token) {
        for (std::int32_t position = 0; position < kHidden; ++position) {
            double sum = 0.0;
            for (std::int32_t stream = 0; stream < kStreams; ++stream) {
                const double logit = static_cast<double>(bf16_round(static_cast<float>(
                    logits[static_cast<std::size_t>(stream * kHidden + position) * tokens +
                           token])));
                const double gate  = 1.0 / (1.0 + std::exp(-logit));
                __nv_bfloat16 value;
                std::memcpy(
                    &value,
                    &normalized[static_cast<std::size_t>(position) +
                                static_cast<std::size_t>(kHidden) * (stream + kStreams * token)],
                    sizeof(value));
                sum += gate * static_cast<double>(__bfloat162float(value));
            }
            want_block_input[static_cast<std::size_t>(position) +
                             static_cast<std::size_t>(kHidden) * token] =
                static_cast<float>(sum * 0.25);
        }
        for (std::int32_t destination = 0; destination < kStreams; ++destination) {
            double sum = 0.0;
            for (std::int32_t source = 0; source < kStreams; ++source) {
                sum += static_cast<double>(
                    partials[static_cast<std::size_t>(token) * hc::kInjectionPartialsPerToken +
                             source * kStreams + destination]);
            }
            want_injection[static_cast<std::size_t>(destination) + kStreams * token] =
                static_cast<float>(sum);
        }
    }

    // THE FP8 WEIGHT IS ONE PAYLOAD, not two allocations: the codes at offset 0 and the BF16 scale
    // plane at align_up(n*k, 256), which is what validate_fp8_weight checks. Its other conditions
    // are that group_size and group equal k, that shape and padded_shape are [n,k,1,1], that
    // scale_ne is [n,1,1,1] and scale_nb is {2, n*2, n*2, n*2}, and that qdata and scales are
    // 16-byte aligned.
    const std::size_t code_plane_bytes = codes.size();
    const std::size_t scale_plane_offset =
        (code_plane_bytes + 255) & ~static_cast<std::size_t>(255);
    const std::size_t scale_plane_bytes = scale_bits.size() * sizeof(std::uint16_t);
    // The payload is assembled on the HOST and uploaded in one copy: weight_base is a DEVICE
    // pointer, so a host std::memcpy into it would fault.
    std::vector<std::byte> host_payload(scale_plane_offset + scale_plane_bytes, std::byte{0});
    std::memcpy(host_payload.data(), codes.data(), code_plane_bytes);
    std::memcpy(host_payload.data() + scale_plane_offset, scale_bits.data(), scale_plane_bytes);
    GuardedDeviceBuffer device_weight_payload(host_payload.size());
    device_weight_payload.copy_from_host(host_payload.data(), host_payload.size());
    auto* weight_base = static_cast<std::byte*>(device_weight_payload.data());
    GuardedDeviceBuffer device_low_rank(low_rank.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_normalized(normalized.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_partials(partials.size() * sizeof(float));
    GuardedDeviceBuffer device_block_input(want_block_input.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_injection(want_injection.size() * sizeof(std::uint16_t));
    device_low_rank.copy_from_host(low_rank.data(), device_low_rank.bytes());
    device_normalized.copy_from_host(normalized.data(), device_normalized.bytes());
    device_partials.copy_from_host(partials.data(), device_partials.bytes());
    device_block_input.fill(0xcd);
    device_injection.fill(0xcd);

    Weight weight;
    weight.device          = Device::cuda(0);
    weight.qtype           = QType::FP8_E4M3FN_ROW_BF16;
    weight.layout          = QuantLayout::RowScale;
    weight.scale_dtype     = DType::BF16;
    weight.payload         = weight_base;
    weight.payload_bytes   = static_cast<std::uint64_t>(scale_plane_offset + scale_plane_bytes);
    weight.qdata           = weight_base;
    weight.scales          = weight_base + scale_plane_offset;
    weight.n               = kRows;
    weight.k               = kRank;
    weight.group           = kRank;
    weight.group_size      = static_cast<std::uint32_t>(kRank);
    weight.ndim            = 2;
    weight.shape[0]        = kRows;
    weight.shape[1]        = kRank;
    weight.shape[2]        = 1;
    weight.shape[3]        = 1;
    weight.padded_shape[0] = kRows;
    weight.padded_shape[1] = kRank;
    weight.padded_shape[2] = 1;
    weight.padded_shape[3] = 1;
    weight.scale_ne[0]     = kRows;
    weight.scale_ne[1]     = 1;
    weight.scale_ne[2]     = 1;
    weight.scale_ne[3]     = 1;
    weight.scale_nb[0]     = 2;
    weight.scale_nb[1]     = kRows * 2;
    weight.scale_nb[2]     = kRows * 2;
    weight.scale_nb[3]     = kRows * 2;

    Tensor low_rank_tensor(device_low_rank.span(), DType::BF16, {kRank, tokens});
    Tensor normalized_tensor(device_normalized.span(), DType::BF16, {kRows, tokens});
    Tensor partials_tensor(device_partials.span(), DType::FP32,
                           {hc::kInjectionPartialsPerToken, tokens});
    Tensor block_input_tensor(device_block_input.span(), DType::BF16, {kHidden, tokens});
    Tensor injection_tensor(device_injection.span(), DType::BF16, {kStreams, tokens});

    try {
        ninfer::ops::detail::flash_next::launch_fp8_hc_up_mix(
            low_rank_tensor, weight, normalized_tensor, &partials_tensor, block_input_tensor,
            &injection_tensor, nullptr);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: the up-mix route threw: " << error.what() << "\n";
        return ++failures;
    }
    cuda_synchronize();

    const auto got_block_input =
        from_device<std::uint16_t>(device_block_input.data(), want_block_input.size());
    double worst = 0.0;
    for (std::size_t index = 0; index < want_block_input.size(); ++index) {
        __nv_bfloat16 value;
        std::memcpy(&value, &got_block_input[index], sizeof(value));
        const double got   = static_cast<double>(__bfloat162float(value));
        const double want  = static_cast<double>(want_block_input[index]);
        const double scale = std::max(1.0, std::fabs(want));
        worst              = std::max(worst, std::fabs(got - want) / scale);
    }
    if (worst > kRouteRelativeTolerance) {
        std::cerr << "FAIL: block_input deviates by " << worst << " relative, above "
                  << kRouteRelativeTolerance << "\n";
        ++failures;
    }

    const auto got_injection =
        from_device<std::uint16_t>(device_injection.data(), want_injection.size());
    for (std::size_t index = 0; index < want_injection.size(); ++index) {
        const __nv_bfloat16 want = __float2bfloat16_rn(want_injection[index]);
        std::uint16_t want_bits  = 0;
        std::memcpy(&want_bits, &want, sizeof(want_bits));
        if (got_injection[index] != want_bits) {
            std::cerr << "FAIL: injection differs at " << index << ": got " << got_injection[index]
                      << " want " << want_bits << "\n";
            ++failures;
            break;
        }
    }
    return failures;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: the HyperConnection math test needs a CUDA device\n";
        return 77;
    }
    failures += run_gate_mix(1);
    failures += run_gate_mix(3);
    failures += run_injection(1);
    failures += run_injection(3);
    failures += run_up_mix_route(3);
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "hyperconnection math: ok\n";
    return 0;
}
