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
#include "ops/launcher/hyperconnection_math.cuh"
#include "ops/op_tester.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

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
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "hyperconnection math: ok\n";
    return 0;
}
