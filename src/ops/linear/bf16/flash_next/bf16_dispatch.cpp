#include "ops/linear/bf16/flash_next/bf16_dispatch.h"

#include "ops/linear/bf16/flash_next/bf16_config.h"
#include "ops/linear/bf16/flash_next/bf16_launch.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail::flash_next {

Bf16Launch select_bf16_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    const bool flash_next_problem =
        (n == 320 && k == 10240) || (n == 48 && k == 2560) ||
        (n == 10240 && k == 320) ||
        (n == 12288 && k == 2560) || (n == 512 && k == 2560) ||
        (n == 128 && k == 2560) ||
        (n == 2560 && k == 6144) || (n == 640 && k == 2560) ||
        (n == 10240 && k == 2560) || (n == 6144 && k == 2560) ||
        (n == 2560 && k == 640) || (n == 2560 && k == 2560) ||
        (n == 248320 && k == 2560) || (n == 117248 && k == 2560);
    if (!flash_next_problem || t <= 0) {
        throw std::invalid_argument("bf16 linear: unsupported shape or T");
    }
    if (t == 1) { return launch_bf16_decode; }
    if (flash_next_problem && t <= 16) { return launch_bf16_small_t; }

    return launch_bf16_mma;
}

Bf16Launch select_bf16_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
        return select_bf16_a16_launch(n, k, t);
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("bf16 linear: unsupported policy");
}

void bf16_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                   cudaStream_t stream) {
    const Bf16Launch launch = select_bf16_launch(weight.n, weight.k, x.ne[1], policy);
    launch(x, weight, out, stream);
}

} // namespace ninfer::ops::detail::flash_next
