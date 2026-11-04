#include "ninfer/ops/rmsnorm_pack_tail.h"
#include "core/device.h"
#include "ops/common/host/operands.h"
#include "ops/launcher/rmsnorm_pack_tail.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {
void require_bf16_contiguous_aligned(const Tensor& tensor, const char* name) {
    if (tensor.dtype != DType::BF16 || !tensor.is_contiguous() || tensor.data == nullptr ||
        (reinterpret_cast<std::uintptr_t>(tensor.data) & 15U) != 0) {
        throw std::invalid_argument(std::string("rmsnorm_pack_tail: ") + name +
                                    " must be contiguous 16-byte-aligned BF16");
    }
}
} // namespace

void rmsnorm_pack_tail(const Tensor& input, const Tensor& weight, Tensor& output,
                       CudaExecutionView execution) {
    if (input.device != execution.device() || weight.device != execution.device() ||
        output.device != execution.device()) {
        throw std::invalid_argument(
            "rmsnorm_pack_tail: operands must belong to the execution device");
    }
    const auto stream = execution.stream();
    require_bf16_contiguous_aligned(input, "input");
    require_bf16_contiguous_aligned(weight, "weight");
    require_bf16_contiguous_aligned(output, "output");

    const std::int32_t batch = input.ne[2];
    const std::int32_t width = input.ne[1];
    const bool registered_rows = input.ne[0] == 5120 || input.ne[0] == 2560;
    if (!registered_rows || (width < 2 || width > 16) || batch < 1 || batch > 8 ||
        input.ne[3] != 1) {
        throw std::invalid_argument(
            "rmsnorm_pack_tail: input must have shape [D,W,B], D in {2560,5120}, W=2..16, B=1..8");
    }
    const std::int32_t rows = input.ne[0];
    if (weight.ne[0] != rows || weight.ne[1] != 1 || weight.ne[2] != 1 || weight.ne[3] != 1) {
        throw std::invalid_argument("rmsnorm_pack_tail: weight must have shape [D]");
    }
    if (output.ne[0] != rows || output.ne[1] != (width - 1) * batch || output.ne[2] != 1 ||
        output.ne[3] != 1) {
        throw std::invalid_argument("rmsnorm_pack_tail: output must have shape [D,(W-1)B]");
    }
    const common::operand::Range ranges[]{
        common::operand::range(input.data, input.bytes(), "rmsnorm_pack_tail"),
        common::operand::range(weight.data, weight.bytes(), "rmsnorm_pack_tail"),
        common::operand::range(output.data, output.bytes(), "rmsnorm_pack_tail"),
    };
    common::operand::disjoint(ranges, "rmsnorm_pack_tail");

    detail::rmsnorm_pack_tail_launch(input, weight, output, stream);
}

} // namespace ninfer::ops
