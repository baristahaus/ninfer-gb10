#pragma once

#include "ninfer/ops/dynamic_grouped_conv.h"

#include <stdexcept>

namespace ninfer::ops::detail::dynamic_grouped_conv {
// The op serves the text hidden (5120, groups 320) and the DFlash2 draft's hidden
// (2560, groups 160); every width-dependent constant derives from the tensor's H.
inline constexpr int kGroupWidth = 16;
inline constexpr int kTaps = 2, kSides = 2;

struct Shape {
    int hidden;
    int groups;
    int coefficient_rows;

    [[nodiscard]] static Shape of(int hidden) {
        if (hidden != 5120 && hidden != 2560) {
            throw std::invalid_argument("dynamic grouped conv: H must be 5120 or 2560");
        }
        const int groups = hidden / kGroupWidth;
        return {hidden, groups, groups * kTaps * kSides};
    }
};
inline constexpr const char* kPrepareOp = "dynamic grouped conv prepare";
inline constexpr const char* kAddOp     = "linear dynamic grouped conv add";

void validate_interval(int min_width, int max_width, int min_batch, int max_batch, const char* op);
void validate_input_rows(int input_rows);
void validate_prepare(const Tensor& residual, const Tensor& norm_weight, float epsilon,
                      const Tensor& base_kernel, const Weight& projection, const Tensor& prepared,
                      const Tensor& finish_delta, const MemoryArena& workspace,
                      CudaExecutionView execution);
// The prepare plan's split scratch scales with the coefficient rows of the shape.
std::size_t prepare_workspace_bytes(int width, int batch, int coefficient_rows, int splits);
void validate_add(const Tensor& x, const Weight& projection, const Tensor& base_kernel,
                  const Tensor& finish_delta, const Tensor& residual, const MemoryArena& workspace,
                  CudaExecutionView execution);
} // namespace ninfer::ops::detail::dynamic_grouped_conv
