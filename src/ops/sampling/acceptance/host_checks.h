#pragma once

// Host-side shape and dtype checks for the speculative round's device ops. Moved here from
// ops/wrapper/speculative_round.cpp when that wrapper was retired: the four entries it held are now
// owned by the restructured acceptance tree, and the checks they share are one contract rather than
// four copies.

#include "core/tensor.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops::acceptance_detail {

inline void require_contiguous_nonnull(const Tensor& t, const char* op, const char* name) {
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

inline void require_dtype(const Tensor& t, DType dtype, const char* op, const char* name) {
    if (t.dtype != dtype) {
        throw std::invalid_argument(std::string(op) + ": invalid dtype for " + name);
    }
    require_contiguous_nonnull(t, op, name);
}

inline void require_vector(const Tensor& t, DType dtype, std::int32_t n, const char* op,
                           const char* name) {
    require_dtype(t, dtype, op, name);
    if (n <= 0 || t.ne[0] != n || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid vector shape for " + name);
    }
}

inline void require_matrix(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t cols,
                           const char* op, const char* name) {
    require_dtype(t, dtype, op, name);
    if (rows <= 0 || cols <= 0 || t.ne[0] != rows || t.ne[1] != cols || t.ne[2] != 1 ||
        t.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid matrix shape for " + name);
    }
}

} // namespace ninfer::ops::acceptance_detail
