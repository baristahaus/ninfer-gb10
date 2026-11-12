#pragma once

#include "core/device.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::common {

// cudaFuncAttributeMaxDynamicSharedMemorySize is a property of a (function, device) pair, not of a
// function. A one-time process-wide set therefore admits the capacity on whichever device happened to
// be current at the first call and leaves every other device at the 48 KiB default, where the same
// kernel launch fails as cudaErrorInvalidValue. No single-device run can observe that, which is why
// it appears only on the second device of a multi-device round. Memoise per device instead.
template <auto Kernel>
void set_max_dynamic_shared(int bytes) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    // Per instantiation, so each kernel keeps its own memo; sized for the largest device ordinal the
    // driver can report, and an ordinal outside it simply sets the attribute every time.
    static std::array<std::uint8_t, 64> prepared{};
    if (device >= 0 && device < static_cast<int>(prepared.size())) {
        if (prepared[device] != 0) { return; }
        CUDA_CHECK(
            cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        prepared[device] = 1;
        return;
    }
    CUDA_CHECK(cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

// AlwaysDynamic is for kernels that declare extern shared storage at every size.
// Their static shared storage also counts against the default limit, so explicitly
// admit the requested dynamic capacity even when it alone fits within 48 KiB.
template <int Bytes, auto Kernel, bool AlwaysDynamic = false>
int prepare_shared() {
    static_assert(Bytes >= 0 && Bytes <= 99 * 1024);
    if constexpr (Bytes > 48 * 1024 || (AlwaysDynamic && Bytes > 0)) {
        set_max_dynamic_shared<Kernel>(Bytes);
        return Bytes;
    } else {
        return AlwaysDynamic ? Bytes : 0;
    }
}

// Slice only when the selected kernel would exceed grid.y; this is not model chunking.
inline constexpr std::int32_t kCudaGridYLimit = 65'535;

template <class Launch>
void for_each_token_slice(std::int32_t tokens, std::int32_t columns_per_block, Launch&& launch) {
    if (tokens <= 0 || columns_per_block <= 0) {
        throw std::invalid_argument("token slicing requires positive dimensions");
    }
    const std::int64_t capacity = static_cast<std::int64_t>(columns_per_block) * kCudaGridYLimit;
    std::int32_t offset         = 0;
    while (offset < tokens) {
        const std::int32_t count = static_cast<std::int32_t>(
            std::min<std::int64_t>(capacity, static_cast<std::int64_t>(tokens) - offset));
        launch(offset, count);
        offset += count;
    }
}

} // namespace ninfer::ops::common
