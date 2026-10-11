#include "poison_device_state.h"

#include "core/device.h"

#include <cuda_runtime.h>

namespace ninfer::test {
namespace {

__global__ void fill_shared_memory(std::uint32_t pattern, std::uint32_t words) {
    extern __shared__ std::uint32_t shared[];
    for (std::uint32_t i = threadIdx.x; i < words; i += blockDim.x) {
        // volatile so the stores are not elided: their only purpose is the stale contents.
        reinterpret_cast<volatile std::uint32_t*>(shared)[i] = pattern;
    }
}

} // namespace

void poison_shared_memory(std::uint32_t pattern) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int bytes = 0;
    int sms   = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaFuncSetAttribute(fill_shared_memory,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    // Several waves of maximal blocks on every SM leave the pattern across each SM's shared
    // memory carve-out for the next kernel to find.
    fill_shared_memory<<<sms * 4, 1024, static_cast<std::size_t>(bytes)>>>(
        pattern, static_cast<std::uint32_t>(bytes) / 4U);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void poison_device_bytes(void* data, std::size_t bytes, unsigned char value) {
    CUDA_CHECK(cudaMemset(data, value, bytes));
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace ninfer::test
