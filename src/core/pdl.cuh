#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace ninfer::pdl {

struct LaunchConfig {
    dim3 grid;
    dim3 block;
    std::size_t dynamic_smem_bytes = 0;
    cudaStream_t stream            = nullptr;
};

// Launches a consumer kernel as a programmatic dependent of the immediately preceding producer
// kernel in the same stream. Every consumer control path that reads producer output must first call
// wait_for_dependencies().
template <class... KernelArgs, class... CallArgs>
[[nodiscard]] inline cudaError_t
launch_dependent(const LaunchConfig& launch, void (*kernel)(KernelArgs...), CallArgs&&... args) {
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config{};
    config.gridDim          = launch.grid;
    config.blockDim         = launch.block;
    config.dynamicSmemBytes = launch.dynamic_smem_bytes;
    config.stream           = launch.stream;
    config.attrs            = &attribute;
    config.numAttrs         = 1;

    return cudaLaunchKernelEx(&config, kernel, std::forward<CallArgs>(args)...);
}

// Every producer CTA must call this at least once or exit. This enables dependent scheduling but
// does not make producer writes visible to the consumer; the trigger belongs where the launch
// overlap is wanted (typically the producer's start, so the dependent's producer-independent
// preamble and grid dispatch hide behind the producer's execution).
__device__ __forceinline__ void trigger_dependents() { cudaTriggerProgrammaticLaunchCompletion(); }

// Call on every consumer control path before its first access to producer-dependent data.
// griddepcontrol.wait blocks only while prerequisite grids are still in flight, and its
// visibility guarantee is tied to that blocking path: when the producer already completed
// before the consumer's CTAs reach the wait (a short producer overlapped by a longer-preamble
// consumer), the wait returns without an acquire and the consumer's first reads race the
// producer's L2 flush. The explicit gpu-scope acquire below pairs with the producer's grid
// completion release under every timing, so the reads after this call are always ordered.
__device__ __forceinline__ void wait_for_dependencies() {
    cudaGridDependencySynchronize();
    __threadfence();
}

} // namespace ninfer::pdl
