#pragma once
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::ops {
inline constexpr int kSteeringMaxRank    = 32;
inline constexpr int kActivationLayers   = 48;
inline constexpr int kActivationWidth    = 2560;
inline constexpr int kActivationElements = 33280; // stream, normalized, gates, mixed
inline constexpr int kActivationRows     = 8;     // one per active request row
inline constexpr int kActivationColumns  = 8;     // widest MTP verification row

struct ActivationRow {
    int lane        = 0;
    int prompt_last = -1;
    int capture     = 0;
    float strength  = 0;
};

struct ActivationSample {
    int fires    = 0;
    int position = -1;
    int token    = -1;
    int pad      = 0;
};

// Device-resident control block. Every pointer and shape remains fixed across graph launches.
struct ActivationDevice {
    const float* directions = nullptr; // [48,MAX_RANK,4,2560]
    const int* ranks        = nullptr;
    const int* masks        = nullptr;
    int norm_preserve       = 0;
    int completion_capacity = 1; // Widest target-verification row for MTP, one otherwise.
    ActivationRow rows[kActivationRows]{};
    float* samples           = nullptr; // [concurrency,1+completion_capacity,48,33280]
    ActivationSample* checks = nullptr; // [concurrency,1+completion_capacity,48]
};

void activation_capture(const Tensor& hyper, const Tensor& normalized, const Tensor& gates,
                        const Tensor& mixed, const Tensor& positions, const Tensor& ids,
                        const Tensor& valid, const ActivationDevice* control, int layer, int width,
                        cudaStream_t stream, bool speculative_columns = false);
} // namespace ninfer::ops
