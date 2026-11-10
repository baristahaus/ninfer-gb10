#pragma once

#include "core/device.h"
#include "core/tensor.h"

#include <cstdint>

namespace ninfer::ops {

/**
 * YaRN (Yet another RoPE extensioN) NTK-by-parts frequency scaling, opt-in. `factor <= 1` is the
 * disabled state and the Op then takes exactly the plain-RoPE path.
 *
 * Only the Text geometries are scaled: Text 1-D at head_dim 256 and Text MRoPE at head_dim 256 /
 * rotary_dim 64. Vision 2-D has its own effective dimension and table, so passing a disabled value
 * there is the only supported use.
 *
 * The remaining fields are the reference's `_compute_yarn_parameters` inputs and must come from the
 * model rather than be guessed: `original_max_position` is the model's TRUE native window (never
 * scaled) and `beta_fast`/`beta_slow` are the paper's 32 and 1 unless the model's configuration
 * says otherwise.
 */
struct RopeYarn {
    float factor                       = 1.0F;    // 1.0 = disabled
    std::int32_t original_max_position = 262144;
    float beta_fast                    = 32.0F;
    float beta_slow                    = 1.0F;

    [[nodiscard]] constexpr bool enabled() const noexcept { return factor > 1.0F; }
};

/**
 * Applies split-half NeoX RoPE in place. For pair i in [0,rotary_dim/2), angle phi(i,t), and
 * each head:
 *
 *   y[i]     = x[i] * cos(phi) - x[i+R/2] * sin(phi)
 *   y[i+R/2] = x[i+R/2] * cos(phi) + x[i] * sin(phi), R = rotary_dim.
 *
 * Dimensions [rotary_dim,head_dim) are unchanged. Supported modes are:
 *
 * - Text 1-D: positions I32 [T], either head_dim=256 with even 0<rotary_dim<=256, or the
 *   head_dim=rotary_dim=128; phi=positions[t]*theta^(-2*i/rotary_dim).
 * - Text MRoPE: positions I32 [T,3], head_dim=256, rotary_dim=64; pair i uses axis i%3 with
 *   the same frequency as Text 1-D.
 * - Vision 2-D: positions I32 [T,2], head_dim=rotary_dim=72; pairs 0..17 use axis 0 and pairs
 *   18..35 use axis 1, each with local frequency theta^(-2*(i%18)/36).
 *
 * With `yarn.enabled()` the Text frequencies are replaced by the YaRN NTK-by-parts table and both
 * cos and sin are multiplied by the reference's attention temperature (HF applies it to cos/sin,
 * so the effective logit multiplier is the temperature squared). YaRN requires rotary_dim 64; any
 * other Text rotary width with `yarn.enabled()` is rejected rather than silently ignored. Vision
 * ignores `yarn` entirely.
 *
 * positions is contiguous and theta is positive and finite. Q/K tensors are BF16
 * [head_dim,heads,T] with positive head counts, contiguous head features and heads, and an optional
 * padded token stride. q and k must not overlap one another or positions. The Op mutates only
 * dimensions [0,rotary_dim); unrotated dimensions and token padding remain bit-exact. Empty token
 * extents perform no device work. The Op uses no workspace or persistent state. `execution`
 * supplies the stream and the selected device's positive physical SM count.
 */
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          CudaExecutionView execution);
void rope(const Tensor& positions, int rotary_dim, float theta, const RopeYarn& yarn, Tensor& q,
          Tensor& k, CudaExecutionView execution);

// Single-tensor form with the same formula and storage contract. The head count comes directly
// from x; Q versus K role does not change the transformation.
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
          CudaExecutionView execution);
void rope(const Tensor& positions, int rotary_dim, float theta, const RopeYarn& yarn, Tensor& x,
          CudaExecutionView execution);


// Stream forms for callers that carry the stream directly; validation is identical.
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          cudaStream_t stream);
void rope(const Tensor& positions, int rotary_dim, float theta, const RopeYarn& yarn, Tensor& q,
          Tensor& k, cudaStream_t stream);
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x, cudaStream_t stream);
void rope(const Tensor& positions, int rotary_dim, float theta, const RopeYarn& yarn, Tensor& x,
          cudaStream_t stream);

} // namespace ninfer::ops
