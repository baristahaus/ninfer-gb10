#pragma once
#include "core/weight.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

class Bf16GemmContext;

// query_key_value, output_gate and output may be BF16 or row-scaled FP8 (fp8_e4m3fn_row_bf16);
// the a/b control projections are BF16.
struct FlashNextGdnWeights {
    Tensor a_log;
    Tensor dt_bias;
    Tensor convolution;
    Weight a_projection;
    Weight b_projection;
    Weight query_key_value;
    Weight output_gate;
    Tensor norm;
    Weight output;
};

[[nodiscard]] std::size_t flash_next_gdn_workspace_capacity_bytes(std::int32_t tokens);

// Exact single-sequence Flash-Next Gated DeltaNet block. The input is the 2560-row HC block
// stream. The width-three BF16 convolution state and [128,128,48] FP32 recurrence state are
// transitioned from the supplied source to destination; exact alias is allowed for each pair.
// The BF16 [2560,T] destination is overwritten with the projected block result. `execution`
// supplies the stream and the physical SM count the chunked recurrence decomposes over.
void flash_next_gdn(const Tensor& input, const FlashNextGdnWeights& weights,
                    const Tensor& convolution_state_in, Tensor& convolution_state_out,
                    const Tensor& recurrent_state_in, Tensor& recurrent_state_out,
                    Tensor& destination, WorkspaceArena& workspace,
                    DeviceExecutionView execution, Bf16GemmContext* bf16_gemm = nullptr);

// One-token exact-B selected-slot transition used by ordinary decode.
void flash_next_gdn_batch_update(const Tensor& input, const FlashNextGdnWeights& weights,
                                 Tensor& convolution_states, Tensor& recurrent_states,
                                 const Tensor& source_slots, const Tensor& destination_slots,
                                 Tensor& destination, WorkspaceArena& workspace,
                                 cudaStream_t stream);

void flash_next_gdn_replay_record(const Tensor& input, const FlashNextGdnWeights& weights,
                                  const Tensor& convolution_states,
                                  const Tensor& recurrent_states,
                                  const Tensor& valid_columns, const Tensor& source_slots,
                                  GdnReplayRecordLayer records, Tensor& destination,
                                  WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
