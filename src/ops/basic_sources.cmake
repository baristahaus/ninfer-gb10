# Restored Flash-Next generation ops: launchers, low-level kernels and the
# no-counterpart public wrappers. Shared semantic ops (sampling, rmsnorm, rope,
# argmax, residual_add, ...) stay owned by the restructured ops layer; Flash-Next
# call sites adapt to those APIs. HyperConnection and PLE kernels live in
# gated_residual/ and ple/ in the restructured layer; the master-generation
# launcher duplicates are not built.

set(NINFER_OPS_DIR "${CMAKE_CURRENT_LIST_DIR}")

# Low-level CUDA kernels for the restored ops.
foreach(_f IN ITEMS
    vision_pos_embed.cuh)
  if(EXISTS "${NINFER_OPS_DIR}/kernel/${_f}")
    target_sources(ninfer_ops PRIVATE "${NINFER_OPS_DIR}/kernel/${_f}")
  endif()
endforeach()

# Launchers for ops without a restructured-layer counterpart.
foreach(_f IN ITEMS
    vision_pos_embed.cu gdn_gating.cu)
  if(EXISTS "${NINFER_OPS_DIR}/launcher/${_f}")
    target_sources(ninfer_ops PRIVATE "${NINFER_OPS_DIR}/launcher/${_f}")
  endif()
endforeach()

# The MTP round advance has no restructured-layer counterpart. target_logprobs used to be listed
# here as a master-generation overload beside the restructured reduction route; the two were merged
# into src/ops/reduction/target_logprobs.cu, which now owns both the scalar and the ranking form.
target_sources(ninfer_ops PRIVATE
  "${NINFER_OPS_DIR}/wrapper/mtp_round.cpp"
  "${NINFER_OPS_DIR}/launcher/mtp_round.cu"
  "${NINFER_OPS_DIR}/wrapper/speculative_round.cpp"
  "${NINFER_OPS_DIR}/launcher/speculative_round.cu"
  "${NINFER_OPS_DIR}/wrapper/mtp_pack.cpp"
  "${NINFER_OPS_DIR}/launcher/mtp_pack.cu"
  "${NINFER_OPS_DIR}/wrapper/argmax.cpp"
  "${NINFER_OPS_DIR}/launcher/argmax.cu")

# Public wrappers for the same no-counterpart ops.
foreach(_f IN ITEMS
    vision_pos_embed.cpp
    gdn_gating.cpp gdn_input_proj_exl3.cpp)
  target_sources(ninfer_ops PRIVATE "${NINFER_OPS_DIR}/wrapper/${_f}")
endforeach()

# Restored Flash-Next BF16 bank kernels (expert-major grouped GEMM/GEMV routes).
file(GLOB _ninfer_fn_bf16_sources "${NINFER_OPS_DIR}/linear/bf16/flash_next/*.cu")
target_sources(ninfer_ops PRIVATE ${_ninfer_fn_bf16_sources})
# Master-generation HyperConnection entry points (stream and stream+Bf16GemmContext forms); the
# Flash-Next family and its qualification tests call these directly.
target_sources(ninfer_ops PRIVATE
  "${NINFER_OPS_DIR}/launcher/hyperconnection.cu"
  "${NINFER_OPS_DIR}/launcher/flash_next_ple.cu"
  "${NINFER_OPS_DIR}/launcher/flash_next_ple_stage.cu"
  "${NINFER_OPS_DIR}/linear_attention/flash_next_gdn.cu"
)
