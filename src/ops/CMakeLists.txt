# Compile policy is owned here; each semantic family explicitly contributes sources.
add_library(ninfer_ops SHARED)
ninfer_internal_includes(ninfer_ops)
ninfer_cuda_library(ninfer_ops)
target_link_libraries(ninfer_ops PUBLIC ninfer_core PRIVATE ninfer_ops_non_rdc)

# TEST-ONLY BUILD VARIANT: -Wl,--wrap makes the library's own CUDA calls go through the shims in
# tests/support/launch_counter.cpp, which are resolved at load from the test executable (which
# exports them). It exists because the counter's launch=0 column was a BLIND SPOT rather than a
# result: --wrap on the executable cannot see calls made from inside this shared library.
# OFF BY DEFAULT. The shipped libninfer_ops.so must never carry a wrap: a shipped library that
# rewrites its own CUDA calls is a liability, and a diagnostic must not change what ships.
option(NINFER_WRAP_OPS_LAUNCHES "Wrap the ops library's CUDA calls for the launch counter (test-only)" OFF)
if(NINFER_WRAP_OPS_LAUNCHES)
  target_link_options(ninfer_ops PRIVATE
    -Wl,--wrap=cudaLaunchKernel -Wl,--wrap=cudaMemcpyAsync -Wl,--wrap=cudaMemsetAsync
    -Wl,--wrap=cudaMemcpy2DAsync -Wl,--wrap=cudaMemcpy3DAsync -Wl,--wrap=cudaGraphLaunch
    -Wl,--wrap=cudaStreamWaitEvent)
  message(STATUS "ninfer_ops: CUDA calls wrapped for the launch counter (test-only build)")
endif()

# Non-RDC kernels keep instruction and register-allocation choices within one translation unit.
# This private PIC archive joins the shared library through host launchers.
add_library(ninfer_ops_non_rdc STATIC)
ninfer_internal_includes(ninfer_ops_non_rdc)
ninfer_cuda_non_rdc_archive(ninfer_ops_non_rdc)
target_link_libraries(ninfer_ops_non_rdc PRIVATE ninfer_core CUDA::cudart CUDA::cuda_driver)

target_sources(ninfer_ops PRIVATE weight_input.cpp)
include("${CMAKE_CURRENT_LIST_DIR}/column_topk/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/collective/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/common/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/elementwise/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/indexing/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/embedding/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/reduction/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sampling/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sequence/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/dynamic_grouped_conv/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gated_residual/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/basic_sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/attn_input_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gdn_gating_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sparse_moe/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/kv_cache/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear/sources.cmake")
# Master-generation BF16 vendor route (Bf16GemmContext) beside the restructured projection layer.
include("${CMAKE_CURRENT_LIST_DIR}/linear/sources_master.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ple/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/normalization/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/rope/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/softmax_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/moe/sources.cmake")
