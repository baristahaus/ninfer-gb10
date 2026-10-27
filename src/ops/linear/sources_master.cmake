# Master-generation additions to the restructured linear layer: the Flash-Next NVFP4 format
# validation shared by the projection wrappers and dispatch routes, and the BF16 vendor
# tensor-core route (Bf16GemmContext over cuBLAS) with the Flash-Next BF16 bank kernels.
include("${CMAKE_CURRENT_LIST_DIR}/bf16/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/fp8/sources.cmake")
target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_format.cpp"
)
