# Master-generation BF16 vendor route: the Bf16GemmContext implementation. The shared dispatch
# and shape kernels stay owned by the restructured projection layer.
target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bf16_cublas.cpp"
)
target_link_libraries(ninfer_ops PRIVATE CUDA::cublas)

include("${CMAKE_CURRENT_LIST_DIR}/flash_next/sources.cmake")
