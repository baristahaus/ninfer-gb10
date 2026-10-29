target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/exl3_dense_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/exl3_prefill.cu"
  "${CMAKE_CURRENT_LIST_DIR}/tensorfold/exl3_linear.cu"
)
