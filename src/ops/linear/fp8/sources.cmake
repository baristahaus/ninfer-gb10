# Master-generation FP8 Flash-Next bank kernels: the HyperConnection up-mix and down-SiLU
# launches the master HyperConnection and PLE entry points call.
target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/fp8_flash_next.cu"
)
