target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a4.cu"
)

# The single owner of the NVFP4 A4 TMA instantiations that more than one Op selects. The kernels are
# non-RDC, so a second copy in another translation unit reaches the module as a duplicate
# registration; see nvfp4_a4_tma_shared.h.
target_sources(ninfer_ops_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a4_tma_shared.cu"
)
