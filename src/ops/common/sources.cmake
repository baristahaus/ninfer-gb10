target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/host/operands.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/device_info.cu"
  "${CMAKE_CURRENT_LIST_DIR}/peer_exchange.cu"
)
