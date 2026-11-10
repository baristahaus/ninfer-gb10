ninfer_add_op_test(ninfer_rope_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/rope.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/rmsnorm_rope.cpp"
  LIBRARIES ninfer_ops)

# Host-only: the YaRN frequency math against an independent double-precision oracle. Needs no
# device, which is why it is a separate binary from the GPU rope suite above.
ninfer_add_op_test(ninfer_yarn_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_yarn.cpp")
