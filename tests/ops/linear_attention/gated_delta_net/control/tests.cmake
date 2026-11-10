ninfer_add_op_test(ninfer_gdn_control_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/test_control.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/fixtures.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/oracle.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/cases.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/runner.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/validation.cpp"
  LIBRARIES ninfer_ops)

# Host-only: the control projection's head-count granularity and the MMA tile straddle.
ninfer_add_op_test(ninfer_gdn_control_granularity_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_geometry_granularity.cu"
  LIBRARIES ninfer_ops)

# Host-only: the control projection's head geometry, the padded M tile and its route table.
ninfer_add_op_test(ninfer_gdn_control_geometry_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_geometry_granularity.cu"
  LIBRARIES ninfer_ops)
