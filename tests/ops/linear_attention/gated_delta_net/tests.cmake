ninfer_add_op_test(ninfer_gated_delta_net_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_recurrence.cpp"
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gated_delta_net_replay_record_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_record.cpp"
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_replay_fold_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fold.cpp"
  LIBRARIES ninfer_ops)

# Host-only: the recurrent path's head-count contract. Compiled as .cu because it reads HeadMap out
# of a kernel header, but it calls only host functions and runs without a device.
ninfer_add_op_test(ninfer_gdn_head_geometry_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_head_geometry.cu"
  LIBRARIES ninfer_ops)
