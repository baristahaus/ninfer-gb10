ninfer_add_op_test(ninfer_column_topk_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/column_topk.cpp"
  LIBRARIES ninfer_ops)
