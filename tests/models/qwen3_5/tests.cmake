ninfer_add_test(ninfer_qwen3_5_loading_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading_real.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen3_5_loading_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading.cpp"
  LIBRARIES ninfer_model_loading)

set_tests_properties(
  ninfer_qwen3_5_loading_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_frontend_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_frontend.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer_core ninfer::json)

ninfer_add_test(ninfer_qwen3_5_runtime_mechanisms_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_runtime_mechanisms.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_state_image_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state_image.cpp"
  LIBRARIES ninfer_engine ninfer_core)

set_tests_properties(
  ninfer_qwen3_5_state_image_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_state_image_layout_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state_image_layout.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_context_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_context_store.cpp"
  LIBRARIES ninfer_engine ninfer_core)

set_tests_properties(
  ninfer_qwen3_5_context_store_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_prefix_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_prefix_real.cpp"
  LIBRARIES ninfer_engine ninfer::json)

set_tests_properties(
  ninfer_qwen3_5_prefix_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_score_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_score_real.cpp"
  LIBRARIES ninfer_engine)

set_tests_properties(
  ninfer_qwen3_5_score_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_vision_workspace_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_workspace.cpp"
  LIBRARIES ninfer_model_runtime ninfer_engine)

set_tests_properties(
  ninfer_qwen3_5_vision_workspace_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_dflash2_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash2_real.cpp"
  LIBRARIES ninfer_engine)

set_tests_properties(
  ninfer_qwen3_5_dflash2_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_moe_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_moe_real.cpp"
  LIBRARIES ninfer_engine)

set_tests_properties(
  ninfer_qwen3_5_moe_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_dflash_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash_real.cpp"
  LIBRARIES ninfer_engine)

set_tests_properties(
  ninfer_qwen3_5_dflash_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_tool_call_parser_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../../test_tool_call_parser.cpp"
  LIBRARIES ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_visual_scatter_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_visual_scatter.cpp"
  LIBRARIES ninfer_engine ninfer_core)

set_tests_properties(
  ninfer_qwen3_5_visual_scatter_test
  PROPERTIES SKIP_RETURN_CODE 77)

# Standalone repros for the open 5090-test engine bugs (ninfer-gb10-hdt, ninfer-gb10-dn9).
# They are skipped under plain CTest (no artifact) and run individually with
# NINFER_TEST_ARTIFACT set to the 27B artifact. See docs/maintainer/5090-test-guidance.md.
ninfer_add_test(ninfer_qwen3_5_repro_vision_reuse_divergence
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/repro_vision_reuse_divergence.cpp"
  LIBRARIES ninfer_engine)

set_tests_properties(
  ninfer_qwen3_5_repro_vision_reuse_divergence
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_repro_host_restore_state
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/repro_host_restore_state.cpp"
  LIBRARIES ninfer_engine)

set_tests_properties(
  ninfer_qwen3_5_repro_host_restore_state
  PROPERTIES SKIP_RETURN_CODE 77)
