# The registered 125B-A6B package qualification.
#
# - The load-plan test re-anchors the load-path qualification on the plan surface
#   (device objects and prepared-resource reads) and skips without the real artifact.
# - The Engine tests live with the Flash-Next family registration (they drive the
#   Flash-Next Engine route through the package's prepared artifact).

ninfer_add_test(ninfer_qwen3_8_flash_next_125b_a6b_load_plan_test
  SOURCES models/qwen3_8_flash_next_125b_a6b/test_load_plan.cpp
  LIBRARIES ninfer_model_loading ninfer_model_control ninfer_model_execution)
set_tests_properties(ninfer_qwen3_8_flash_next_125b_a6b_load_plan_test PROPERTIES
  SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
