# Flash-Next mathematical and complete Engine qualification.
ninfer_add_op_test(ninfer_hyperconnection_test SOURCES ops/test_hyperconnection.cpp LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_flash_next_ple_test SOURCES ops/test_flash_next_ple.cpp LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_flash_next_ple_stage_test SOURCES ops/test_flash_next_ple_stage.cpp LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_flash_next_moe_test
  SOURCES ops/test_flash_next_moe.cpp ops/poison_device_state.cu LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_flash_next_qsa_test SOURCES ops/test_flash_next_qsa.cpp LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_flash_next_gdn_test SOURCES ops/test_flash_next_gdn.cpp LIBRARIES ninfer_ops)
# The complete Engine qualification binaries register with the Flash-Next engine route
# (FlashNextInstance port, engine-side statistics surfaces, fault-injection seam).

ninfer_add_test(ninfer_qwen3_8_flash_next_engine_real_test
  SOURCES models/qwen3_8_flash_next_125b_a6b/test_engine_real.cpp
  LIBRARIES ninfer_engine ninfer_artifact)
# Each exercise loads the ~70 GiB artifact; the tests need the GPU to themselves.
set_tests_properties(ninfer_qwen3_8_flash_next_engine_real_test PROPERTIES
  SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)

ninfer_add_test(ninfer_qwen3_8_flash_next_engine_fault_test
  SOURCES models/qwen3_8_flash_next_125b_a6b/test_engine_fault.cpp
  LIBRARIES ninfer_engine ninfer_artifact)
target_compile_definitions(ninfer_qwen3_8_flash_next_engine_fault_test PRIVATE
  NINFER_ENGINE_FAULT_INJECTION=1)
set_tests_properties(ninfer_qwen3_8_flash_next_engine_fault_test PROPERTIES
  SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)

add_test(NAME ninfer_flash_next_performance_report_test
  COMMAND ${Python3_EXECUTABLE} -m unittest tests/test_flash_next_performance.py)
set_tests_properties(ninfer_flash_next_performance_report_test PROPERTIES WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

ninfer_add_test(ninfer_qwen3_8_flash_next_ple_table_test
  SOURCES models/qwen3_8_flash_next/test_ple_table.cpp LIBRARIES ninfer_model_loading ninfer_model_control ninfer_model_execution)
# The load-path qualification test reads the master MaterializationPlan surface; it registers
# with the engine-port item that adopts that plan.

if(NINFER_PERFORMANCE_TRACE)
  ninfer_add_test(ninfer_performance_trace_fixture
    SOURCES performance_trace_fixture.cu LIBRARIES ninfer_ops)
endif()
ninfer_add_test(ninfer_qwen3_8_flash_next_tool_call_parser_test
  SOURCES models/qwen3_8_flash_next/test_tool_call_parser.cpp LIBRARIES ninfer_model_loading ninfer_model_control ninfer_model_execution)
ninfer_add_test(ninfer_qwen3_8_flash_next_frontend_test
  SOURCES models/qwen3_8_flash_next/test_frontend.cpp LIBRARIES ninfer_model_loading ninfer_model_control ninfer_model_execution)
set_tests_properties(ninfer_qwen3_8_flash_next_frontend_test PROPERTIES SKIP_RETURN_CODE 77)
