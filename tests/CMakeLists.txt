find_package(Python3 REQUIRED COMPONENTS Interpreter)

include(cmake/NinferTests.cmake)
include(cmake/CoreTests.cmake)
include(artifact/tests.cmake)
include(cmake/BenchSupportTests.cmake)
if(NINFER_BUILD_ENGINE)
  include(text/tests.cmake)
  include(cmake/RuntimeTests.cmake)
  include(cmake/ProductTests.cmake)
  include(models/qwen3_5/tests.cmake)
  include(models/common/tests.cmake)
  include(models/qwen3_8_flash_next/tests.cmake)
  include(models/qwen3_8_flash_next_125b_a6b/tests.cmake)
  if(NINFER_TARGET STREQUAL "rtx5090")
    include(models/common/moe/tests.cmake)
  endif()
  include(models/qwen4_exp/tests.cmake)

  # The row-parallel head's logits gather: the host cases check the padded-sum layout and the
  # refusals, the device cases run the real exchange between two devices and skip without them.
  ninfer_add_test(ninfer_sharded_logits_gather_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sharded_logits_gather.cpp"
    LIBRARIES ninfer_model_execution ninfer_core CUDA::cudart)
  set_tests_properties(ninfer_sharded_logits_gather_test PROPERTIES SKIP_RETURN_CODE 77)
endif()
include(ops/tests.cmake)
include(distributed/tests.cmake)

# Resolve the test-artifact manifest last: it sets each artifact variable only on the tests that
# are registered above and only when the artifact is present, so an absent entry means skip.
include(cmake/TestArtifacts.cmake)
ninfer_register_test_artifacts()
