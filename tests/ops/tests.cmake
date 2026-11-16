add_library(ninfer_op_test_support STATIC "${CMAKE_CURRENT_LIST_DIR}/suite.cpp")
ninfer_test_includes(ninfer_op_test_support)
ninfer_op_oracle_options(ninfer_op_test_support)
target_link_libraries(ninfer_op_test_support PUBLIC ninfer_cpu CUDA::cudart)

include("${CMAKE_CURRENT_LIST_DIR}/elementwise/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/column_topk/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/embedding/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/indexing/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/reduction/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/normalization/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/rope/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sequence/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sampling/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear/tests.cmake")

# The HyperConnection elementwise mathematics on its own, against an FP64 oracle. It is a standalone
# binary rather than a case in an op suite because the helpers are __device__ functions the op
# suites never reach directly, and because the two fp8 kernel trees are merged on the strength of
# this oracle. Compiled as a .cu, skips without a device.
ninfer_add_test(ninfer_hyperconnection_math_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/hyperconnection_math_test.cu"
  LIBRARIES ninfer_core CUDA::cudart)
set_tests_properties(ninfer_hyperconnection_math_test PROPERTIES SKIP_RETURN_CODE 77)
include("${CMAKE_CURRENT_LIST_DIR}/dynamic_grouped_conv/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gated_residual/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_attention/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/kv_cache/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/softmax_attention/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/moe/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ple/tests.cmake")

# Host-only: the EXL3 codec decoder against the exllamav3 fixture (skips until it is generated).
ninfer_add_op_test(ninfer_exl3_codec_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_exl3_codec.cpp"
  LIBRARIES ninfer::json
  NEEDS_SOURCE_DIR)
