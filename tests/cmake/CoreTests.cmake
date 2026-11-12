# This target deliberately receives no src/, CUDA, artifact, kernel, or target
# include root. It proves that the public product headers stand alone.
add_executable(ninfer_public_api_test "${CMAKE_CURRENT_LIST_DIR}/../test_public_api.cpp")
target_include_directories(ninfer_public_api_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME ninfer_public_api_test COMMAND ninfer_public_api_test)

foreach(part IN ITEMS system task_pool)
  ninfer_add_test(ninfer_cpu_${part}_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/../cpu/test_${part}.cpp"
    LIBRARIES ninfer_cpu)
  set_tests_properties(ninfer_cpu_${part}_test PROPERTIES TIMEOUT 30)
endforeach()

if(NINFER_TARGET STREQUAL "rtx5090")
  foreach(part IN ITEMS x86 executor)
    ninfer_add_test(ninfer_cpu_${part}_test
      SOURCES "${CMAKE_CURRENT_LIST_DIR}/../cpu/test_${part}.cpp"
      LIBRARIES ninfer_cpu_executor)
    set_tests_properties(ninfer_cpu_${part}_test PROPERTIES TIMEOUT 30)
  endforeach()
  ninfer_add_test(ninfer_cpu_tensor_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/../cpu/test_tensor.cpp"
    LIBRARIES ninfer_core ninfer_cpu_executor)
endif()

ninfer_add_test(ninfer_device_test       SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_device.cpp"
  LIBRARIES ninfer_core)

set_tests_properties(ninfer_device_test PROPERTIES
  ENVIRONMENT_MODIFICATION "NINFER_CUDA_SYNC=unset:")
set(sync_modes spin blocking yield auto)
set(sync_flags 1 4 2 0)
foreach(mode flags IN ZIP_LISTS sync_modes sync_flags)
  add_test(NAME ninfer_device_sync_${mode}_test COMMAND ninfer_device_test ${flags})
  set_tests_properties(ninfer_device_sync_${mode}_test PROPERTIES
    ENVIRONMENT "NINFER_CUDA_SYNC=${mode}" SKIP_RETURN_CODE 77)
endforeach()
# A CUDA_CHECK failure reports through an exception instead of aborting; this asserts the report
# carries the file, the line and the driver's text, which is what makes a lane failure diagnosable.
add_test(NAME ninfer_device_cuda_check_test COMMAND ninfer_device_test --cuda-check-reports)
set_tests_properties(ninfer_device_cuda_check_test PROPERTIES SKIP_RETURN_CODE 77)

foreach(mode IN ITEMS invalid empty)
  add_test(NAME ninfer_device_sync_${mode}_test COMMAND ninfer_device_test --invalid-sync)
endforeach()
set_tests_properties(ninfer_device_sync_invalid_test PROPERTIES
  ENVIRONMENT "NINFER_CUDA_SYNC=invalid")
set_tests_properties(ninfer_device_sync_empty_test PROPERTIES
  ENVIRONMENT "NINFER_CUDA_SYNC=")

ninfer_add_test(ninfer_decode_graph_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_decode_graph.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_kv_mirror_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_kv_mirror.cpp"
  LIBRARIES ninfer_core)
set_tests_properties(ninfer_kv_mirror_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_tensor_test       SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_tensor.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_memory_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_memory.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_nf4_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_nf4.cpp"
  LIBRARIES ninfer_core)
add_test(NAME ninfer_nf4_exact_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/../test_nf4_exact.py"
    $<TARGET_FILE:ninfer_nf4_test>)

ninfer_add_test(ninfer_direct_io_filesystem_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../io/test_direct_io_filesystem.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_read_queue_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../io/test_read_queue.cpp"
  LIBRARIES ninfer_core PkgConfig::URING)
ninfer_test_interpose(ninfer_read_queue_test io_uring_submit)
set_tests_properties(ninfer_read_queue_test PROPERTIES TIMEOUT 30)

ninfer_add_test(ninfer_arena_test        SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_arena.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_spill_kv_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_spill_kv_store.cpp"
  LIBRARIES ninfer_core)
set_tests_properties(ninfer_spill_kv_store_test PROPERTIES TIMEOUT 30)

ninfer_add_test(ninfer_disk_kv_cache_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_disk_kv_cache.cpp"
  LIBRARIES ninfer_core)
set_tests_properties(ninfer_disk_kv_cache_test PROPERTIES TIMEOUT 300)

ninfer_add_test(ninfer_state_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_state_store.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_gdn_replay_records_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_gdn_replay_records.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_gdn_replay_shard_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_gdn_replay_shard.cpp"
  LIBRARIES ninfer_core)
set_tests_properties(ninfer_gdn_replay_shard_test PROPERTIES TIMEOUT 30)

set_tests_properties(
  ninfer_device_test
  ninfer_decode_graph_test
  ninfer_arena_test
  ninfer_state_store_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_host_timing_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_host_timing.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_weight_bank_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_weight_bank.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_expert_cache_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_expert_cache.cu"
          "${CMAKE_CURRENT_LIST_DIR}/../expert_cache_cuda_errors.cpp"
  LIBRARIES ninfer_core)
# The expert cache needs a device; a stub-driver build reports 77 and that is a skip, not a failure.
set_tests_properties(ninfer_expert_cache_test PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77)

ninfer_test_interpose(ninfer_expert_cache_test
  cudaEventRecord cudaMemcpyAsync cudaMemcpyBatchAsync cudaStreamSynchronize cudaStreamWaitEvent
  cudaFree cudaFreeHost)

# The documentation and source guard is a host-only script over the checked-in tree. It runs with
# the suite so a stale flag, path, preset or unregistered source fails a normal test run.
if(Python3_EXECUTABLE)
  add_test(NAME ninfer_docs_guard_test
    COMMAND ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/tools/check_docs.py
            ${PROJECT_SOURCE_DIR})
  set_tests_properties(ninfer_docs_guard_test PROPERTIES LABELS "host")

  # The guard's own teeth: it plants one defect of each class in a temporary tree.
  add_test(NAME ninfer_docs_guard_self_test
    COMMAND ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/tools/check_docs.py --self-test)
  set_tests_properties(ninfer_docs_guard_self_test PROPERTIES LABELS "host")

  # The artifact-fetch tool's own teeth: --link must not leave a cache entry on a filesystem the
  # engine's direct-I/O loader refuses, and --check must report one. It runs on temporary trees and
  # needs no artifact.
  add_test(NAME ninfer_fetch_artifacts_self_test
    COMMAND ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/tools/fetch_test_artifacts.py --self-test)
  set_tests_properties(ninfer_fetch_artifacts_self_test PROPERTIES LABELS "host")

  # The repository-weight guard: a tracked file above the per-file limit without an allowlist entry,
  # any tracked profiler trace, or a tracked tree above the total ceiling fails the suite. It exists
  # because a large blob that reaches history cannot be removed without rewriting every commit, so
  # the rule has to fire before the commit rather than after it.
  add_test(NAME ninfer_repo_weight_guard_test
    COMMAND ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/tools/check_repo_weight.py
            ${PROJECT_SOURCE_DIR})
  set_tests_properties(ninfer_repo_weight_guard_test PROPERTIES LABELS "host")
endif()

# The test-skip guard: a skip must carry a message on stderr and must not be counted as a pass.
# It runs with the suite so the next silent `return 77` fails a normal test run.
if(Python3_EXECUTABLE)
  add_test(NAME ninfer_test_skip_guard_test
    COMMAND ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/tools/check_test_skips.py
            ${PROJECT_SOURCE_DIR})
  set_tests_properties(ninfer_test_skip_guard_test PROPERTIES LABELS "host")

  add_test(NAME ninfer_test_skip_guard_self_test
    COMMAND ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/tools/check_test_skips.py --self-test)
  set_tests_properties(ninfer_test_skip_guard_self_test PROPERTIES LABELS "host")
endif()
