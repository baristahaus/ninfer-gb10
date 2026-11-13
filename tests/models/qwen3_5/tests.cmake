ninfer_add_test(ninfer_sharded_group_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sharded_group.cpp"
  LIBRARIES ninfer_model_execution)
set_tests_properties(ninfer_sharded_group_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_loading_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading_real.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen3_5_loading_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading.cpp"
  LIBRARIES ninfer_model_loading)

# The two-GPU lane loads but refuses to infer until the sharded forward is wired. Needs two GPUs
# and a 27B artifact; skips otherwise.
# NEEDS_SOURCE_DIR: the test reads its prompt and goldens from tests/fixtures/tp2_accept/, so both
# arms consume the same token ids without a tokenizer and the reference chain is recorded rather
# than only compared across hosts.
# The per-rank launch counter is linked into this test only: -Wl,--wrap makes every
# cudaLaunchKernel/cudaMemcpyAsync/cudaMemsetAsync/cudaGraphLaunch/cudaStreamWaitEvent in the
# process go through the shims in tests/support/launch_counter.cpp, which count per device (the
# device ordinal IS the rank here) and print the per-round delta at LocalExecutor's round
# boundary. It exists because the collective audit counts labels, not launches.
ninfer_add_test(ninfer_qwen3_5_tp2_accept_real_test NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_tp2_accept_real.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/../../support/launch_counter.cpp"
  LIBRARIES ninfer_engine CUDA::cudart CUDA::cuda_driver)
target_link_options(ninfer_qwen3_5_tp2_accept_real_test PRIVATE
  -Wl,--wrap=cudaLaunchKernel -Wl,--wrap=__cudaLaunchKernel -Wl,--wrap=cudaLaunchKernelExC -Wl,--wrap=cuLaunchKernel -Wl,--wrap=cudaMemcpyAsync -Wl,--wrap=cudaMemsetAsync
  -Wl,--wrap=cudaGraphLaunch -Wl,--wrap=cudaStreamWaitEvent
  -Wl,--wrap=cudaMemcpy2DAsync -Wl,--wrap=cudaMemcpy3DAsync
  # The per-site tags live in the model code inside the shared library; a weak symbol referenced
  # from there binds to this executable only if the executable exports it. --export-dynamic is on
  # the TEST binary alone; the shipped library's link line is not touched.
  -Wl,--export-dynamic)
set_tests_properties(ninfer_qwen3_5_tp2_accept_real_test PROPERTIES
  SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)

ninfer_add_test(ninfer_qwen3_5_tp2_refusal_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_tp2_refusal_real.cpp"
  LIBRARIES ninfer_engine CUDA::cudart)
set_tests_properties(ninfer_qwen3_5_tp2_refusal_real_test PROPERTIES
  SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)

ninfer_add_test(ninfer_qwen3_5_runtime_mechanisms_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_runtime_mechanisms.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_state_image_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state_image.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_state_image_layout_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state_image_layout.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_native_transactions_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_native_transactions.cpp"
  LIBRARIES ninfer_model_execution ninfer_model_loading ninfer_core)

ninfer_add_test(ninfer_qwen3_5_mtp_history_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_history_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_model_loading ninfer_core)

ninfer_add_test(ninfer_qwen3_5_prefix_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_prefix_real.cpp"
  LIBRARIES ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_preemption_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_preemption_real.cpp"
  LIBRARIES ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_grammar_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_grammar_real.cpp"
  LIBRARIES ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_tools_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_tools_real.cpp"
  LIBRARIES ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_score_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_score_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_score_w2_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_score_w2_real.cpp"
  LIBRARIES ninfer_engine CUDA::cudart)
# The test skips with 77 when it has no two-device host, no artifact or no fixture; the skip guard
# caught the missing registration before the suite would have counted a skip as a failure.
set_tests_properties(ninfer_qwen3_5_score_w2_real_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen3_5_vision_workspace_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_workspace.cpp"
  LIBRARIES ninfer_model_execution ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_dflash2_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash2_real.cpp"
  LIBRARIES ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_speculative_invariant_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_speculative_invariant_real.cpp"
  LIBRARIES ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_dflash_prefill_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dflash_prefill_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_model_loading ninfer_core)

ninfer_add_test(ninfer_qwen3_5_moe_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_moe_real.cpp"
  LIBRARIES ninfer_engine ninfer::json)

add_test(NAME ninfer_qwen3_5_moe_offload_real_test
  COMMAND ninfer_qwen3_5_moe_real_test --offload)
set_tests_properties(ninfer_qwen3_5_moe_offload_real_test PROPERTIES
  ENVIRONMENT "NINFER_VRAM_BUDGET_MIB=16384")

ninfer_add_test(ninfer_qwen3_5_dflash_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash_real.cpp"
  LIBRARIES ninfer_engine ninfer::json)

add_test(NAME ninfer_qwen3_5_agent_continuation_real_test
  COMMAND ninfer_qwen3_5_prefix_real_test)
set_tests_properties(ninfer_qwen3_5_agent_continuation_real_test PROPERTIES
  ENVIRONMENT "NINFER_PREFIX_REAL_SCENARIO=agent-continuation")

# A real-model test owns the single GPU while its artifact is resident.
set(ninfer_qwen3_5_real_tests
  ninfer_qwen3_5_loading_real_test
  ninfer_qwen3_5_native_transactions_test
  ninfer_qwen3_5_mtp_history_real_test
  ninfer_qwen3_5_prefix_real_test
  ninfer_qwen3_5_agent_continuation_real_test
  ninfer_qwen3_5_preemption_real_test
  ninfer_qwen3_5_grammar_real_test
  ninfer_qwen3_5_tools_real_test
  ninfer_qwen3_5_score_real_test
  ninfer_qwen3_5_vision_workspace_test
  ninfer_qwen3_5_dflash2_real_test
  ninfer_qwen3_5_speculative_invariant_real_test
  ninfer_qwen3_5_dflash_prefill_real_test
  ninfer_qwen3_5_moe_real_test
  ninfer_qwen3_5_moe_offload_real_test
  ninfer_qwen3_5_dflash_real_test)
set_tests_properties(${ninfer_qwen3_5_real_tests} PROPERTIES
  SKIP_RETURN_CODE 77
  RUN_SERIAL TRUE
  LABELS "gpu;real")

set_tests_properties(
  ninfer_qwen3_5_state_image_test
  PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu")
