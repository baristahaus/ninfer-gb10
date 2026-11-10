ninfer_add_test(ninfer_peer_mailbox_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_peer_mailbox.cpp"
  LIBRARIES ninfer_ops)
set_tests_properties(ninfer_peer_mailbox_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_execution_context_cuda_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_execution_context_cuda.cpp"
  LIBRARIES ninfer_core)
set_tests_properties(ninfer_execution_context_cuda_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_execution_context_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_execution_context.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_flash_next_shard_rules_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_flash_next_shard_rules.cpp"
  LIBRARIES ninfer_artifact ninfer_model_execution)

ninfer_add_test(ninfer_qwen3_5_shard_rules_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_qwen3_5_shard_rules.cpp"
  LIBRARIES ninfer_artifact ninfer_model_loading)

ninfer_add_test(ninfer_repacked_weights_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_repacked_weights.cpp"
  LIBRARIES ninfer_model_loading)
ninfer_add_test(ninfer_consumer_geometry_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_consumer_geometry.cpp"
  LIBRARIES ninfer_core)

# A plan containing ONLY a sliced request must bind, by the plan's request rather than by object
# handle. Host-only.
ninfer_add_test(ninfer_sliced_binding_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_sliced_binding.cpp"
  LIBRARIES ninfer_model_loading ninfer_artifact)

ninfer_add_test(ninfer_shard_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_shard.cpp"
  LIBRARIES ninfer_artifact)

ninfer_add_test(ninfer_sharded_forward_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_sharded_forward.cpp"
  LIBRARIES ninfer_artifact)

ninfer_add_test(ninfer_layout_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_layout.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_media_decode_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_media_decode.cpp"
  LIBRARIES ninfer_media_decode)

ninfer_add_test(ninfer_prompt_input_test
  NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_prompt_input.cpp"
  LIBRARIES ninfer_product_prompt_input)

ninfer_add_test(ninfer_pretty_logging_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_pretty_logging.cpp"
  LIBRARIES ninfer_product_logging)

ninfer_add_test(ninfer_perplexity_evaluation_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_perplexity_evaluation.cpp"
          ${PROJECT_SOURCE_DIR}/apps/perplexity/evaluation.cpp
  LIBRARIES ninfer_core)

target_include_directories(ninfer_perplexity_evaluation_test PRIVATE
  ${PROJECT_SOURCE_DIR}/apps/perplexity)

ninfer_add_test(ninfer_cli_options_test
  NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_cli_options.cpp"
  LIBRARIES ninfer_product_config)

ninfer_add_test(ninfer_openai_schema_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_openai_schema.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_serve_security_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_serve_security.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_cors_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_cors.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_openai_responses_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_openai_responses.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_openai_responses_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_openai_responses_store.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_anthropic_schema_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_anthropic_schema.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_serve_options_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_serve_options.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_turn_anchors_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_turn_anchors.cpp"
  LIBRARIES ninfer_serve)
set_tests_properties(ninfer_turn_anchors_test PROPERTIES TIMEOUT 30)

ninfer_add_test(ninfer_request_log_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_request_log.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_metrics_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_metrics.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_http_error_handler_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_http_error_handler.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_http_transport_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_http_transport.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_tool_streaming_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_tool_streaming.cpp"
  LIBRARIES ninfer_serve ninfer_model_execution ninfer::json)

ninfer_add_test(ninfer_reasoning_retry_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_reasoning_retry.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_gpu_telemetry_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_gpu_telemetry.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_prefill_miss_dump_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_prefill_miss_dump.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_speculation_detail_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_speculation_detail.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_reasoning_retry_wire_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_reasoning_retry_wire.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_chat_timings_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_chat_timings.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_startup_resources_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_startup_resources.cpp"
  LIBRARIES ninfer_engine ninfer_product_logging ninfer::json)
