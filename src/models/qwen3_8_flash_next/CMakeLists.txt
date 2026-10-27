# Flash-Next family sources on the split model targets: frontend policy on the
# control target, the tokenizer on loading, and state, PLE, Vision and runtime
# glue on execution.

target_sources(ninfer_model_control PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/chat_template.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/digest.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/frontend.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/media_cache.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/processor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/tool_call_parser.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/vision/control.cpp)

target_sources(ninfer_model_loading PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/tokenizer.cpp)

target_sources(ninfer_model_execution PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/ple_stage.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/ple_table.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/prefix_identity.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/runtime/visual_scatter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/state/decoder_state.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/state/state_image.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/state/round_state.cpp)

foreach(_target IN ITEMS ninfer_model_control ninfer_model_loading ninfer_model_execution)
  target_include_directories(${_target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/export)
endforeach()
