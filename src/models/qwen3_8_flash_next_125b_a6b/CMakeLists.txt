# The registered 125B-A6B package: binding, loading plan and execution variant
# construction run on the execution target beside the family runtime glue.

target_sources(ninfer_model_execution PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/package.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load_plan.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/model.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/variant.cpp)

target_include_directories(ninfer_model_execution PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/export)
