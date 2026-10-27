add_library(ninfer_artifact STATIC
  schema.cpp
  formats.cpp
  layouts.cpp
  reader.cpp
  mapped_range.cpp
  backing.cpp
  binder.cpp
  placement.cpp
  encoded_slice.cpp
  materializer.cpp
  tier_plan.cpp
  views.cpp
)
ninfer_internal_includes(ninfer_artifact)
target_link_libraries(ninfer_artifact PUBLIC ninfer_core ninfer::json)
