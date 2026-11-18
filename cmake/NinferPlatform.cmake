set(NINFER_TARGET rtx5090 CACHE STRING "NInfer execution target")
set_property(CACHE NINFER_TARGET PROPERTY STRINGS rtx5090 dgx_spark)

if(DEFINED NINFER_CONFIGURED_TARGET AND
   NOT NINFER_CONFIGURED_TARGET STREQUAL NINFER_TARGET)
  message(FATAL_ERROR
    "This build directory belongs to ${NINFER_CONFIGURED_TARGET}; "
    "use a separate directory for ${NINFER_TARGET}")
endif()

set(NINFER_TARGET_RTX5090 0)
set(NINFER_TARGET_SPARK 0)
if(NINFER_TARGET STREQUAL "rtx5090")
  set(NINFER_TARGET_RTX5090 1)
  set(NINFER_CUDA_ARCHITECTURE 120a)
  set(NINFER_PROCESSOR_PATTERN "^(x86_64|amd64|AMD64)$")
elseif(NINFER_TARGET STREQUAL "dgx_spark")
  set(NINFER_TARGET_SPARK 1)
  set(NINFER_CUDA_ARCHITECTURE 121a)
  set(NINFER_PROCESSOR_PATTERN "^(aarch64|arm64|ARM64)$")
else()
  message(FATAL_ERROR "Unknown NINFER_TARGET '${NINFER_TARGET}'")
endif()

# CUDA compiler detection must see the selected architecture, not a host default.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES ${NINFER_CUDA_ARCHITECTURE}
    CACHE STRING "CUDA architectures to build")
endif()
function(ninfer_validate_cuda_architecture)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL NINFER_CUDA_ARCHITECTURE)
    message(FATAL_ERROR
      "${NINFER_TARGET} requires CMAKE_CUDA_ARCHITECTURES=${NINFER_CUDA_ARCHITECTURE}; "
      "got '${CMAKE_CUDA_ARCHITECTURES}'")
  endif()
endfunction()
ninfer_validate_cuda_architecture()

function(ninfer_validate_platform)
  # The toolchain file can change CUDA architectures during project() initialization.
  ninfer_validate_cuda_architecture()
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
     NOT CMAKE_SYSTEM_PROCESSOR MATCHES "${NINFER_PROCESSOR_PATTERN}")
    message(FATAL_ERROR
      "${NINFER_TARGET} does not support ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}")
  endif()
  set(NINFER_CONFIGURED_TARGET ${NINFER_TARGET} CACHE INTERNAL "Configured NInfer target")
  configure_file(${PROJECT_SOURCE_DIR}/src/core/platform.h.in
    ${PROJECT_BINARY_DIR}/generated/core/platform.h @ONLY)
  message(STATUS "NInfer target: ${NINFER_TARGET} (${CMAKE_SYSTEM_PROCESSOR}, sm_${NINFER_CUDA_ARCHITECTURE})")
endfunction()

# --- Kernel-source content tracking ------------------------------------------
# Tar-based source syncs with preserved timestamps set source mtimes OLD while
# the content is NEW, so ninja sees objects as up to date and silently skips
# recompiles of changed dispatch-table sources (how pre-pin kernels shipped in
# ninfer-gb10-1r0). With NINFER_TRACK_KERNEL_SOURCES ON, every CUDA compile of
# a kernel-shape source leaves an <object>.srcsha sidecar recording the SHA-256
# of the source as compiled. tools/check_kernel_staleness.py compares those
# records against the current sources and fails on an object that is stale
# even when every mtime lies. A skipped recompile leaves the old record in
# place, which is exactly what makes the mismatch visible.
option(NINFER_TRACK_KERNEL_SOURCES
  "Record per-object source hashes (.srcsha) for tools/check_kernel_staleness.py" ON)
if(NINFER_TRACK_KERNEL_SOURCES)
  set(NINFER_STALENESS_LAUNCHER "${CMAKE_CURRENT_LIST_DIR}/../tools/kernel_source_hash_launcher.py")
  if(EXISTS "${NINFER_STALENESS_LAUNCHER}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${NINFER_STALENESS_LAUNCHER}")
  endif()
endif()
