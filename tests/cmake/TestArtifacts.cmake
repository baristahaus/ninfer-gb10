# Test-artifact plumbing.
#
# tests/fixtures/artifacts/manifest.json names every artifact a test can consume, the environment
# variable the test reads, and the file name in the cache. This module resolves each entry against
# the cache at configure time and sets the variable on the tests that consume it **only when the
# file is there**. An absent entry leaves the variable unset, and the test's own guard returns 77,
# so a fresh checkout skips the real-artifact tests instead of failing them.
#
# Precedence for one variable: an explicit environment value at configure time, then the manifest
# entry's cached file. The cache directory is NINFER_TEST_ARTIFACT_CACHE (CMake variable or
# environment), else the manifest default.
#
# Populate the cache with tools/fetch_test_artifacts.py, then re-run configure. The cache is never
# in git; the manifest is.

set(NINFER_TEST_ARTIFACT_MANIFEST "${PROJECT_SOURCE_DIR}/tests/fixtures/artifacts/manifest.json")
set(NINFER_TEST_ARTIFACT_CACHE "" CACHE PATH
    "Test-artifact cache directory (default: NINFER_TEST_ARTIFACT_CACHE env, else the manifest's)")

function(_ninfer_test_artifact_cache_dir out)
  set(dir "${NINFER_TEST_ARTIFACT_CACHE}")
  if(dir STREQUAL "")
    set(dir "$ENV{NINFER_TEST_ARTIFACT_CACHE}")
  endif()
  if(dir STREQUAL "")
    file(READ "${NINFER_TEST_ARTIFACT_MANIFEST}" _manifest)
    string(JSON dir GET "${_manifest}" cache default)
  endif()
  if(dir MATCHES "^~")
    string(REPLACE "~" "$ENV{HOME}" dir "${dir}")
  endif()
  set(${out} "${dir}" PARENT_SCOPE)
endfunction()

function(ninfer_register_test_artifacts)
  if(NOT EXISTS "${NINFER_TEST_ARTIFACT_MANIFEST}")
    message(WARNING "No test-artifact manifest at ${NINFER_TEST_ARTIFACT_MANIFEST}; "
                    "real-artifact tests will skip.")
    return()
  endif()

  file(READ "${NINFER_TEST_ARTIFACT_MANIFEST}" json)
  string(JSON count ERROR_VARIABLE error LENGTH "${json}" artifacts)
  if(error)
    message(FATAL_ERROR "Test-artifact manifest is not valid JSON: ${error}")
  endif()

  _ninfer_test_artifact_cache_dir(cache)
  set(absent 0)
  set(configured 0)
  # The tests that will skip, not just the artifacts that are absent: a green run and a run where
  # nothing ran must not look the same, and the count is what tells them apart.
  set(skipping "")

  math(EXPR last "${count} - 1")
  foreach(index RANGE 0 ${last})
    string(JSON name GET "${json}" artifacts ${index} name)
    string(JSON env GET "${json}" artifacts ${index} env)
    string(JSON file GET "${json}" artifacts ${index} file)
    string(JSON kind GET "${json}" artifacts ${index} source kind)

    # A committed fixture is used in place; the rest resolve against the cache.
    if(kind STREQUAL "committed")
      string(JSON relative GET "${json}" artifacts ${index} source path)
      set(path "${PROJECT_SOURCE_DIR}/${relative}")
    else()
      set(path "${cache}/${file}")
    endif()

    string(JSON test_count ERROR_VARIABLE error LENGTH "${json}" artifacts ${index} tests)
    if(error)
      message(FATAL_ERROR "Manifest entry ${name}: 'tests' must be a list")
    endif()
    if(test_count EQUAL 0)
      continue()
    endif()
    math(EXPR test_last "${test_count} - 1")
    set(tests "")
    foreach(t RANGE 0 ${test_last})
      string(JSON test GET "${json}" artifacts ${index} tests ${t})
      list(APPEND tests "${test}")
    endforeach()

    # Explicit environment value wins over the cache; a stale explicit value is ignored.
    set(chosen "")
    if(DEFINED ENV{${env}} AND NOT "$ENV{${env}}" STREQUAL "" AND EXISTS "$ENV{${env}}")
      set(chosen "$ENV{${env}}")
    elseif(EXISTS "${path}")
      set(chosen "${path}")
    endif()

    set(registered "")
    foreach(test IN LISTS tests)
      if(TEST ${test})
        list(APPEND registered ${test})
      endif()
    endforeach()
    if(NOT registered)
      continue()
    endif()

    # Every artifact-consuming test skips cleanly when its artifact is absent. Labels stay with
    # the test's own registration: the manifest does not know whether a test is host-only.
    set_tests_properties(${registered} PROPERTIES SKIP_RETURN_CODE 77)

    if(chosen STREQUAL "")
      math(EXPR absent "${absent} + 1")
      list(APPEND skipping ${registered})
      message(STATUS "Test artifact ${name}: absent (${path}); ${env} left unset, "
                     "${test_count} test(s) will skip")
    else()
      math(EXPR configured "${configured} + 1")
      set_tests_properties(${registered} PROPERTIES ENVIRONMENT "${env}=${chosen}")
      message(STATUS "Test artifact ${name}: ${env}=${chosen}")
    endif()
  endforeach()

  if(skipping)
    list(REMOVE_DUPLICATES skipping)
    list(LENGTH skipping skipping_count)
  else()
    set(skipping_count 0)
  endif()
  message(STATUS "Test artifacts: ${configured} resolved, ${absent} absent, "
                 "${skipping_count} test(s) will skip (cache ${cache})")

  # The manifest is checked in and must stay consistent with the tests and sources.
  if(Python3_EXECUTABLE)
    add_test(NAME ninfer_test_artifacts_manifest_test
      COMMAND ${Python3_EXECUTABLE} -B
              ${PROJECT_SOURCE_DIR}/tools/check_test_artifacts.py ${PROJECT_SOURCE_DIR})
    set_tests_properties(ninfer_test_artifacts_manifest_test PROPERTIES LABELS "host")
  endif()
endfunction()
