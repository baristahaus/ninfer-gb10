#!/usr/bin/env bash
# Step 0: configure build/ for sm_121a with tests and benchmarks, build, and run ctest plus the
# Flash-Next real-artifact tests. Writes profiles/bench/gb10/step0/summary.md.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
dir=$(step_dir step0)

generator=()
if [[ ! -f build/CMakeCache.txt ]] && command -v ninja >/dev/null; then generator=(-G Ninja); fi

log "configuring build/"
set +e
cmake -S . -B build "${generator[@]}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121a \
    -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON -DNINFER_PERFORMANCE_TRACE=OFF \
    >"$dir/configure.log" 2>&1
configure_rc=$?
build_rc=1
if ((configure_rc == 0)); then
    log "building (log: $dir/build.log)"
    cmake --build build -j >"$dir/build.log" 2>&1
    build_rc=$?
fi
ctest_rc=1
real_rc=1
if ((build_rc == 0)); then
    log "running ctest (log: $dir/ctest.log)"
    ctest --test-dir build --output-on-failure >"$dir/ctest.log" 2>&1
    ctest_rc=$?
    log "running Flash-Next real-artifact tests (log: $dir/ctest_flash_next_real.log)"
    NINFER_QWEN38_FLASH_NEXT_WEIGHTS="$ART" ctest --test-dir build \
        -R 'ninfer_qwen3_8_flash_next_(real|load_plan|frontend)_test' --output-on-failure \
        >"$dir/ctest_flash_next_real.log" 2>&1
    real_rc=$?
fi
set -e

{
    echo "## Step 0 — build and tests"
    echo
    machine_summary
    echo "- Configure: exit $configure_rc; build: exit $build_rc"
    if ((configure_rc != 0)); then
        echo; echo '```'; tail -n 40 "$dir/configure.log"; echo '```'
    elif ((build_rc != 0)); then
        echo; echo "First build errors:"; echo '```'
        grep -m 20 -E 'error|Error' "$dir/build.log" || tail -n 40 "$dir/build.log"
        echo '```'
    else
        echo
        echo "Full ctest (exit $ctest_rc):"
        echo
        "${SUMMARIZE[@]}" ctest "$dir/ctest.log"
        echo
        echo "Flash-Next real-artifact tests (exit $real_rc):"
        echo
        "${SUMMARIZE[@]}" ctest "$dir/ctest_flash_next_real.log"
        if ((ctest_rc != 0 || real_rc != 0)); then
            echo; echo "Failure output (last 60 lines of each failing log):"
            for f in ctest ctest_flash_next_real; do
                if grep -q 'tests FAILED' "$dir/$f.log"; then
                    echo; echo "$f.log:"; echo '```'; tail -n 60 "$dir/$f.log"; echo '```'
                fi
            done
        fi
    fi
} >"$dir/summary.md"

log "summary: $dir/summary.md"
cat "$dir/summary.md"
((configure_rc == 0 && build_rc == 0 && ctest_rc == 0 && real_rc == 0))
