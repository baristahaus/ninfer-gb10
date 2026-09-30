#!/usr/bin/env bash
# PR #36 fix verification: eb9e87fa (the capacity fix) against its base ed6525fa. Opus's four
# checks, one GPU job at a time. Runs on the GB10 machine; resumable via DONE markers.
#
#   A-BUILD  tree A = verify/pr36fix-2026-09-30 (eb9e87fa + the engine-level driver), into build/
#   B-SETUP  tree B = base worktree at ../ninfer-gb10-base (ed6525fa + the same driver patch);
#            configure + start the build in the background (CPU only; overlaps the A GPU phases)
#   A-TESTS  check 1: ctest test_flash_next_qsa (must pass on the fix) + the Flash-Next real
#            suite (the real test is expected to fail on the stale golden - its mtp line is the
#            evidence; the other three tests must pass)
#   A-Driver checks 2a/2b/3a: the driver (mtp line, prefix reuse cross-path, near-tie logprobs)
#   A-PPL    check 4a: the step 7 4K PPL gate (target 3.9982, run-to-run floor 1.3e-4)
#   B-TESTS  checks 2a/2b/3b on the base: ctest real suite (must pass fully) + the driver
#   B-PPL    check 4b: the PPL gate on the base
#   REPORT   the result table
#
# The base tree's real test passing confirms the old golden holds there; the fix tree's driver
# confirms the golden-free cross-path machinery while the MTP golden awaits re-recording.
X925_CORES=${X925_CORES:-5-9,15-19}
if [[ -z ${PINNED:-} ]]; then
    PINNED=1 exec taskset -c "$X925_CORES" "$0" "$@"
fi
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

PHASES=${PHASES:-A-BUILD,B-SETUP,A-TESTS,A-Driver,A-PPL,B-TESTS,B-PPL,REPORT}
ROOT=$OUT_ROOT/pr36fix-verify
mkdir -p "$ROOT"
A_TREE=$GB10_ROOT
A_BRANCH=verify/pr36fix-2026-09-30
B_TREE=/home/apollo11/ninfer-gb10-base
B_COMMIT=ed6525fa
want() { case ",$PHASES," in *",$1,"*) [[ ! -f $ROOT/$1/DONE ]] ;; *) return 1 ;; esac; }
begin() { rm -rf "${ROOT:?}/$1"; mkdir -p "$ROOT/$1"; log "phase $1: $2"; }
finish() { touch "$ROOT/$1/DONE"; log "phase $1 done"; }
drop_caches() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; }
gpu_idle() {
    if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]]; then
        echo "GPU is held by another job; single-instance discipline: aborting." >&2
        exit 1
    fi
}
wait_base_build() {
    local rc_file="$ROOT/B-SETUP/build.rc" pid
    pid=$(cat "$ROOT/B-SETUP/build.pid" 2>/dev/null || true)
    if [[ -f $rc_file ]]; then return 0; fi
    if [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null || while kill -0 "$pid" 2>/dev/null; do sleep 60; done
    fi
    [[ -f $rc_file ]]
}

gpu_idle
log "pr36fix-verify on $(basename "$ART"), A=$A_BRANCH B=$B_COMMIT, phases $PHASES"

# ---- A-BUILD: the fix tree, incremental ----------------------------------------------------
if want A-BUILD; then
    begin A-BUILD "build the fix tree (eb9e87fa + driver)"
    cd "$A_TREE"
    [[ $(git branch --show-current) == "$A_BRANCH" ]] || { log "not on $A_BRANCH; aborting"; exit 1; }
    git merge-base --is-ancestor eb9e87fa HEAD || { log "eb9e87fa is not an ancestor of HEAD"; exit 1; }
    cmake --build build -j >"$ROOT/A-BUILD/build.log" 2>&1 || {
        log "A-BUILD failed; aborting"; tail -n 40 "$ROOT/A-BUILD/build.log" >&2; exit 1; }
    finish A-BUILD
fi

# ---- B-SETUP: the base worktree, its build overlapping the A GPU phases ---------------------
if want B-SETUP; then
    begin B-SETUP "base worktree + background build (ed6525fa + driver patch)"
    if [[ ! -e $B_TREE/.git ]]; then
        git -C "$A_TREE" worktree add --detach "$B_TREE" "$B_COMMIT" \
            >"$ROOT/B-SETUP/worktree.log" 2>&1 || { log "worktree setup failed"; exit 1; }
    fi
    git -C "$A_TREE" diff eb9e87fa "$A_BRANCH" -- \
        tests/models/qwen3_8_flash_next/tests.cmake \
        tests/models/qwen3_8_flash_next_125b_a6b/test_pr36fix_checks.cpp \
        >"$ROOT/B-SETUP/driver.patch"
    git -C "$B_TREE" apply "$A_TREE/$ROOT/B-SETUP/driver.patch" \
        >"$B_TREE/apply.log" 2>&1 || { log "driver patch failed to apply"; exit 1; }
    cmake -S "$B_TREE" -B "$B_TREE/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        >"$ROOT/B-SETUP/configure.log" 2>&1 || { log "B configure failed"; exit 1; }
    (cd "$B_TREE" && cmake --build build -j; echo $? >"$A_TREE/$ROOT/B-SETUP/build.rc") \
        >"$A_TREE/$ROOT/B-SETUP/build.log" 2>&1 &
    echo $! >"$ROOT/B-SETUP/build.pid"
    log "base build started in the background (pid $(cat "$ROOT/B-SETUP/build.pid"))"
    finish B-SETUP
fi
# ---- A-TESTS: check 1 (the op test must pass) + the real suite (mtp line evidence) ----------
if want A-TESTS; then
    begin A-TESTS "ctest on the fix tree"
    cd "$A_TREE"
    gpu_idle
    ctest --test-dir build -R ninfer_flash_next_qsa_test --output-on-failure \
        >"$ROOT/A-TESTS/ctest-qsa.log" 2>&1 || qsa_rc=$?
    NINFER_QWEN38_FLASH_NEXT_WEIGHTS="$ART" ctest --test-dir build \
        -R 'ninfer_qwen3_8_flash_next_(real|load_plan|frontend|fault)_test' --output-on-failure \
        >"$ROOT/A-TESTS/ctest-real.log" 2>&1 || real_rc=$?
    {
        echo "## A-TESTS - ctest on $A_BRANCH (eb9e87fa + driver)"
        echo
        echo "check 1, test_flash_next_qsa (exit ${qsa_rc:-0}; must pass on the fix):"
        echo
        tail -n 5 "$ROOT/A-TESTS/ctest-qsa.log"
        echo
        echo "Flash-Next real suite (exit ${real_rc:-0}; the real test is expected to fail on the"
        echo "stale golden until Opus re-records it; the other three tests must pass):"
        echo
        tail -n 12 "$ROOT/A-TESTS/ctest-real.log"
        echo
        echo "mtp line from the real test:"
        grep -A 1 "^mtp:" "$ROOT/A-TESTS/ctest-real.log" || echo "(no mtp line printed)"
    } >"$ROOT/A-TESTS/summary.md" 2>&1
    finish A-TESTS
fi

# ---- A-Driver: checks 2a/2b/3a ------------------------------------------------------------
if want A-Driver; then
    begin A-Driver "engine-level checks on the fix tree"
    cd "$A_TREE"
    gpu_idle
    NINFER_QWEN38_FLASH_NEXT_WEIGHTS="$ART" ./build/tests/ninfer_qwen3_8_flash_next_pr36fix_checks \
        >"$ROOT/A-Driver/log" 2>&1 || driver_rc=$?
    cp "$ROOT/A-Driver/log" "$ROOT/A-Driver/summary.md"
    finish A-Driver
fi

# ---- A-PPL: check 4a, the step 7 4K PPL gate ----------------------------------------------
if want A-PPL; then
    begin A-PPL "4K PPL gate on the fix tree"
    cd "$A_TREE"
    gpu_idle
    drop_caches
    ./build/apps/ninfer-perplexity "$ART" --corpus eval/corpora/perplexity-1m/manifest.json \
        --kv-dtype "$KV_DTYPE" --token-scores --output "$ROOT/A-PPL/perplexity" \
        >"$ROOT/A-PPL/perplexity.log" 2>&1 || ppl_rc=$?
    {
        echo "## A-PPL - 4K PPL gate on $A_BRANCH (target 3.9982, run-to-run floor 1.3e-4)"
        echo
        echo "exit ${ppl_rc:-0}"
        echo
        "$PYTHON" - "$ROOT/A-PPL/perplexity/report.json" <<'EOF'
import json
import sys

try:
    d = json.load(open(sys.argv[1]))
except Exception as error:  # report missing: the log tail below is the evidence
    print(f"report.json unreadable: {error}")
    raise SystemExit
overall = d["overall"]
ppl = overall["perplexity"]
delta = ppl - 3.9982
within = abs(delta) <= 1.3e-4
print(f"overall PPL {ppl:.6f} (mean NLL {overall['mean_nll']:.6f} over "
      f"{overall['scored_tokens']:,} tokens); delta vs 3.9982 = {delta:+.2e} "
      f"({'within' if within else 'OUTSIDE'} the 1.3e-4 floor)")
EOF
        echo
        echo "log tail:"
        echo
        tail -n 6 "$ROOT/A-PPL/perplexity.log"
    } >"$ROOT/A-PPL/summary.md" 2>&1
    finish A-PPL
fi

# ---- B-TESTS: checks 2a/2b/3b on the base ---------------------------------------------------
if want B-TESTS; then
    begin B-TESTS "ctest real suite + driver on the base"
    wait_base_build || { log "base build incomplete; aborting"; exit 1; }
    build_rc=$(cat "$ROOT/B-SETUP/build.rc" 2>/dev/null || echo 1)
    [[ $build_rc == 0 ]] || { log "base build failed (rc $build_rc)"; exit 1; }
    tail -n 3 "$ROOT/B-SETUP/build.log"
    cd "$B_TREE"
    gpu_idle
    NINFER_QWEN38_FLASH_NEXT_WEIGHTS="$ART" ctest --test-dir build \
        -R 'ninfer_qwen3_8_flash_next_(real|load_plan|frontend|fault)_test' --output-on-failure \
        >"$ROOT/B-TESTS/ctest-real.log" 2>&1 || real_rc=$?
    NINFER_QWEN38_FLASH_NEXT_WEIGHTS="$ART" ./build/tests/ninfer_qwen3_8_flash_next_pr36fix_checks \
        >"$ROOT/B-TESTS/driver.log" 2>&1 || driver_rc=$?
    {
        echo "## B-TESTS - base tree $B_COMMIT (the old golden must pass fully)"
        echo
        echo "real suite (exit ${real_rc:-0}):"
        echo
        tail -n 8 "$ROOT/B-TESTS/ctest-real.log"
        echo
        echo "driver (exit ${driver_rc:-0}):"
        echo
        cat "$ROOT/B-TESTS/driver.log"
    } >"$ROOT/B-TESTS/summary.md" 2>&1
    finish B-TESTS
fi

# ---- B-PPL: check 4b, the PPL gate on the base ----------------------------------------------
if want B-PPL; then
    begin B-PPL "4K PPL gate on the base"
    cd "$B_TREE"
    gpu_idle
    drop_caches
    ./build/apps/ninfer-perplexity "$ART" --corpus eval/corpora/perplexity-1m/manifest.json \
        --kv-dtype "$KV_DTYPE" --token-scores --output "$ROOT/B-PPL/perplexity" \
        >"$ROOT/B-PPL/perplexity.log" 2>&1 || ppl_rc=$?
    {
        echo "## B-PPL - 4K PPL gate on base $B_COMMIT"
        echo
        echo "exit ${ppl_rc:-0}"
        echo
        "$PYTHON" - "$ROOT/B-PPL/perplexity/report.json" <<'EOF'
import json
import sys

try:
    d = json.load(open(sys.argv[1]))
except Exception as error:
    print(f"report.json unreadable: {error}")
    raise SystemExit
overall = d["overall"]
ppl = overall["perplexity"]
delta = ppl - 3.9982
within = abs(delta) <= 1.3e-4
print(f"overall PPL {ppl:.6f} (mean NLL {overall['mean_nll']:.6f} over "
      f"{overall['scored_tokens']:,} tokens); delta vs 3.9982 = {delta:+.2e} "
      f"({'within' if within else 'OUTSIDE'} the 1.3e-4 floor)")
EOF
        echo
        echo "log tail:"
        echo
        tail -n 6 "$ROOT/B-PPL/perplexity.log"
    } >"$ROOT/B-PPL/summary.md" 2>&1
    finish B-PPL
fi

# ---- REPORT ----------------------------------------------------------------------------------
if want REPORT; then
    begin REPORT "result table"
    {
        echo "# PR #36 fix verification - results (eb9e87fa vs base ed6525fa)"
        echo
        echo "## A: the fix tree ($A_BRANCH)"
        echo
        sed -n '1,40p' "$ROOT/A-TESTS/summary.md" 2>/dev/null
        echo
        sed -n '1,60p' "$ROOT/A-Driver/summary.md" 2>/dev/null
        echo
        sed -n '1,20p' "$ROOT/A-PPL/summary.md" 2>/dev/null
        echo
        echo "## B: the base tree ($B_COMMIT)"
        echo
        sed -n '1,60p' "$ROOT/B-TESTS/summary.md" 2>/dev/null
        echo
        sed -n '1,20p' "$ROOT/B-PPL/summary.md" 2>/dev/null
    } >"$ROOT/summary.md" 2>&1
    finish REPORT
fi
log "pr36fix-verify finished (phases $PHASES); results under $ROOT"
