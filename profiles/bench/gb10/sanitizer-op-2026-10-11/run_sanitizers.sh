#!/usr/bin/env bash
# Sanitizer battery on the Flash-Next MoE op test (ninfer-gb10-dnf), 2026-10-11.
# The v2 diag run showed the 24-row crash is timing/memory-state dependent (no
# invalid id, all finite, no crash under the instrumented build, while the
# unpatched server faults deterministically at the same config). Opus: run the
# four sanitizers on the MoE op test alone (it exercises route_kernel at
# 20/22/24/32 rows, no model load). Sequential, one GPU job at a time.
set -u
cd /home/apollo11/ninfer-gb10
OUT=profiles/bench/gb10/sanitizer-op-2026-10-11
mkdir -p "$OUT"
clog() { echo "[san-op $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$OUT/run.log"; }

# Pre-launch checks: sanitizer present, no GPU jobs, no phantom counter,
# free unified memory
command -v compute-sanitizer >/dev/null || { clog "compute-sanitizer not found"; exit 1; }
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
if [ -n "$apps" ]; then clog "GPU guard: compute app(s) '$apps' held - aborting"; exit 1; fi
if [ -n "$util" ] && [ "$util" -gt 0 ]; then clog "GPU guard: phantom counter? util ${util}% with no compute app - aborting"; exit 1; fi
free_g=$(free -g | awk '/^Mem:/{print $7}')
clog "start; HEAD $(git rev-parse --short HEAD); GPU idle (util ${util:-?}%, no apps); free memory ${free_g}G available"
(( free_g >= 100 )) || { clog "insufficient free memory (<100G); aborting"; exit 1; }

# Binary freshness: rebuild the test target at HEAD (no-op or a fast relink)
clog "building ninfer_flash_next_moe_test (freshness)"
cmake --build build -j --target ninfer_flash_next_moe_test >"$OUT/build.log" 2>&1 || { clog "BUILD FAILED: see build.log"; exit 1; }
clog "test binary ready"

run_tool() { # $1 = tool name, rest = extra flags
  local tool=$1; shift
  clog "running compute-sanitizer --tool $tool $*"
  compute-sanitizer --tool "$tool" "$@" build/tests/ninfer_flash_next_moe_test >"$OUT/${tool}.log" 2>&1
  local rc=$?
  clog "$tool done rc=$rc (0 = clean, nonzero = findings or failure)"
}

run_tool racecheck --racecheck-report all
run_tool synccheck
run_tool initcheck
run_tool memcheck

clog "done"
