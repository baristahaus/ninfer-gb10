#!/usr/bin/env bash
# Poisoned MoE op test (ninfer-gb10-dnf), 2026-10-11.
# Opus (ef5c1e2b): the op test reruns the fused-grouping verify shapes
# 18/20/22/24/32 after poisoning the workspace (0xFF = BF16 NaN) and every
# SM's shared memory (0x7FFF0320, an impossible expert id) - the state an
# Engine round can leave. One GPU job, no model load.
# If the test fails or faults: one rerun under memcheck, then a dead-context
# check (a faulted context can leave an unreaped channel).
set -u
cd /home/apollo11/ninfer-gb10
OUT=profiles/bench/gb10/poison-op-2026-10-11
mkdir -p "$OUT"
clog() { echo "[poison-op $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$OUT/run.log"; }

# Pre-launch checks: no GPU jobs, no phantom counter, free unified memory
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
if [ -n "$apps" ]; then clog "GPU guard: compute app(s) '$apps' held - aborting"; exit 1; fi
if [ -n "$util" ] && [ "$util" -gt 0 ]; then clog "GPU guard: phantom counter? util ${util}% with no compute app - aborting"; exit 1; fi
free_g=$(free -g | awk '/^Mem:/{print $7}')
clog "start; HEAD $(git rev-parse --short HEAD); GPU idle (util ${util:-?}%, no apps); free memory ${free_g}G available"
(( free_g >= 100 )) || { clog "insufficient free memory (<100G); aborting"; exit 1; }

clog "building ninfer_flash_next_moe_test (poisoned)"
cmake --build build -j --target ninfer_flash_next_moe_test >"$OUT/build.log" 2>&1 || { clog "BUILD FAILED: see build.log"; exit 1; }
clog "test binary ready"

clog "running the poisoned MoE op test"
build/tests/ninfer_flash_next_moe_test >"$OUT/test.log" 2>&1
rc=$?
clog "test rc=$rc"

if [ "$rc" -ne 0 ]; then
  clog "test failed/faulted - one rerun under memcheck"
  compute-sanitizer --tool memcheck build/tests/ninfer_flash_next_moe_test >"$OUT/memcheck.log" 2>&1
  clog "memcheck rc=$?"
  # Dead-context check: a faulted context can leave an unreaped channel
  # (phantom counter: utilization > 0 with no compute app listed)
  apps2=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
  util2=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  if [ -z "$apps2" ] && [ -n "$util2" ] && [ "$util2" -gt 0 ]; then
    clog "REBOOT REQUIRED: no compute app but util ${util2}% - dead context held (phantom counter)"
  elif [ -n "$apps2" ]; then
    clog "REBOOT REQUIRED: process(es) '$apps2' still holding the GPU after the fault"
  else
    clog "no dead context: GPU clean (util ${util2:-0}%, no apps)"
  fi
else
  clog "test passed"
fi

grep 'poisoned T=' "$OUT/test.log" > "$OUT/poisoned-lines.txt" 2>/dev/null || true
clog "poisoned lines: $(wc -l < "$OUT/poisoned-lines.txt" 2>/dev/null || echo 0)"
clog "done (test rc=$rc)"
exit "$rc"
