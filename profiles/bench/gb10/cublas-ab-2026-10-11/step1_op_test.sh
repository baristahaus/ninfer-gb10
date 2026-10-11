#!/usr/bin/env bash
# Step 1 of the cublas-above-48 A/B run list (ninfer-gb10-dnf), 2026-10-11.
# Op test on the branch head (e13cdf4b): does the cuBLAS-backed linear() overload
# fail at some verify row count? GPU guard first: no compute apps, no phantom
# counter (util > 0 with no apps = dead context from a crashed run).
set -u
cd /home/apollo11/ninfer-gb10
OUT=profiles/bench/gb10/cublas-ab-2026-10-11
mkdir -p "$OUT"
clog() { echo "[ab-step1 $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$OUT/step1.log"; }

# GPU guard
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
if [ -n "$apps" ]; then clog "GPU guard: compute app(s) '$apps' held - aborting"; exit 1; fi
if [ -n "$util" ] && [ "$util" -gt 0 ]; then
  clog "GPU guard: phantom counter? util ${util}% with no compute app - aborting (reboot needed)"; exit 1
fi
clog "start; HEAD $(git rev-parse --short HEAD); GPU idle (util ${util:-?}%, no apps)"

# Binary freshness
cmake --build build -j --target ninfer_linear_bf16_a16_test >"$OUT/step1-build.log" 2>&1 \
  || { clog "BUILD FAILED: see step1-build.log"; exit 1; }
clog "test binary ready"

ctest --test-dir build -R ninfer_linear_bf16_a16_test --output-on-failure \
  >"$OUT/step1-ctest.log" 2>&1
rc=$?
clog "ctest rc=$rc"
clog "verdict: $(grep -m1 -E '^(OK|FAIL|SKIP)' "$OUT/step1-ctest.log")"
if [ $rc -ne 0 ]; then
  clog "first failing lines:"
  grep -m10 -E 'not finite|guard|FAILED|Error|error' "$OUT/step1-ctest.log" | sed 's/^/    /'
fi

# Post-run GPU check (dead context leaves a phantom counter)
sleep 3
apps2=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
util2=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
if [ -z "$apps2" ] && [ -n "$util2" ] && [ "$util2" -gt 0 ]; then
  clog "REBOOT REQUIRED: no compute app but util ${util2}% - dead context held (phantom counter)"
else
  clog "GPU clean after run (util ${util2:-0}%, no apps)"
fi
clog "done"
