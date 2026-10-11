#!/usr/bin/env bash
# Step 3 of the cublas-above-48 A/B run list (ninfer-gb10-dnf), 2026-10-11.
# Run ONLY if step 2 starts the patched server. Measures the patch's cost where
# cuBLAS used to run: C5 K3 decode tok/s on the natural corpus (the usual
# serving protocol: concurrency_sweep.py --n 5 against
# ninfer-serve --max-concurrency 5 --spec mtp --draft-tokens 3 --lm-head-draft,
# 512-token greedy requests), unpatched vs patched. One GPU job per variant.
# Ends with the patch dropped (unpatched tree).
set -u
cd /home/apollo11/ninfer-gb10
OUT=profiles/bench/gb10/cublas-ab-2026-10-11
mkdir -p "$OUT"
clog() { echo "[ab-step3 $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$OUT/step3.log"; }

srv=""
on_exit() {
  if [ -n "$srv" ] && kill -0 "$srv" 2>/dev/null; then
    kill -TERM "$srv" 2>/dev/null
    wait "$srv" 2>/dev/null
  fi
  git checkout -- src/ops/linear/linear.cpp 2>/dev/null
}
trap on_exit INT TERM

source tools/gb10/config.local.sh
test -n "${ART:-}" && test -e "$ART" || { clog "ART missing or not set: ${ART:-<unset>}"; exit 1; }

gpu_guard() {
  local apps util
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  if [ -n "$apps" ]; then clog "GPU guard: compute app(s) '$apps' held - aborting"; exit 1; fi
  if [ -n "$util" ] && [ "$util" -gt 0 ]; then
    clog "GPU guard: phantom counter? util ${util}% with no compute app - aborting (reboot needed)"; exit 1
  fi
  local free_g
  free_g=$(free -g | awk '/^Mem:/{print $7}')
  (( free_g >= 100 )) || { clog "insufficient free memory (<100G); aborting"; exit 1; }
  clog "GPU idle (util ${util:-?}%, no apps); free ${free_g}G"
}

gpu_check() { # post-run dead-context check
  sleep 3
  local apps2 util2
  apps2=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
  util2=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  if [ -z "$apps2" ] && [ -n "$util2" ] && [ "$util2" -gt 0 ]; then
    clog "REBOOT REQUIRED: no compute app but util ${util2}% - dead context held (phantom counter)"
  else
    clog "GPU clean after run (util ${util2:-0}%, no apps)"
  fi
}

run_variant() { # $1 = label (unpatched|patched)
  local label=$1 port
  [ "$label" = "patched" ] && port=8003 || port=8002
  clog "=== variant: $label (port $port) ==="
  gpu_guard
  cmake --build build -j --target ninfer-serve >"$OUT/step3-build-$label.log" 2>&1 \
    || { clog "BUILD FAILED ($label): see step3-build-$label.log"; exit 1; }
  build/apps/ninfer-serve "$ART" --host 0.0.0.0 --port "$port" --max-context 73728 \
    --max-concurrency 5 --kv-dtype fp8 --spec mtp --draft-tokens 3 --lm-head-draft \
    >"$OUT/step3-serve-$label.log" 2>&1 &
  srv=$!
  local ready=0
  for i in $(seq 1 300); do
    if grep -q 'CUDA graphs ready' "$OUT/step3-serve-$label.log" 2>/dev/null; then
      ready=1
      break
    fi
    kill -0 "$srv" 2>/dev/null || break
    sleep 2
  done
  if [ "$ready" -ne 1 ]; then
    wait "$srv" 2>/dev/null
    clog "server ($label) failed to reach CUDA graphs ready; rc=$?; last lines:"
    tail -15 "$OUT/step3-serve-$label.log" 2>/dev/null | sed 's/^/    /'
    srv=""
    gpu_check
    return 1
  fi
  clog "server ($label) up: CUDA graphs ready"
  python3 tools/gb10/concurrency_sweep.py "http://127.0.0.1:$port" \
    "$OUT/step3-sweep-$label.json" --n 5 --max-tokens 512 >"$OUT/step3-sweep-$label.log" 2>&1
  clog "sweep ($label) done; aggregate lines:"
  grep -E 'tok/s|aggregate|batch' "$OUT/step3-sweep-$label.log" | sed 's/^/    /'
  kill -TERM "$srv" 2>/dev/null
  wait "$srv" 2>/dev/null
  srv=""
  clog "server ($label) stopped"
  gpu_check
}

clog "start; HEAD $(git rev-parse --short HEAD)"

# Variant 1: unpatched (branch head state; the patch is already dropped by step 2)
test -z "$(git diff --name-only -- src/ops/linear/linear.cpp)" \
  || { git checkout -- src/ops/linear/linear.cpp; clog "linear.cpp was modified; restored to branch head"; }
run_variant unpatched || exit 1

# Variant 2: patched
git apply share/diag-dnf/cublas-above-48.patch || { clog "PATCH FAILED to apply"; exit 1; }
clog "patch applied (cutoff 16 -> 48 tokens)"
run_variant patched || exit 1

git checkout -- src/ops/linear/linear.cpp
clog "patch dropped (linear.cpp restored to branch head)"
trap - INT TERM
clog "done"
