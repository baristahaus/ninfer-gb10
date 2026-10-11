#!/usr/bin/env bash
# Step 2 of the cublas-above-48 A/B run list (ninfer-gb10-dnf), 2026-10-11.
# A/B on the configuration that faults today: apply the patch that keeps
# T <= 48 on the native kernels (cublas-above-48.patch), rebuild ninfer-serve,
# start it at the faulting flags (--max-concurrency 6, 24 verify rows), wait
# for "CUDA graphs ready" (the unpatched build faults during graph preparation),
# then six concurrent greedy chat requests, 256 tokens each.
# The patch is dropped at the end in all outcomes.
set -u
cd /home/apollo11/ninfer-gb10
OUT=profiles/bench/gb10/cublas-ab-2026-10-11
mkdir -p "$OUT"
clog() { echo "[ab-step2 $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$OUT/step2.log"; }

srv=""
on_exit() {
  if [ -n "$srv" ] && kill -0 "$srv" 2>/dev/null; then
    kill -TERM "$srv" 2>/dev/null
    wait "$srv" 2>/dev/null
  fi
  git checkout -- src/ops/linear/linear.cpp 2>/dev/null
}
trap on_exit INT TERM

# GPU guard
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
if [ -n "$apps" ]; then clog "GPU guard: compute app(s) '$apps' held - aborting"; exit 1; fi
if [ -n "$util" ] && [ "$util" -gt 0 ]; then
  clog "GPU guard: phantom counter? util ${util}% with no compute app - aborting (reboot needed)"; exit 1
fi
free_g=$(free -g | awk '/^Mem:/{print $7}')
clog "start; HEAD $(git rev-parse --short HEAD); GPU idle (util ${util:-?}%, no apps); free ${free_g}G"
(( free_g >= 100 )) || { clog "insufficient free memory (<100G); aborting"; exit 1; }

# Apply the A/B patch
test -z "$(git status --short -- src/ops/linear/linear.cpp | grep -v '^??')" \
  || { clog "linear.cpp already modified - aborting"; exit 1; }
git apply share/diag-dnf/cublas-above-48.patch || { clog "PATCH FAILED to apply"; exit 1; }
clog "patch applied (cutoff 16 -> 48 tokens)"

source tools/gb10/config.local.sh
test -n "${ART:-}" && test -e "$ART" || { clog "ART missing or not set: ${ART:-<unset>}"; exit 1; }

cmake --build build -j --target ninfer-serve >"$OUT/step2-build.log" 2>&1 \
  || { clog "BUILD FAILED: see step2-build.log"; exit 1; }
clog "serve binary ready (patched)"

clog "launching patched server (port 8001, max-concurrency 6 - the faulting config)"
build/apps/ninfer-serve "$ART" --host 0.0.0.0 --port 8001 --max-concurrency 6 \
  --kv-dtype fp8 --spec mtp --draft-tokens 3 --lm-head-draft --prefill-chunk 4096 \
  >"$OUT/step2-serve.log" 2>&1 &
srv=$!

ready=0
for i in $(seq 1 300); do    # up to 10 min for startup + graph preparation
  if grep -q 'CUDA graphs ready' "$OUT/step2-serve.log" 2>/dev/null; then
    ready=1
    break
  fi
  kill -0 "$srv" 2>/dev/null || break
  sleep 2
done

if [ "$ready" -eq 1 ]; then
  clog "server up: CUDA graphs ready (the unpatched build faults here)"
  clog "sending six concurrent greedy requests, 256 tokens each"
  req_pids=()
  for i in 1 2 3 4 5 6; do
    ( curl -s --max-time 600 http://127.0.0.1:8001/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"qwen3.8-flash-next-125b-a6b\",\"messages\":[{\"role\":\"user\",\"content\":\"Write the first $i multiples of 7, one per line, then stop.\"}],\"max_tokens\":256,\"temperature\":0}" \
        >"$OUT/req-$i.json" 2>&1; echo "req $i rc=$?" ) &
    req_pids+=($!)
  done
  wait "${req_pids[@]}"
  sleep 2
  python3 - "$OUT" <<'EOF'
import json, sys, os
out = sys.argv[1]
for i in range(1, 7):
    p = os.path.join(out, f"req-{i}.json")
    try:
        with open(p) as f:
            d = json.load(f)
        ch = d.get("choices", [{}])[0]
        usage = d.get("usage", {})
        print(f"req {i}: finish={ch.get('finish_reason')} completion_tokens={usage.get('completion_tokens')}")
    except Exception as e:
        raw = open(p, errors="replace").read() if os.path.exists(p) else "<missing>"
        print(f"req {i}: PARSE/HTTP FAILURE ({e.__class__.__name__}): {raw[:200]!r}")
EOF
  clog "requests done; stopping server"
  kill -TERM "$srv" 2>/dev/null
  wait "$srv" 2>/dev/null
  clog "server stopped"
else
  wait "$srv" 2>/dev/null
  rc=$?
  clog "server exited rc=$rc WITHOUT 'CUDA graphs ready' - the patched build still faults"
  clog "fault context (last lines of the serve log):"
  tail -15 "$OUT/step2-serve.log" 2>/dev/null | sed 's/^/    /'
  srv=""
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

git checkout -- src/ops/linear/linear.cpp
clog "patch dropped (linear.cpp restored to branch head)"
trap - INT TERM
clog "done"
