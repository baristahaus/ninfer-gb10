#!/usr/bin/env bash
# C4 K=0 determinism follow-up (2026-10-02, twoFour): the K=1 probe showed C4 output varies
# run-to-run. This repeats the same pattern with no MTP (K=0, plain batching) to separate
# speculative decoding from plain batched decode. 370a5247 arm; C4 K=0 sweep x3 on one server,
# then x1 on a fresh server. One GPU job at a time. Data: .../qsa-verify/determinism-k0/.
set -euo pipefail
source tools/gb10/common.sh

ROOT=$OUT_ROOT/qsa-verify/determinism-k0
SERVE=$OUT_ROOT/qsa-verify/arms/370a5247.ninfer-serve
require_binary "$SERVE"
mkdir -p "$ROOT"
SERVER_PID=0

gpu_idle() {
  if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]]; then
    echo "GPU held by another job; single-instance discipline: aborting." >&2
    exit 1
  fi
}

probe_start_server() { # $1 = tag, $2 = concurrency
  local tag=${1:?tag required} conc=${2:?conc required}
  local args=(--port "$PORT" --max-context 73728 --max-concurrency "$conc" --kv-dtype fp8
              --preserve-thinking --request-log-jsonl "$ROOT/$tag.requests.jsonl")
  log "start $tag: C$conc K=0 (no MTP)"
  "$SERVE" "$ART" "${args[@]}" >"$ROOT/$tag.serve.log" 2>&1 &
  SERVER_PID=$!
  local waited=0
  until curl -sf "$BASE_URL/health" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "$tag: server exited during startup. Last log lines:" >&2
      tail -n 40 "$ROOT/$tag.serve.log" >&2
      exit 1
    fi
    ((waited < 1800)) || { echo "$tag: not healthy after 30 min" >&2; exit 1; }
    sleep 5
    waited=$((waited + 5))
  done
}

probe_stop_server() { # $1 = tag (SIGTERM, then verify the GPU is actually released)
  local tag=${1:?tag required}
  kill "$SERVER_PID" 2>/dev/null || true
  for _ in $(seq 1 120); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null && \
       [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]]; then
      break
    fi
    sleep 2
  done
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "$tag: survived 4 min after SIGTERM; SIGKILL" >&2
    kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]]; then
    echo "$tag: GPU still held after teardown; aborting." >&2
    exit 1
  fi
  log "stop $tag: GPU released"
}

sweep() { # $1 = out tag, $2 = concurrency
  "$PYTHON" tools/gb10/concurrency_sweep.py "$BASE_URL" "$ROOT/$1.load.json" \
      --n "$2" --max-tokens 512 --prompt-chars 2000 --ignore-eos >"$ROOT/$1.sweep.log" 2>&1 \
    || log "$1: sweep reported errors; see $ROOT/$1.sweep.log"
  log "sweep $1 done"
}

gpu_idle
probe_start_server c4k0-server 4
sweep c4k0-run1 4
sweep c4k0-run2 4
sweep c4k0-run3 4
probe_stop_server c4k0-server
probe_start_server c4k0-fresh 4
sweep c4k0-run4 4
probe_stop_server c4k0-fresh

"$PYTHON" - "$ROOT" <<'EOF'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
tags = ["c4k0-run1", "c4k0-run2", "c4k0-run3", "c4k0-run4"]
table = {}
for tag in tags:
    p = root / (tag + ".load.json")
    if not p.exists():
        print(tag + ": MISSING " + p.name)
        continue
    data = json.loads(p.read_text())
    reqs = data["runs"][0]["requests"]
    rows = {r["stream"]: r.get("content_sha256", "?") for r in reqs}
    table[tag] = rows
    print(tag + ": " + " ".join(h[:12] for h in rows.values()))
c4 = [table.get("c4k0-run%d" % i) for i in (1, 2, 3, 4)]
if all(c4):
    for k in sorted(set().union(*[set(v) for v in c4])):
        vals = {v[k] for v in c4 if k in v}
        verdict = "IDENTICAL across 4 runs" if len(vals) == 1 else "DIFFERS (%d distinct)" % len(vals)
        print("C4K0 %s: %s" % (k, verdict))
EOF
echo "probe data: $ROOT"
