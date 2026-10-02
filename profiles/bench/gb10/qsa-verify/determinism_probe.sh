#!/usr/bin/env bash
# C4 determinism probe (2026-10-02, twoFour), per the handoff run list:
# 370a5247 arm; C4 K=1 sweep x3 on one server, then x1 on a fresh server; C1 K=1 x1 control.
# Reports content_sha256 per stream per run (sweep records it since 1c2e979b).
# Follow-up if C4 outputs differ between runs: same pattern with K=0 (no MTP), tag k0.
# One GPU job at a time. Data: profiles/bench/gb10/qsa-verify/determinism/.
set -euo pipefail
source tools/gb10/common.sh

ROOT=$OUT_ROOT/qsa-verify/determinism
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

probe_start_server() { # $1 = tag, $2 = concurrency, $3 = K (0 = no MTP)
  local tag=$1 conc=$2 K=$3
  local args=(--port "$PORT" --max-context 73728 --max-concurrency "$conc" --kv-dtype fp8
              --preserve-thinking --request-log-jsonl "$ROOT/$tag.requests.jsonl")
  if ((K > 0)); then args+=(--spec mtp --draft-tokens "$K" --lm-head-draft); fi
  log "start $tag: C$conc K=$K"
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
  local tag=$1
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

# C4 K=1: three runs on one server
probe_start_server c4k1-server 4 1
sweep c4k1-run1 4
sweep c4k1-run2 4
sweep c4k1-run3 4
probe_stop_server c4k1-server

# C4 K=1: one run on a fresh server
probe_start_server c4k1-fresh 4 1
sweep c4k1-run4 4
probe_stop_server c4k1-fresh

# C1 K=1 control: one run
probe_start_server c1k1-server 1 1
sweep c1k1-run1 1
probe_stop_server c1k1-server

# Report content_sha256 per stream per run
"$PYTHON" - "$ROOT" <<'EOF'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
tags = ["c4k1-run1", "c4k1-run2", "c4k1-run3", "c4k1-run4", "c1k1-run1"]
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
c4 = [table.get("c4k1-run%d" % i) for i in (1, 2, 3, 4)]
if all(c4):
    for k in sorted(set().union(*[set(v) for v in c4])):
        vals = {v[k] for v in c4 if k in v}
        verdict = "IDENTICAL across 4 runs" if len(vals) == 1 else "DIFFERS (%d distinct)" % len(vals)
        print("C4K1 %s: %s" % (k, verdict))
EOF
echo "probe data: $ROOT"
