#!/usr/bin/env bash
# Control run (step 1 of the 2026-10-11 discrepancy analysis): the published 2026-10-03
# protocol (DGPP's scripts/serve_load.py) on the current head, MTP K=3, with the exact
# 2026-10-03 NInfer arm server flags. The protocol drops the page cache before the server
# start. Compares the C1 (single-stream) and C4 (four concurrent) wall tok/s to the
# published figures (59.89 C1, 121.39 C4).
set -euo pipefail
cd /home/apollo11/ninfer-gb10

ROOT=profiles/bench/gb10/control-2026-10-11
mkdir -p "$ROOT"
ART=/home/apollo11/ninfer-gb10/out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer
PORT=18087
SERVE_BIN=build/apps/ninfer-serve
PYTHON=python3
DGPP_DIR=/home/apollo11/dgpp

log() { printf '[control %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

# 1. The GPU must be idle (one GPU job at a time).
log "checking GPU is free"
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
if ((util > 5)); then
  log "GPU is busy ($util%); aborting"
  exit 1
fi

# Record the tree under test.
git log -1 --format='tree %h %cd' >"$ROOT/tree.txt"
"$SERVE_BIN" --version >"$ROOT/serve-version.txt" 2>&1 || true

# 2. Drop the page cache (the protocol drops caches before each server start).
log "dropping page cache"
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null

# 3. Start the server with the 2026-10-03 server flags (K=3).
log "starting server (K=3, 2026-10-03 flags)"
"$SERVE_BIN" "$ART" --port "$PORT" \
  --max-context 73728 --max-concurrency 4 --kv-dtype fp8 --preserve-thinking \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --request-log-jsonl "$ROOT/ninfer-k3.jsonl" >"$ROOT/server.log" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

# 4. Wait for the server to be healthy.
log "waiting for server to be healthy"
waited=0
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "server exited during startup"
    tail -n 40 "$ROOT/server.log" >&2
    exit 1
  fi
  if ((waited >= 1800)); then
    log "server not healthy after 30 minutes"
    tail -n 40 "$ROOT/server.log" >&2
    exit 1
  fi
  sleep 5
  waited=$((waited + 5))
done
log "server healthy after ${waited}s"

# 5. Run the serve_load.py protocol: C1, C2, C4; five classes; 256 output tokens;
#    greedy (temp 0); thinking off; 3 repeats per class-concurrency.
log "running serve_load.py protocol"
"$PYTHON" "$DGPP_DIR/scripts/serve_load.py" 127.0.0.1 "$PORT" \
  --concurrency 1,2,4 --classes prose,code,json,math,chat --max-tokens 256 --repeat 3 \
  --json-out "$ROOT/ninfer-k3.json" >"$ROOT/ninfer-k3.txt" 2>&1 \
  || log "serve_load.py exited nonzero"

# 6. Stop the server.
log "stopping server"
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
trap - EXIT

# 7. Request-log summary (the decode-only rates).
"$PYTHON" tools/gb10/request_log_summary.py "$ROOT/ninfer-k3.jsonl" >"$ROOT/ninfer-k3.md" 2>&1 \
  || log "request_log_summary exited nonzero"

log "control run complete; results under $ROOT"
