#!/usr/bin/env bash
# Discriminator (2026-10-11): the sweep's own requests (8000-char prompts, --ignore-eos to a
# fixed 512 tokens, the perplexity-corpus prompt set) at C1 and C4, on a server started with
# --max-concurrency 4 (the sweep used 8). This isolates the server-sizing variable: if the C1/C4
# aggregate tok/s move toward the published 2026-10-03 figures (59.9 / 121.4) the gap is the
# 8-lane server sizing; if they stay near the sweep's 45.3 / 81.5, the sizing is not the cause.
set -euo pipefail
cd /home/apollo11/ninfer-gb10

ROOT=profiles/bench/gb10/control-2026-10-11
mkdir -p "$ROOT"
ART=/home/apollo11/ninfer-gb10/out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer
PORT=8001
SERVE_BIN=build/apps/ninfer-serve
PYTHON=python3

log() { printf '[discr %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

# 1. The GPU must be idle (one GPU job at a time).
log "checking GPU is free"
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
if ((util > 5)); then
  log "GPU is busy ($util%); aborting"
  exit 1
fi

# 2. Start the server with the sweep's flags but --max-concurrency 4 (instead of 8).
log "starting server (sweep flags, --max-concurrency 4)"
"$SERVE_BIN" "$ART" --host 0.0.0.0 --port "$PORT" --max-context 73728 --kv-capacity auto \
  --max-concurrency 4 --kv-dtype fp8 --spec mtp --draft-tokens 3 --lm-head-draft \
  --preserve-thinking --request-log-jsonl "$ROOT/discr-k3.jsonl" >"$ROOT/discr-server.log" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

# 3. Wait for the server to be healthy.
log "waiting for server to be healthy"
waited=0
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "server exited during startup"
    tail -n 40 "$ROOT/discr-server.log" >&2
    exit 1
  fi
  if ((waited >= 1800)); then
    log "server not healthy after 30 minutes"
    tail -n 40 "$ROOT/discr-server.log" >&2
    exit 1
  fi
  sleep 5
  waited=$((waited + 5))
done
log "server healthy after ${waited}s"

# 4. Run the sweep's own requests at C1 and C4 (8000-char prompts, --ignore-eos to 512).
log "running the sweep's own requests at C1 and C4"
"$PYTHON" tools/gb10/concurrency_sweep.py "http://127.0.0.1:$PORT" "$ROOT/discr-k3.json" \
  --n 1,4 --max-tokens 512 --prompt-chars 8000 --ignore-eos >"$ROOT/discr-k3.txt" 2>&1 \
  || log "concurrency_sweep exited nonzero"

# 5. Stop the server.
log "stopping server"
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
trap - EXIT

log "discriminator complete; results under $ROOT"
