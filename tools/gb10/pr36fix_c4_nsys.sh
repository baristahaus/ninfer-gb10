#!/usr/bin/env bash
# C4 K=1 nsys pair: one nsys rep per tree (fix, then base), Block J workload
# (serve_load prose c=4, max-tokens 256, repeat 3, server with --spec mtp --draft-tokens 1
# --lm-head-draft). Capture flags identical to Block J / the C4 runbook.
# Usage: pr36fix_c4_nsys.sh <TAG> <SERVE_BIN>
set -euo pipefail
cd /home/apollo11/ninfer-gb10

TAG=$1
SERVE_BIN=$2
T=profiles/bench/gb10/pr36fix-verify/C4-K1
OUT=$T/nsys-$TAG
ART=/home/apollo11/ninfer-gb10/out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer
PORT=18007
BASE_URL="http://127.0.0.1:$PORT"
LOAD=/home/apollo11/dgpp/scripts/serve_load.py
mkdir -p "$OUT"

if curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
    echo "Something already answers on $BASE_URL; stop it first." >&2
    exit 2
fi

echo "[$(date -u +%FT%TZ)] nsys profile: $TAG ($SERVE_BIN)" >&2
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
    --force-overwrite true -o "$OUT/trace" \
    "$SERVE_BIN" "$ART" --port "$PORT" --max-context 73728 --max-concurrency 4 \
    --kv-dtype fp8 --preserve-thinking --spec mtp --draft-tokens 1 --lm-head-draft \
    --request-log-jsonl "$OUT/request.jsonl" >"$OUT/server.log" 2>&1 &
SRV=$!
trap 'kill "$SRV" 2>/dev/null || true' EXIT
for i in $(seq 1 360); do
    if curl -sf "$BASE_URL/health" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$SRV" 2>/dev/null; then
        echo "Server exited during startup. Last log lines:" >&2
        tail -n 40 "$OUT/server.log" >&2
        exit 1
    fi
    sleep 5
done
curl -sf "$BASE_URL/health" >/dev/null 2>&1 || { echo "Server not healthy. Last log lines:" >&2; tail -n 40 "$OUT/server.log" >&2; exit 1; }
echo "[$(date -u +%FT%TZ)] server healthy; load (Block J workload)" >&2
python3 "$LOAD" 127.0.0.1 "$PORT" --concurrency 4 --classes prose --max-tokens 256 \
    --repeat 3 >"$OUT/load.txt" 2>&1

echo "[$(date -u +%FT%TZ)] stopping server (nsys finalizes rep)" >&2
kill "$SRV" 2>/dev/null || true
wait "$SRV" 2>/dev/null || true
trap - EXIT
ls -la "$OUT/trace.nsys-rep" >&2
echo "[$(date -u +%FT%TZ)] done: $OUT" >&2
