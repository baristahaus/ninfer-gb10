#!/usr/bin/env bash
# PR #36 decode speed sweep: N=1 and N=8 active requests against one tree's ninfer-serve.
# fp8_mtp artifact, the local serving configuration (MTP draft 2), decode-only rates from the
# structured request log. Run one tree at a time (GPU is single-instance).
#
# Usage: pr36fix_decode_sweep.sh <SERVE_BIN> <OUT_DIR>
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

SERVE_BIN=$1
OUT=$2
ART=/home/apollo11/ninfer-gb10/out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer
PORT=18087
BASE_URL="http://127.0.0.1:$PORT"
mkdir -p "$OUT"

if curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
    echo "Something already answers on $BASE_URL; stop it first." >&2
    exit 2
fi

echo "[$(date -u +%FT%TZ)] starting server: $SERVE_BIN" >&2
"$SERVE_BIN" "$ART" --port "$PORT" \
    --max-context 73728 --max-concurrency 8 --kv-dtype fp8 \
    --spec mtp --draft-tokens 2 --lm-head-draft --preserve-thinking \
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
if ! curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
    echo "Server not healthy after 30 minutes. Last log lines:" >&2
    tail -n 40 "$OUT/server.log" >&2
    exit 1
fi
echo "[$(date -u +%FT%TZ)] server healthy; running sweep N=1,8" >&2
python3 tools/gb10/concurrency_sweep.py "$BASE_URL" "$OUT/sweep.json" --n 1,8
echo "[$(date -u +%FT%TZ)] stopping server" >&2
kill "$SRV" 2>/dev/null || true
wait "$SRV" 2>/dev/null || true
trap - EXIT
python3 tools/gb10/request_log_summary.py "$OUT/request.jsonl" >"$OUT/request_summary.md" 2>&1
echo "[$(date -u +%FT%TZ)] done: $OUT" >&2
