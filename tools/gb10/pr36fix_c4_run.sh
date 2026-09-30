#!/usr/bin/env bash
# C4 K=1 A/B single run: server start, throwaway warm-up pass, measured pass (3 waves),
# stop, per-round summary over the measured pass only, and the width check.
# Usage: pr36fix_c4_run.sh <SERVE_BIN> <OUT_DIR>
set -euo pipefail
cd /home/apollo11/ninfer-gb10

SERVE_BIN=$1
OUT=$2
ART=/home/apollo11/ninfer-gb10/out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer
PORT=18087
BASE_URL="http://127.0.0.1:$PORT"
LOAD=/home/apollo11/dgpp/scripts/serve_load.py
mkdir -p "$OUT"

if curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
    echo "Something already answers on $BASE_URL; stop it first." >&2
    exit 2
fi

echo "[$(date -u +%FT%TZ)] starting server: $SERVE_BIN" >&2
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
echo "[$(date -u +%FT%TZ)] server healthy" >&2

echo "[$(date -u +%FT%TZ)] warm-up pass (throwaway)" >&2
python3 "$LOAD" 127.0.0.1 "$PORT" --concurrency 4 --classes prose --max-tokens 256 --repeat 1 \
    >"$OUT/warmup.txt" 2>&1
echo "[$(date -u +%FT%TZ)] measured pass (3 waves)" >&2
python3 "$LOAD" 127.0.0.1 "$PORT" --concurrency 4 --classes prose --max-tokens 256 --repeat 3 \
    --warm 0 --json-out "$OUT/load.json" >"$OUT/load.txt" 2>&1

echo "[$(date -u +%FT%TZ)] stopping server" >&2
kill "$SRV" 2>/dev/null || true
wait "$SRV" 2>/dev/null || true
trap - EXIT

# The warm-up invocation (default --warm 1) completes 5 requests (1 warm + 4 wave);
# the measured pass (--warm 0) completes exactly 12. Summarize the measured 12 only.
python3 - "$OUT/request.jsonl" "$OUT/request_measured.jsonl" <<'EOF'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
done = [l for l in open(src) if json.loads(l).get("event") == "request_done"]
assert len(done) >= 17, f"expected 17 request_done records, got {len(done)}"
with open(dst, "w") as f:
    f.writelines(done[5:])
print(f"split: {len(done)} total, {len(done) - 5} measured")
EOF
python3 tools/gb10/request_log_summary.py "$OUT/request_measured.jsonl" >"$OUT/request_summary.md" 2>&1
echo "=== LOAD (measured pass) ==="
cat "$OUT/load.txt"
echo "=== WIDTH CHECK (throughput lines in server.log) ==="
grep -E "INFO  throughput" "$OUT/server.log" | sed -n '1,40p'
echo "[$(date -u +%FT%TZ)] done: $OUT" >&2
