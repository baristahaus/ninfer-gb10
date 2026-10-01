#!/bin/bash
# Route-naming run (a5581fce): C4 config with the page pool shrunk to 1,152
# (the C1 config's pool). Serve enforces --kv-capacity >= --max-context, so both
# are set to 18432: page pool = 18432/64 x 4 = 288 x 4 = 1,152, matching C1.
# The probe request (~356 tokens) is far below the 18432 cap, so max-context
# does not bind. If the cold first-request output is the 1212 B C1 output, the
# 1172/1212 split tracks KV page-pool size; if it stays 1172 B, it tracks
# max_concurrency.
set -u
cd /home/apollo11/ninfer-gb10
ART="$PWD/out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer"
BIN="$PWD/build/apps/ninfer-serve"
D="profiles/bench/gb10/k4-speed"
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ] || { echo "ABORT GPU busy"; exit 1; }
[ -x "$BIN" ] || { echo "ABORT no binary"; exit 1; }
PORT=18810
TAG="routename-c4-cap18432"
echo "=== $TAG port $PORT"
"$BIN" "$ART" --port "$PORT" --max-context 18432 --max-concurrency 4 \
  --kv-capacity 18432 --token-logprobs --kv-dtype fp8 --preserve-thinking --spec mtp \
  --draft-tokens 1 --lm-head-draft --request-log-jsonl "$D/$TAG.jsonl" \
  > "$D/$TAG.log" 2>&1 &
P=$!
ok=0
for i in $(seq 1 90); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
  kill -0 $P 2>/dev/null || break
  sleep 4
done
[ "$ok" = "1" ] || { echo "NOT HEALTHY"; awk 'NR<=10' "$D/$TAG.log"; kill -9 $P 2>/dev/null; exit 1; }
echo "--- capacity line:"
awk '/capacity/' "$D/$TAG.log" | awk 'NR<=1'
echo "--- cold first request (logprob_probe):"
python3 "$D/../k4-verify/logprob_probe.py" "$PORT" "$D/$TAG.json"
echo "--- Totals:"
python3 tools/gb10/request_log_summary.py "$D/$TAG.jsonl" 2>&1 | awk '/^Totals/'
kill -TERM $P; wait $P 2>/dev/null || true
echo "routename_done"
