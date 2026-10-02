#!/bin/bash
# C1 sequential MTP decode speed on the exact upstream commit e5c45144 (GB10).
# C4-concurrent MTP decode is blocked on this commit by the view-mismatch
# defect (recorded separately); sequential requests stay single-in-flight.
set -u
WT=$HOME/ninfer-gb10-upstream
cd $HOME/ninfer-gb10
ART="$PWD/../models/qwen3_8_flash_next_v3_fork/qwen3_8_flash_next_125b_a6b_nvfp4.ninfer"
BIN="$WT/build/apps/ninfer-serve"
D="$PWD/profiles/bench/gb10/k4-speed/plefix-upstream"
mkdir -p "$D"
busy() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l; }
[ "$(busy)" = "0" ] || { echo "ABORT GPU busy"; exit 1; }
"$BIN" "$ART" --port 18896 --max-context 73728 --max-concurrency 4 \
  --kv-dtype fp8 --preserve-thinking --no-cuda-graph --spec mtp --draft-tokens 1 \
  --lm-head-draft --request-log-jsonl "$D/up-c1k1.jsonl" > "$D/up-c1k1.log" 2>&1 &
p=$!
ok=0
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:18896/health" >/dev/null 2>&1 && { ok=1; break; }
  kill -0 $p 2>/dev/null || break
  sleep 5
done
[ "$ok" = "1" ] || { echo "NOT HEALTHY"; awk 'NR<=14' "$D/up-c1k1.log"; kill -9 $p 2>/dev/null; exit 1; }
awk '/capacity/' "$D/up-c1k1.log" | awk 'NR<=1'
python3 $HOME/dgpp/scripts/serve_load.py 127.0.0.1 18896 \
  --concurrency 1 --classes prose --max-tokens 256 --repeat 3 --warm 1 \
  > "$D/up-c1k1-load.txt" 2>&1
awk '/aggregate wall|per-class/' "$D/up-c1k1-load.txt"
kill -TERM $p; wait $p 2>/dev/null || true
python3 tools/gb10/request_log_summary.py "$D/up-c1k1.jsonl" 2>&1 | awk '/^Totals/'
echo "plefix_up_c1k1_done"
