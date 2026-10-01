#!/bin/bash
# K4 speed gate (a5581fce): C4 K=1 warm serve, --pipelined-decode off vs on, ABBA.
# Reports per arm: serve_load wall tok/s + request-log Totals (decode tok/s,
# host exposed ms/round).
set -u
cd /home/apollo11/ninfer-gb10
ART="$PWD/out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer"
BIN="$PWD/build/apps/ninfer-serve"
D="profiles/bench/gb10/k4-speed"
mkdir -p "$D"
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ] || { echo "ABORT GPU busy"; exit 1; }
[ -x "$BIN" ] || { echo "ABORT no binary"; exit 1; }
sha256sum "$BIN" | awk '{print "serve sha256:", $1}'

arm() { # $1=tag $2=extra
  local tag=$1 extra=$2 port=$((18790 + RANDOM % 10))
  echo "=== arm $tag (extra: [$extra]) port $port"
  "$BIN" "$ART" --port "$port" --max-context 73728 --max-concurrency 4 \
    --kv-dtype fp8 --preserve-thinking --spec mtp --draft-tokens 1 --lm-head-draft \
    --request-log-jsonl "$D/$tag.jsonl" $extra > "$D/$tag.log" 2>&1 &
  local p=$!
  local ok=0
  for i in $(seq 1 90); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $p 2>/dev/null || break
    sleep 4
  done
  [ "$ok" = "1" ] || { echo "NOT HEALTHY ($tag)"; awk 'NR<=10' "$D/$tag.log"; kill -9 $p 2>/dev/null; exit 1; }
  awk '/capacity/' "$D/$tag.log" | awk 'NR<=1'
  python3 /home/apollo11/dgpp/scripts/serve_load.py 127.0.0.1 "$port" \
    --concurrency 4 --classes prose --max-tokens 256 --repeat 3 --warm 1 \
    > "$D/$tag-load.txt" 2>&1
  awk '/aggregate wall|per-class/' "$D/$tag-load.txt"
  kill -TERM $p; wait $p 2>/dev/null || true
  echo "--- $tag request log Totals:"
  python3 tools/gb10/request_log_summary.py "$D/$tag.jsonl" 2>&1 | awk '/^Totals/'
  echo
}

# ABBA: A=off, B=on
arm off-a1 ""
arm on-b1  "--pipelined-decode"
arm on-b2  "--pipelined-decode"
arm off-a2 ""
echo "speed_abba_done"
