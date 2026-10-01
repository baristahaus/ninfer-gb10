#!/bin/bash
# a5581fce reruns (Opus's bisect + shutdown 503 fix), GB10, 2026-10-01.
# 1. bisect: cold first 256-token request with and without --pipelined-decode
# 2. double SIGTERM during a stream: client must see a 503 error event
set -u
cd /home/apollo11/ninfer-gb10
ART="$PWD/out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer"
BIN="$PWD/build/apps/ninfer-serve"
D="profiles/bench/gb10/k4-verify"
SERVE_FLAGS="--max-context 73728 --max-concurrency 1 --kv-dtype fp8 --preserve-thinking --spec mtp --draft-tokens 1 --lm-head-draft"
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ] || { echo "ABORT GPU busy"; exit 1; }
[ -x "$BIN" ] || { echo "ABORT no binary"; exit 1; }

healthy() { # $1=port
  for i in $(seq 1 90); do
    curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && return 0
    sleep 4
  done
  return 1
}

run_serve() { # $1=name $2=port $3=extra flags; sets SERVE_PID
  local name=$1 port=$2 extra=$3
  SERVE_PID=""
  echo "=== $name (port $port, extra: [$extra])"
  "$BIN" "$ART" --port "$port" $SERVE_FLAGS $extra > "$D/a5581-$name.log" 2>&1 &
  local p=$!
  if ! healthy $port; then echo "NOT HEALTHY ($name)"; awk 'NR<=8' "$D/a5581-$name.log"; kill -9 $p 2>/dev/null; return 1; fi
  SERVE_PID=$p
  echo "healthy ($name, pid $p)"
  return 0
}

# --- 1. bisect: cold first request, survivor 256 tokens ---
if run_serve bisect-off 18781 ""; then
  python3 "$D/discard_load.py" 18781 "$D/a5581-bisect-off.jsonl" solo survivor
  kill -TERM $SERVE_PID; wait $SERVE_PID 2>/dev/null || true
  echo "bisect-off serve log:"; awk '/released|stopped|error|invariant/' "$D/a5581-bisect-off.log" | awk 'NR<=4'
  echo
fi

if run_serve bisect-on 18782 "--pipelined-decode"; then
  python3 "$D/discard_load.py" 18782 "$D/a5581-bisect-on.jsonl" solo survivor
  kill -TERM $SERVE_PID; wait $SERVE_PID 2>/dev/null || true
  echo "bisect-on serve log:"; awk '/released|stopped|error|invariant/' "$D/a5581-bisect-on.log" | awk 'NR<=4'
  echo
fi

# --- 2. double SIGTERM during a 512-token stream ---
echo "=== double-term (port 18783)"
"$BIN" "$ART" --port 18783 $SERVE_FLAGS > "$D/a5581-doubleterm.log" 2>&1 &
p3=$!
healthy 18783 || { echo "NOT HEALTHY (doubleterm)"; awk 'NR<=8' "$D/a5581-doubleterm.log"; kill -9 $p3 2>/dev/null; exit 1; }
echo "healthy (doubleterm, pid $p3)"
python3 "$D/stream_probe.py" 18783 512 "$D/a5581-shC-stream.json" > "$D/a5581-shC-probe.out" 2>&1 &
pp=$!
sleep 2.5
echo "first TERM at $(date -u +%H:%M:%S.%3N)"
kill -TERM $p3
sleep 2.0
echo "second TERM at $(date -u +%H:%M:%S.%3N)"
kill -TERM $p3
wait $pp 2>/dev/null || true
wait $p3 2>/dev/null || true
echo "probe result:"; cat "$D/a5581-shC-probe.out"
echo
echo "serve log (shutdown/cancel lines):"
awk '/shutdown|drain|cancelled|released|stopped|error/' "$D/a5581-doubleterm.log" | awk 'NR<=8'
echo "rerun-a5581_done"
