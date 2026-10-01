#!/usr/bin/env bash
# K4 verify (per tree): unit tests, bitwise real test, warm-cache C4 K=1 nsys serve,
# NVTX + request-log + capacity stats.
# Usage: verify.sh <dir> <port>
set -uo pipefail
cd /home/apollo11/ninfer-gb10
DIR="profiles/bench/gb10/$1"
PORT="$2"
mkdir -p "$DIR"
ART="$PWD/out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer"
BIN=build/apps/ninfer-serve

echo "=== sha256 serve"
sha256sum "$BIN" | awk '{print $1}'

echo "=== unit tests"
"$PWD/build/tests/ninfer_sampling_test" || exit 1
"$PWD/build/tests/ninfer_flash_next_ple_stage_test" || exit 1
"$PWD/build/tests/ninfer_mtp_round_test" || exit 1
"$PWD/build/tests/ninfer_gdn_replay_fold_test" || exit 1
"$PWD/build/tests/ninfer_flash_next_ple_test" || exit 1
echo "unit tests OK"

busy() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l; }
[ "$(busy)" = "0" ] || { echo "ABORT: GPU busy"; exit 1; }

echo "=== real test (bitwise goldens)"
NINFER_QWEN38_FLASH_NEXT_WEIGHTS="$ART" "$PWD/build/tests/ninfer_qwen3_8_flash_next_real_test" || exit 1
echo "real test OK"

echo "=== nsys serve trace (C4 K=1, prose, warm cache)"
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  -o "$DIR/serve-c4k1" \
  "$BIN" "$ART" --port "$PORT" \
    --max-context 73728 --max-concurrency 4 --kv-dtype fp8 --preserve-thinking \
    --spec mtp --draft-tokens 1 --lm-head-draft \
    --request-log-jsonl "$DIR/serve.jsonl" > "$DIR/serve.log" 2>&1 &
pid=$!
ok=0
for i in $(seq 1 150); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
  kill -0 "$pid" 2>/dev/null || break
  sleep 4
done
[ "$ok" = "1" ] || { echo "server not healthy"; awk 'NR<=30' "$DIR/serve.log"; kill "$pid" 2>/dev/null; exit 1; }
echo "server healthy"
python3 /home/apollo11/dgpp/scripts/serve_load.py 127.0.0.1 "$PORT" \
  --concurrency 4 --classes prose --max-tokens 256 --repeat 3 --warm 1 \
  > "$DIR/load.txt" 2>&1
kill -INT "$pid"
wait "$pid" || true
echo "server stopped"

# The nsys sqlite export can race the profile process exit; re-export if it came out empty.
if [ ! -s "$DIR/serve-c4k1.sqlite" ]; then
  nsys export --type sqlite --force-overwrite true -o "$DIR/serve-c4k1.sqlite" \
    "$DIR/serve-c4k1.nsys-rep" >/dev/null 2>&1
fi

echo "=== capacity (startup memory) from serve log"
grep -m1 'capacity |' "$DIR/serve.log"

echo "=== nvtx stats"
python3 - "$DIR/serve-c4k1.sqlite" <<'EOF'
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
cur = db.cursor()
rounds = cur.execute("SELECT COUNT(*) FROM NVTX_EVENTS e JOIN StringIds s ON e.textId=s.id WHERE s.value='decode'").fetchone()[0]
print(f"decode rounds: {rounds}")
for name in ("decode.mtp_round", "flash_next_ple_wait_staged",
             "decode.mtp.submit.frame_upload", "decode.mtp.submit.ingress",
             "decode.mtp.submit", "program.submit", "cuda_graph.launch",
             "engine.commit_output", "engine.maintenance"):
    r = cur.execute("""SELECT COUNT(*), SUM(e.end - e.start), MAX(e.end - e.start)
        FROM NVTX_EVENTS e JOIN StringIds s ON e.textId=s.id
        WHERE s.value=? AND e.end > e.start""", (name,)).fetchone()
    if r and r[0]:
        print(f"{name:38s} n={r[0]:5d} total={r[1]/1e9:8.3f}s  mean={1e3*r[1]/1e9/r[0]:9.1f}us  max={r[2]/1e3:9.1f}us  ms/round={1e3*r[1]/1e9/rounds:6.2f}")
    else:
        print(f"{name:38s} (none)")
EOF

echo "=== request log totals"
python3 tools/gb10/request_log_summary.py "$DIR/serve.jsonl" 2>&1 | awk '/^Totals/'
echo "K4 VERIFY DONE ($1)"
