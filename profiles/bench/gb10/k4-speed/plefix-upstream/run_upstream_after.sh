#!/bin/bash
# PLE fork fix - AFTER battery on the exact upstream commit e5c45144 (GB10).
# Artifact: the v3_fork NVFP4 artifact (the upstream binder rejects the fork's
# fp8_mtp representation). Local verification deltas on the worktree: CMake
# arch 121a + device-gate 12.0|12.1 (uncommitted, "Do not commit" marked).
# 1) PLE op test; 2) real test; 3) MTP-on C1/C4; 4) MTP-off C1/C4; 5) C4 K=1 decode.
set -u
WT=$HOME/ninfer-gb10-upstream
cd $HOME/ninfer-gb10
ART="$PWD/../models/qwen3_8_flash_next_v3_fork/qwen3_8_flash_next_125b_a6b_nvfp4.ninfer"
BIN="$WT/build/apps/ninfer-serve"
D="$PWD/profiles/bench/gb10/k4-speed/plefix-upstream"
mkdir -p "$D"
busy() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l; }
git -C "$WT" log --oneline -1 | awk '{print "upstream commit:", $0}'
sha256sum "$BIN" | awk '{print "upstream serve sha256:", $1}'

echo "=== [1/5] PLE op test (in-place + fork)"
[ "$(busy)" = "0" ] || { echo "ABORT GPU busy"; exit 1; }
"$WT/build/tests/ninfer_flash_next_ple_test" > "$D/up_ple_test.out" 2>&1 \
  && echo "ple op test PASS" || { echo "ple op test FAIL"; awk 'NR<=10' "$D/up_ple_test.out"; exit 1; }

echo "=== [2/5] real test (v3_fork NVFP4 artifact)"
[ "$(busy)" = "0" ] || { echo "ABORT GPU busy"; exit 1; }
NINFER_QWEN38_FLASH_NEXT_WEIGHTS="$ART" "$WT/build/tests/ninfer_qwen3_8_flash_next_real_test" \
  > "$D/up_real_test.out" 2>&1 && echo "real test PASS" || { echo "real test FAIL (exit $?)"; awk 'NR<=15' "$D/up_real_test.out"; }

probe() { # $1=tag $2=conc $3=port $4... extra flags
  local tag=$1 conc=$2 port=$3; shift 3
  "$BIN" "$ART" --port "$port" --max-context 73728 --max-concurrency "$conc" \
    --kv-dtype fp8 --preserve-thinking --no-cuda-graph "$@" \
    > "$D/up-$tag.log" 2>&1 &
  local p=$! ok=0
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $p 2>/dev/null || break
    sleep 5
  done
  [ "$ok" = "1" ] || { echo "NOT HEALTHY ($tag)"; awk 'NR<=14' "$D/up-$tag.log"; kill -9 $p 2>/dev/null; return 1; }
  python3 profiles/bench/gb10/k4-speed/plefix-upstream/route_probe_nologprobs.py "$port" "$D/up-$tag-probe.json"
  kill -TERM $p; wait $p 2>/dev/null || true
  sleep 3
}

echo "=== [3/5] MTP ON: C1 then C4 (expect IDENTICAL)"
sleep 3
[ "$(busy)" = "0" ] || { echo "ABORT GPU busy"; exit 1; }
probe mtp-c1 1 18891 --spec mtp --draft-tokens 1 --lm-head-draft
[ "$(busy)" = "0" ] || { echo "ABORT GPU busy"; exit 1; }
probe mtp-c4 4 18892 --spec mtp --draft-tokens 1 --lm-head-draft

echo "=== [4/5] MTP OFF: C1 then C4 (expect IDENTICAL)"
[ "$(busy)" = "0" ] || { echo "ABORT GPU busy"; exit 1; }
probe nomtp-c1 1 18893
[ "$(busy)" = "0" ] || { echo "ABORT GPU busy"; exit 1; }
probe nomtp-c4 4 18894

python3 - "$D" <<'PY'
import json, sys
d = sys.argv[1]
def sha(f):
    j = json.load(open(f))
    return j["text_sha"], j["text_len"]
for pair, tag in ((("mtp-c1", "mtp-c4"), "MTP ON"), (("nomtp-c1", "nomtp-c4"), "MTP OFF")):
    a, b = pair
    try:
        sa, la = sha(f"{d}/up-{a}-probe.json")
        sb, lb = sha(f"{d}/up-{b}-probe.json")
        print(f"{tag}: C1 {sa} (len {la})  C4 {sb} (len {lb})  " + ("IDENTICAL" if sa == sb else "SPLIT"))
    except FileNotFoundError:
        print(f"{tag}: probe json missing")
PY

echo "=== [5/5] C4 K=1 decode run (speed + acceptance)"
sleep 3
[ "$(busy)" = "0" ] || { echo "ABORT GPU busy"; exit 1; }
"$BIN" "$ART" --port 18895 --max-context 73728 --max-concurrency 4 \
  --kv-dtype fp8 --preserve-thinking --no-cuda-graph --spec mtp --draft-tokens 1 --lm-head-draft \
  --request-log-jsonl "$D/up-c4k1.jsonl" > "$D/up-c4k1.log" 2>&1 &
p=$!
ok=0
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:18895/health" >/dev/null 2>&1 && { ok=1; break; }
  kill -0 $p 2>/dev/null || break
  sleep 5
done
[ "$ok" = "1" ] || { echo "NOT HEALTHY (c4k1)"; awk 'NR<=14' "$D/up-c4k1.log"; kill -9 $p 2>/dev/null; exit 1; }
awk '/capacity/' "$D/up-c4k1.log" | awk 'NR<=1'
python3 $HOME/dgpp/scripts/serve_load.py 127.0.0.1 18895 \
  --concurrency 4 --classes prose --max-tokens 256 --repeat 3 --warm 1 \
  > "$D/up-c4k1-load.txt" 2>&1
awk '/aggregate wall|per-class/' "$D/up-c4k1-load.txt"
kill -TERM $p; wait $p 2>/dev/null || true
python3 tools/gb10/request_log_summary.py "$D/up-c4k1.jsonl" 2>&1 | awk '/^Totals/'
echo "plefix_up_after_done"
