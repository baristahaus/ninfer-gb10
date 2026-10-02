#!/bin/bash
# PLE fork fix - BEFORE battery on the PR base 2aa87467 (GB10).
# MTP-on and MTP-off C1/C4 probes. Expect SPLIT (the unfixed defect).
set -u
WT=$HOME/ninfer-gb10-upstream
cd $HOME/ninfer-gb10
ART="$PWD/../models/qwen3_8_flash_next_v3_fork/qwen3_8_flash_next_125b_a6b_nvfp4.ninfer"
BIN="$WT/build/apps/ninfer-serve"
D="$PWD/profiles/bench/gb10/k4-speed/plefix-upstream"
mkdir -p "$D"
busy() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l; }
git -C "$WT" log --oneline -1 | awk '{print "base commit:", $0}'
sha256sum "$BIN" | awk '{print "base serve sha256:", $1}'

probe() { # $1=tag $2=conc $3=port $4... extra flags
  local tag=$1 conc=$2 port=$3; shift 3
  "$BIN" "$ART" --port "$port" --max-context 73728 --max-concurrency "$conc" \
    --kv-dtype fp8 --preserve-thinking --no-cuda-graph "$@" \
    > "$D/base-$tag.log" 2>&1 &
  local p=$! ok=0
  for i in $(seq 1 90); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $p 2>/dev/null || break
    sleep 4
  done
  [ "$ok" = "1" ] || { echo "NOT HEALTHY ($tag)"; awk 'NR<=10' "$D/base-$tag.log"; kill -9 $p 2>/dev/null; return 1; }
  python3 profiles/bench/gb10/k4-speed/plefix-upstream/route_probe_nologprobs.py "$port" "$D/base-$tag-probe.json"
  kill -TERM $p; wait $p 2>/dev/null || true
  sleep 3
}

echo "=== base, MTP ON: C1 then C4 (expect SPLIT)"
[ "$(busy)" = "0" ] || { echo "ABORT GPU busy"; exit 1; }
probe mtp-c1 1 18901 --spec mtp --draft-tokens 1 --lm-head-draft
[ "$(busy)" = "0" ] || { echo "ABORT GPU busy"; exit 1; }
probe mtp-c4 4 18902 --spec mtp --draft-tokens 1 --lm-head-draft

echo "=== base, MTP OFF: C1 then C4 (expect SPLIT)"
[ "$(busy)" = "0" ] || { echo "ABORT GPU busy"; exit 1; }
probe nomtp-c1 1 18903
[ "$(busy)" = "0" ] || { echo "ABORT GPU busy"; exit 1; }
probe nomtp-c4 4 18904

python3 - "$D" <<'PY'
import json, sys
d = sys.argv[1]
def sha(f):
    j = json.load(open(f))
    return j["text_sha"], j["text_len"]
for pair, tag in ((("mtp-c1", "mtp-c4"), "MTP ON"), (("nomtp-c1", "nomtp-c4"), "MTP OFF")):
    a, b = pair
    sa, la = sha(f"{d}/base-{a}-probe.json")
    sb, lb = sha(f"{d}/base-{b}-probe.json")
    print(f"BASE {tag}: C1 {sa} (len {la})  C4 {sb} (len {lb})  " + ("IDENTICAL" if sa == sb else "SPLIT"))
PY
echo "plefix_up_before_done"
