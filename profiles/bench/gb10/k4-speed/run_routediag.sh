#!/bin/bash
# Route diagnostics (a5581fce, --no-cuda-graph): does the 1172/1212 first-request
# split come from a config-dependent prefill route or a decode route?
#  R1: C1, logits capture (NINFER_FLASH_NEXT_LOGITS_DIR)      -> round-by-round logits
#  R2: C4, logits capture                                     -> round-by-round logits
#  R3: C1, GDN state at frontier = prompt_tokens (NINFER_FLASH_NEXT_STATE_DIR/FRONTIER)
#  R4: C4, GDN state at the same frontier
# Cold first request per serve (the 256-token probe, greedy).
set -u
cd /home/apollo11/ninfer-gb10
ART="$PWD/out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer"
BIN="$PWD/build/apps/ninfer-serve"
D="profiles/bench/gb10/k4-speed/routediag"
mkdir -p "$D"
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ] || { echo "ABORT GPU busy"; exit 1; }
[ -x "$BIN" ] || { echo "ABORT no binary"; exit 1; }
sha256sum "$BIN" | awk '{print "serve sha256:", $1}'

serve() { # $1=tag $2=conc $3..=env pairs (KEY=VAL)
  local tag=$1 conc=$2; shift 2
  local port=$((18820 + RANDOM % 20))
  echo "=== $tag (conc=$conc) port $port"
  env "$@" "$BIN" "$ART" --port "$port" --max-context 73728 --max-concurrency "$conc" \
    --kv-dtype fp8 --preserve-thinking --spec mtp --draft-tokens 1 --lm-head-draft \
    --no-cuda-graph --token-logprobs --request-log-jsonl "$D/$tag.jsonl" \
    > "$D/$tag.log" 2>&1 &
  local p=$!
  local ok=0
  for i in $(seq 1 90); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 $p 2>/dev/null || break
    sleep 4
  done
  [ "$ok" = "1" ] || { echo "NOT HEALTHY ($tag)"; awk 'NR<=10' "$D/$tag.log"; kill -9 $p 2>/dev/null; exit 1; }
  awk '/capacity/' "$D/$tag.log" | awk 'NR<=1'
  python3 "$D/../route_probe.py" "$port" "$D/$tag-probe.json" | tee "$D/$tag-probe.out"
  kill -TERM $p; wait $p 2>/dev/null || true
  echo
}

# R1/R2: logits capture. STATE_DIR set with a frontier that cannot match (1),
# so no state copies happen.
serve c1-logits 1 NINFER_FLASH_NEXT_LOGITS_DIR="$D/c1-logits" \
                   NINFER_FLASH_NEXT_STATE_DIR="$D/c1-state" \
                   NINFER_FLASH_NEXT_STATE_FRONTIER=1
PT=$(python3 -c "import json; print(json.load(open('$D/c1-logits-probe.json'))['prompt_tokens'])")
echo "prompt_tokens=$PT"
serve c4-logits 4 NINFER_FLASH_NEXT_LOGITS_DIR="$D/c4-logits" \
                   NINFER_FLASH_NEXT_STATE_DIR="$D/c4-state" \
                   NINFER_FLASH_NEXT_STATE_FRONTIER=1
# R3/R4: GDN state at frontier = prompt_tokens (end of prefill).
serve c1-state 1 NINFER_FLASH_NEXT_STATE_DIR="$D/c1-state" \
                NINFER_FLASH_NEXT_STATE_FRONTIER="$PT"
serve c4-state 4 NINFER_FLASH_NEXT_STATE_DIR="$D/c4-state" \
                NINFER_FLASH_NEXT_STATE_FRONTIER="$PT"
echo "routediag_done"
