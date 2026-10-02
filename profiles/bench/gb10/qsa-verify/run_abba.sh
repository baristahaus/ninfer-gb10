#!/usr/bin/env bash
# QSA split-decode staging verification (2026-10-02, twoFour), per the handoff run list:
# B arm 370a5247 (stage each KV tile once, in 16-byte pieces) vs A arm a5e4a219 (M3 tree).
#
# Gates (run before this script, one at a time):
#   build/tests/ninfer_flash_next_qsa_test
#   NINFER_QWEN38_FLASH_NEXT_WEIGHTS=$ART build/tests/ninfer_qwen3_8_flash_next_real_test
#
# ABBA decode, per M3 protocol: A1 B1 B2 A2 per config; fresh serve per leg; drop_caches per
# leg; concurrency_sweep --max-tokens 512 --prompt-chars 2000 --ignore-eos. Configs: C4 K=1,
# C1 K=1, C4 K=3. Round attribution is separate (tools/gb10/round_attribution.sh).
#
# One GPU job at a time. Resumable: legs with a DONE marker are skipped.
# Data: profiles/bench/gb10/qsa-verify/.
set -euo pipefail
source tools/gb10/common.sh

ROOT=$OUT_ROOT/qsa-verify
A_SERVE=$ROOT/arms/a5e4a219.ninfer-serve
B_SERVE=$ROOT/arms/370a5247.ninfer-serve
require_binary "$A_SERVE"
require_binary "$B_SERVE"

gpu_idle() {
  if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]]; then
    echo "GPU held by another job; single-instance discipline: aborting." >&2
    exit 1
  fi
}
drop_caches() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; }

run_leg() { # $1 = leg dir name, $2 = arm (base|new), $3 = concurrency, $4 = K
  local name=$1 arm=$2 conc=$3 K=$4
  local dir=$ROOT/$name
  local serve_bin=$B_SERVE
  [[ $arm == base ]] && serve_bin=$A_SERVE
  gpu_idle
  drop_caches
  local args=(--port "$PORT" --max-context 73728 --max-concurrency "$conc" --kv-dtype fp8
              --preserve-thinking --request-log-jsonl "$dir/requests.jsonl")
  if ((K > 0)); then args+=(--spec mtp --draft-tokens "$K" --lm-head-draft); fi
  log "leg $name: $arm arm, C$conc K=$K (fresh serve + drop_caches)"
  "$serve_bin" "$ART" "${args[@]}" >"$dir/serve.log" 2>&1 &
  local spid=$!
  local waited=0
  until curl -sf "$BASE_URL/health" >/dev/null 2>&1; do
    if ! kill -0 "$spid" 2>/dev/null; then
      echo "leg $name: server exited during startup. Last log lines:" >&2
      tail -n 40 "$dir/serve.log" >&2
      exit 1
    fi
    ((waited < 1800)) || { echo "leg $name: not healthy after 30 min" >&2; exit 1; }
    sleep 5
    waited=$((waited + 5))
  done
  "$PYTHON" tools/gb10/concurrency_sweep.py "$BASE_URL" "$dir/load.json" \
      --n "$conc" --max-tokens 512 --prompt-chars 2000 --ignore-eos >"$dir/load.log" 2>&1 \
    || log "leg $name: sweep reported errors; see $dir/load.log"
  "$PYTHON" tools/gb10/request_log_summary.py "$dir/requests.jsonl" >"$dir/summary.txt" 2>&1 \
    || log "leg $name: request-log summary failed"
  # Teardown: SIGTERM (drain), then verify the GPU is actually released (M3 teardown lesson:
  # an orphaned server holding the pool must abort the campaign, not be papered over).
  kill "$spid" 2>/dev/null || true
  for _ in $(seq 1 120); do
    if ! kill -0 "$spid" 2>/dev/null && \
       [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]]; then
      break
    fi
    sleep 2
  done
  if kill -0 "$spid" 2>/dev/null; then
    echo "leg $name: server survived 4 min after SIGTERM; sending SIGKILL" >&2
    kill -9 "$spid" 2>/dev/null || true
    wait "$spid" 2>/dev/null || true
  fi
  if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]]; then
    echo "leg $name: GPU still held after teardown; aborting before the next leg." >&2
    exit 1
  fi
  touch "$dir/DONE"
  log "leg $name done: $(awk '/^Totals/ {print}' "$dir/summary.txt")"
}

for cfg in "c4-1:4:1" "c1-1:1:1" "c4-3:4:3"; do
  IFS=: read -r tag conc K <<<"$cfg"
  for leg in 1 2 3 4; do
    arm=new
    if ((leg == 1 || leg == 4)); then arm=base; fi   # A1 B1 B2 A2
    name=abba-${tag}-${leg}-${arm}
    if [[ -f $ROOT/$name/DONE ]]; then
      log "leg $name: done (skip)"
      continue
    fi
    mkdir -p "$ROOT/$name"
    run_leg "$name" "$arm" "$conc" "$K"
  done
done

# ---- ABBA pair averages -----------------------------------------------------------
field() { # $1 = leg dir, $2 = sed program run against the Totals line
  awk '/^Totals/ {print}' "$ROOT/$1/summary.txt" | sed -n "$2"
}
tps()      { field "$1" 's/.*decode [0-9.]* s (\([0-9.]*\) tok\/s).*/\1/p'; }
tokround() { field "$1" 's/.*rounds [0-9]*, \([0-9.]*\) tok\/round.*/\1/p'; }
devwait()  { field "$1" 's/.*device wait \([0-9.]*\) ms\/round.*/\1/p'; }
queue()    { field "$1" 's/.*queue wait mean \([0-9]*\) ms.*/\1 ms/p'; }

{
  echo "# QSA split-decode staging verification: 370a5247 vs a5e4a219, GB10, 2026-10-02, twoFour"
  echo
  echo "B arm (new): 370a5247, sha256 $(sha256sum "$B_SERVE" | awk '{print $1}')"
  echo "A arm (base): a5e4a219, sha256 $(sha256sum "$A_SERVE" | awk '{print $1}')"
  echo
  machine_summary
  echo "- Protocol: ABBA (A1 B1 B2 A2) per config; fresh serve + drop_caches per leg;"
  echo "  concurrency_sweep --max-tokens 512 --prompt-chars 2000 --ignore-eos (M3 protocol)."
  echo
  for cfg in "c4-1:4:1" "c1-1:1:1" "c4-3:4:3"; do
    IFS=: read -r tag conc K <<<"$cfg"
    echo "## C$conc K=$K"
    echo
    echo "| Leg | Arm | Decode tok/s | Tok/round | Device wait ms/round | Queue wait mean |"
    echo "|---|---|---:|---:|---:|---:|"
    for leg in 1 2 3 4; do
      arm=new
      if ((leg == 1 || leg == 4)); then arm=base; fi
      d=abba-${tag}-${leg}-${arm}
      echo "| $leg | $arm | $(tps "$d") | $(tokround "$d") | $(devwait "$d") | $(queue "$d") |"
    done
    a1=$(tps "abba-${tag}-1-base"); b1=$(tps "abba-${tag}-2-new")
    b2=$(tps "abba-${tag}-3-new"); a2=$(tps "abba-${tag}-4-base")
    aavg=$(awk -v x="$a1" -v y="$a2" 'BEGIN {printf "%.2f", (x + y) / 2}')
    bavg=$(awk -v x="$b1" -v y="$b2" 'BEGIN {printf "%.2f", (x + y) / 2}')
    gain=$(awk -v a="$aavg" -v b="$bavg" 'BEGIN {printf "%+.1f%%", (b - a) / a * 100}')
    echo
    echo "Pair averages: A (base) $aavg tok/s (legs $a1, $a2) vs B (new) $bavg tok/s (legs $b1, $b2) → $gain."
    echo
  done
} >"$ROOT/abba-summary.md"
log "ABBA summary: $ROOT/abba-summary.md"
cat "$ROOT/abba-summary.md"
