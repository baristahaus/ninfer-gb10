#!/usr/bin/env bash
# MTP capture crash probe (ninfer-gb10-1du), 2026-10-11. The K=5 x 4-request
# server core-dumped during CUDA graph capture (ninfer-gb10-dnf). Question: is
# the crash row-count-dependent (K=3 at 6/8 requests verifies 24/32 rows -
# supported settings) or K-specific (6 tokens per request, unreachable since
# d10d21fe)? Cap is back to 3 (d10d21fe). One GPU job at a time; the chain
# aborts on any crash (a crashed context needs a node reboot before the next
# GPU job).
set -uo pipefail
cd /home/apollo11/ninfer-gb10
source tools/gb10/config.local.sh
OUT=profiles/bench/gb10/crash-probe-2026-10-11
mkdir -p "$OUT"
clog() { echo "[probe $(date -u +%Y-%m-%d\ %H:%M:%SZ)] $*" | tee -a "$OUT/probe.log"; }
gpu_free() { [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]]; }
PORT=8001
MODEL=qwen3.8-flash-next-125b-a6b

free_g=$(free -g | awk '/^Mem:/{print $7}')
clog "start; free memory ${free_g}G available"
((free_g >= 100)) || { clog "insufficient free memory (<100G); aborting."; exit 1; }
clog "building (incremental)"
cmake --build build -j >"$OUT/build.log" 2>&1
brc=$?
clog "build rc=$brc"
((brc == 0)) || { clog "build FAILED; stopping. See build.log."; exit 1; }

probe_one() { # $1=conc
  local conc=$1
  local log="$OUT/serve-conc${conc}-k3.log"
  gpu_free || { clog "conc $conc: GPU is held; aborting."; exit 1; }
  clog "conc $conc: starting server (K=3)"
  build/apps/ninfer-serve "$ART" --host 127.0.0.1 --port "$PORT" \
    --max-context 73728 --max-concurrency "$conc" --kv-dtype fp8 \
    --preserve-thinking --lm-head-draft --host-kv-mib 0 --prefill-chunk 4096 \
    --spec mtp --draft-tokens 3 >"$log" 2>&1 &
  local srv=$!
  local ok=0
  for _ in $(seq 1 240); do
    kill -0 "$srv" 2>/dev/null || break
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    sleep 2
  done
  if [[ $ok -eq 1 ]]; then
    if grep -q 'CUDA graphs ready' "$log"; then
      clog "conc $conc: CUDA graphs ready"
    else
      clog "conc $conc: WARNING - no 'CUDA graphs ready' line in $log"
    fi
    local resp rc_curl=0
    resp=$(curl -sf -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word.\"}],\"max_tokens\":16,\"temperature\":0,\"stream\":false}") || rc_curl=$?
    if ((rc_curl == 0)); then
      clog "conc $conc: short chat OK: $(printf '%s' "$resp" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(repr(d["choices"][0]["message"]["content"][:100]))' 2>&1)"
    else
      clog "conc $conc: short chat FAILED (rc=$rc_curl): $(printf '%s' "$resp" | grep -m1 -v '^$')"
    fi
    kill -TERM "$srv" 2>/dev/null || true
    wait "$srv" 2>/dev/null || true
    clog "conc $conc: PASS (started, graphs ready, chat served, stopped)"
    return 0
  fi
  local rc=0
  wait "$srv" || rc=$?
  local marker
  marker=$(grep -m1 -E 'CUDA_CHECK|illegal memory access|Aborted|core dumped|Segmentation' "$log" || true)
  clog "conc $conc: CRASH (exit rc=$rc); marker: ${marker:-<none>}; log: $log"
  return 1
}

crashed=
if probe_one 6; then
  probe_one 8 || crashed=8
else
  crashed=6
fi

if [[ -z $crashed ]]; then
  clog "BOTH STARTED - the crash needs 6 tokens per request; row count alone is not the cause"
  exit 0
fi

command -v compute-sanitizer >/dev/null || { clog "compute-sanitizer not found; cannot run memcheck. REBOOT REQUIRED."; exit 2; }
clog "conc $crashed crashed - compute-sanitizer memcheck on the smallest crashing config"
compute-sanitizer --tool memcheck --print-limit 20 build/apps/ninfer-serve "$ART" \
  --host 127.0.0.1 --port "$PORT" --max-context 73728 --max-concurrency "$crashed" \
  --kv-dtype fp8 --preserve-thinking --lm-head-draft --host-kv-mib 0 --prefill-chunk 4096 \
  --spec mtp --draft-tokens 3 >"$OUT/memcheck-conc${crashed}.log" 2>&1
smrc=$?
python3 - "$OUT/memcheck-conc${crashed}.log" "$OUT/memcheck-first-report.txt" <<'PY'
import re, sys
lines = open(sys.argv[1]).read().splitlines()
out, in_rep = [], False
for ln in lines:
    if in_rep:
        if ln.startswith('========='):
            break
        out.append(ln)
    elif re.search(r'Invalid|Unspecified system|out of (shared|global|local) memory', ln):
        in_rep = True
        out.append(ln)
open(sys.argv[2], 'w').write(('\n'.join(out) + '\n') if out else 'no memcheck error report found\n')
PY
clog "memcheck rc=$smrc; first report: $OUT/memcheck-first-report.txt"
clog "REBOOT REQUIRED before the next GPU job (crashed context left unreaped)"
exit 1
