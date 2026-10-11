#!/usr/bin/env bash
# initcheck on the unpatched server (ninfer-gb10-dnf), 2026-10-11.
# Opus: the disassembly pinned the fault to group_offsets[512] (expert id
# exactly 512, the all-NaN-scores sentinel) in route_kernel. initcheck on
# the UNPATCHED binary records which bytes of which allocation were never
# written, so the result applies to the build that crashes. Unpatched
# HEAD binary, --max-concurrency 6 (24 verify rows - the crashing config).
# Stop at "CUDA graphs ready" or at the fault. Budget: up to 1 hour.
set -u
cd /home/apollo11/ninfer-gb10
OUT=profiles/bench/gb10/initcheck-server-2026-10-11
mkdir -p "$OUT"
clog() { echo "[ic-run $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$OUT/run.log"; }

srv=""
on_exit() { if [ -n "$srv" ] && kill -0 "$srv" 2>/dev/null; then kill -TERM "$srv" 2>/dev/null; wait "$srv" 2>/dev/null; fi; }
trap on_exit INT TERM

# Pre-launch checks: sanitizer present, no GPU jobs, no phantom counter,
# free unified memory >= model size + headroom
command -v compute-sanitizer >/dev/null || { clog "compute-sanitizer not found"; exit 1; }
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
if [ -n "$apps" ]; then clog "GPU guard: compute app(s) '$apps' held - aborting"; exit 1; fi
if [ -n "$util" ] && [ "$util" -gt 0 ]; then clog "GPU guard: phantom counter? util ${util}% with no compute app - aborting"; exit 1; fi
free_g=$(free -g | awk '/^Mem:/{print $7}')
clog "start; HEAD $(git rev-parse --short HEAD); GPU idle (util ${util:-?}%, no apps); free memory ${free_g}G available"
(( free_g >= 100 )) || { clog "insufficient free memory (<100G); aborting"; exit 1; }

# Artifact: exact path from the machine-local config (never by glob)
source tools/gb10/config.local.sh
test -n "${ART:-}" && test -e "$ART" || { clog "ART missing or not set: ${ART:-<unset>}"; exit 1; }
clog "artifact: $ART"

# Binary freshness: unpatched HEAD binary (no diagnostic patch applied)
test -z "$(git status --short -- include/ninfer src/ops src/models | grep -v '^??')" \
  || { clog "working tree has source modifications - not the unpatched product build"; exit 1; }
clog "building ninfer-serve (freshness, unpatched)"
cmake --build build -j --target ninfer-serve >"$OUT/build.log" 2>&1 || { clog "BUILD FAILED: see build.log"; exit 1; }
clog "serve binary ready"

# 2026-10-11 14:42Z: Opus's flag form "--track-unused-memory no" is invalid in this
# compute-sanitizer (CUDA 13.0.88): the option is a value-less flag, so "no" is
# consumed as the target name ("Target application doesn't exist"). Fallback: omit
# the flag - unused-memory tracking is off by default, which is the intended
# semantics. Recorded as a fallback per the standing rules.
ICFLAGS=(--tool initcheck --print-limit 50)
# Preflight: confirm compute-sanitizer accepts the target before the heavy run.
# --help exits before CUDA init, so this is GPU-free. The 2026-10-11 14:43Z and
# 14:45Z runs never launched the sanitizer at all (an edit had removed the launch
# line; the "Target application doesn't exist" log seen then was stale from the
# 14:41Z broken-flag run) - this preflight keeps a launch failure out of the
# 1-hour window.
for attempt in 1 2; do
  if compute-sanitizer "${ICFLAGS[@]}" build/apps/ninfer-serve --help 2>&1 | grep -q 'usage:'; then
    clog "preflight ok: sanitizer accepted the target (attempt $attempt)"
    break
  fi
  if [ "$attempt" -eq 2 ]; then
    clog "preflight failed twice: sanitizer rejected build/apps/ninfer-serve; aborting"
    exit 1
  fi
  clog "preflight: transient target rejection; retrying in 10 s"
  sleep 10
done
clog "launching initcheck on the unpatched server (port 8001, max-concurrency 6)"
compute-sanitizer "${ICFLAGS[@]}" build/apps/ninfer-serve "$ART" --port 8001 \
  --max-concurrency 6 --kv-dtype fp8 --spec mtp --draft-tokens 3 --lm-head-draft \
  --prefill-chunk 4096 >"$OUT/initcheck.log" 2>&1 &
srv=$!
ok=0
for i in $(seq 1 1800); do   # up to 60 min (Opus's budget)
  if grep -q 'CUDA graphs ready' "$OUT/initcheck.log" 2>/dev/null; then ok=1; break; fi
  kill -0 "$srv" 2>/dev/null || break
  sleep 2
done

if [ "$ok" -eq 1 ]; then
  sleep 3
  clog "graphs ready under initcheck (unexpected for the unpatched binary); stopping server"
  kill -TERM "$srv" 2>/dev/null
  wait "$srv" 2>/dev/null
  clog "server stopped"
elif kill -0 "$srv" 2>/dev/null; then
  clog "no graph-ready line within 60 min; server still running - stopping it"
  kill -TERM "$srv" 2>/dev/null
  wait "$srv" 2>/dev/null
  clog "server stopped (no graph-ready - partial log captured)"
else
  wait "$srv" 2>/dev/null
  rc=$?
  clog "process exited rc=$rc (expected: the unpatched fault during graph preparation)"
  # Dead-context check: a faulted context can leave an unreaped channel
  # (phantom counter: utilization > 0 with no compute app listed)
  sleep 3
  apps2=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
  util2=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  if [ -z "$apps2" ] && [ -n "$util2" ] && [ "$util2" -gt 0 ]; then
    clog "REBOOT REQUIRED: no compute app but util ${util2}% - dead context held (phantom counter)"
  elif [ -n "$apps2" ]; then
    clog "REBOOT REQUIRED: process(es) '$apps2' still holding the GPU after the fault"
  else
    clog "no dead context: GPU clean (util ${util2:-0}%, no apps)"
  fi
fi
srv=""
trap - INT TERM

clog "summary: $(grep -c 'ERROR SUMMARY' "$OUT/initcheck.log" 2>/dev/null) ERROR SUMMARY line(s); log $(wc -l < "$OUT/initcheck.log" 2>/dev/null) lines"
clog "done"
