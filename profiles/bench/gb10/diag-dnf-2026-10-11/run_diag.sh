#!/usr/bin/env bash
# diag-dnf nonfinite-scan run (ninfer-gb10-dnf), 2026-10-11.
# Apply share/diag-dnf/nonfinite-scan.patch (Opus, a122fe65), build the
# instrumented ninfer-serve, run it at --max-concurrency 6 (24 verify rows -
# the crashing config), stop it once the post-graph-preparation record is
# printed, collect every [diag line, then drop the patch and restore the
# clean product binary. One GPU job.
set -u
cd /home/apollo11/ninfer-gb10
OUT=profiles/bench/gb10/diag-dnf-2026-10-11
mkdir -p "$OUT"
clog() { echo "[diag-run $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$OUT/run.log"; }

srv=""
on_exit() { if [ -n "$srv" ] && kill -0 "$srv" 2>/dev/null; then kill -TERM "$srv" 2>/dev/null; wait "$srv" 2>/dev/null; fi; }
trap on_exit INT TERM

# GPU guard: no compute apps, no phantom counter (high util + no process),
# and free unified memory >= model size + headroom
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

clog "applying share/diag-dnf/nonfinite-scan.patch"
git apply --check share/diag-dnf/nonfinite-scan.patch || { clog "patch does not apply cleanly"; exit 1; }
git apply share/diag-dnf/nonfinite-scan.patch

clog "building ninfer-serve (instrumented)"
cmake --build build -j --target ninfer-serve >"$OUT/build.log" 2>&1 || { clog "BUILD FAILED: see build.log"; exit 1; }
clog "build OK"

clog "launching server (port 8001, max-concurrency 6, draft-tokens 3)"
build/apps/ninfer-serve "$ART" --port 8001 --max-concurrency 6 --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 --lm-head-draft --prefill-chunk 4096 >"$OUT/diag.log" 2>&1 &
srv=$!

ok=0
for i in $(seq 1 360); do   # up to 12 min for startup + instrumented graph prep
  if grep -q 'non-finite records' "$OUT/diag.log" 2>/dev/null; then ok=1; break; fi
  kill -0 "$srv" 2>/dev/null || break
  sleep 2
done

if [ "$ok" -eq 1 ]; then
  sleep 2
  clog "diag record complete; stopping server"
  kill -TERM "$srv" 2>/dev/null
  wait "$srv" 2>/dev/null
  clog "server stopped"
elif kill -0 "$srv" 2>/dev/null; then
  clog "no diag record within 12 min; server still running - stopping it"
  kill -TERM "$srv" 2>/dev/null
  wait "$srv" 2>/dev/null
  clog "server stopped (no diag record - partial log captured)"
else
  wait "$srv" 2>/dev/null
  rc=$?
  clog "server exited early rc=$rc; crash markers:"
  grep -m 15 -nE 'CUDA_CHECK|illegal|Aborted|core dumped|Segmentation|Error' "$OUT/diag.log" >> "$OUT/run.log" 2>/dev/null
  clog "REBOOT REQUIRED before the next GPU job (crashed context)"
fi
srv=""
trap - INT TERM

clog "collecting [diag lines"
grep '^\[diag' "$OUT/diag.log" > "$OUT/diag-lines.txt" 2>/dev/null || true
clog "collected $(wc -l < "$OUT/diag-lines.txt" 2>/dev/null || echo 0) [diag lines"

clog "dropping the diag patch (git checkout + scoped clean)"
git checkout -- . && git clean -fd include/ninfer/ops src/ops/launcher
clog "patch dropped; working tree back to $(git rev-parse --short HEAD)"

clog "rebuilding clean ninfer-serve"
cmake --build build -j --target ninfer-serve >"$OUT/build-clean.log" 2>&1 \
  && clog "clean rebuild OK" || clog "CLEAN REBUILD FAILED: see build-clean.log"
clog "done"
