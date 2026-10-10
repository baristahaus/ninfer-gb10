#!/usr/bin/env bash
# Deeper-drafts campaign (ninfer-gb10-4sh), 2026-10-10. Arms A -> B -> C, strictly
# sequential, one GPU job at a time. Per-arm logs and the campaign log land here.
# Resumable: k_sweep skips runs whose acceptance.json exists; parity arms rerun
# idempotently; long_context is one ~40 min unit.
set -uo pipefail
cd /home/apollo11/ninfer-gb10
source tools/gb10/config.local.sh
OUT=profiles/bench/gb10/depth-2026-10-10
mkdir -p "$OUT"
clog() { echo "[depth $(date -u +%Y-%m-%d\ %H:%M:%SZ)] $*" | tee -a "$OUT/campaign.log"; }

clog "arm A start: k_sweep K=3 4 5 REPEATS=2 (natural corpus, 1 request)"
SWEEP="fp8mtp=$ART" KS="3 4 5" REPEATS=2 NAME=depth tools/gb10/k_sweep.sh >>"$OUT/armA.log" 2>&1
rcA=$?
clog "arm A rc=$rcA"

for k in 3 4 5; do
    clog "arm B start: parity ours-k$k (105 DGPP requests, conc 1/2/4)"
    profiles/bench/gb10/lk-parity-2026-10-10/parity_arm.sh "ours-k$k" build/apps/ninfer-serve "$ART" 8001 \
        profiles/bench/gb10/lk-parity-2026-10-10/results/ours-k$k \
        --spec mtp --draft-tokens "$k" >>"$OUT/armB-k$k.log" 2>&1
    clog "arm B k$k rc=$?"
done

clog "arm C start: long_context K=3 5 SIZES=15000 REPS=2 tasks=script,edit"
KS="3 5" SIZES=15000 REPS=2 DGPP=0 DRIVER_ARGS="--tasks script,edit" \
    tools/gb10/long_context.sh >>"$OUT/armC.log" 2>&1
clog "arm C rc=$?"
clog "campaign complete"
