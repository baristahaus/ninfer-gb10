#!/usr/bin/env bash
# Deeper-drafts round 2 (ninfer-gb10-h3l), 2026-10-10. Cap raised to 5
# (4b62efce, fb1c44a7). Gate: build HEAD + the real-artifact Engine test, now
# including the K=5 pass; on gate failure stop. Then arms A -> B -> C, strictly
# sequential, one GPU job at a time.
set -uo pipefail
cd /home/apollo11/ninfer-gb10
source tools/gb10/config.local.sh
OUT=profiles/bench/gb10/depth2-2026-10-10
mkdir -p "$OUT"
clog() { echo "[depth2 $(date -u +%Y-%m-%d\ %H:%M:%SZ)] $*" | tee -a "$OUT/campaign.log"; }
gpu_free() { [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]]; }

clog "gate start: build HEAD + ninfer_qwen3_8_flash_next_real_test (K=5 pass)"
gpu_free || { clog "GPU is held by another job; aborting before the gate."; exit 1; }
if [[ ! -f build/CMakeCache.txt ]]; then
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121a \
        -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON -DNINFER_PERFORMANCE_TRACE=OFF \
        >"$OUT/configure.log" 2>&1 || { clog "gate FAILED at configure; stopping."; exit 1; }
fi
cmake --build build -j >"$OUT/build.log" 2>&1
brc=$?
clog "gate: build rc=$brc"
((brc == 0)) || { clog "gate FAILED at build; stopping. See build.log."; exit 1; }
NINFER_QWEN38_FLASH_NEXT_WEIGHTS="$ART" ctest --test-dir build \
    -R 'ninfer_qwen3_8_flash_next_real_test' --output-on-failure >"$OUT/gate-test.log" 2>&1
trc=$?
clog "gate: real test rc=$trc"
((trc == 0)) || { clog "gate FAILED at the real test (K=5 pass); stopping. See gate-test.log."; exit 1; }
clog "gate PASSED; starting arms"

clog "arm A start: k_sweep K=3 4 5 REPEATS=2 NAME=depth2 (natural corpus, 1 request)"
SWEEP="fp8mtp=$ART" KS="3 4 5" REPEATS=2 NAME=depth2 tools/gb10/k_sweep.sh >"$OUT/armA.log" 2>&1
clog "arm A rc=$?"

for k in 3 4 5; do
    gpu_free || { clog "arm B k$k: GPU is held; aborting."; exit 1; }
    clog "arm B start: parity ours-k$k (105 DGPP requests, conc 1/2/4)"
    profiles/bench/gb10/lk-parity-2026-10-10/parity_arm.sh "ours-k$k" build/apps/ninfer-serve "$ART" 8001 \
        profiles/bench/gb10/lk-parity-2026-10-10-r2/results/ours-k$k \
        --spec mtp --draft-tokens "$k" >"$OUT/armB-k$k.log" 2>&1
    clog "arm B k$k rc=$?"
done

gpu_free || { clog "arm C: GPU is held; aborting."; exit 1; }
clog "arm C start: long_context K=3 4 5 SIZES=15000 REPS=2 tasks=script,edit"
KS="3 4 5" SIZES=15000 REPS=2 DGPP=0 DRIVER_ARGS="--tasks script,edit" \
    tools/gb10/long_context.sh >"$OUT/armC.log" 2>&1
clog "arm C rc=$?"
clog "campaign complete"
