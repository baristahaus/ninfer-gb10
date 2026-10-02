#!/usr/bin/env bash
# Flash-Next MoE microbench (no model load): is the 8-row NVFP4 routed path at the memory
# roofline for the experts a round reads? Runs ninfer_flash_next_moe_bench (a few GB of synthetic
# expert banks) for a timing sweep over distinct experts per call, then, with NCU=1, profiles one
# eager pass with Nsight Compute.
#
#   tools/gb10/moe_microbench.sh            # timing sweep only
#   NCU=1 tools/gb10/moe_microbench.sh      # plus ncu --set full on the routed kernels
#
# Memory guard. ncu replay saves and restores device memory per kernel pass. On GB10 the GPU
# shares the 121 GB pool with the host, and an ncu run against a model-size process exhausted it
# and rebooted the node (2026-10-01). This script refuses to start unless no other process holds
# the GPU and MemFree exceeds the bench's footprint plus 30 GiB of headroom. MemFree, not
# MemAvailable: the 2026-10-01 incident happened with reclaimable page cache still resident. Never point ncu
# at ninfer-serve or ninfer_bench with the real artifact.
#
# Writes profiles/bench/gb10/moe-microbench/summary.md.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
LAYERS=${LAYERS:-4}
TOKENS=${TOKENS:-8}
DISTINCT=${DISTINCT:-10,20,40,64,80}
NCU=${NCU:-0}
out=profiles/bench/gb10/moe-microbench
mkdir -p "$out"
bin=build/bench/ninfer_flash_next_moe_bench

if [[ ! -x $bin ]]; then
    cmake -S . -B build -DNINFER_BUILD_BENCHMARKS=ON >/dev/null
    cmake --build build -j --target ninfer_flash_next_moe_bench
fi

guard() { # $1 = layers the run allocates
    if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]]; then
        echo "GPU busy: another process holds device memory. Stop it first." >&2
        exit 2
    fi
    # 1.42 GB per layer bank plus ~1 GiB of workspace, CUDA context and shared expert.
    local need_kib=$(( ($1 * 1420 + 1024) * 1024 + 30 * 1024 * 1024 ))
    local free_kib
    free_kib=$(awk '/^MemFree:/ {print $2}' /proc/meminfo)
    if (( free_kib < need_kib )); then
        echo "MemFree $((free_kib / 1048576)) GiB < required $((need_kib / 1048576)) GiB" \
             "(footprint + 30 GiB). Drop the page cache (sync; echo 3 > /proc/sys/vm/drop_caches)" \
             "or wait for memory to settle." >&2
        exit 2
    fi
}

guard "$LAYERS"
"$bin" --tokens "$TOKENS" --layers "$LAYERS" --distinct "$DISTINCT" | tee "$out/timing.txt"

if [[ $NCU == 1 ]]; then
    # Two layers keep replay save/restore small; one eager pass, routed kernels only.
    guard 2
    ncu --profile-from-start off --set full --kernel-name regex:'nvfp4_w4a4|quantize_decode_routes|reduce_decode_routes' \
        --launch-count 12 --force-overwrite --export "$out/moe-ncu" \
        "$bin" --tokens "$TOKENS" --layers 2 --distinct 80 --warmup 1 --repeat 1 --profile \
        >"$out/ncu.log" 2>&1 || { tail -n 30 "$out/ncu.log" >&2; exit 1; }
    ncu --import "$out/moe-ncu.ncu-rep" --page details --print-units base \
        --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy \
        >"$out/ncu-details.txt" 2>&1 || true
fi

{
    echo "## Flash-Next MoE microbench"
    echo
    echo "- Commit: $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (with local changes)')"
    echo "- GPU/driver: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
    echo "- T=$TOKENS rows, $LAYERS layer banks, distinct experts $DISTINCT"
    echo
    echo '```'
    cat "$out/timing.txt"
    echo '```'
    if [[ -f $out/ncu-details.txt ]]; then
        echo
        echo "ncu (one eager pass, 2 layers, 80 distinct experts): see ncu-details.txt and moe-ncu.ncu-rep."
    fi
} >"$out/summary.md"
cat "$out/summary.md"
