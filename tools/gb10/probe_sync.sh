#!/usr/bin/env bash
# Round-boundary probe (plan step 6): builds tools/gb10/sync_probe.cu for sm_121a and runs it
# once per CUDA device schedule: how late a synchronize returns after the GPU finishes, and what
# a graph launch costs per node. Writes profiles/bench/gb10/sync_probe/summary.md. Needs no model.
# PIN_CPUS="0 10" additionally runs the blocking schedule pinned to each listed CPU, to compare
# core types. Run on an otherwise idle machine; takes about a minute.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
dir=profiles/bench/gb10/sync_probe
rm -rf "$dir"
mkdir -p "$dir"

echo "[gb10] building sync probe" >&2
"$NVCC" -O3 -std=c++17 -arch=sm_121a tools/gb10/sync_probe.cu -o "$dir/sync_probe"

{
    echo "## Round-boundary probe — $(date -u +%FT%TZ)"
    echo
    echo "CPU idle states (cpu0: name, exit latency µs):"
    echo
    for state in /sys/devices/system/cpu/cpu0/cpuidle/state*; do
        [[ -d $state ]] && echo "- $(cat "$state/name"): $(cat "$state/latency")"
    done
    echo
    for schedule in blocking yield spin auto; do
        "$dir/sync_probe" --schedule "$schedule" || echo "schedule $schedule failed (rc=$?)"
    done
    for cpu in ${PIN_CPUS:-}; do
        echo "#### Pinned to CPU $cpu"
        echo
        taskset -c "$cpu" "$dir/sync_probe" --schedule blocking || echo "pinned run failed (rc=$?)"
    done
} | tee "$dir/summary.md"
echo "[gb10] summary: $dir/summary.md" >&2
