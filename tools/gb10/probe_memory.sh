#!/usr/bin/env bash
# Unified-memory probe: builds tools/gb10/memory_probe.cu for sm_121a and runs it. Needs no model
# or config. Writes profiles/bench/gb10/probe/summary.md, which report.sh includes.
# Run on an otherwise idle machine (stop any server first). Extra arguments go to the probe,
# e.g. --gib 2 --seconds 1.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
dir=profiles/bench/gb10/probe
rm -rf "$dir"
mkdir -p "$dir"

echo "[gb10] building memory probe" >&2
"$NVCC" -O3 -std=c++17 -arch=sm_121a tools/gb10/memory_probe.cu -o "$dir/memory_probe"
echo "[gb10] running memory probe (about a minute)" >&2
"$dir/memory_probe" "$@" | tee "$dir/summary.md"
echo "[gb10] summary: $dir/summary.md" >&2
