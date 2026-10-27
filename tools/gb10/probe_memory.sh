#!/usr/bin/env bash
# Unified-memory probe: builds tools/gb10/memory_probe.cu for sm_121a and runs it. Writes
# profiles/bench/gb10/probe/summary.md, which report.sh includes.
# Needs no model. When ART is set (in the environment or tools/gb10/config.local.sh) and names
# the Flash-Next artifact, it also copies 64 MiB of each decode-path weight class out of it and
# repeats the compression comparison on those bytes (section 2b); the copies are deleted after.
# Run on an otherwise idle machine (stop any server first); it takes about 10 seconds, plus a few
# more for the weight samples. Extra arguments go to the probe, e.g. --gib 2 --seconds 1.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if [[ -z ${ART:-} && -f tools/gb10/config.local.sh ]]; then
    # shellcheck source=config.example.sh
    source tools/gb10/config.local.sh
fi
PYTHON=${PYTHON:-python3}
NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
dir=profiles/bench/gb10/probe
rm -rf "$dir"
mkdir -p "$dir"

samples=()
trap 'rm -rf "$dir/samples"' EXIT
if [[ -n ${ART:-} && -f $ART ]]; then
    echo "[gb10] copying weight samples from $ART" >&2
    while IFS= read -r line; do samples+=(--sample "$line"); done < <(
        "$PYTHON" tools/gb10/weight_samples.py "$ART" "$dir/samples")
else
    echo "[gb10] ART not set; skipping the weight-sample compression check" >&2
fi

echo "[gb10] building memory probe" >&2
"$NVCC" -O3 -std=c++17 -arch=sm_121a tools/gb10/memory_probe.cu -o "$dir/memory_probe"
echo "[gb10] running memory probe" >&2
"$dir/memory_probe" "${samples[@]}" "$@" | tee "$dir/summary.md"
echo "[gb10] summary: $dir/summary.md" >&2
