#!/usr/bin/env bash
# File-backed page GPU-read probe (plan step 6.3 gate): builds tools/gb10/file_page_probe.cu
# for sm_121a and runs the phases in order, each as its own invocation so a hung cold-fault
# phase times out without taking the rest. Writes
# profiles/bench/gb10/file_page_probe/summary.md. Needs no model.
#
# The evictor defaults to the largest regular file under $MODELS (default: the directory of
# ART from config.local.sh, i.e. the Flash-Next artifact volumes) so the 512 MiB scratch
# mapping is pushed out of the page cache before the cold phases. NOTE: the evicting read (~125 GiB) recycles the whole page cache — the PLE table's
# residency is gone afterwards; the next campaign's warm run re-establishes it.
# Run on an otherwise idle machine (stop any server first). Takes ~2-4 minutes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
if [[ -z ${MODELS:-} && -z ${ART:-} && -f tools/gb10/config.local.sh ]]; then
    # shellcheck source=config.example.sh
    source tools/gb10/config.local.sh
fi
MODELS=${MODELS:-$(dirname "${ART:-/nonexistent/x}")}
dir=profiles/bench/gb10/file_page_probe
rm -rf "$dir"
mkdir -p "$dir"

evictor=$(find "$MODELS" -maxdepth 1 -type f -printf '%s %p\n' 2>/dev/null | sort -rn | awk 'NR==1{print $2}')
if [[ -z $evictor ]]; then
    echo "[gb10] no evictor file under $MODELS (set MODELS)" >&2
    exit 1
fi
scratch=$dir/scratch.bin

echo "[gb10] building file-page probe" >&2
"$NVCC" -O3 -std=c++17 -arch=sm_121a tools/gb10/file_page_probe.cu -o "$dir/file_page_probe"
echo "[gb10] evictor: $evictor" >&2

{
    echo "# File-backed page GPU-read probe — $(date -u +%FT%TZ)"
    echo
    echo "scratch 512 MiB (page-cache-backed file); evictor $evictor (125 GiB read); row = 2560 B"
    echo "at a 4 KiB-aligned offset; row offsets deterministic (splitmix64, fixed seed)."
    echo "Decision gate: device-side PLE gather (plan step 6.3) — what does a GPU read of a"
    echo "file-backed page cost when resident vs. cold (NVMe page-in via the fault path)?"
    echo
    for phase in prep warm res-seq res-rows evict cold-serial cold-rows res-rows-again; do
        echo "## phase: $phase"
        case $phase in
            evict)
                timeout 900 "$dir/file_page_probe" --phase evict --file "$scratch" \
                    --evict "$evictor" --evict-bytes 125G || echo "TIMEOUT or FAIL (rc=$?)"
                ;;
            *)
                timeout 600 "$dir/file_page_probe" --phase "$phase" --file "$scratch" \
                    || echo "TIMEOUT or FAIL (rc=$?)"
                ;;
        esac
        echo
    done
    echo "## cleanup"
    rm -f "$scratch"
    echo "scratch removed"
} | tee "$dir/summary.md"

echo "[gb10] summary: $dir/summary.md" >&2
