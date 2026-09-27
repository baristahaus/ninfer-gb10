#!/usr/bin/env bash
# Join every finished step summary into profiles/bench/gb10/report.md, the one file to share.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
report="$OUT_ROOT/report.md"
mkdir -p "$OUT_ROOT"
{
    echo "# GB10 report — $(date -u +%Y-%m-%dT%H:%MZ)"
    found=0
    for step in step0 step1 step2 step3; do
        if [[ -f $OUT_ROOT/$step/summary.md ]]; then
            echo
            cat "$OUT_ROOT/$step/summary.md"
            found=1
        fi
    done
    ((found)) || echo "No step summaries found under $OUT_ROOT."
} >"$report"
log "wrote $report ($(wc -c <"$report") bytes); paste its contents back"
