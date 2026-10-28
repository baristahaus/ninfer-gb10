#!/usr/bin/env bash
# Tune --proposal-head-rows for a workload: run one bench pass with the full
# optimized proposal head and MTP round capture, then rank candidate truncation
# sizes by their measured acceptance loss (accepted drafts outside the prefix)
# and wasted verify bandwidth (rejected drafts outside the prefix).
#
# Usage: tune_proposal_rows.sh ARTIFACT [CORPUS] [OUTDIR]
# The recommendation is startup-fixed: CUDA graphs bake the head row count, so
# the engine picks N at load time from this analysis, not per request.
set -euo pipefail

ART="${1:?usage: tune_proposal_rows.sh ARTIFACT [CORPUS] [OUTDIR]}"
CORPUS="${2:-bench/fixtures/bench_corpus.ids}"
OUT="${3:-/tmp/proposal_rows_tune}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

mkdir -p "$OUT"
CAP="$OUT/full_capture.bin"
"$REPO/build/bench/ninfer_bench" \
  --weights "$ART" --corpus "$CORPUS" \
  --kv-dtype fp8 --spec mtp --draft-tokens 2 \
  --sampling greedy --repetitions 3 --concurrency 2 --max-ctx 2048 \
  --mtp-capture "$CAP" \
  --output json --output-file "$OUT/full.json" \
  > "$OUT/bench.log" 2>&1

printf '%9s %14s %14s\n' rows accepted_out rejected_out
for rows in 6144 12288 16384 24576 32768 47104 65536 98304 131072; do
  line=$(python3 "$REPO/tools/mtp_capture_summary.py" "$CAP" --artifact "$ART" --rows "$rows" |
    awk '/accepted drafts outside prefix/ {a=$5}
         /rejected drafts outside prefix/ {r=$5; sub("/.*","",r)}
         END {printf "%s %s", a+0, r+0}')
  printf '%9s %14s %14s\n' "$rows" $line
done
echo "Pick the smallest rows whose accepted_outside loss is at noise level"
echo "(compare against the accepted draft total in $OUT/full.json)."
