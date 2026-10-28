#!/bin/bash
# Fixed MTP acceptance battery for A/B artifact comparison.
# Usage: mtp_acceptance_battery.sh <artifact> <outdir>
# Runs a fixed prompt set greedily and extracts the per-run MTP acceptance lines.
set -u
ART="$1"
OUT="$2"
mkdir -p "$OUT"
i=0
while IFS= read -r prompt; do
    i=$((i+1))
    build/apps/ninfer "$ART" --spec mtp --draft-tokens 2 \
        --max-new 128 --prompt "$prompt" --greedy --no-thinking --log-level warning \
        > "$OUT/run$i.out" 2> "$OUT/run$i.err"
done << 'PROMPTS'
Summarize the plot of Hamlet in three sentences.
Explain how a transformer attention layer works, step by step.
Write a Python function that merges two sorted lists without duplicates.
Describe the water cycle for a ten-year-old.
What are the main causes of the 1930s Great Depression?
Give a short recipe for lentil soup with exact quantities.
Translate "the weather is nice today" into French and German.
List five practical uses of a standard deck of playing cards.
PROMPTS
grep -h 'mtp acceptance' "$OUT"/*.err > "$OUT/acceptance.txt"
cat "$OUT/acceptance.txt"
