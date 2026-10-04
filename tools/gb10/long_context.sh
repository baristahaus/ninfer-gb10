#!/usr/bin/env bash
# Long-prompt operations workload: 15K / 30K / 60K-token synthetic incident bundles (ops_corpus.py),
# two tasks (scripts; root-cause triage), a prefix-reusing follow-up turn, and one interference
# probe (a 60K prompt arriving while a 15K request decodes). Measures TTFT, prefill and decode
# rates per size, follow-up TTFT, and the decode stall a new long prompt causes.
#
#   tools/gb10/long_context.sh                       # NInfer K=1 and K=3, then DGPP if configured
#   KS="3" SIZES=15000,60000 REPS=1 tools/gb10/long_context.sh
#
# NInfer runs with two lanes (the interference probe needs both) and a request log, so the
# summaries carry server-side prefill seconds, recomputed prompt tokens on follow-ups and MTP
# tokens per round. DGPP runs when DGPP_DIR, DGPP_START and DGPP_STOP are set, as for block_i.sh
# I9 (its config decides its slot count). One GPU job at a time; page cache dropped before every
# server start. Writes profiles/bench/gb10/long-context/summary.md.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
KS=${KS:-1 3}
SIZES=${SIZES:-15000,30000,60000}
REPS=${REPS:-2}
MAX_TOKENS=${MAX_TOKENS:-1536}
dir=$(step_dir long-context)

drop_caches() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; }
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]]; then
    echo "GPU busy; one device-allocating job at a time." >&2
    exit 2
fi
driver=("$PYTHON" tools/gb10/long_context.py)
driver_args=(--tokens "$SIZES" --reps "$REPS" --max-tokens "$MAX_TOKENS")

for k in $KS; do
    SERVE_ARGS=(--max-context 73728 --max-concurrency 2 --kv-dtype "$KV_DTYPE" --preserve-thinking
                --spec mtp --draft-tokens "$k" --lm-head-draft)
    drop_caches
    start_server "$dir/ninfer-k$k.log" --request-log-jsonl "$dir/ninfer-k$k.jsonl"
    log "NInfer K=$k: long-context workload"
    "${driver[@]}" "$BASE_URL" "$dir/ninfer-k$k.json" "${driver_args[@]}" >"$dir/ninfer-k$k.txt" 2>&1 ||
        log "NInfer K=$k driver exited nonzero; see $dir/ninfer-k$k.txt"
    stop_server
    "$PYTHON" tools/gb10/request_log_summary.py "$dir/ninfer-k$k.jsonl" >"$dir/ninfer-k$k.md"
done

if [[ -n ${DGPP_DIR:-} && -n ${DGPP_START:-} && -n ${DGPP_STOP:-} ]]; then
    drop_caches
    git -C "$DGPP_DIR" log -1 --format='DGPP commit %h %cd' >"$dir/dgpp-version.txt"
    ( cd "$DGPP_DIR" && export DGPP_RESIDENT_CACHE=off && eval "$DGPP_START" ) >"$dir/dgpp-start.log" 2>&1
    dgpp_url="http://127.0.0.1:${DGPP_PORT:-8000}"
    waited=0
    until curl -sf "$dgpp_url/v1/models" >/dev/null 2>&1; do
        if ((waited >= 1800)); then log "DGPP not ready after 30 minutes"; break; fi
        sleep 5
        waited=$((waited + 5))
    done
    log "DGPP: long-context workload"
    "${driver[@]}" "$dgpp_url" "$dir/dgpp.json" "${driver_args[@]}" >"$dir/dgpp.txt" 2>&1 ||
        log "DGPP driver exited nonzero; see $dir/dgpp.txt"
    ( cd "$DGPP_DIR" && eval "$DGPP_STOP" ) >>"$dir/dgpp-start.log" 2>&1 || true
else
    echo "skipped: DGPP_DIR, DGPP_START and DGPP_STOP not all set" >"$dir/dgpp-skipped.txt"
fi

{
    echo "# Long-context operations workload"
    echo
    machine_summary
    echo
    echo "Sizes $SIZES tokens, $REPS reps, max_tokens $MAX_TOKENS, greedy, thinking off."
    for f in "$dir"/ninfer-k*.txt "$dir"/dgpp.txt; do
        [[ -f $f ]] || continue
        echo
        echo "## $(basename "$f" .txt)"
        echo
        sed -n '/^| tokens/,$p' "$f"
        grep -h '"interference"' "$f" | sed 's/^/interference: /'
    done
} >"$dir/summary.md"
log "summary: $dir/summary.md"
