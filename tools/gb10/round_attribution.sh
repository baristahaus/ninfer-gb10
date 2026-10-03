#!/usr/bin/env bash
# Block L item 1: where a steady-state MTP decode round's GPU time goes at a fixed concurrency.
# Builds an annotated ninfer-serve in build-trace/ (NINFER_PERFORMANCE_TRACE=ON), runs it under
# Nsight Systems from launch (graph construction must be in the trace), drives CONC simultaneous
# greedy requests twice (ignore_eos, so every request decodes MAX_TOKENS), and attributes the
# GPU work launched inside decode.mtp_round ranges with exactly CONC active rows, per stage and
# per kernel. Writes profiles/bench/gb10/round-attribution/summary.md.
#
#   CONC=4 DRAFT=1 MAX_TOKENS=256 tools/gb10/round_attribution.sh
#   ROUTE_STATS=1 ...   # also tally distinct routed experts per MoE call (moe_route_stats.txt)
#
# Timings carry profiler overhead: use them for shares, not for speed claims.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
CONC=${CONC:-4}
DRAFT=${DRAFT:-1}
MAX_TOKENS=${MAX_TOKENS:-256}
dir=$(step_dir round-attribution)
if [[ ${ROUTE_STATS:-0} == 1 ]]; then export NINFER_MOE_ROUTE_STATS="$(realpath "$dir")/moe_route_stats.txt"; fi
command -v nsys >/dev/null || { echo "nsys not found" >&2; exit 2; }
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]]; then
    echo "GPU busy; one device-allocating job at a time." >&2
    exit 2
fi

generator=()
if [[ ! -f build-trace/CMakeCache.txt ]] && command -v ninja >/dev/null; then generator=(-G Ninja); fi
log "building annotated ninfer-serve in build-trace/ (log: $dir/build_trace.log)"
cmake -S . -B build-trace "${generator[@]}" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=121a -DNINFER_PERFORMANCE_TRACE=ON >"$dir/build_trace.log" 2>&1
cmake --build build-trace -j --target ninfer-serve >>"$dir/build_trace.log" 2>&1

serve_args=(--port "$PORT" --max-context 73728 --max-concurrency "$CONC" --kv-dtype "$KV_DTYPE"
            --preserve-thinking --spec mtp --draft-tokens "$DRAFT" --lm-head-draft
            --request-log-jsonl "$dir/requests.jsonl")
log "nsys capture: C$CONC MTP draft $DRAFT, two batches of $CONC x $MAX_TOKENS tokens"
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
    --export=sqlite --force-overwrite=true --output="$dir/trace" \
    build-trace/apps/ninfer-serve "$ART" "${serve_args[@]}" >"$dir/serve.log" 2>&1 &
NSYS_PID=$!
waited=0
until curl -sf "$BASE_URL/health" >/dev/null 2>&1; do
    kill -0 "$NSYS_PID" 2>/dev/null || { tail -n 40 "$dir/serve.log" >&2; exit 1; }
    ((waited < 1800)) || { echo "server not healthy after 30 min" >&2; exit 1; }
    sleep 5; waited=$((waited + 5))
done
"$PYTHON" tools/gb10/concurrency_sweep.py "$BASE_URL" "$dir/load.json" \
    --n "$CONC,$CONC" --max-tokens "$MAX_TOKENS" --prompt-chars 2000 --ignore-eos \
    >"$dir/load.log" 2>&1 ||
    log "load driver reported errors; see $dir/load.log"
# SIGTERM the server (not nsys) so it drains; nsys then finalizes the trace and exports SQLite.
# nsys re-execs the target with an absolute path, so match the process name, not the cmdline prefix.
pkill -TERM -x ninfer-serve || true
finalize_waited=0
while kill -0 "$NSYS_PID" 2>/dev/null; do
    ((finalize_waited < 900)) || { echo "nsys did not finalize within 15 min; killing it" >&2; kill -TERM "$NSYS_PID" 2>/dev/null; exit 1; }
    sleep 5; finalize_waited=$((finalize_waited + 5))
done
[[ -f $dir/trace.sqlite ]] || { echo "no SQLite export; see $dir/serve.log" >&2; exit 1; }

"$PYTHON" tools/bench/flash_next_performance.py "$dir/trace.sqlite" \
    --hardware tools/bench/hardware/gb10.json --serve-rounds decode.mtp_round \
    --batch "$CONC" --trim 0 --output "$dir/report.json" >"$dir/report.log" 2>&1 ||
    { cat "$dir/report.log" >&2; exit 1; }
{
    echo "## Round attribution: C$CONC, MTP draft $DRAFT"
    echo
    machine_summary
    echo "- nsys: $(nsys --version 2>/dev/null | head -1)"
    echo "- Load: two batches of $CONC simultaneous greedy requests, $MAX_TOKENS output tokens each (ignore_eos)"
    echo "- Request log totals: $("$PYTHON" tools/gb10/request_log_summary.py "$dir/requests.jsonl" 2>&1 | awk '/^Totals/')"
    echo
    sed '1d' "$dir/report.md"
} >"$dir/summary.md"
log "summary: $dir/summary.md"
cat "$dir/summary.md"
