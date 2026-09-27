#!/usr/bin/env bash
# Step 3: where decode time goes. Builds a separate annotated ninfer_bench in build-trace/,
# captures Nsight Systems traces for MTP off and MTP with DRAFT_TOKENS, attributes GPU work per
# stage against tools/bench/hardware/gb10.json, and checks PLE page-cache behaviour with the
# unannotated build. Writes profiles/bench/gb10/step3/summary.md.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
dir=$(step_dir step3)
require_binary "$BENCH_BIN"
require_untraced_build
if ! command -v nsys >/dev/null; then
    echo "nsys not found; install Nsight Systems or add it to PATH." >&2
    exit 2
fi

generator=()
if [[ ! -f build-trace/CMakeCache.txt ]] && command -v ninja >/dev/null; then generator=(-G Ninja); fi
log "building annotated ninfer_bench in build-trace/ (log: $dir/build_trace.log)"
cmake -S . -B build-trace "${generator[@]}" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=121a -DNINFER_BUILD_BENCHMARKS=ON -DNINFER_PERFORMANCE_TRACE=ON \
    >"$dir/build_trace.log" 2>&1
cmake --build build-trace -j --target ninfer_bench >>"$dir/build_trace.log" 2>&1

capture() { # $1 = name, remaining = speculative flags
    local name=$1
    shift
    log "nsys capture $name (8K prompt, 32 decode outputs)"
    nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
        --export=sqlite --force-overwrite=true --output="$dir/$name" \
        build-trace/bench/ninfer_bench --weights "$ART" \
        --corpus bench/fixtures/qwen3_8_flash_next_context.ids \
        -pg 8192,32 --max-ctx 16384 --prefill-chunk 8192 --kv-dtype "$KV_DTYPE" "$@" \
        --warmup 1 -r 1 -o json --output-file "$dir/$name-bench.json" >"$dir/$name-nsys.log" 2>&1 ||
        { log "capture $name failed; see $dir/$name-nsys.log"; return 0; }
    "$PYTHON" tools/bench/flash_next_performance.py "$dir/$name.sqlite" \
        --hardware tools/bench/hardware/gb10.json --benchmark "$dir/$name-bench.json" \
        --output "$dir/$name-report.json" >>"$dir/$name-nsys.log" 2>&1 ||
        log "report for $name failed; see $dir/$name-nsys.log"
}
capture mtp0 --spec none
capture "mtp$DRAFT_TOKENS" --spec mtp --draft-tokens "$DRAFT_TOKENS" --lm-head-draft

log "PLE residency: unannotated decode run, counting major page faults"
timer=()
[[ -x /usr/bin/time ]] && timer=(/usr/bin/time -v)
free -g >"$dir/free_before.txt"
faults_before=$(awk '/^pgmajfault /{print $2}' /proc/vmstat)
seconds_before=$SECONDS
"${timer[@]}" "$BENCH_BIN" --weights "$ART" \
    --corpus bench/fixtures/qwen3_8_flash_next_context.ids -pg 8192,512 --max-ctx 16384 \
    --prefill-chunk 8192 --kv-dtype "$KV_DTYPE" --spec none --warmup 1 -r 3 -o table \
    >"$dir/residency.log" 2>&1 || log "residency run failed; see $dir/residency.log"
faults=$(($(awk '/^pgmajfault /{print $2}' /proc/vmstat) - faults_before))
elapsed=$((SECONDS - seconds_before))
free -g >"$dir/free_after.txt"

{
    echo "## Step 3 — decode attribution and PLE residency"
    echo
    machine_summary
    echo "- nsys: $(nsys --version 2>/dev/null | head -1)"
    echo "- Hardware profile: tools/bench/hardware/gb10.json (273 GB/s; derived compute rates)"
    for name in mtp0 "mtp$DRAFT_TOKENS"; do
        echo
        echo "### $name"
        echo
        if [[ -f $dir/$name-report.md ]]; then
            sed '1d' "$dir/$name-report.md"
        else
            echo "Report missing. Last log lines:"; echo '```'; tail -n 30 "$dir/$name-nsys.log"; echo '```'
        fi
    done
    echo
    echo "### PLE residency (unannotated build, 8K prompt, 512 decode outputs, 1 warmup + 3 runs)"
    echo
    echo "System-wide major page faults during the run: $faults over ${elapsed}s (includes model load)"
    echo
    echo '```'
    grep -E 'Major|Maximum resident|Elapsed|decode|tg' "$dir/residency.log" || tail -n 20 "$dir/residency.log"
    echo
    echo "free -g before:"; cat "$dir/free_before.txt"
    echo "free -g after:"; cat "$dir/free_after.txt"
    echo '```'
} >"$dir/summary.md"

log "summary: $dir/summary.md"
cat "$dir/summary.md"
