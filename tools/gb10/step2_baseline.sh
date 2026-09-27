#!/usr/bin/env bash
# Step 2: unprofiled GB10 baseline. Measures sustained memory bandwidth, then ninfer_bench at 8K
# and 64K prompts with 512 decode outputs for MTP off, MTP with DRAFT_TOKENS, and MTP3. With
# RUN_SERVING=1 it also runs the Flash-Next serving matrix. Run on an otherwise idle machine.
# Writes profiles/bench/gb10/step2/summary.md. Takes roughly 30-60 minutes without the matrix.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
dir=$(step_dir step2)
require_binary "$BENCH_BIN"
require_untraced_build

log "building and running the memory bandwidth probe"
if "$NVCC" -O3 -std=c++17 -arch=sm_121a tools/hbm_bandwidth_probe.cu -o "$dir/hbm_bandwidth_probe" \
    >"$dir/probe_build.log" 2>&1; then
    "$dir/hbm_bandwidth_probe" --peak-gbps 273 >"$dir/bandwidth.txt" 2>&1 || true
else
    cp "$dir/probe_build.log" "$dir/bandwidth.txt"
fi

bench() { # $1 = output name, remaining = extra ninfer_bench arguments
    local name=$1
    shift
    if ! "$BENCH_BIN" --weights "$ART" --corpus bench/fixtures/qwen3_8_flash_next_context.ids \
        --max-ctx 73728 --prefill-chunk 8192 --kv-dtype "$KV_DTYPE" "$@" \
        >"$dir/$name.log" 2>&1; then
        echo "ninfer_bench failed ($name). Last lines of $dir/$name.log:" >&2
        tail -n 30 "$dir/$name.log" >&2
        exit 1
    fi
}

log "warming the page cache (discarded run)"
bench warm -pg 8192,32 --spec none --warmup 0 -r 1 -o table

draft_counts=(0 "$DRAFT_TOKENS")
[[ $DRAFT_TOKENS != 3 ]] && draft_counts+=(3)
reports=()
for k in "${draft_counts[@]}"; do
    [[ $k == 0 ]] && spec=(--spec none) || spec=(--spec mtp --draft-tokens "$k" --lm-head-draft)
    log "ninfer_bench K=$k (8K and 64K prompts, 1 warmup + 5 measured)"
    bench "mtp$k" -pg '8192,512;65536,512' "${spec[@]}" --warmup 1 -r 5 -o json \
        --output-file "$dir/mtp$k.json"
    reports+=("$dir/mtp$k.json")
done

serving_status="skipped (set RUN_SERVING=1 and TOKENIZER in the config to enable)"
if [[ $RUN_SERVING == 1 ]]; then
    : "${TOKENIZER:?RUN_SERVING=1 needs TOKENIZER}"
    log "building the serving prompt fixture"
    "$PYTHON" - "$TOKENIZER" "$dir/contexts.json" <<'EOF'
import json, sys
from pathlib import Path
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
ids = list(map(int, Path("bench/fixtures/qwen3_8_flash_next_context.ids").read_text().split()))
Path(sys.argv[2]).write_text(json.dumps({str(n): tokenizer.decode(ids[:n]) for n in (1980, 2049, 8192, 65536)}))
EOF
    # Cold-prompt cases must not reuse prefixes (tools/bench/README.md).
    start_server "$dir/server.log" --no-prefix-reuse
    log "serving matrix (log: $dir/serving.log)"
    "$PYTHON" tools/bench/run_flash_next_serving.py --url "$BASE_URL" \
        --contexts "$dir/contexts.json" --output "$dir/serving.jsonl" >"$dir/serving.log" 2>&1
    stop_server
    serving_status="completed"
fi

{
    echo "## Step 2 — unprofiled baseline"
    echo
    machine_summary
    echo "- Benchmark: -pg 8192,512 and 65536,512; --max-ctx 73728 --prefill-chunk 8192 --kv-dtype $KV_DTYPE; 1 warmup + 5 measured"
    echo "- Power mode: $(nvidia-smi --query-gpu=power.limit --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
    echo
    echo "Memory bandwidth probe:"
    echo '```'
    grep -E 'SMs / L2|Best sustained|VRAM total' "$dir/bandwidth.txt" || tail -n 5 "$dir/bandwidth.txt"
    echo '```'
    echo
    "${SUMMARIZE[@]}" bench "${reports[@]}"
    echo
    echo "Serving matrix: $serving_status"
    if [[ -f $dir/serving.jsonl ]]; then
        echo
        "${SUMMARIZE[@]}" serving "$dir/serving.jsonl"
    fi
} >"$dir/summary.md"

log "summary: $dir/summary.md"
cat "$dir/summary.md"
