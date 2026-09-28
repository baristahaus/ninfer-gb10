#!/usr/bin/env bash
# Speed check for the T=2..4 TC re-route: step-2 MTP K=2/K=3 rows on the FP8
# artifact, same protocol as step2_baseline.sh (8K/64K, 512 decode, 1 warmup +
# 5 measured, X925-pinned). Reference: step2-fp8-postsync (SIMT route).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
DIR=profiles/bench/gb10/step2-fp8-teroute
mkdir -p "$DIR"

X925_CORES=${X925_CORES:-5-9,15-19}
if [[ -z ${PINNED:-} ]]; then
    echo "pinning to X925 cores: $X925_CORES" >&2
    PINNED=1 exec env taskset -c "$X925_CORES" "$0" "$@"
fi
source tools/gb10/config.local.sh
export ART=/home/apollo11/models/fp8/qwen3_8_flash_next_125b_a6b_nvfp4_fp8.ninfer
BENCH_BIN=build/bench/ninfer_bench

sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
echo "page cache dropped; warming (discarded run)" >&2
"$BENCH_BIN" --weights "$ART" --corpus bench/fixtures/qwen3_8_flash_next_context.ids \
    --max-ctx 73728 --prefill-chunk 8192 --kv-dtype "$KV_DTYPE" \
    -pg 8192,512 --spec mtp --draft-tokens 2 --lm-head-draft --warmup 0 -r 1 -o table \
    >"$DIR/warm.log" 2>&1

for k in 2 3; do
    logf="$DIR/mtp$k.json"
    echo "ninfer_bench K=$k (1 warmup + 5 measured)" >&2
    "$BENCH_BIN" --weights "$ART" --corpus bench/fixtures/qwen3_8_flash_next_context.ids \
        --max-ctx 73728 --prefill-chunk 8192 --kv-dtype "$KV_DTYPE" \
        -pg '8192,512;65536,512' --spec mtp --draft-tokens "$k" --lm-head-draft \
        --warmup 1 -r 5 -o json --output-file "$logf" >"$DIR/mtp$k.log" 2>&1
done

python3 - "$DIR" <<'EOF'
import json, sys
d = sys.argv[1]
for k in (2, 3):
    o = json.load(open(f"{d}/mtp{k}.json"))
    rows = []
    for t in o["tests"]:
        spec = t.get("speculative", {})
        rows.append(
            f"{t['n_prompt']}K: decode {t['decode_output_tok_s_mean']:.1f} "
            f"(±{t['decode_output_tok_s_stddev']:.2f}) prefill {t['prefill_tok_s_mean']:.0f} "
            f"acc {spec.get('acceptance_ratio', 0):.4f}"
        )
    print(f"K={k}: " + " | ".join(rows))
EOF
echo "done: $DIR" >&2
