#!/usr/bin/env bash
# PR #18 verification battery (GB10 box, sequential, single GPU).
#   1. full ctest (includes serving schema tests and the device test)
#   2. Flash-Next real-artifact tests on the BF16 fork artifact
#   3. Flash-Next real-artifact tests on the FP8 artifact
#   4. step-2 decode without speculation on both artifacts (8K/64K)
#   5. 7a baseline on the FP8 artifact (explicit ART)
#   6. TEB multi-turn prefix-cache hit count, BF16 (directive change)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
DIR=profiles/bench/gb10/pr18-verification
mkdir -p "$DIR"

X925_CORES=${X925_CORES:-5-9,15-19}
if [[ -z ${PINNED:-} ]]; then
    echo "pinning to X925 cores: $X925_CORES" >&2
    PINNED=1 exec env taskset -c "$X925_CORES" "$0" "$@"
fi
source tools/gb10/config.local.sh
ART_BF16=/home/apollo11/models/qwen3_8_flash_next_v3_fork/qwen3_8_flash_next_125b_a6b_nvfp4.ninfer
ART_FP8=/home/apollo11/models/fp8/qwen3_8_flash_next_125b_a6b_nvfp4_fp8.ninfer
BENCH_BIN=build/bench/ninfer_bench

echo "=== 1. full ctest"
ctest --test-dir build --output-on-failure >"$DIR/1-ctest.log" 2>&1
echo "ctest exit: $?"

real_tests() { # $1 = artifact
    NINFER_QWEN38_FLASH_NEXT_WEIGHTS="$1" ctest --test-dir build \
        -R 'ninfer_qwen3_8_flash_next_(real|load_plan|frontend|fault)_test' --output-on-failure
}
echo "=== 2. real-artifact tests (BF16)"
real_tests "$ART_BF16" >"$DIR/2-real-bf16.log" 2>&1
echo "real bf16 exit: $?"
echo "=== 3. real-artifact tests (FP8)"
real_tests "$ART_FP8" >"$DIR/3-real-fp8.log" 2>&1
echo "real fp8 exit: $?"

nospec() { # $1 = artifact, $2 = name
    echo "=== 4. step 2 without speculation ($2)"
    "$BENCH_BIN" --weights "$1" --corpus bench/fixtures/qwen3_8_flash_next_context.ids \
        --max-ctx 73728 --prefill-chunk 8192 --kv-dtype fp8 \
        -pg '8192,512;65536,512' --warmup 1 -r 5 -o json \
        --output-file "$DIR/4-nospec-$2.json" >"$DIR/4-nospec-$2.log" 2>&1
    echo "nospec $2 exit: $?"
}
nospec "$ART_BF16" bf16
nospec "$ART_FP8" fp8

echo "=== 5. 7a baseline (FP8, explicit ART)"
ART="$ART_FP8" LABEL=pr18-fp8 tools/gb10/step7a_baseline.sh >"$DIR/5-7a-fp8.log" 2>&1
echo "7a fp8 exit: $?"

echo "=== 6. TEB multi-turn prefix-cache hit count (BF16)"
tools/gb10/teb_directive_check.sh >"$DIR/6-teb-directive.log" 2>&1
echo "teb directive exit: $?"

python3 - "$DIR" <<'EOF'
import json, os, re, sys
d = sys.argv[1]
print()
print("=== summary ===")
m = re.search(r"tests passed\n(\d+)% tests", open(f"{d}/1-ctest.log").read() or "")
tail = open(f"{d}/1-ctest.log").read()[-400:]
print(f"ctest: {tail.splitlines()[-2] if len(tail.splitlines()) > 1 else tail.splitlines()[-1]}")
for name in ("bf16", "fp8"):
    o = json.load(open(f"{d}/4-nospec-{name}.json"))
    rows = " | ".join(
        f"{t['n_prompt'] // 1024}K decode {t['decode_output_tok_s_mean']:.1f}"
        f"(±{t['decode_output_tok_s_stddev']:.2f}) prefill {t['prefill_tok_s_mean']:.0f}"
        for t in o["tests"]
    )
    print(f"nospec {name}: {rows}")
s = open(f"{d}/5-7a-fp8.log").read()
for line in s.splitlines():
    if "perplexity" in line and ("4096" in line or "65536" in line or "score" in line.lower()):
        print("7a:", line.strip()[:120])
t = open(f"{d}/6-teb-directive.log").read()
for line in t.splitlines():
    if "hit" in line.lower() or "score" in line.lower() or "prefix" in line.lower():
        print("teb:", line.strip()[:120])
EOF
echo "battery done: $DIR" >&2
