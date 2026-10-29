#!/usr/bin/env bash
# Golden-check: run the Flash-Next engine-real golden on a built tree and report the first
# divergent token of the greedy MTP prefix. Used to localize kernel-generation changes that
# alter greedy numerics (SP-7/EXL3 integration, 2026-10-29).
#
# Usage: tools/gb10/golden_check.sh [build-dir]
# Environment: NINFER_QWEN38_FLASH_NEXT_WEIGHTS points at the fp8mtp artifact; it defaults to
# the workstation's out/fp8mtp generation when unset.
set -u
: "${NINFER_QWEN38_FLASH_NEXT_WEIGHTS:=out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer}"
export NINFER_QWEN38_FLASH_NEXT_WEIGHTS
DIR=${1:-build}
TEST="$DIR/tests/ninfer_qwen3_8_flash_next_engine_real_test"
[ -x "$TEST" ] || { echo "missing test binary: $TEST" >&2; exit 2; }
LOG=$(mktemp /tmp/golden_check.XXXXXX.log)
timeout 900 "$TEST" >"$LOG" 2>&1
status=$?
if [ $status -eq 0 ]; then
    echo "PASS $(cd "$DIR" >/dev/null 2>&1 && git rev-parse --show-toplevel 2>/dev/null | xargs -r git -C /dev/stdin rev-parse HEAD 2>/dev/null)"
    rm -f "$LOG"
    exit 0
fi
mtp=$(grep -m1 '^mtp:' "$LOG" | cut -d' ' -f2-)
exp=$(grep -m1 '^expected:' "$LOG" | cut -d' ' -f2-)
first=$(python3 - "$mtp" "$exp" <<'EOF'
import sys
mtp = sys.argv[1].split() if len(sys.argv) > 1 and sys.argv[1] else []
exp = sys.argv[2].split() if len(sys.argv) > 2 and sys.argv[2] else []
for i, (a, b) in enumerate(zip(mtp, exp)):
    if a != b:
        print(f"index={i} got={a} want={b}")
        break
else:
    print("index=0 (no aligned divergence found)")
EOF
)
echo "FAIL first divergence: $first"
echo "log: $LOG"
exit 1
