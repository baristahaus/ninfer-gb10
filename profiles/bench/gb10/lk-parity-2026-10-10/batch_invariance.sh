#!/usr/bin/env bash
# Batch invariance (run-list item 4): the same 4 class prompts replayed twice
# at C4 against the lk build. Two rounds, fresh server and dropped caches
# between rounds; the outputs must match exactly.
#
# usage: batch_invariance.sh BINARY ART PORT OUTDIR [SPEC_FLAGS...]
set -euo pipefail
BINARY=$1; ART=$2; PORT=$3; OUTDIR=$4; shift 4
SPEC_FLAGS=("$@")
DIR=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUTDIR"

run_round() {
    local tag=$1
    sync
    if sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
        echo "caches dropped (sudo)" > "$OUTDIR/caches.$tag"
    else
        echo "cache drop unavailable (no permission); round ran with warm caches" > "$OUTDIR/caches.$tag"
    fi
    timeout 900 "$BINARY" "$ART" --host 127.0.0.1 --port "$PORT" \
      --max-context 73728 --max-concurrency 4 --kv-dtype fp8 --preserve-thinking \
      --lm-head-draft --host-kv-mib 0 --prefill-chunk 4096 "${SPEC_FLAGS[@]}" \
      > "$OUTDIR/server.$tag.log" 2>&1 &
    local SRV=$!
    local ok=0
    for _ in $(seq 1 180); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
        sleep 2
    done
    if [[ $ok -eq 0 ]]; then
        kill "$SRV" 2>/dev/null || true
        echo "server never became healthy (round $tag); see $OUTDIR/server.$tag.log" >&2
        exit 1
    fi
    python3 "$DIR/batch_invariance.py" 127.0.0.1 "$PORT" "$OUTDIR/round_$tag.json"
    kill "$SRV" 2>/dev/null || true
    wait "$SRV" 2>/dev/null || true
}

run_round a
run_round b

python3 - "$OUTDIR" <<'EOF'
import json
import sys

out = sys.argv[1]
a = {r["class"]: r["text"] for r in json.load(open(f"{out}/round_a.json"))}
b = {r["class"]: r["text"] for r in json.load(open(f"{out}/round_b.json"))}
ok = all(a[k] == b[k] for k in a) and len(a) == len(b)
for k in sorted(a):
    same = a[k] == b.get(k)
    print(f"{k}: {'MATCH' if same else 'MISMATCH'} ({len(a[k])} chars vs {len(b.get(k, ''))} chars)")
print("BATCH_INVARIANCE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
EOF
