#!/usr/bin/env bash
# One parity arm: fresh server, 105 DGPP serve_load requests
# (C1/C2/C4 x 5 classes x 3 repeats, 256 tokens, greedy, thinking off).
#
# usage: parity_arm.sh NAME BINARY ART PORT OUTDIR [SPEC_FLAGS...]
#   NAME      arm name (e.g. lk-k3)
#   BINARY    the ninfer-serve binary for this arm
#   ART       the .ninfer entry artifact
#   PORT      serve port
#   OUTDIR    output directory
#   SPEC_FLAGS...  e.g. --spec mtp --draft-tokens 3
set -euo pipefail
NAME=$1; BINARY=$2; ART=$3; PORT=$4; OUTDIR=$5; shift 5
SPEC_FLAGS=("$@")
mkdir -p "$OUTDIR"

sync
if sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
    echo "caches dropped (sudo)" > "$OUTDIR/$NAME.caches"
else
    echo "cache drop unavailable (no permission); arm ran with warm caches" > "$OUTDIR/$NAME.caches"
fi

timeout 3600 "$BINARY" "$ART" --host 127.0.0.1 --port "$PORT" \
  --max-context 73728 --max-concurrency 4 --kv-dtype fp8 --preserve-thinking \
  --lm-head-draft --host-kv-mib 0 --prefill-chunk 4096 "${SPEC_FLAGS[@]}" \
  > "$OUTDIR/$NAME.server.log" 2>&1 &
SRV=$!

ok=0
for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
    sleep 2
done
if [[ $ok -eq 0 ]]; then
    kill "$SRV" 2>/dev/null || true
    echo "server never became healthy; see $OUTDIR/$NAME.server.log" >&2
    exit 1
fi

rc=0
python3 /home/apollo11/dgpp/scripts/serve_load.py 127.0.0.1 "$PORT" \
  --concurrency 1,2,4 --classes all --max-tokens 256 --repeat 3 \
  --json-out "$OUTDIR/$NAME.json" > "$OUTDIR/$NAME.txt" 2>&1 || rc=$?

kill "$SRV" 2>/dev/null || true
wait "$SRV" 2>/dev/null || true
echo "$rc" > "$OUTDIR/$NAME.rc"
exit $rc
