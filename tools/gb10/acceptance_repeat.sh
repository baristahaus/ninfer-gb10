#!/usr/bin/env bash
# Acceptance-protocol repeat: fp8 artifact on the post-sync binary, to firm up
# the 65.01 -> 63.61 engine-code delta (step7a phase-2 protocol, no PPL/TEB).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
DIR=profiles/bench/gb10/step7a-fp8-postsync-acceptance-repeat
rm -rf "$DIR"; mkdir -p "$DIR"

X925_CORES=${X925_CORES:-5-9,15-19}
if [[ -z ${PINNED:-} ]]; then
    echo "pinning to X925 cores: $X925_CORES" >&2
    PINNED=1 exec env taskset -c "$X925_CORES" "$0" "$@"
fi
source tools/gb10/config.local.sh
export ART=/home/apollo11/models/fp8/qwen3_8_flash_next_125b_a6b_nvfp4_fp8.ninfer
PORT=18088
SERVE_ARGS=(
  --max-context 73728
  --max-concurrency 2
  --kv-dtype fp8
  --spec mtp --draft-tokens 2 --lm-head-draft
  --preserve-thinking
)
BASE_URL="http://127.0.0.1:$PORT"
PYTHON=${PYTHON:-python3}

sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null

build/apps/ninfer-serve "$ART" --port "$PORT" "${SERVE_ARGS[@]}" \
    --request-log-jsonl "$DIR/request.jsonl" >"$DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
    if curl -sf "$BASE_URL/health" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "server exited during startup:" >&2; tail -n 40 "$DIR/server.log" >&2; exit 1
    fi
    sleep 1
done
echo "server healthy" >&2

"$PYTHON" tools/gb10/step7a_acceptance.py "$BASE_URL" "$DIR/request.jsonl" "$DIR" \
    >"$DIR/acceptance.log" 2>&1
kill "$SERVER_PID" 2>/dev/null || true
trap - EXIT

"$PYTHON" - "$DIR/acceptance.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
p = d["accepted_per_position_per_round"]
print(f"acceptance {d['acceptance_ratio']:.4f}  (drafted {d['drafted_tokens']}, "
      f"accepted {d['accepted_tokens']}, rounds {d['rounds']})  pos0 {p['0']:.4f} pos1 {p['1']:.4f}")
EOF
echo "done: $DIR" >&2
