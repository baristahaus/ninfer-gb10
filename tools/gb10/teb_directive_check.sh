#!/usr/bin/env bash
# Section 9.3 verification: TEB hard-mode trial with the new directive placement
# on the BF16 fork artifact (before = step7a-bf16-postsync, old code, 87/100).
# Same protocol as step7a_baseline.sh phases 2-3 (server + TEB), no perplexity.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
DIR=profiles/bench/gb10/step7a-bf16-postsync-directive
rm -rf "$DIR"; mkdir -p "$DIR/serve"

X925_CORES=${X925_CORES:-5-9,15-19}
if [[ -z ${PINNED:-} ]]; then
    echo "pinning to X925 cores: $X925_CORES" >&2
    PINNED=1 exec env taskset -c "$X925_CORES" "$0" "$@"
fi
source tools/gb10/config.local.sh
export ART=$HOME/models/qwen3_8_flash_next_v3_fork/qwen3_8_flash_next_125b_a6b_nvfp4.ninfer
PORT=18087
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
    --request-log-jsonl "$DIR/serve/request.jsonl" >"$DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
    if curl -sf "$BASE_URL/health" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "server exited during startup:" >&2; tail -n 40 "$DIR/server.log" >&2; exit 1
    fi
    sleep 1
done
echo "server healthy" >&2

tool-eval-bench run --base-url "$BASE_URL" --backend ninfer --seed 42 \
    --hardmode --trials 1 --no-live --json \
    --output-dir "$DIR/teb" >"$DIR/teb.log" 2>&1
kill "$SERVER_PID" 2>/dev/null || true
trap - EXIT

"$PYTHON" - "$DIR/teb" <<'EOF'
import json, re, sys
txt = open(sys.argv[1] + "/teb.log").read()
dec = json.JSONDecoder()
best = None
for m in re.finditer(r"\{", txt):
    try:
        o, _ = dec.raw_decode(txt[m.start():])
        if isinstance(o, dict) and "final_score" in o and "scores" in o:
            best = o
    except Exception:
        continue
if best:
    print(f"TEB final: {best['final_score']}/100 "
          f"({best['scores']['total_points']}/{best['scores']['max_points']} points)")
else:
    print("TEB: no final score found in log")
EOF
echo "done: $DIR" >&2
