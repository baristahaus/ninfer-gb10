#!/usr/bin/env bash
# Block L item 2: out-of-bounds checks at prefill chunk boundaries (after dgpp's 257-token
# head-offset bug). Compute Sanitizer memcheck (not initcheck, which misreports on GB10):
#   A. ninfer_bench at exact prompt lengths 255/256/257/513 with a 256-token chunk, MTP off and
#      on, eager (graphs off), 4 output tokens.
#   B. ninfer-serve, 128-token chunk, 24 requests whose system message grows by one word each,
#      so the context-cache capture frontier at the end of the system turn steps across a chunk
#      boundary (StateImage fork continuing exactly at a boundary).
# Writes profiles/bench/gb10/chunk-memcheck/summary.md. Memcheck is slow: expect hours.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
dir=$(step_dir chunk-memcheck)
require_binary "$BENCH_BIN"
require_binary "$SERVE_BIN"
command -v compute-sanitizer >/dev/null || { echo "compute-sanitizer not found" >&2; exit 2; }
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]]; then
    echo "GPU busy; one device-allocating job at a time." >&2
    exit 2
fi
sanitize=(compute-sanitizer --tool memcheck --leak-check no --print-limit 20 --error-exitcode 99)

declare -a rows=()
for spec in "" "--spec mtp --draft-tokens $DRAFT_TOKENS --lm-head-draft"; do
    tag=$([[ -z $spec ]] && echo mtp0 || echo "mtp$DRAFT_TOKENS")
    for length in 255 256 257 513; do
        log "A: $tag prompt $length"
        status=0
        # shellcheck disable=SC2086
        "${sanitize[@]}" "$BENCH_BIN" --weights "$ART" \
            --corpus bench/fixtures/qwen3_8_flash_next_context.ids -pg "$length,4" \
            --max-ctx 2048 --prefill-chunk 256 --kv-dtype "$KV_DTYPE" --no-cuda-graph $spec \
            --warmup 0 -r 1 -o table >"$dir/A-$tag-$length.log" 2>&1 || status=$?
        rows+=("| A | $tag | prompt $length, chunk 256 | exit $status | $(grep -m1 -o 'ERROR SUMMARY: [0-9]* error' "$dir/A-$tag-$length.log" || echo 'no summary') |")
    done
done

log "B: serve under memcheck, chunk 128, capture-frontier sweep"
"${sanitize[@]}" "$SERVE_BIN" "$ART" --port "$PORT" --max-context 4096 --max-concurrency 1 \
    --kv-dtype "$KV_DTYPE" --prefill-chunk 128 --spec mtp --draft-tokens "$DRAFT_TOKENS" \
    --lm-head-draft --no-cuda-graph >"$dir/B-serve.log" 2>&1 &
SERVER_PID=$!
waited=0
until curl -sf "$BASE_URL/health" >/dev/null 2>&1; do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    ((waited < 7200)) || break
    sleep 10; waited=$((waited + 10))
done
"$PYTHON" - "$BASE_URL" >"$dir/B-requests.log" 2>&1 <<'PY'
import json, sys, urllib.request
base = sys.argv[1]
model = json.load(urllib.request.urlopen(base + "/v1/models"))["data"][0]["id"]
words = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
         "november oscar papa quebec romeo sierra tango uniform victor whiskey xray").split()
base_system = "You are a careful assistant. " + "Keep every answer short and exact. " * 12
for n in range(len(words)):
    body = {"model": model, "max_tokens": 4, "temperature": 0,
            "messages": [{"role": "system", "content": base_system + " ".join(words[:n + 1])},
                         {"role": "user", "content": "Name one prime number."}]}
    request = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    try:
        reply = json.load(urllib.request.urlopen(request, timeout=7200))
        print(n, "prompt_tokens", reply.get("usage", {}).get("prompt_tokens"), flush=True)
    except Exception as error:  # record and continue; the sanitizer log is the verdict
        print(n, "error", error, flush=True)
PY
[[ $waited -lt 7200 ]] || log "B: server never became healthy; see $dir/B-serve.log"
kill -TERM "$SERVER_PID" 2>/dev/null || true
status=0
wait "$SERVER_PID" || status=$?
SERVER_PID=
rows+=("| B | mtp$DRAFT_TOKENS | 24 system lengths, chunk 128 | exit $status | $(grep -m1 -o 'ERROR SUMMARY: [0-9]* error' "$dir/B-serve.log" || echo 'no summary') |")

{
    echo "## Chunk-boundary memcheck"
    echo
    machine_summary
    echo
    echo "| Part | MTP | Case | Exit | Sanitizer |"
    echo "|---|---|---|---|---|"
    printf '%s\n' "${rows[@]}"
    echo
    echo "B prompt lengths (the system-turn capture frontier moves by about one token per row):"
    echo '```'
    cat "$dir/B-requests.log"
    echo '```'
} >"$dir/summary.md"
log "summary: $dir/summary.md"
cat "$dir/summary.md"
