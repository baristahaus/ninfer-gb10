#!/usr/bin/env bash
# Step 1: start a temporary server, check streamed and non-streamed JSON response_format output,
# and run the tool-eval-bench structured-output scenarios TC-64..TC-69 when tool-eval-bench is
# installed. Writes profiles/bench/gb10/step1/summary.md.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
dir=$(step_dir step1)
start_server "$dir/server.log"
MODEL=$(curl -s --max-time 60 "$BASE_URL/v1/models" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
: "${MODEL:?no model listed at $BASE_URL/v1/models}"
log "model: $MODEL"

schema_body() { # $1 = true|false (stream)
    cat <<EOF
{"model":"$MODEL","stream":$1,"max_tokens":4096,
 "response_format":{"type":"json_schema","json_schema":{"name":"film","schema":{
   "type":"object","properties":{"title":{"type":"string"},"year":{"type":"integer"}},
   "required":["title","year"]}}},
 "messages":[{"role":"user","content":"Give me one classic film as JSON, in a \`\`\`json fence."}]}
EOF
}
object_body='{"model":"MODEL_ID","stream":true,"max_tokens":4096,
 "response_format":{"type":"json_object"},
 "messages":[{"role":"user","content":"Reply with a JSON object with keys city and country for the capital of France. Put a sentence before it and wrap it in a ```json fence."}]}'
object_body=${object_body//MODEL_ID/$MODEL}

post() { # $1 = body, $2 = output file
    curl -sN --max-time 900 "$BASE_URL/v1/chat/completions" \
        -H 'content-type: application/json' -d "$1" >"$2" || true
}
log "request 1/3: streamed json_schema"
post "$(schema_body true)" "$dir/json_schema_stream.sse"
log "request 2/3: non-streamed json_schema"
post "$(schema_body false)" "$dir/json_schema.json"
log "request 3/3: streamed json_object with prose and a fence"
post "$object_body" "$dir/json_object_stream.sse"

teb_status="tool-eval-bench not installed (install: uv tool install git+https://github.com/SeraphimSerapis/tool-eval-bench.git); skipped"
if command -v tool-eval-bench >/dev/null; then
    log "tool-eval-bench probe and TC-64..TC-69 (log: $dir/teb.log)"
    set +e
    NO_COLOR=1 TERM=dumb tool-eval-bench probe --base-url "$BASE_URL" >"$dir/teb_probe.log" 2>&1
    NO_COLOR=1 TERM=dumb tool-eval-bench run --base-url "$BASE_URL" --backend ninfer --seed 42 \
        --no-live --output-dir "$dir/teb" \
        --scenarios TC-64 TC-65 TC-66 TC-67 TC-68 TC-69 >"$dir/teb.log" 2>&1
    teb_rc=$?
    set -e
    teb_status="exit $teb_rc"
fi
stop_server

strip_ansi() { sed -e 's/\x1b\[[0-9;]*[A-Za-z]//g' "$1"; }
{
    echo "## Step 1 — JSON response_format and tool-eval-bench structured output"
    echo
    machine_summary
    echo "- Server flags: ${SERVE_ARGS[*]}"
    echo
    echo "Streamed json_schema, fenced prompt:"
    echo
    "${SUMMARIZE[@]}" stream "$dir/json_schema_stream.sse"
    echo
    echo "Non-streamed json_schema, fenced prompt:"
    echo
    "${SUMMARIZE[@]}" chat "$dir/json_schema.json"
    echo
    echo "Streamed json_object, prose and fence requested:"
    echo
    "${SUMMARIZE[@]}" stream "$dir/json_object_stream.sse"
    echo
    echo "tool-eval-bench: $teb_status"
    if [[ -f $dir/teb.log ]]; then
        echo; echo "Probe:"; echo '```'; strip_ansi "$dir/teb_probe.log" | tail -n 15; echo '```'
        echo; echo "TC-64..TC-69 (last 60 lines):"; echo '```'; strip_ansi "$dir/teb.log" | tail -n 60; echo '```'
    fi
    if grep -qiE 'error|exception|abort' "$dir/server.log"; then
        echo; echo "Server log error lines:"; echo '```'
        grep -iE 'error|exception|abort' "$dir/server.log" | tail -n 20; echo '```'
    fi
} >"$dir/summary.md"

log "summary: $dir/summary.md"
cat "$dir/summary.md"
