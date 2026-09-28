# Shared setup for the GB10 step scripts. Sourced, not executed.
set -euo pipefail

GB10_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$GB10_ROOT"

GB10_CONFIG=${GB10_CONFIG:-tools/gb10/config.local.sh}
if [[ ! -f $GB10_CONFIG ]]; then
    echo "Missing $GB10_CONFIG. Create it with:" >&2
    echo "  cp tools/gb10/config.example.sh $GB10_CONFIG   # then edit ART and SERVE_ARGS" >&2
    exit 2
fi
# A value passed in the environment (each script's usage line documents ART=...)
# takes precedence over the config file default.
ART_FROM_ENV=${ART-}
# shellcheck source=config.example.sh
source "$GB10_CONFIG"
if [[ -n $ART_FROM_ENV ]]; then
    ART=$ART_FROM_ENV
fi
: "${ART:?ART must name the Flash-Next artifact}" "${PORT:?}" "${KV_DTYPE:?}" "${DRAFT_TOKENS:?}"
PYTHON=${PYTHON:-python3}
RUN_SERVING=${RUN_SERVING:-0}
TOKENIZER=${TOKENIZER:-}
if [[ ! -f $ART ]]; then
    echo "ART does not name a file: $ART" >&2
    exit 2
fi

NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
export CUDACXX=${CUDACXX:-$NVCC}
OUT_ROOT=profiles/bench/gb10
SERVE_BIN=build/apps/ninfer-serve
BENCH_BIN=build/bench/ninfer_bench
BASE_URL="http://127.0.0.1:$PORT"
SUMMARIZE=("$PYTHON" tools/gb10/summarize.py)

# step_dir NAME: fresh output directory for one step.
step_dir() {
    local dir="$OUT_ROOT/$1"
    rm -rf "$dir"
    mkdir -p "$dir"
    echo "$dir"
}

log() { printf '[gb10 %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

require_binary() {
    if [[ ! -x $1 ]]; then
        echo "Missing $1. Run tools/gb10/step0_build_test.sh first." >&2
        exit 2
    fi
}

# The timing build must not carry performance annotations.
require_untraced_build() {
    if grep -q '^NINFER_PERFORMANCE_TRACE:BOOL=ON' build/CMakeCache.txt 2>/dev/null; then
        echo "build/ has NINFER_PERFORMANCE_TRACE=ON; rerun step0 (it configures it OFF)." >&2
        exit 2
    fi
}

SERVER_PID=
stop_server() {
    if [[ -n $SERVER_PID ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "stopping server (pid $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=
}

# start_server LOGFILE [EXTRA_FLAGS...]: start ninfer-serve in the background and wait until
# /health is ok.
start_server() {
    local server_log=$1
    shift
    require_binary "$SERVE_BIN"
    if curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
        echo "Something already answers on $BASE_URL; stop it or change PORT." >&2
        exit 2
    fi
    log "starting server on port $PORT (log: $server_log)"
    "$SERVE_BIN" "$ART" --port "$PORT" "${SERVE_ARGS[@]}" "$@" >"$server_log" 2>&1 &
    SERVER_PID=$!
    trap stop_server EXIT
    local waited=0
    until curl -sf "$BASE_URL/health" >/dev/null 2>&1; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server exited during startup. Last log lines:" >&2
            tail -n 40 "$server_log" >&2
            exit 1
        fi
        if ((waited >= 1800)); then
            echo "Server not healthy after 30 minutes. Last log lines:" >&2
            tail -n 40 "$server_log" >&2
            exit 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    log "server healthy after ${waited}s"
}

# Machine description shared by every summary.
machine_summary() {
    echo "- Host: $(uname -srm); $( (. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") || echo unknown OS)"
    echo "- Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git diff --quiet 2>/dev/null || echo ' (with local changes)')"
    echo "- nvcc: $("$NVCC" --version 2>/dev/null | grep -o 'release [0-9.]*, V[0-9.]*' || echo 'not found')"
    echo "- GPU/driver: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || echo 'nvidia-smi failed')"
    local art_size
    art_size=$( { ls -1 "$ART" "$ART".part-* 2>/dev/null || true; } | xargs -r du -ch | awk 'END { print $1 }')
    echo "- Artifact: $(basename "$ART"), ${art_size:--} (multi-volume total)"
}
