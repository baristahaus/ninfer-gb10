#!/usr/bin/env bash
# Plan Block I: the decision-facts campaign (about 8 hours on GB10). Each phase answers one open
# question in the plan; the plan's Block I section says which decision each result feeds.
#
#   I0  build, ctest, Flash-Next real tests; 4K perplexity; Qwen3.5 27B smoke (upstream attention)
#   I1  allocation versus page cache: what the memory-sizing fix must read
#   I2  served decode attribution: request logs and ninfer_bench on natural text versus the fixture
#   I3  acceptance run-to-run floor at K=2 and K=3
#   I4  concurrency scaling at MC=8 for K=0/2/3, distinct and shared prompts
#   I5  context scaling 1K..128K for K=0 and K=3
#   I6  structured output without constrained decoding (step 4)
#   I7  step 3 attribution refresh at K=3 on this artifact
#   I8  PLE int6 side-check (7f; CPU only, needs BF16_SOURCE and FP8_SOURCE)
#   I9  DGPP against NInfer on this machine, one client and one prompt set (needs DGPP_DIR)
#
# Settings (environment, besides tools/gb10/config.local.sh):
#   ART           the adopted artifact (fp8mtp), required as always
#   PHASES        comma list, default I0,I1,I2,I3,I4,I5,I6,I7,I8,I9
#   TOKENIZER     Flash-Next tokenizer directory; I2 and I5 then also run on natural text
#   QWEN35_ART    Qwen3.5 27B artifact for the I0 smoke (skipped when unset)
#   BF16_SOURCE / FP8_SOURCE  checkpoints for I8 (skipped when either is unset); CONVERT_PYTHON is
#                 an interpreter with torch and safetensors (defaults to PYTHON)
#   DGPP_DIR      a built HawkBearPig/dgpp checkout for I9 (skipped when unset), with
#   DGPP_START / DGPP_STOP  commands that start and stop its single-Spark RadixArk server (run
#                 from DGPP_DIR), and DGPP_PORT its HTTP port (default 8000)
# A phase whose DONE marker exists is skipped, so an interrupted campaign resumes. One GPU job at
# a time throughout; phases drop the page cache where a cold start is part of the protocol.
X925_CORES=${X925_CORES:-5-9,15-19}
if [[ -z ${PINNED:-} ]]; then
    PINNED=1 exec taskset -c "$X925_CORES" "$0" "$@"
fi
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

PHASES=${PHASES:-I0,I1,I2,I3,I4,I5,I6,I7,I8,I9}
# BLOCK_I_ROOT keeps a rerun (for example PHASES=I9) from replacing an earlier campaign's records.
ROOT=${BLOCK_I_ROOT:-$OUT_ROOT/block-i}
mkdir -p "$ROOT"
want() { case ",$PHASES," in *",$1,"*) [[ ! -f $ROOT/$1/DONE ]] ;; *) return 1 ;; esac; }
begin() { rm -rf "${ROOT:?}/$1"; mkdir -p "$ROOT/$1"; log "phase $1: $2"; }
finish() { touch "$ROOT/$1/DONE"; log "phase $1 done"; }
drop_caches() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; }
meminfo() { grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo | tr -s ' '; }
gpu_idle() {
    if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]]; then
        echo "GPU is held by another job; single-instance discipline: aborting." >&2
        exit 1
    fi
}
bench() { # $1 = log path, then ninfer_bench arguments after the artifact
    local out=$1
    shift
    "$BENCH_BIN" --weights "$ART" --kv-dtype "$KV_DTYPE" "$@" >"$out" 2>&1
}
serve_args() { # $1 = max concurrency, $2 = K (0 = no speculation)
    SERVE_ARGS=(--max-context 73728 --max-concurrency "$1" --kv-dtype "$KV_DTYPE" --preserve-thinking)
    if (($2 > 0)); then SERVE_ARGS+=(--spec mtp --draft-tokens "$2" --lm-head-draft); fi
}
NATURAL=$ROOT/natural_corpus.ids
natural_corpus() {
    [[ -n $TOKENIZER ]] || return 1
    [[ -f $NATURAL ]] || "$PYTHON" tools/gb10/natural_corpus_ids.py "$TOKENIZER" "$NATURAL" >&2
}

gpu_idle
log "Block I on $(basename "$ART"), phases $PHASES"

# ---- I0: the merged tree builds, tests pass, perplexity holds --------------------------------
if want I0; then
    begin I0 "build, tests, perplexity, 27B smoke"
    tools/gb10/step0_build_test.sh >"$ROOT/I0/step0.log" 2>&1 || true
    cp "$OUT_ROOT/step0/summary.md" "$ROOT/I0/step0-summary.md"
    drop_caches
    ./build/apps/ninfer-perplexity "$ART" --corpus eval/corpora/perplexity-1m/manifest.json \
        --kv-dtype "$KV_DTYPE" --output "$ROOT/I0/perplexity" \
        >"$ROOT/I0/perplexity.log" 2>&1
    if [[ -n ${QWEN35_ART:-} ]]; then
        drop_caches
        "$BENCH_BIN" --weights "$QWEN35_ART" --corpus bench/fixtures/bench_corpus.ids \
            -pg '8192,256' --warmup 1 -r 3 -o table >"$ROOT/I0/qwen35-bench.log" 2>&1
        ./build/apps/ninfer "$QWEN35_ART" --greedy --max-new 128 --no-thinking \
            --prompt "Explain in three sentences why the sky is blue." \
            >"$ROOT/I0/qwen35-smoke.txt" 2>"$ROOT/I0/qwen35-smoke.log"
    fi
    finish I0
fi

# ---- I1: does cudaMalloc reclaim clean page cache? ------------------------------------------
if want I1; then
    begin I1 "allocation versus page cache"
    "$NVCC" -O2 -std=c++17 -arch=sm_121a tools/gb10/alloc_probe.cu -o "$ROOT/I1/alloc_probe"
    {
        echo "## cold (after drop_caches)"
        drop_caches
        meminfo
        "$ROOT/I1/alloc_probe"
        echo
        echo "## warm (artifact volumes read into the page cache first)"
        for volume in "$ART" "$ART".part-*; do [[ -f $volume ]] && cat "$volume" >/dev/null; done
        meminfo
        "$ROOT/I1/alloc_probe"
    } >"$ROOT/I1/alloc_probe.md" 2>&1
    # Server startup, warm cache and no drop: does the engine's own check refuse it, and what
    # KV capacity does automatic sizing choose?
    serve_args 8 3
    for volume in "$ART" "$ART".part-*; do [[ -f $volume ]] && cat "$volume" >/dev/null; done
    meminfo >"$ROOT/I1/warm-start-meminfo.txt"
    if ( start_server "$ROOT/I1/warm-start.log" --kv-capacity auto \
            --request-log-jsonl "$ROOT/I1/warm-start.jsonl"; stop_server ); then
        echo "warm start: ok" >>"$ROOT/I1/warm-start-meminfo.txt"
    else
        echo "warm start: refused" >>"$ROOT/I1/warm-start-meminfo.txt"
    fi
    drop_caches
    meminfo >"$ROOT/I1/cold-start-meminfo.txt"
    start_server "$ROOT/I1/cold-start.log" --kv-capacity auto \
        --request-log-jsonl "$ROOT/I1/cold-start.jsonl"
    meminfo >>"$ROOT/I1/cold-start-meminfo.txt"
    stop_server
    finish I1
fi

# ---- I2: where the served rate goes ----------------------------------------------------------
if want I2; then
    begin I2 "served decode attribution"
    for log in "$OUT_ROOT"/step7a-*/serve/request.jsonl; do
        [[ -f $log ]] && "$PYTHON" tools/gb10/request_log_summary.py "$log"
    done >"$ROOT/I2/gate-request-logs.md"
    drop_caches
    for corpus in bench/fixtures/qwen3_8_flash_next_context.ids $(natural_corpus && echo "$NATURAL"); do
        name=$(basename "$corpus" .ids)
        for k in 0 3; do
            spec=()
            ((k > 0)) && spec=(--spec mtp --draft-tokens "$k" --lm-head-draft)
            bench "$ROOT/I2/$name-k$k.log" --corpus "$corpus" --max-ctx 73728 \
                --prefill-chunk 8192 -pg '2048,512;8192,512' "${spec[@]}" --warmup 1 -r 3 \
                -o json --output-file "$ROOT/I2/$name-k$k.json"
        done
    done
    finish I2
fi

# ---- I3: acceptance floor ---------------------------------------------------------------------
if want I3; then
    begin I3 "acceptance run-to-run floor"
    SWEEP="adopted=$ART" KS="2 3" REPEATS=3 NAME=floor tools/gb10/k_sweep.sh \
        >"$ROOT/I3/k_sweep.md" 2>"$ROOT/I3/k_sweep.log"
    finish I3
fi

# ---- I4: concurrency scaling -----------------------------------------------------------------
if want I4; then
    begin I4 "concurrency scaling at MC=8"
    for k in 0 2 3; do
        serve_args 8 "$k"
        drop_caches
        start_server "$ROOT/I4/server-k$k.log" --request-log-jsonl "$ROOT/I4/request-k$k.jsonl"
        "$PYTHON" tools/gb10/concurrency_sweep.py "$BASE_URL" "$ROOT/I4/distinct-k$k.json" \
            --n 1,2,4,8 --max-tokens 1024 >"$ROOT/I4/distinct-k$k.txt" 2>&1 ||
            log "I4 k=$k distinct sweep exited nonzero; partial results in distinct-k$k.txt"
        "$PYTHON" tools/gb10/concurrency_sweep.py "$BASE_URL" "$ROOT/I4/shared-k$k.json" \
            --n 8 --shared --max-tokens 1024 >"$ROOT/I4/shared-k$k.txt" 2>&1 ||
            log "I4 k=$k shared sweep exited nonzero; partial results in shared-k$k.txt"
        stop_server
        "$PYTHON" tools/gb10/request_log_summary.py "$ROOT/I4/request-k$k.jsonl" \
            >"$ROOT/I4/request-k$k.md"
    done
    finish I4
fi

# ---- I5: context scaling ---------------------------------------------------------------------
if want I5; then
    begin I5 "context scaling"
    corpus=bench/fixtures/qwen3_8_flash_next_context.ids
    natural_corpus && corpus=$NATURAL
    drop_caches
    for k in 0 3; do
        spec=()
        ((k > 0)) && spec=(--spec mtp --draft-tokens "$k" --lm-head-draft)
        bench "$ROOT/I5/k$k.log" --corpus "$corpus" --max-ctx 139264 --prefill-chunk 8192 \
            -pg '1024,256;8192,256;32768,256;131072,256' "${spec[@]}" --warmup 1 -r 3 \
            -o json --output-file "$ROOT/I5/k$k.json"
    done
    finish I5
fi

# ---- I6: structured output without constrained decoding --------------------------------------
if want I6; then
    begin I6 "structured output"
    serve_args 1 3
    drop_caches
    start_server "$ROOT/I6/server.log" --request-log-jsonl "$ROOT/I6/request.jsonl"
    "$PYTHON" tools/gb10/structured_probe.py "$BASE_URL" "$ROOT/I6/structured.json" \
        >"$ROOT/I6/structured.md" 2>&1
    stop_server
    finish I6
fi

# ---- I7: attribution refresh -----------------------------------------------------------------
if want I7; then
    begin I7 "step 3 attribution at K=3"
    drop_caches
    DRAFT_TOKENS=3 tools/gb10/step3_attribution.sh >"$ROOT/I7/step3.log" 2>&1
    cp "$OUT_ROOT/step3/summary.md" "$ROOT/I7/step3-summary.md"
    finish I7
fi

# ---- I8: PLE int6 side-check (CPU) -----------------------------------------------------------
if want I8; then
    begin I8 "PLE int6 side-check"
    if [[ -n ${BF16_SOURCE:-} && -n ${FP8_SOURCE:-} ]]; then
        "${CONVERT_PYTHON:-$PYTHON}" tools/gb10/ple_int6_check.py "$BF16_SOURCE" "$FP8_SOURCE" \
            >"$ROOT/I8/ple_int6.md" 2>&1
    else
        echo "skipped: BF16_SOURCE and FP8_SOURCE not both set" >"$ROOT/I8/ple_int6.md"
    fi
    finish I8
fi

# ---- I9: DGPP against NInfer on the same machine ---------------------------------------------
# One client for both engines: DGPP's own scripts/serve_load.py (its published C1/C2/C4 protocol:
# five prompt classes, greedy, thinking off, 256 tokens, three repetitions). NInfer runs its
# adopted artifact at K=1 (DGPP's published single-Spark depth), 2 and 3 with four slots.
if want I9; then
    begin I9 "DGPP against NInfer"
    if [[ -n ${DGPP_DIR:-} && -n ${DGPP_START:-} && -n ${DGPP_STOP:-} ]]; then
        load=("$PYTHON" "$DGPP_DIR/scripts/serve_load.py" 127.0.0.1)
        load_args=(--concurrency 1,2,4 --classes prose,code,json,math,chat --max-tokens 256
            --repeat 3)
        git -C "$DGPP_DIR" log -1 --format='DGPP commit %h %cd' >"$ROOT/I9/dgpp-version.txt"
        for k in 1 2 3; do
            serve_args 4 "$k"
            drop_caches
            start_server "$ROOT/I9/ninfer-k$k.log" --request-log-jsonl "$ROOT/I9/ninfer-k$k.jsonl"
            "${load[@]}" "$PORT" "${load_args[@]}" --json-out "$ROOT/I9/ninfer-k$k.json" \
                >"$ROOT/I9/ninfer-k$k.txt" 2>&1 || log "I9 NInfer K=$k load exited nonzero"
            stop_server
            "$PYTHON" tools/gb10/request_log_summary.py "$ROOT/I9/ninfer-k$k.jsonl" \
                >"$ROOT/I9/ninfer-k$k.md"
        done
        drop_caches
        gpu_idle
        # DGPP writes a prepacked weight image (~82 GiB) to ~/.cache/dgpp/resident on first load
        # unless told not to; this comparison must not consume that disk.
        ( cd "$DGPP_DIR" && export DGPP_RESIDENT_CACHE=off && eval "$DGPP_START" ) \
            >"$ROOT/I9/dgpp-start.log" 2>&1
        waited=0
        until curl -sf "http://127.0.0.1:${DGPP_PORT:-8000}/v1/models" >/dev/null 2>&1; do
            if ((waited >= 1800)); then log "DGPP not ready after 30 minutes"; break; fi
            sleep 5
            waited=$((waited + 5))
        done
        "${load[@]}" "${DGPP_PORT:-8000}" "${load_args[@]}" --json-out "$ROOT/I9/dgpp.json" \
            >"$ROOT/I9/dgpp.txt" 2>&1 || log "I9 DGPP load exited nonzero"
        curl -sf "http://127.0.0.1:${DGPP_PORT:-8000}/v1/metrics" >"$ROOT/I9/dgpp-metrics.json" || true
        ( cd "$DGPP_DIR" && eval "$DGPP_STOP" ) >>"$ROOT/I9/dgpp-start.log" 2>&1 || true
        "$PYTHON" "$DGPP_DIR/scripts/bench_compare.py" "$ROOT/I9/dgpp.json" \
            "$ROOT/I9/ninfer-k1.json" >"$ROOT/I9/compare-k1.txt" 2>&1 || true
    else
        echo "skipped: DGPP_DIR, DGPP_START and DGPP_STOP not all set" >"$ROOT/I9/skipped.txt"
    fi
    finish I9
fi

log "Block I finished; results under $ROOT"
