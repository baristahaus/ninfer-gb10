#!/usr/bin/env bash
# Step 7a quality-gate baseline. For the named artifact, measure:
#   1. fixed-token perplexity scores (full corpus, default 4096/2048 protocol,
#      fp8 KV, --token-scores), then the same at 65536/32768 for the drift
#      comparison (PPL_LONG=0 skips it);
#   2. MTP acceptance from a serving run (16 corpus streams, greedy, 1024
#      generated tokens each; counters from the server request log);
#   3. TEB hard-mode tool-call score (--hardmode --seed 42; single trial by
#      default, the revised gate; TEB_TRIALS overrides for escalation).
# The plan's 7a/7b/7c artifacts must reproduce these within the stated tolerances.
#
# Usage: ART=/path/to/artifact.ninfer [LABEL=name] tools/gb10/step7a_baseline.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# GB10 is heterogeneous: 10x Cortex-X925 (cpus 5-9,15-19; 3.9 GHz,
# cpu_capacity ~1000) plus 10x A725 (~2.8 GHz, ~720). Pin the whole campaign to
# the X925 half (plan section 8; +2 tok/s measured on vcruz305's box).
# Override with X925_CORES=...; PINNED=1 guards the re-exec.
if [[ -z ${PINNED:-} ]]; then
    X925_CORES=${X925_CORES:-5-9,15-19}
    log "pinning campaign to X925 cores: $X925_CORES"
    PINNED=1 exec env taskset -c "$X925_CORES" "$0" "$@"
fi


LABEL=${LABEL:-$(basename "$(dirname "$ART")")}
PPL_KV=${PPL_KV:-$KV_DTYPE}
if [[ ${RESUME:-0} == 1 ]]; then
    DIR=$OUT_ROOT/step7a-$LABEL
    if [[ ! -f $DIR/perplexity/report.json ]]; then
        echo "RESUME=1 but $DIR/perplexity/report.json is missing; run without RESUME." >&2
        exit 2
    fi
else
    DIR=$(step_dir "step7a-$LABEL")
fi
log "label: $LABEL"
log "dir: $DIR"

require_binary build/apps/ninfer-perplexity
require_binary "$SERVE_BIN"
require_untraced_build

# GB10 unified memory: the engine startup check counts only physical free RAM,
# and each weights-loading phase (perplexity, server) streams the full ~77 GiB
# from NVMe into the page cache. Start each phase cold so the startup check
# sees enough physical free (reload cost ~25 s at ~5 GiB/s; a no-op when the
# cache is already cold).
drop_caches() {
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
}

if [[ ${RESUME:-0} != 1 ]]; then
    # ---- 1. Perplexity token scores ------------------------------------------
    PPL_DIR=$DIR/perplexity
    mkdir -p "$PPL_DIR"
    log "phase 1: perplexity token scores (kv-dtype $PPL_KV, full corpus)"
    log "dropping page cache before phase 1"
    drop_caches
    ./build/apps/ninfer-perplexity "$ART" \
        --corpus eval/corpora/perplexity-1m/manifest.json \
        --kv-dtype "$PPL_KV" --token-scores --output "$PPL_DIR" \
        >"$DIR/perplexity-stdout.log" 2>"$DIR/perplexity-stderr.log"
    log "phase 1 done: $PPL_DIR"

    # ---- 1b. Long-window token scores for the drift comparison ---------------
    # Every window starts from empty state, so drift through the FP32 GDN state
    # can only show within one window: 4096-token windows cannot reveal it.
    # tools/bench/compare_token_drift.py compares two artifacts' runs.
    if [[ ${PPL_LONG:-1} == 1 ]]; then
        PPL_LONG_DIR=$DIR/perplexity-64k
        mkdir -p "$PPL_LONG_DIR"
        log "phase 1b: perplexity token scores, 65536/32768 windows"
        ./build/apps/ninfer-perplexity "$ART" \
            --corpus eval/corpora/perplexity-1m/manifest.json \
            --context 65536 --stride 32768 \
            --kv-dtype "$PPL_KV" --token-scores --output "$PPL_LONG_DIR" \
            >"$DIR/perplexity-64k-stdout.log" 2>"$DIR/perplexity-64k-stderr.log"
        log "phase 1b done: $PPL_LONG_DIR"
    fi
fi

# ---- 2. Server: MTP acceptance serving run -----------------------------------
mkdir -p "$DIR/serve"
REQ_JSONL=$DIR/serve/request.jsonl

# Phase 1 streams the weights from NVMe, leaving them in the page cache; drop
# again before the server's startup check (see drop_caches).
log "dropping page cache before server start"
drop_caches


start_server "$DIR/server.log" --request-log-jsonl "$REQ_JSONL"
log "phase 2: MTP acceptance serving run"
"$PYTHON" tools/gb10/step7a_acceptance.py "$BASE_URL" "$REQ_JSONL" "$DIR" \
    >"$DIR/acceptance.log" 2>&1
log "phase 2 done: $DIR/acceptance.json"

# ---- 3. TEB hardmode ----------------------------------------------------------
# Quality-gate protocol (revised 2026-09-27): fixed-token scores are the primary
# gate; one TEB hard-mode trial is the structural smoke test. Escalate to more
# trials only when the single trial lands more than ~2 points below the BF16
# artifact's single trial (90/100 on 2026-09-28; one-trial scatter ~±0.9).
TEB_TRIALS=${TEB_TRIALS:-1}
log "phase 3: TEB hardmode (--trials $TEB_TRIALS --seed 42; 92 scenarios)"
command -v tool-eval-bench >/dev/null
tool-eval-bench run --base-url "$BASE_URL" --backend ninfer --seed 42 \
    --hardmode --trials "$TEB_TRIALS" --no-live --json \
    --output-dir "$DIR/teb" >"$DIR/teb.log" 2>&1
log "phase 3 done: $DIR/teb"

stop_server

# ---- Summary ------------------------------------------------------------------
{
    echo "# Step 7a baseline: $LABEL"
    echo
    machine_summary
    echo
    echo "- Artifact: \`$ART\`"
    echo "- Serve args: ${SERVE_ARGS[*]}"
    echo "- KV dtype (perplexity): $PPL_KV"
    echo "- CPU: pinned to X925 cores ${X925_CORES:-5-9,15-19} (taskset)"

} >"$DIR/summary.md"
"$PYTHON" tools/gb10/step7a_summary.py "$DIR" >>"$DIR/summary.md"
log "summary: $DIR/summary.md"
cat "$DIR/summary.md"
