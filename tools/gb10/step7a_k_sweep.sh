#!/usr/bin/env bash
# Step 7a K-sweep acceptance comparison (gate #26, plan 2026-09-gb10).
#
# Measures MTP acceptance and tok/s at --max-concurrency 1 for
# K = 0..3 draft tokens (K=0: no speculation) on the new fp8_mtp artifact
# and the fp8_projections baseline. Same 16 corpus prompts and the same
# greedy protocol as step 7a phase 2. K=0 is artifact-independent
# (identical main-model tensors; the MTP side is never executed) and runs
# once, on the new artifact.
#
# Sequential by construction: one server at a time (single-instance GPU).
# A pre-launch nvidia-smi guard aborts if another job holds the GPU.
# Re-running skips labels whose acceptance.json already exists (resume).
#
# Usage: tools/gb10/step7a_k_sweep.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

NEW_ART=/home/apollo11/ninfer-gb10/out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer
PROJ_ART=/home/apollo11/models/fp8/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_projections.ninfer

if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]]; then
    echo "GPU is held by another job; single-instance discipline: aborting." >&2
    exit 1
fi

CFGDIR=$(mktemp -d /tmp/kcfg.XXXXXX)
trap 'rm -rf "$CFGDIR"' EXIT

run_k() { # $1=K $2=ART $3=LABEL
    local k=$1 art=$2 label=$3
    local dir="$OUT_ROOT/step7a-$label" spec
    if [[ -f $dir/acceptance.json ]]; then
        log "skip $label (acceptance.json exists)"
        return 0
    fi
    if [[ $k == 0 ]]; then spec=""; else spec="--spec mtp --draft-tokens $k --lm-head-draft"; fi
    local cfg="$CFGDIR/k${k}.sh"
    cat >"$cfg" <<EOF
PORT=$PORT
KV_DTYPE=$KV_DTYPE
DRAFT_TOKENS=$k
SERVE_ARGS=(
  --max-context 73728
  --max-concurrency 1
  --kv-dtype $KV_DTYPE
  $spec
  --preserve-thinking
)
PYTHON=$PYTHON
RUN_SERVING=0
TOKENIZER=
EOF
    GB10_CONFIG=$cfg ART=$art LABEL=$label PHASES=2 tools/gb10/step7a_baseline.sh
}

run_k 0 "$NEW_ART"  acc-k0-new
run_k 1 "$NEW_ART"  acc-k1-new
run_k 2 "$NEW_ART"  acc-k2-new
run_k 3 "$NEW_ART"  acc-k3-new
run_k 1 "$PROJ_ART" acc-k1-proj
run_k 2 "$PROJ_ART" acc-k2-proj
run_k 3 "$PROJ_ART" acc-k3-proj

log "K-sweep done; acceptance summaries:"
"$PYTHON" - <<'EOF'
import json, pathlib

root = pathlib.Path("profiles/bench/gb10")
for d in sorted(root.glob("step7a-acc-*")):
    p = d / "acceptance.json"
    if not p.is_file():
        continue
    a = json.loads(p.read_text())
    streams = a.get("streams", [])
    toks = sum(s["generated_tokens"] for s in streams)
    secs = sum(s["seconds"] for s in streams)
    rate = f"{toks / secs:.1f}" if secs else "-"
    print(f"{d.name}: ratio={a.get('acceptance_ratio')} "
          f"drafted={a.get('drafted_tokens')} accepted={a.get('accepted_tokens')} "
          f"fallback={a.get('fallback_steps')} "
          f"per_pos={a.get('accepted_per_position_per_round')} tok/s(sum)={rate}")
EOF
