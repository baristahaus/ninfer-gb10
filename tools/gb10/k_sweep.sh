#!/usr/bin/env bash
# MTP acceptance and tok/s sweep at --max-concurrency 1 (the step 7a phase 2 protocol: 16 corpus
# streams, greedy, 1024 generated tokens each; counters from the server request log).
#
#   SWEEP="label=/abs/artifact.ninfer ..."  artifacts to compare (required, explicit paths)
#   KS="0 1 2 3"                            draft depths; K=0 runs without speculation
#   REPEATS=1                               independent server runs per (artifact, K)
#   LOOKUP=1                                serve with --prompt-lookup (labels gain -lookup)
#   NAME=ksweep                             output prefix: profiles/bench/gb10/step7a-NAME-...
#
# One server at a time (the GPU is single-instance); aborts if another job holds the GPU.
# A run whose acceptance.json already exists is skipped, so an interrupted sweep resumes.
# Ends with one table over every run of this NAME.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

: "${SWEEP:?SWEEP must list label=/abs/artifact.ninfer pairs}"
KS=${KS:-0 1 2 3}
REPEATS=${REPEATS:-1}
NAME=${NAME:-ksweep}
LOOKUP=${LOOKUP:-0}

if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]]; then
    echo "GPU is held by another job; single-instance discipline: aborting." >&2
    exit 1
fi

CFGDIR=$(mktemp -d)
trap 'rm -rf "$CFGDIR"; cleanup_background' EXIT

run_one() { # $1=K $2=ART $3=LABEL
    local k=$1 art=$2 label=$3 spec=""
    if [[ -f $OUT_ROOT/step7a-$label/acceptance.json ]]; then
        log "skip $label (acceptance.json exists)"
        return 0
    fi
    [[ $k == 0 ]] || spec="--spec mtp --draft-tokens $k --lm-head-draft"
    [[ $k == 0 || $LOOKUP == 0 ]] || spec+=" --prompt-lookup"
    local cfg="$CFGDIR/$label.sh"
    cat >"$cfg" <<CFG
PORT=$PORT
KV_DTYPE=$KV_DTYPE
DRAFT_TOKENS=$k
SERVE_ARGS=(--max-context 73728 --max-concurrency 1 --kv-dtype $KV_DTYPE $spec --preserve-thinking)
PYTHON=$PYTHON
CFG
    GB10_CONFIG=$cfg ART=$art LABEL=$label PHASES=2 tools/gb10/step7a_baseline.sh
}

for pair in $SWEEP; do
    label=${pair%%=*} art=${pair#*=}
    [[ -f $art ]] || { echo "SWEEP artifact is not a file: $art" >&2; exit 2; }
    for k in $KS; do
        for ((r = 1; r <= REPEATS; ++r)); do
            run_one "$k" "$art" "$NAME-$label-k$k$([[ $LOOKUP == 0 ]] || echo -lookup)-r$r"
        done
    done
done

"$PYTHON" - "$OUT_ROOT" "$NAME" <<'PY'
import json, pathlib, sys

root, name = pathlib.Path(sys.argv[1]), sys.argv[2]
print("| run | acceptance | drafted | accepted | tok/s (sum over streams) |")
print("|---|---:|---:|---:|---:|")
for d in sorted(root.glob(f"step7a-{name}-*")):
    p = d / "acceptance.json"
    if not p.is_file():
        continue
    a = json.loads(p.read_text())
    toks = sum(s["generated_tokens"] for s in a.get("streams", []))
    secs = sum(s["seconds"] for s in a.get("streams", []))
    ratio = a.get("acceptance_ratio")
    print(f"| {d.name.removeprefix('step7a-')} | {ratio if ratio is None else f'{ratio:.4f}'} | "
          f"{a.get('drafted_tokens')} | {a.get('accepted_tokens')} | "
          f"{toks / secs if secs else float('nan'):.1f} |")
PY
