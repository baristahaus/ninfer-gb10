#!/usr/bin/env bash
# PR #36 verification (d1efb24f: QSA dense in-budget route and the kDenseQsa graph class).
#
#   V0  build, test_flash_next_qsa, Flash-Next real-artifact tests
#   V1  server matrix MC=1/4/8 x K=0/3: cold start, MC-wide smoke generation; the new graph
#       class must not raise DifferentTopology and every request must complete
#   V2  4K perplexity: must stay within the 1.3e-4 run-to-run floor of 3.9982 (short prefill
#       tails now take the dense route; long windows are unchanged)
#   V3  the I9 C1/C4 rows rerun (K=1/2/3, the DGPP serve_load protocol, five-class phase
#       average) against the recorded I9 rows: C1 42.0/48.2/50.3, C4 81.2/84.3/82.6
#   V4  the step 3 attribution refresh at K=3 (the QSA share; recorded I7: 8.8%)
#
# Settings (environment, besides tools/gb10/config.local.sh):
#   PHASES    comma list, default V0,V1,V2,V3,V4
#   DGPP_DIR  a HawkBearPig/dgpp checkout (V3 needs its scripts/serve_load.py)
# A phase whose DONE marker exists is skipped, so an interrupted campaign resumes. One GPU job
# at a time throughout; phases drop the page cache where a cold start is part of the protocol.
X925_CORES=${X925_CORES:-5-9,15-19}
if [[ -z ${PINNED:-} ]]; then
    PINNED=1 exec taskset -c "$X925_CORES" "$0" "$@"
fi
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

PHASES=${PHASES:-V0,V1,V2,V3,V4}
ROOT=$OUT_ROOT/pr36-verify
mkdir -p "$ROOT"
want() { case ",$PHASES," in *",$1,"*) [[ ! -f $ROOT/$1/DONE ]] ;; *) return 1 ;; esac; }
begin() { rm -rf "${ROOT:?}/$1"; mkdir -p "$ROOT/$1"; log "phase $1: $2"; }
finish() { touch "$ROOT/$1/DONE"; log "phase $1 done"; }
drop_caches() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; }
gpu_idle() {
    if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]]; then
        echo "GPU is held by another job; single-instance discipline: aborting." >&2
        exit 1
    fi
}
serve_args() { # $1 = max concurrency, $2 = K (0 = no speculation)
    SERVE_ARGS=(--max-context 73728 --max-concurrency "$1" --kv-dtype "$KV_DTYPE" --preserve-thinking)
    if (($2 > 0)); then SERVE_ARGS+=(--spec mtp --draft-tokens "$2" --lm-head-draft); fi
}

gpu_idle
log "pr36-verify on $(basename "$ART"), phases $PHASES, tree $(git log -1 --format=%h)"

# ---- V0: the tree builds, the QSA op test and the Flash-Next real tests pass ---------------
if want V0; then
    begin V0 "build, test_flash_next_qsa, Flash-Next real tests"
    cmake --build build -j >"$ROOT/V0/build.log" 2>&1 || {
        log "V0 build failed; aborting the campaign"
        tail -n 40 "$ROOT/V0/build.log" >&2
        exit 1
    }
    ctest --test-dir build -R ninfer_flash_next_qsa_test --output-on-failure \
        >"$ROOT/V0/ctest-qsa.log" 2>&1 || qsa_rc=$?
    NINFER_QWEN38_FLASH_NEXT_WEIGHTS="$ART" ctest --test-dir build \
        -R 'ninfer_qwen3_8_flash_next_(real|load_plan|frontend|fault)_test' --output-on-failure \
        >"$ROOT/V0/ctest-real.log" 2>&1 || real_rc=$?
    {
        echo "## V0 — build and tests (PR #36, $(git log -1 --format=%h))"
        echo
        echo "test_flash_next_qsa (exit ${qsa_rc:-0}):"
        echo
        tail -n 4 "$ROOT/V0/ctest-qsa.log"
        echo
        echo "Flash-Next real tests (exit ${real_rc:-0}):"
        echo
        tail -n 4 "$ROOT/V0/ctest-real.log"
    } >"$ROOT/V0/summary.md"
    if ((${qsa_rc:-0} != 0 || ${real_rc:-0} != 0)); then
        log "V0 tests failed; aborting the campaign (logs under $ROOT/V0)"
        exit 1
    fi
    finish V0
fi

# ---- V1: the new graph class across the MC x K matrix --------------------------------------
if want V1; then
    begin V1 "server matrix MC=1/4/8 x K=0/3"
    for mc in 1 4 8; do
        for k in 0 3; do
            serve_args "$mc" "$k"
            drop_caches
            start_server "$ROOT/V1/server-mc${mc}-k${k}.log"
            "$PYTHON" tools/gb10/concurrency_sweep.py "$BASE_URL" \
                "$ROOT/V1/sweep-mc${mc}-k${k}.json" --n "$mc" --max-tokens 256 \
                >"$ROOT/V1/sweep-mc${mc}-k${k}.txt" 2>&1 \
                || log "V1 mc=$mc k=$k sweep exited nonzero"
            stop_server
        done
    done
    "$PYTHON" - "$ROOT/V1" >"$ROOT/V1/matrix.md" <<'EOF'
import glob
import json
import re
import sys

root = sys.argv[1]
rows = []
for f in sorted(glob.glob(root + "/sweep-mc*-k*.json")):
    mc, k = re.search(r"mc(\d+)-k(\d+)", f).groups()
    run = json.load(open(f))["runs"][0]
    errors = [r for r in run["requests"] if "error" in r]
    topo = open(f.replace("sweep", "server")).read().count("DifferentTopology")
    rows.append((int(mc), int(k), run, errors, topo))
rows.sort(key=lambda r: (r[0], r[1]))
print("| MC | K | smoke (MC x 256 tok) | requests ok | DifferentTopology |")
print("|---:|---:|---|---:|---:|")
for mc, k, run, errors, topo in rows:
    ok = run["n"] - len(errors)
    detail = f"{run['aggregate_tok_s']:.1f} tok/s, {run['completion_tokens']} tok"
    if errors:
        detail += "; " + "; ".join(e["error"][:80] for e in errors[:2])
    print(f"| {mc} | {k} | {detail} | {ok}/{run['n']} | {topo} |")
fail = any(r[4] != 0 or r[3] for r in rows)
print()
print("V1 " + ("FAIL: DifferentTopology raised or request errors above" if fail
               else "PASS: six cold starts, zero DifferentTopology, zero request errors"))
EOF
    finish V1
fi

# ---- V2: 4K perplexity against the 1.3e-4 run-to-run floor ----------------------------------
if want V2; then
    begin V2 "4K perplexity"
    drop_caches
    ./build/apps/ninfer-perplexity "$ART" --corpus eval/corpora/perplexity-1m/manifest.json \
        --kv-dtype "$KV_DTYPE" --output "$ROOT/V2/perplexity" \
        >"$ROOT/V2/perplexity.log" 2>&1 || ppl_rc=$?
    {
        echo "## V2 — 4K perplexity (target 3.9982, run-to-run floor 1.3e-4, recorded 3.998124)"
        echo
        echo "exit ${ppl_rc:-0}"
        echo
        "$PYTHON" - "$ROOT/V2/perplexity/report.json" <<'EOF'
import json
import math
import sys

try:
    d = json.load(open(sys.argv[1]))
except Exception as error:  # report missing: the log tail below is the evidence
    print(f"report.json unreadable: {error}")
    raise SystemExit
overall = d["overall"]
ppl = overall["perplexity"]
delta = ppl - 3.9982
within = abs(delta) <= 1.3e-4
print(f"overall PPL {ppl:.6f} (mean NLL {overall['mean_nll']:.6f} over "
      f"{overall['scored_tokens']:,} tokens); delta vs 3.9982 = {delta:+.2e} "
      f"({ 'within' if within else 'OUTSIDE' } the 1.3e-4 floor)")
for domain in d["domains"]:
    print(f"  {domain['domain']:>20s}  {domain['perplexity']:.4f}")
EOF
        echo
        echo "log tail:"
        echo
        tail -n 6 "$ROOT/V2/perplexity.log"
    } >"$ROOT/V2/summary.md" 2>&1
    finish V2
fi

# ---- V3: the I9 C1/C4 rows rerun ------------------------------------------------------------
if want V3; then
    begin V3 "I9 C1/C4 rows rerun"
    if [[ -n ${DGPP_DIR:-} && -f "$DGPP_DIR/scripts/serve_load.py" ]]; then
        load=("$PYTHON" "$DGPP_DIR/scripts/serve_load.py" 127.0.0.1)
        load_args=(--concurrency 1,4 --classes prose,code,json,math,chat --max-tokens 256
            --repeat 3)
        for k in 1 2 3; do
            serve_args 4 "$k"
            drop_caches
            start_server "$ROOT/V3/ninfer-k$k.log" --request-log-jsonl "$ROOT/V3/ninfer-k$k.jsonl"
            "${load[@]}" "$PORT" "${load_args[@]}" --json-out "$ROOT/V3/ninfer-k$k.json" \
                >"$ROOT/V3/ninfer-k$k.txt" 2>&1 \
                || log "V3 K=$k load exited nonzero"
            stop_server
        done
        "$PYTHON" - "$ROOT/V3" >"$ROOT/V3/rows.md" <<'EOF'
import json
import statistics
import sys

root = sys.argv[1]
recorded = {1: {1: 42.0, 4: 81.2}, 2: {1: 48.2, 4: 84.3}, 3: {1: 50.3, 4: 82.6}}
print("| K | C1 recorded | C4 recorded | C1 rerun | C4 rerun |")
print("|---:|---:|---:|---:|---:|")
for k in (1, 2, 3):
    d = json.load(open(f"{root}/ninfer-k{k}.json"))
    rows = []
    for conc in (1, 4):
        rates = [p["metrics"]["wall_tokens_per_s"]
                 for p in d["phases"] if p["concurrency"] == conc]
        rows.append(f"{statistics.mean(rates):.1f}")
    print(f"| {k} | {recorded[k][1]:.1f} | {recorded[k][4]:.1f} | {rows[0]} | {rows[1]} |")
print()
print("Phase average of wall tok/s over the five classes x three repetitions, the I9 row "
      "aggregation. Per class:")
for k in (1, 2, 3):
    d = json.load(open(f"{root}/ninfer-k{k}.json"))
    for conc in (1, 4):
        per = {}
        for p in d["phases"]:
            if p["concurrency"] != conc:
                continue
            per.setdefault(p["class"], []).append(p["metrics"]["wall_tokens_per_s"])
        cells = "  ".join(f"{c} {statistics.mean(v):.1f}"
                          for c, v in sorted(per.items()))
        print(f"K={k} C{conc}: {cells}")
EOF
    else
        echo "skipped: DGPP_DIR not set or scripts/serve_load.py missing" >"$ROOT/V3/skipped.txt"
    fi
    finish V3
fi

# ---- V4: the step 3 attribution refresh at K=3 ---------------------------------------------
if want V4; then
    begin V4 "step 3 attribution at K=3"
    drop_caches
    DRAFT_TOKENS=3 tools/gb10/step3_attribution.sh >"$ROOT/V4/step3.log" 2>&1 \
        || log "V4 step3 exited nonzero; see step3.log"
    cp "$OUT_ROOT/step3/summary.md" "$ROOT/V4/step3-summary.md" 2>/dev/null || true
    {
        echo "## V4 — step 3 attribution at K=3 (recorded I7: qsa.select 8.8% of decode-verify)"
        echo
        if [[ -f $ROOT/V4/step3-summary.md ]]; then
            # The mtp3 section of the step 3 summary: the K=3 stage shares.
            awk '/^### mtp3/{on=1; print; next} /^### /{on=0} on' "$ROOT/V4/step3-summary.md"
        else
            echo "summary missing; step3.log tail:"
            tail -n 30 "$ROOT/V4/step3.log"
        fi
    } >"$ROOT/V4/summary.md" 2>&1
    finish V4
fi
log "pr36-verify finished (phases $PHASES); results under $ROOT"
