#!/usr/bin/env bash
# ncu comparison of the 8-row MoE stage (build-order item 2, Block J): NInfer's NVFP4 w4a4
# MoE against DGPP's moe_w4a4 (moe_slot_*), at the served C8 shape (4 verify + 4 draft rows
# per step at K=1). Block J (nsys) measured the stage at 45.7 ms/step (NInfer) against
# 38.4 (DGPP); this run profiles the kernels directly: per-kernel time, DRAM traffic and
# achieved GB/s against the 246 GB/s GB10 roofline, grid/block shape, stage totals.
#
#   ncu_moe_8row.sh ninfer   start the NInfer C8 K=1 server under ncu, drive 8 concurrent
#                            256-token requests, export the profile
#   ncu_moe_8row.sh dgpp     same against the DGPP server (DGPP_DIR must be a built checkout
#                            with DGPP_START/DGPP_STOP, as in block_i.sh I9)
#
# Output: profiles/ncu/moe-8row/{side}/ {ncu.ncu-rep, raw.csv, kernels.md, smoke json/log}.
# Post-merge only: the NInfer side runs on the merged tree (master + #36).
#
# ncu overhead: each profiled kernel is replayed, so the serve is much slower than normal
# (SKIP+COUNT launches only; the rest of the run is unprofiled). One GPU job at a time.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SIDE=${1:-}
case "$SIDE" in
    ninfer|dgpp) ;;
    *) echo "usage: $0 {ninfer|dgpp}" >&2; exit 2 ;;
esac

ROOT=profiles/ncu/moe-8row/$SIDE
mkdir -p "$ROOT"
gpu_idle() {
    if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]]; then
        echo "GPU is held by another job; single-instance discipline: aborting." >&2
        exit 1
    fi
}
drop_caches() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; }
serve_args() { # $1 = max concurrency, $2 = K (0 = no speculation)
    SERVE_ARGS=(--max-context 73728 --max-concurrency "$1" --kv-dtype "$KV_DTYPE" --preserve-thinking)
    if (($2 > 0)); then SERVE_ARGS+=(--spec mtp --draft-tokens "$2" --lm-head-draft); fi
}
gpu_idle

# Steady-state decode only: skip the first 300 launches (startup graphs, warm, prefill) and
# profile the next 400 (a few hundred decode kernels per stage at 8 rows).
NCU_OPTS=(--target-processes all --launch-skip 300 --launch-count 400
    --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,
    sm__cycles_active.avg.pct_of_peak_sustained_elapsed,launch__grid_size,launch__block_size
    --export "$ROOT/ncu" --force-overwrite=true)

if [[ $SIDE == ninfer ]]; then
    serve_args 8 3
    SERVE_ARGS+=(--lm-head-draft)
    drop_caches
    ncu "${NCU_OPTS[@]}" "$SERVE_BIN" "$ART" --port "$PORT" "${SERVE_ARGS[@]}" \
        >"$ROOT/server.log" 2>&1 &
    server_pid=$!
    local_waited=0
    until curl -sf "$BASE_URL/health" >/dev/null 2>&1; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "server exited during startup" >&2
            tail -n 40 "$ROOT/server.log" >&2
            exit 1
        fi
        sleep 5; local_waited=$((local_waited + 5))
        ((local_waited >= 1800)) && { tail -n 40 "$ROOT/server.log" >&2; exit 1; }
    done
    "$PYTHON" tools/gb10/concurrency_sweep.py "$BASE_URL" "$ROOT/smoke.json" --n 8 \
        --max-tokens 256 >"$ROOT/smoke.txt" 2>&1 || log "smoke exited nonzero"
    # let the profiled launches drain, then stop
    sleep 30
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
else
    [[ -n ${DGPP_DIR:-} && -n ${DGPP_START:-} && -n ${DGPP_STOP:-} ]] || {
        echo "DGPP_DIR, DGPP_START and DGPP_STOP must be set for the dgpp side" >&2; exit 2; }
    drop_caches
    ( cd "$DGPP_DIR" && export DGPP_RESIDENT_CACHE=off \
        && eval "ncu ${NCU_OPTS[*]} $DGPP_START" ) >"$ROOT/server.log" 2>&1
    waited=0
    until curl -sf "http://127.0.0.1:${DGPP_PORT:-8000}/v1/models" >/dev/null 2>&1; do
        sleep 5; waited=$((waited + 5))
        ((waited >= 3600)) && { echo "DGPP not ready after an hour" >&2; break; }
    done
    python3 - "$DGPP_PORT" "$ROOT" <<'EOF'
import json
import sys
import threading
import time
import urllib.request

port, root = sys.argv[1], sys.argv[2]
model = json.load(urllib.request.urlopen(
    f"http://127.0.0.1:{port}/v1/models", timeout=60))["data"][0]["id"]
prompt = open("eval/corpora/perplexity-1m/data/wikitext/00.txt", encoding="utf-8").read()[:8000]
results = [None] * 8

def worker(i):
    payload = {"model": model, "stream": False, "temperature": 0, "max_tokens": 256,
               "messages": [{"role": "user",
                             "content": prompt + f"\n\nContinue, part {i}."}]}
    try:
        resp = json.load(urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}), timeout=3600))
        results[i] = resp["usage"]
    except Exception as error:
        results[i] = {"error": str(error)}

threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
[t.start() for t in threads]
[t.join() for t in threads]
json.dump(results, open(f"{root}/smoke.json", "w"), indent=1)
print(results)
EOF
    ( cd "$DGPP_DIR" && eval "$DGPP_STOP" ) >>"$ROOT/server.log" 2>&1 || true
fi

ncu --import "$ROOT/ncu.ncu-rep" --csv --page raw >"$ROOT/raw.csv" 2>"$ROOT/import.log" \
    || ncu --import "$ROOT/ncu.ncu-rep" --csv >"$ROOT/raw.csv" 2>>"$ROOT/import.log" \
    || { log "ncu import failed; see import.log"; exit 1; }

python3 - "$ROOT" "$SIDE" >"$ROOT/kernels.md" <<'EOF'
import csv
import sys

root, side = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(f"{root}/raw.csv")))
kernel_col = next(c for c in rows[0] if "Name" in c or "name" in c.lower())
dur_col = next(c for c in rows[0] if "gpu__time_duration" in c)
rd_col = next(c for c in rows[0] if "dram__bytes_read" in c)
wr_col = next(c for c in rows[0] if "dram__bytes_write" in c)
sm_col = next(c for c in rows[0] if "sm__cycles_active" in c)

ROOFLINE_GBS = 246.0
moe_names = ("route", "routes", "routed", "grouped", "reduce_decode", "reduce_grouped",
             "w4a4", "gemv_fp4", "moe_slot", "moe_router", "expert")
by_kernel = {}
for r in rows:
    name = r[kernel_col].split("<")[0].strip()
    k = by_kernel.setdefault(name, [0, 0.0, 0.0, 0.0, 0.0])
    k[0] += 1
    k[1] += float(r[dur_col] or 0)
    k[2] += float(r[rd_col] or 0)
    k[3] += float(r[wr_col] or 0)
    k[4] += float(r[sm_col] or 0)

moe_rows = {n: v for n, v in by_kernel.items()
            if any(m in n.lower() for m in moe_names)}
total_ms = sum(v[1] for v in moe_rows.values()) / 1e6
total_bytes = sum(v[2] + v[3] for v in moe_rows.values())
print(f"# {side} — MoE stage, 8-row decode (ncu 2025.3.1, replayed)")
print()
print(f"MoE-stage total: {total_ms:.1f} ms over {sum(v[0] for v in moe_rows.values())} "
      f"profiled launches, {total_bytes / 1e6:.0f} MB DRAM "
      f"({total_bytes / (sum(v[1] for v in moe_rows.values()) or 1):.0f} GB/s "
      f"of {ROOFLINE_GBS:.0f} roofline)")
print()
print("| kernel | launches | ms | MB | GB/s | % of stage |")
print("|---|---:|---:|---:|---:|---:|")
for name, v in sorted(moe_rows.items(), key=lambda kv: -kv[1][1]):
    ms = v[1] / 1e6
    mb = (v[2] + v[3]) / 1e6
    gbs = (v[2] + v[3]) / (v[1] or 1)
    print(f"| `{name}` | {v[0]} | {ms:.2f} | {mb:.1f} | {gbs:.0f} | {100 * v[1] / (sum(v2[1] for v2 in moe_rows.values()) or 1):.1f}% |")
EOF
log "$SIDE ncu done; results under $ROOT"
