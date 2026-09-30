#!/usr/bin/env bash
# PR #33 verification on GB10 (2026-09-30), per Opus's protocol. Verifies the
# integrated-device startup sizing (7e23083c + 7bfa75fb: budget MemAvailable, the pinned
# host KV and a fixed 6 GiB host reserve, keyed on cudaDeviceProp::integrated) and the
# seal fix (66744eb5), on the adopted fp8mtp artifact.
#
# One GPU job at a time. Phases with a DONE marker are skipped, so an interrupted run
# resumes. Results under profiles/bench/gb10/pr33-verify/.
#
#   S1  warm and cold starts at MC=8, --kv-capacity auto; record the resolved KV capacity
#       and MemAvailable around each start (the warm start must succeed without
#       drop_caches)
#   S2  Flash-Next real / fault / load_plan / frontend ctest (completes Block I's I0)
#   S3  8 concurrent requests on the auto-sized server; major page faults and MemAvailable
#       before, during and after (I7 reference: 666 major faults over 512 tokens)
#   S4  I4 rerun on current master (block_i.sh), then a forced-eviction case: MC=8,
#       --kv-capacity 16384, 4 requests sharing one prefix plus 4 distinct, K=3
#   S6  no GPU: I5 k3 accepted lengths per row; I4 N=1 decode-only rates from the rerun
#
# PHASES=S1,S3 limits the run. Requires the build/ tree of current master.
X925_CORES=${X925_CORES:-5-9,15-19}
if [[ -z ${PINNED:-} ]]; then
    PINNED=1 exec taskset -c "$X925_CORES" "$0" "$@"
fi
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

PHASES=${PHASES:-S1,S2,S3,S4,S6}
ROOT=$OUT_ROOT/pr33-verify
mkdir -p "$ROOT"
want() { case ",$PHASES," in *",$1,"*) [[ ! -f $ROOT/$1/DONE ]] ;; *) return 1 ;; esac; }
begin() { rm -rf "${ROOT:?}/$1"; mkdir -p "$ROOT/$1"; log "phase $1: $2"; }
finish() { touch "$ROOT/$1/DONE"; log "phase $1 done"; }
cleanup() {
    if [[ -n ${SERVER_PID:-} ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        stop_server || true
    fi
}
trap cleanup EXIT
drop_caches() { sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; }
meminfo() { grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo | tr -s ' '; }
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

# ---- S1: warm and cold starts at MC=8, auto KV capacity -----------------------------------
if want S1; then
    begin S1 "warm + cold MC=8 auto-KV starts"
    run_auto_start() { # $1 = warm|cold
        echo
        echo "## $1 start at MC=8, --kv-capacity auto"
        echo "before:"; meminfo
        serve_args 8 3
        start_server "$ROOT/S1/$1.log" --kv-capacity auto \
            --request-log-jsonl "$ROOT/S1/$1.jsonl"
        echo "after (server healthy):"; meminfo
        stop_server
    }
    {
        echo "# S1: warm and cold starts, --kv-capacity auto"
        echo "The warm start reads the page cache and must succeed without drop_caches"
        echo "(that is the point of PR #33)."
        for volume in "$ART" "$ART".part-*; do [[ -f $volume ]] && cat "$volume" >/dev/null; done
        run_auto_start warm
        drop_caches
        run_auto_start cold
    } >"$ROOT/S1/starts.md" 2>&1
    {
        echo "# S1: server_start memory sections (request logs)"
        for tag in warm cold; do
            echo "### $tag"
            "$PYTHON" - "$ROOT/S1/$tag.jsonl" <<'EOF'
import json, sys
with open(sys.argv[1]) as f:
    for line in f:
        ev = json.loads(line)
        if ev.get("event") == "server_start":
            print(json.dumps(ev, indent=1))
            break
EOF
        done
    } >"$ROOT/S1/server-start.md" 2>&1
    finish S1
fi

# ---- S2: Flash-Next ctest (completes Block I's I0) ----------------------------------------
if want S2; then
    begin S2 "Flash-Next ctest (completes I0)"
    for volume in "$ART" "$ART".part-*; do [[ -f $volume ]] && cat "$volume" >/dev/null; done
    NINFER_QWEN38_FLASH_NEXT_WEIGHTS="$ART" ctest --test-dir build \
        -R 'ninfer_qwen3_8_flash_next_(real|load_plan|frontend|fault)_test' \
        --output-on-failure >"$ROOT/S2/ctest.md" 2>&1
    finish S2
fi

# ---- S3: 8 concurrent requests, memory pressure -------------------------------------------
if want S3; then
    begin S3 "8 concurrent, memory pressure"
    {
        echo "before:"; meminfo
        serve_args 8 3
        start_server "$ROOT/S3/serve.log" --kv-capacity auto \
            --request-log-jsonl "$ROOT/S3/request.jsonl"
        base_faults=$(awk '$1 == "pgmajfault" { print $2 }' /proc/vmstat)
        {
            while :; do
                awk -v ts="$(date +%s)" '$1 == "MemAvailable:" { print ts, $2 }' /proc/meminfo
                sleep 2
            done
        } >"$ROOT/S3/memavail.csv" &
        sampler=$!
        "$PYTHON" tools/gb10/concurrency_sweep.py "$BASE_URL" "$ROOT/S3/distinct-k8.json" \
            --n 8 --max-tokens 1024
        kill "$sampler" 2>/dev/null || true
        wait "$sampler" 2>/dev/null || true
        after_faults=$(awk '$1 == "pgmajfault" { print $2 }' /proc/vmstat)
        echo "after:"; meminfo
        stop_server
        echo
        echo "pgmajfault: before=$base_faults after=$after_faults delta=$((after_faults - base_faults))"
        "$PYTHON" - "$ROOT/S3/distinct-k8.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
run = d["runs"][0]
print(f"sweep: {run['n']} requests, {run['failed']} failed, "
      f"{run['completion_tokens']} completion tokens, "
      f"aggregate {run['aggregate_tok_s']:.1f} tok/s, wall {run['wall_seconds']:.1f} s")
EOF
    } >"$ROOT/S3/pressure.md" 2>&1
    {
        echo "# S3: summary"
        awk 'NR == 1 || $2 < min { min = $2 } END { printf "min MemAvailable during run: %.1f GiB over %d samples (~2 s); flag if below ~2 GiB\n", min / 1048576, NR }' "$ROOT/S3/memavail.csv"
        echo "I7 reference: 666 major faults over 512 tokens (~1.3 per token)."
    } >"$ROOT/S3/summary.md" 2>&1
    finish S3
fi

# ---- S4: I4 rerun on current master + forced-eviction case ---------------------------------
if want S4; then
    begin S4 "I4 rerun + forced-eviction case"
    # Part A: I4 on current master. block_i.sh owns the protocol (drop_caches per K server,
    # 4K prompts, 1024 tokens, K=0/2/3, distinct and shared). The sibling marker (not
    # inside $ROOT/S4, which begin wipes) lets a S4 rerun skip the 40-minute I4 rerun;
    # remove $ROOT/S4.partA for a fresh full campaign.
    if [[ ! -f $ROOT/S4.partA ]]; then
        rm -rf "$OUT_ROOT/block-i/I4"
        PINNED=1 PHASES=I4 bash tools/gb10/block_i.sh >"$ROOT/S4/block-i-I4.log" 2>&1
        touch "$ROOT/S4.partA"
    else
        log "S4 part A: I4 rerun already recorded (marker); skipping"
    fi
    # Part B: forced eviction. MC=8, --kv-capacity 16384 with --max-context 16384 (the
    # server requires kv-capacity >= max-context). Eight ~3K-token streams over-subscribe
    # the pool, so eviction happens mid-run. 4 requests share one prefix plus 4 distinct;
    # K=3 (the adopted draft depth). Pass: no errors, and the shared-prefix outputs match
    # the shared prefix run alone.
    {
        SERVE_ARGS=(--max-context 16384 --max-concurrency 8 --kv-dtype "$KV_DTYPE" --preserve-thinking --spec mtp --draft-tokens 3 --lm-head-draft)
        start_server "$ROOT/S4/evict.log" --kv-capacity 16384 \
            --request-log-jsonl "$ROOT/S4/evict.jsonl"
        "$PYTHON" - "$BASE_URL" "$ROOT/S4/evict-result.json" <<'EOF'
import json, sys, threading, time, urllib.request

base_url, out_path = sys.argv[1], sys.argv[2]
MAX_TOKENS = 1024
PROMPT_CHARS = 8000
MANIFEST = "eval/corpora/perplexity-1m/manifest.json"
CONTINUE = "\n\nContinue the text above. Write only the continuation."


def post(payload):
    req = urllib.request.Request(base_url + "/v1/chat/completions",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.load(resp)


with urllib.request.urlopen(base_url + "/v1/models", timeout=60) as resp:
    model = json.load(resp)["data"][0]["id"]
manifest = json.load(open(MANIFEST, encoding="utf-8"))
streams = manifest["streams"]


def text_of(stream):
    with open(f"eval/corpora/perplexity-1m/{stream['path']}", encoding="utf-8") as f:
        return f.read()[:PROMPT_CHARS] + CONTINUE


def payload_for(text):
    return {"model": model, "stream": False, "temperature": 0, "max_tokens": MAX_TOKENS,
            "reasoning_effort": "none", "messages": [{"role": "user", "content": text}]}


shared, distinct = streams[0], streams[1:5]
batch = [(shared, text_of(shared))] * 4 + [(s, text_of(s)) for s in distinct]
results = [None] * len(batch)
barrier = threading.Barrier(len(batch))


def worker(i, stream, text):
    t0 = time.time()
    barrier.wait()
    try:
        resp = post(payload_for(text))
        results[i] = {"stream": stream["id"],
                      "role": "shared" if stream is shared else "distinct",
                      "seconds": time.time() - t0,
                      "completion_tokens": resp["usage"]["completion_tokens"],
                      "finish_reason": resp["choices"][0]["finish_reason"],
                      "text": resp["choices"][0]["message"]["content"]}
    except Exception as error:
        results[i] = {"stream": stream["id"], "seconds": time.time() - t0,
                      "error": str(error)}


threads = [threading.Thread(target=worker, args=(i, s, t)) for i, (s, t) in enumerate(batch)]
for t in threads:
    t.start()
for t in threads:
    t.join()

t0 = time.time()
solo = post(payload_for(text_of(shared)))
solo_seconds = time.time() - t0
solo_text = solo["choices"][0]["message"]["content"]

shared_matches = sum(1 for r in results[:4] if "error" not in r and r["text"] == solo_text)
failed = sum(1 for r in results if "error" in r)
ok = failed == 0 and shared_matches == 4
json.dump({"model": model, "kv_capacity": 16384, "max_tokens": MAX_TOKENS,
           "shared_stream": shared["id"],
           "distinct_streams": [s["id"] for s in distinct],
           "requests": results, "shared_matches": f"{shared_matches}/4",
           "failed": failed, "pass": ok,
           "solo_seconds": solo_seconds}, open(out_path, "w", encoding="utf-8"), indent=1)
print(f"forced eviction: {len(batch)} concurrent (4 shared + 4 distinct), "
      f"failed={failed}, shared matches solo={shared_matches}/4, pass={ok}")
if not ok:
    raise SystemExit(1)
EOF
        stop_server
    } >"$ROOT/S4/evict.md" 2>&1
    finish S4
fi

# ---- S6: no-GPU checks from Block I data ----------------------------------------------------
if want S6; then
    begin S6 "no-GPU checks from Block I data"
    "$PYTHON" - >"$ROOT/S6/i5-accepted-lengths.md" <<'EOF'
import json

d = json.load(open("profiles/bench/gb10/block-i/I5/k3.json"))
print("## I5 k3 accepted length per row (K=3, MC=1)")
print()
print("| context | reps | rounds | drafted | accepted | tokens/round | accept ratio |")
print("|---|---:|---:|---:|---:|---:|---:|")
for t in d["tests"]:
    s = t["reps"][0]["speculative"]
    tpr = 1 + s["accepted_tokens"] / s["rounds"]
    ratio = s["accepted_tokens"] / max(1, s["drafted_tokens"])
    print(f"| {t['label']} | {len(t['reps'])} | {s['rounds']} | {s['drafted_tokens']} | "
          f"{s['accepted_tokens']} | {tpr:.3f} | {ratio:.3f} |")
print()
print("The 1K, 8K and 32K rows continue different text (each ends within the first")
print("~69K-token Wikipedia stream of the corpus), so a 32K acceptance dip could be a")
print("text effect rather than a context-length effect (2026-09-30 review).")
EOF
    {
        echo "## I4 N=1 decode-only rates (rerun on current master, 4K prompt, 1024 tokens)"
        echo
        echo "req 1 of each K request log is the N=1 request (the sweep runs N=1 first)."
        echo
        echo "| K | decode tok/s | tok/round | device wait ms/round |"
        echo "|---:|---:|---:|---:|"
        for k in 0 2 3; do
            awk -F'|' -v k="$k" \
                '$2 ~ /^[[:space:]]*1[[:space:]]*$/ {
                    printf "| %s | %s | %s | %s |\n", k, $8, $10, $11; exit
                }' "profiles/bench/gb10/block-i/I4/request-k$k.md"
        done
        echo
        echo "I2 reference (served, MC=1, natural text, decode-only): 42.0 tok/s at K=1,"
        echo "47.0 at K=3 (2026-09-30 review)."
    } >"$ROOT/S6/i4-n1.md" 2>&1
    finish S6
fi

log "pr33 verify: all requested phases complete"
