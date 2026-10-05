# Operations: running the engine on GB10 nodes

Runbook for operating NInfer (the C++/CUDA inference engine) on a fleet of GB10
(Grace Blackwell) nodes: node setup, build, start, monitor, failure response,
and upgrades. One file for both humans and agents: every procedure leads with
the exact command, and the machine-readable contract at the end restates the
same facts for orchestration.

This is not the user guide. Request/response semantics: [CLI](cli.md),
[HTTP serving](serving.md). Performance methodology and baselines:
[performance](performance/qwen3.8-flash-next-125b-a6b.md). Numerical gates:
[perplexity](perplexity.md).

## 1. Node model

One node runs one engine instance. The node profile for the fleet:

- NVIDIA GB10: 20-core Arm (aarch64), one Blackwell GPU (compute capability
  12.1, build target `sm_121a`), 128 GiB unified LPDDR5x shared by CPU and GPU
  (about 121 GiB visible to the host).
- One resident model: Qwen3.8 Flash-Next 125B-A6B NVFP4 with the FP8 MTP layer
  (the "FP8 MTP" v3 artifact) and the Flash-Next Vision tower. Weights are
  70.5 GiB once resident (70.2 GiB text-only).
- One process owns the GPU. **One device-allocating job per node at a time**
  (serve, benchmark, or test). The step scripts under `tools/gb10/` abort
  (`exit 2`) when `nvidia-smi --query-compute-apps=pid` is non-empty; a
  monitoring daemon that appears only transiently can trip the guard, so a
  spurious "GPU busy" abort is retryable, but never start a second
  device-allocating job to "share" the node.
- Bounded FIFO ingress, no request preemption: at most 8 concurrent requests
  (`--max-concurrency 1..8`, startup-fixed), 16 pending requests, 30 s
  admission timeout.
- No automatic restart: a dead server stays dead until it is started again.

Disk budget per node: artifact 119 GiB (4 files, below), repo plus `build/`
(about 32 GiB), plus logs. Keep the page cache warm for fast restarts;
a cold load reads the artifact (up to 119 GiB across the four volumes)
from local storage.

## 2. Prerequisites

Toolchain (all on the node image): `cmake` ≥ 3.28, `ninja`, the CUDA 13.0
toolkit (`nvcc`), `python3`, `curl`, `nvidia-smi`.

Artifact: the four files plus the conversion report, kept together in one
directory and selected **by explicit path, never by glob or "latest"**:
| File | Size (bytes) |
|---|---:|
| `qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer` (entry) | 32,000,000,000 |
| `qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer.part-0001` | 32,000,000,000 |
| `qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer.part-0002` | 32,000,000,000 |
| `qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer.part-0003` | 30,921,610,496 |
| `qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer.conversion.json` | 103,941 |

The entry is the only path the engine opens; the `.part-NNNN` volumes are
continuations of the same artifact. The conversion report is provenance
(source recipe, verification) and must travel with the artifact. Never start
from a partially copied set; verify the five sizes before the first start.
The production artifact carries the vision component (`vision` in the
component table plus the `vision/*` ViT tensors); the canonical profile
(section 4) requires it. A server started with `--vision` against an artifact
without the component exits at startup with a missing-component error. Check
an artifact before the first start: `cd tools && python3 -m
artifact.inspect --objects --bindings "$ART" | grep -c vision` must be
non-zero (1,224 for the documented generation).


Node configuration: `tools/gb10/config.local.sh` (not committed) supplies
`ART` (artifact path), `PORT` (production: 8000), `KV_DTYPE`, `DRAFT_TOKENS`, `PYTHON`;
environment scalars override the file. The examples below use `$ART` and
`$PORT`.

## 3. Build

Canonical (build plus the full verification suite, about 21 minutes; the
build itself is about 3 minutes):

```bash
tools/gb10/step0_build_test.sh   # exit 0 only if everything below is green
```

Manual equivalent (the flags `step0` uses):

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121a \
  -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON -DNINFER_PERFORMANCE_TRACE=OFF
cmake --build build -j
```

`NINFER_PERFORMANCE_TRACE=OFF` is mandatory for any build you will time; the
step scripts check the cache and refuse a traced build.

Products:

| Binary | Purpose |
|---|---|
| `build/apps/ninfer-serve` | HTTP server (the fleet workhorse) |
| `build/apps/ninfer` | one-shot CLI request (smoke tests) |
| `build/apps/ninfer-perplexity` | fixed-corpus perplexity gate |
| `build/bench/ninfer_bench` | benchmarks |

`NINFER_CUDA_SYNC` needs no action on GB10: the default resolves to `yield`
on integrated devices, which is the measured-best choice there.

## 4. Starting the server

The canonical GB10 production profile (full model context, two lanes, FP8 KV,

```bash
build/apps/ninfer-serve "$ART" \
  --host 127.0.0.1 --port 8000 \
  --max-context 262144 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --vision \
  --preserve-thinking \
  --request-log-jsonl "$LOG_DIR/requests.jsonl" \
  >"$LOG_DIR/server.log" 2>&1 &
```

Startup is a memory check first: if free unified memory cannot hold the load
(weights plus workspace, about 73.7 GiB), the process exits during startup
with `weights exceed free GPU memory`. With a warm page cache the server is
healthy about 23–25 s after launch (measured 24.2 s with vision; log line
`engine ready | ... | weights 70.5 GiB`). Healthy means `GET /health`
returns 200.

```bash
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do sleep 5; done
curl -sf "http://127.0.0.1:$PORT/v1/models"
```

`/v1/models` reports the advertised model alias (the artifact's `metadata.name`,
`qwen3.8-flash-next-125b-a6b`) and the effective `max_model_len` (262,144).

Operational flag subset (defaults in the right column; the full table is in
[serving.md](serving.md#server-options)):

| Option | Effect | Default / GB10 value |
|---|---|---|
| `--max-context N` | per-sequence logical ceiling | 8192 → **262144** (the model's native context) |
| `--kv-capacity N\|auto` | shared KV pool; `auto` maximizes from free memory | `--max-context` (≈147,456 tokens with `auto` on GB10) |
| `--max-concurrency N` | concurrent requests, `1..8`, startup-fixed | 1 → **2** |
| `--max-pending-requests N` | requests allowed to wait for admission | 16 |
| `--pending-timeout-ms N` | max prepare-plus-admission wait | 30000 |
| `--prefill-chunk N` | prefill chunk (multiple of 128) | device default **4096** on GB10; do not override |
| `--long-prefill-wait-ms N` | a multi-chunk prefill waits this long for running requests before admitting anyway; must be below `--pending-timeout-ms` | 20000 |
| `--prefill-decode-share PCT` | decode time owed to running requests per prefill chunk | 50 |
| `--kv-dtype` | KV storage | bf16 → **fp8** |
| `--spec mtp --draft-tokens N --lm-head-draft` | MTP speculation (Flash-Next cap K=3) | off → **mtp, 3** |
| `--shutdown-timeout-seconds N` | admitted requests may finish this long after SIGTERM before being cancelled | 30 |
| `--request-log-jsonl FILE` | full-precision JSON-lines request log | disabled → **on** |
| `--api-key KEY` | required bearer / `x-api-key`; `/health` stays open | unset |
| `--vision` | media input (extra weights and workspace) | off → **on** |
| `--no-thinking` / `--preserve-thinking` | thinking default / closed-turn reasoning | thinking on / off |

Long-prefill policy (the three behaviors above interact; measured on GB10):

- A prompt that needs more than one prefill chunk is **deferred** up to 20 s
  while a running request decodes; the running request keeps decoding at the
  50% share (about 25 tok/s during a 60K prefill; the newcomer's first token
  comes 46% later than solo).
- A short (one-chunk) request **yields** a staged prefill at a chunk boundary
  (first token about 1.7 s) and **backfills** past a deferred long request
  (first token about 1.5 s while the held 60K prompt waits the full 20 s).
- The request log's `context_cache` counters name each event:
  `admission_deferred_long_prefill`, `admission_prefill_yields`,
  `admission_short_backfills`.

Stop:

```bash
kill "$PID"        # SIGTERM: admitted requests drain up to 30 s, then the process exits
wait "$PID"
```

## 5. Monitoring

```bash
# readiness (poll every 5 s during startup; every 30 s in steady state)
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && echo OK

# liveness
pgrep -f 'ninfer-serve .*\.ninfer'

# GPU telemetry
nvidia-smi --query-gpu=utilization.gpu,power.draw,clocks.sm,temperature.gpu --format=csv

# request log: one JSON object per line (server_start, request_done, per-request records)
tail -f "$LOG_DIR/requests.jsonl"
```

- The server log (`$LOG_DIR/server.log`, stderr) carries startup milestones,
  per-request errors, and an aggregate throughput line every 5 s
  (`--log-stats-interval-ms`).
- Each per-request log record carries `context_cache` (admission counters,
  blocked categories `no_feasible_plan`, `no_free_lane`,
  `unsettled_state_fork`, `context_transaction`, KV transfers), the
  `speculative` object (drafts/accepts per position), and
  `request_done.timings_seconds` (`prepare`, `ttft`, `vision`, `prefill`,
  `decode`, `total`).

Operational baselines on GB10 (measured on the 73,728 text profile, MTP K=3,
FP8 KV, greedy — full numbers and method in the [performance doc](performance/qwen3.8-flash-next-125b-a6b.md)):

| Quantity | Value |
|---|---:|
| weights resident | 70.2 GiB |
| startup to healthy (warm page cache) | ≈ 23–25 s |
| prefill rate | ≈ 2,150 tok/s |
| TTFT, 15K / 60K prompt | 7.2 s / 28.3 s |
| decode, solo 1K–8K MTP | 56–57 tok/s |
| decode, natural text, 16 streams | ≈ 49 tok/s |
| decode, script-heavy answer | ≈ 74 tok/s |

Expect requests to time out (HTTP 504 on the OpenAI/Anthropic surface) when
all lanes are busy and the wait exceeds 30 s — under fleet load that is
admission backpressure, not a fault. Check the blocked counters before
escalating.

## 6. Failure modes and response

| Symptom | Evidence | Action |
|---|---|---|
| Startup exit, `weights exceed free GPU memory` | server log tail | Another allocation holds the pool. Find it (`nvidia-smi --query-compute-apps`), stop it, verify the memory is actually released, then start. Never launch a second copy to "make one work". |
| Startup exit, missing Vision component | server log tail | `--vision` selected but the artifact has no `vision` component (check with `tools/artifact/inspect`, section 2). Drop `--vision` or supply the production (vision) artifact. |
| `Something already answers on <url>` / bind failure | tool exit 2 or server log | Stop the old server or change `PORT`. Verify with `/health` and `pgrep` that exactly one server exists. |
| Requests 504 / queue timeouts under load | request log: `pending` waits, `no_free_lane` | Normal at `--max-concurrency` saturation. Raise `--max-concurrency` (≤ 8) or `--kv-capacity` if the reservation math (`no_feasible_plan`) is the bottleneck, then restart. |
| Decode stalls while a long prompt prefills | counters: `deferred_long_prefill`, `prefill_yields`, `short_backfills` | Expected policy behavior (20 s wait, 50% share). Act only if the running request outlives the wait *and* the stall exceeds the newcomer's whole prefill. |
| First load after a reboot is very slow | startup log: materialization time | Cold page cache (up to 119 GiB read from storage). Accept it or issue one warm start after reboots. |
| Node rebooted, NVRM errors in `dmesg` | `dmesg` tail | OOM class: an allocation exceeded free unified memory and the driver tore down. Verify every device process is dead and the memory is released before the next start; start exactly one server. |
| Server dead, unknown cause | server log tail (last 40 lines) | Build/commit mismatch → rebuild and re-verify (section 7). Otherwise restart and watch the startup; if it dies at the memory check, the node is short of free memory. |

Hard rules (the incident history is in the plan doc's ncu section):

- **Pre-launch free-memory check before every device allocation**: a full
  second load needs about 73.7 GiB free unified memory. Any ad-hoc diagnostic
  that reserves large memory is subject to the same check.
- **Never run `ncu` (Nsight Compute) against a large-model process.** Its
  replay allocations on top of a ~70 GiB model exhausted the pool and took the
  node down (driver state corruption, reboot). `nsys` trace-only is the
  profiling tool for real models; profile small targets if `ncu` is needed.
- **Verify a killed run is dead before the next large allocation**: process
  gone, memory released (an orphaned allocation outlives its launcher).
- Stop with SIGTERM (30 s drain). SIGKILL only after the drain, then verify
  release.
- Drop the page cache (`sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`)
  only to force a cold-load measurement; it needs root and stalls the node.

## 7. Verifying a node

Run after a fresh build, after every code or artifact change, and after any
reboot you are unsure about. Steps are sequential; each has a pass criterion.

1. **Build and tests** (about 21 min):
   `tools/gb10/step0_build_test.sh` → exit 0, `summary.md` shows
   `100% tests passed` out of **140** (ctest) and **4** Flash-Next
   real-artifact tests.
2. **Start** (section 4) → `/health` 200 within 60 s warm.
3. **Model advertisement**: `GET /v1/models` → alias
   `qwen3.8-flash-next-125b-a6b`, `max_model_len` 262,144.
4. **Smoke request** (short, greedy; thinking disabled so the 16-token
   budget is all answer):

   ```bash
   curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" \
     -H 'Content-Type: application/json' \
     -d '{"model": "qwen3.8-flash-next-125b-a6b",
          "messages": [{"role": "user", "content": "Return one sentence."}],
          "max_tokens": 16, "temperature": 0,
          "chat_template_kwargs": {"enable_thinking": false}}'
   ```

   Pass: HTTP 200 with content, and a `request_done` line appears in
   `$LOG_DIR/requests.jsonl`.
5. **Vision probe** (required — the canonical profile enables media input):
   a 64×64 solid-red image must be answered `Red`:

   ```bash
   B64=$(python3 -c 'import io, base64; from PIL import Image; b = io.BytesIO(); Image.new("RGB", (64, 64), (255, 0, 0)).save(b, "PNG"); print(base64.b64encode(b.getvalue()).decode())')
   curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" \
     -H 'Content-Type: application/json' \
     -d "{\"model\": \"qwen3.8-flash-next-125b-a6b\",
          \"messages\": [{\"role\": \"user\", \"content\": [
            {\"type\": \"text\", \"text\": \"What single color is this image? Reply with one word.\"},
            {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,$B64\"}}
          ]}],
          \"max_tokens\": 200,
          \"chat_template_kwargs\": {\"enable_thinking\": false}}'"
   ```

   Pass: the response content is `Red` (measured 2026-10-05, GB10).
6. **Max-context probe** (required — the ceiling is the model's native
   262,144, not the old 73,728 test ceiling): a ~160K-token prompt must be
   admitted (about 70 s of prefill at the measured rate):

   ```bash
   P=$(python3 -c 'print(("The lighthouse keeper logged the weather each morning before the fog lifted, noting wind, swell, and the state of the two lamp wicks in the same slim ledger. ") * 4300, end="")')
   curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" \
     -H 'Content-Type: application/json' \
     -d "{\"model\": \"qwen3.8-flash-next-125b-a6b\",
          \"messages\": [{\"role\": \"user\", \"content\": \"$P What does the keeper do each morning? Reply in one short sentence.\"}],
          \"max_tokens\": 16, \"temperature\": 0,
          \"chat_template_kwargs\": {\"enable_thinking\": false}}'"
   ```

   Pass: HTTP 200 with a short answer (the same prompt was rejected at the old
   ceiling).
7. **Numerical gate** (only when a change could alter numerics — kernels,
   quantization, state): the fixed-corpus perplexity protocol from
   [perplexity.md](perplexity.md) against the previous-build baseline before
   trusting the outputs.

## 8. Upgrades

Code (new commit on the shared branch):

```bash
git pull                          # record the commit: git rev-parse --short HEAD
tools/gb10/step0_build_test.sh    # section 7, steps 1 + 5 as applicable
kill "$OLD_PID"                   # SIGTERM, drain up to 30 s
# start per section 4, then steps 2-4
```

The old `build/` directory is the rollback: if the new server misbehaves,
stop it and start the previous binary (kept as `build/apps/ninfer-serve.old`)
or re-checkout the previous commit and rebuild. Artifacts are immutable; an
artifact change is its own operation: verify the five files and sizes (section
2), point `ART` at the new directory, run the real-artifact test
(`NINFER_QWEN38_FLASH_NEXT_WEIGHTS="$ART" ctest --test-dir build -R
'ninfer_qwen3_8_flash_next_(real|load_plan|frontend|fault)_test'`), restart.

## 9. Fleet operations

The node is self-contained; an orchestrator needs only the node contract:

- **Start** is idempotent: probe `/health` first; a live healthy server is the
  "started" state.
- **State machine**: absent → building → built → starting (healthy within
  60 s warm) → serving → draining (≤ 30 s) → absent. Any step that fails stays
  where it is; the orchestrator retries the step, not a fresh node.
- **Requests** go only through `127.0.0.1:$PORT` (`/v1/*`, Anthropic
  routes); routing between nodes is external to the engine.
- **Collection on incident**: `git rev-parse --short HEAD` at the repo, the
  last 200 server-log lines, the request JSONL since startup, one
  `nvidia-smi` snapshot, and the `dmesg` tail. That set is enough for every
  failure mode in section 6.

Node readiness checklist (agent-executable, in order):

```bash
set -e
: "${PORT:=8000}"          # production port
cd "$NINFER_ROOT"
test -f "$ART"                                     # 1. artifact entry present
test -x build/apps/ninfer-serve                    # 2. built
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null \
  || ( build/apps/ninfer-serve "$ART" --host 127.0.0.1 --port "$PORT" \
        --max-context 262144 --max-concurrency 2 --kv-dtype fp8 \
        --spec mtp --draft-tokens 3 --lm-head-draft --vision --preserve-thinking \
        --request-log-jsonl "$LOG_DIR/requests.jsonl" \
        >"$LOG_DIR/server.log" 2>&1 & echo $! >"$LOG_DIR/serve.pid" ; \
       for i in $(seq 1 12); do sleep 5; curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; done )
curl -sf "http://127.0.0.1:$PORT/v1/models"        # 3. model advertised
curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model": "qwen3.8-flash-next-125b-a6b", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": false}}'
B64=$(python3 -c 'import io, base64; from PIL import Image; b = io.BytesIO(); Image.new("RGB", (64, 64), (255, 0, 0)).save(b, "PNG"); print(base64.b64encode(b.getvalue()).decode())')
curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\": \"qwen3.8-flash-next-125b-a6b\", \"messages\": [{\"role\": \"user\", \"content\": [
       {\"type\": \"text\", \"text\": \"What single color is this image? Reply with one word.\"},
       {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,$B64\"}}]}],
       \"max_tokens\": 200, \"chat_template_kwargs\": {\"enable_thinking\": false}}" \
  | python3 -c 'import json, sys; a = json.load(sys.stdin)["choices"][0]["message"]["content"].strip().lower(); assert a == "red", a'   # 4.5 vision probe: 'Red'
```

Artifact distribution between nodes: copy all five files (section 2) as one
atomic set (for example a tar of the directory), verify the five sizes on the
receiving node before the first start, and keep the conversion report
alongside.

## 10. Machine-readable contract

For orchestrators and agents. Same facts as above; commands assume
`$NINFER_ROOT` (repo), `$ART` (artifact entry), `$PORT`, `$LOG_DIR`.

```yaml
node:
  platform: GB10 (aarch64, sm_121a, CUDA 13.0)
  unified_memory_gib: 128        # ~121 visible to the host
  single_device_job: true        # one serve/bench/test process at a time
  auto_restart: false
artifact:
  files:
    - {name: qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer,              bytes: 32000000000}
    - {name: qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer.part-0001,    bytes: 32000000000}
    - {name: qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer.part-0002,    bytes: 32000000000}
    - {name: qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer.part-0003,    bytes: 30921610496}
    - {name: qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer.conversion.json, bytes: 103941}
  select: explicit entry path only (never glob or latest)
  vision: component required by the canonical profile (verify before start, section 2)
  weights_resident_gib: 70.5        # with vision (70.2 text-only)
build:
  verify: tools/gb10/step0_build_test.sh
  pass: {exit: 0, ctest: "100% out of 140", real_artifact_tests: "4/4"}
serve:
  binary: build/apps/ninfer-serve
  canonical_flags: [--max-context, "262144", --max-concurrency, "2", --kv-dtype, fp8,
                    --spec, mtp, --draft-tokens, "3", --lm-head-draft, --vision,
                    --preserve-thinking, --request-log-jsonl, "$LOG_DIR/requests.jsonl"]
  health: {url: "http://127.0.0.1:$PORT/health", ok: 200}
  model: {alias: qwen3.8-flash-next-125b-a6b, max_model_len: 262144}
  ready_seconds_warm: 25
  stop: {signal: SIGTERM, drain_seconds: 30}
limits:
  concurrency: {min: 1, max: 8, fleet: 2}
  pending: {requests: 16, timeout_ms: 30000}
  long_prefill_wait_ms: 20000
  prefill_decode_share_pct: 50
memory:
  single_load_gib: 73.7
  prelaunch_free_memory_check: required
  ncu_on_large_models: forbidden      # nsys trace-only instead
monitor:
  liveness: pgrep -f 'ninfer-serve .*\.ninfer'
  gpu: nvidia-smi --query-gpu=utilization.gpu,power.draw,clocks.sm,temperature.gpu --format=csv
  request_log: $LOG_DIR/requests.jsonl
  counters: [admission_deferred_long_prefill, admission_prefill_yields, admission_short_backfills,
             admission_blocked.no_feasible_plan, admission_blocked.no_free_lane,
             admission_blocked.unsettled_state_fork, admission_blocked.context_transaction]
baselines_gb10:   # measured on the 73728 text profile, MTP K=3, FP8 KV, greedy
  prefill_tok_s: 2150
  ttft_s: {15k: 7.2, 60k: 28.3}
  decode_tok_s: {solo_1k_8k: 57, natural_16_stream: 49, script_heavy: 74}
```
