# K4 verify - 2026-10-01

Trees: K4a = `3d3d7a25` + K3 fix `6d851f69` (detached gate build); K4b = branch tip
`57b0c264` (`f86d582a` K4b + `6d851f69` fix + docs). Host: GB10, driver 580.173.02,
CUDA 13.0.88. Artifact: `out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer`
(same as K1-K3). Serve flags for all serves: `--max-context 73728 --kv-dtype fp8
--preserve-thinking --spec mtp --lm-head-draft` (`--draft-tokens 1` unless noted,
`--max-concurrency 4` unless noted).

Serve binary sha256: K4a `6aacacc9...5a763fa0d5` (single build); K4b `f97fbe4d...3de3c5`
(S1/S2 build) and `7f5ae421...62e7d8` (S6/K=3 rebuild of the same tree - the two K4b
builds of identical sources hash differently; build-metadata nondeterminism, not chased).

## Verdict

- The user's discard check (requests leaving the MTP cohort via stop strings and
  max_tokens, at C4, must match the serial loop): **PASS** - every short request's
  output is byte-identical across K4a batch, K4b batch, K4b solo cold and K4a solo cold.
- The K4b tree as a whole: **FAILS serve-level exactness**. Three K4b-specific findings
  (F1-F3); K4a is clean under every shape tested. The real-test bitwise gate (passed)
  does not cover the shapes in F1-F3. K4b is not verified for the serving workload and
  must not be considered correct until F1-F3 are resolved. Issues opened for the
  findings; `ninfer-gb10-nb2` (K4 verify) stays open.

## Gates that passed (K4b, 57b0c264)

| Gate | Result |
|---|---|
| ctest (140 tests, incl. `test_sampling`, `test_mtp_round`, `test_flash_next_ple_stage`) | 140/140, 10 expected skips (real/fault/load-plan/frontend need the large artifact or the Qwen3.5 artifact), 964 s |
| Real test bitwise vs K1-K3 goldens | PASS, no goldens moved (77 s) |
| Unit tests (sampling, ple_stage, mtp_round, gdn_replay_fold, flash_next_ple) | PASS |
| Discard path: stop strings + max_tokens vs serial loop | PASS - table below |
| C4 K=1 warm-cache serve (K4a gate + K4b) | Ran; data below - host ms/round 2.15 (K4a), capacity +0.43 GiB |
| C1 K=3 serve smoke (K4b) | 3/3 requests completed, no crash; exactness vs serial not verified (no K4a K=3 reference; tree already failed the K=1 exactness gate) |

Precondition: neither K4 commit wrapped the MTP `instantiate_graph_family` call in
`begin_round`/`end_round`/`check`, so the K3 bug (ninfer-gb10-nkw) still blocked the
real-test gate. The documented 4-line fix was applied and pushed as `6d851f69`
(k3-verify/RESULTS.md); the K4a gate build is `3d3d7a25` + that fix.

## Discard-path check (the user's question) - PASS

Four requests at C4 (greedy, temperature 0): `maxtok10` (prose, max_tokens 10),
`stopword` ("Reply with exactly: STOPWORD", stop=["STOPWORD"], max 128), `maxtok16`
(code, max_tokens 16), `survivor` (prose, max_tokens 256). Batch = all four in
parallel; solo = sequential C1. Reasoning+content text compared byte-for-byte.

| Request (finish) | K4a batch | K4b batch | K4b solo cold | K4a solo cold |
|---|---|---|---|---|
| maxtok10 (length, 10 tok) | 37 B `6a966a6b` | 37 B `6a966a6b` | 37 B `6a966a6b` | 37 B `6a966a6b` |
| stopword (stop, 14 tok) | 50 B `f284eda2` | 50 B `f284eda2` | 50 B `f284eda2` | 50 B `f284eda2` |
| maxtok16 (length, 16 tok) | 80 B `b62a186c` | 80 B `b62a186c` | 80 B `b62a186c` | 80 B `b62a186c` |
| survivor (length, 256 tok) | 1234 B `847314b2` | **1212 B `9f1893c5`** | (KV error, F3) | 1234 B `847314b2` |

- The stop string fires correctly (finish=stop at token 14); the max_tokens cuts are
  exact (10/16 tokens). The three requests that leave the cohort mid-decode produce
  byte-identical output on every binary and shape: the discard path
  (`discard_successor_rows`) does not corrupt rows that leave.
- The survivor never leaves the cohort; its row is where the K4a/K4b divergence shows
  up (F1).

## Finding F1: batch-shape-dependent greedy output (K4b only)

Same request (survivor prompt, max_tokens 256, greedy, MTP draft 1):

| Shape | K4a | K4b |
|---|---|---|
| C4 batch, cold (3 short requests in the cohort) | 1234 B `847314b2` | 1212 B `9f1893c5` |
| C1 solo, first request, cold | 1234 B `847314b2` (S4) | 1172 B `86bd9884` (S6) |
| C1 solo, 4th request (3 short requests before), cold | 1234 B `847314b2` (S4) | KV error (F3, S2) |
| C1 solo, partial 68-token prefix hit | 1234 B `847314b2` (S3) | (not run) |

Greedy decode must be batch-shape-invariant. K4b changes the output with the batch
shape (1212 vs 1172) and both differ from the serial-loop output (1234). K4a is
identical across all four shapes. MTP acceptance is identical between the K4a and K4b
C4 batches (103/151, 68.2%) - the round structure matches; the token values differ.
Suspects: the two-in-flight round loop (state/KV handoff across the pipelined rounds)
and/or the K4b sampling op changes (new launcher/wrapper + `decrement_token_counts`);
the real-test bitwise gate passed, so the divergence is latent in shapes it does not
cover.

## Finding F2: response-cache entry corrupted for the first discarded row (K4b only)

K4b S1: after the C4 batch, the same `maxtok10` request re-run solo took the response
replay path (serve log: `cache 75 (93.8%, response replay)`) and returned **41 B
`7aa119c4`** ("We need to answer user's request: ...") - different text than the batch
request's live output (37 B, "We need to respond to user: ...") and than any fresh
decode of the same request (37 B on K4b solo cold and on K4a). The K4b replay is
faithful for the other three requests; the K4a replay is faithful for all four under
the same shape. The corrupted entry belongs to the shortest request - the one that
left the cohort first. The live response sent to the batch client was correct (37 B);
the cache entry stored a different generation. Consequence: a client re-asking a
prompt whose first cached response came from a pipelined batch can get text the
engine never sent for that request.

## Finding F3: KV growth entitlement error, C1 after short requests (K4b only)

K4b S2: C1, cold cache, sequence maxtok10 -> stopword -> maxtok16 -> survivor (max
256). The first three complete; the survivor fails after ~4.5 s of decode (~180
tokens): `[engine] WORKER INVARIANT: KV materialization exceeds active entitlement -
failing all requests` (check at `logical_kv_store.h:1441`, `materialize_to_tokens`;
that check predates K4 - the file is untouched by both K4 commits). The growth
reservation for the C1 256-token request is undersized and the per-round
materialization demand overtakes it. Shape matrix:

| Shape | K4a | K4b |
|---|---|---|
| C1, survivor as 4th request after 3 short | OK (1234 B, S4) | **KV error** (S2) |
| C1, survivor as first request | OK (1234 B, S4) | OK but wrong text (1172 B, F1; S6) |
| C4 batch with 3 short rows | OK (1234 B, S3) | OK but wrong text (1212 B, F1; S1) |


## Reruns after the fix (2026-10-01, tree `b61f9867`, serve sha256 `d3b07d5f...`)

Opus landed `97beaaaf` (F1/F3: successors now take the exact frontier from the pending
round's advanced frame; no successor for a row the pending round exhausts; loud ownership
guard) and `b61f9867` (shutdown drain + startup memory wait). Build: **zero warnings** (the
two `-Wnarrowing` findings are gone). Gates on the fixed tree: 5 unit tests PASS, real test
PASS bitwise (80 s).

### F1 rerun: the 256-token request on all four shapes

| Shape | Before fix | After fix | K4a reference |
|---|---|---|---|
| C4 batch, cold (3 short rows) | 1212 B `9f1893c5` | **1234 B `847314b2`** | 1234 B |
| C1 solo cold, 4th of 4 (F3 sequence) | KV error (F3) | **1234 B `847314b2`** | 1234 B |
| C1 solo, partial prefix (batch first) | 1212 B (replay of batch) | **1234 B `847314b2`** | 1234 B |
| C1 solo cold, **first request** | 1172 B `86bd9884` | **1172 B `86bd9884` - UNCHANGED** | 1234 B |

Three of four shapes now match the serial loop exactly. The remaining divergence is confined
to the first request on a fresh serve that crosses a profile boundary (256 tokens cross 127):
the value is byte-identical to the pre-fix value, so `97beaaaf` did not touch its cause. The
short first requests (maxtok10/maxtok16, no boundary crossed) are correct as the first
request, consistent with a first-request x profile-boundary interaction. A second F1 cause
remains; the real-test bitwise gate still passes, so it is outside the real-test shapes.

### F3 rerun: C1 sequence of three short requests then the 256-token one

PASS: all four complete (37/50/80 B, survivor 1234 B `847314b2`); no entitlement error.

### F2 rerun: C4 batch, then solo replay of each request

All four solo replays are byte-identical to the batch's live outputs (37/50/80/1234 B, same
shas), including stopword, which still gets a successor round (14 of a 128 budget, ending on
the stop string - the requested repro shape). No "MTP successor frame" guard error in any
serve log. Per the fix's own caveat, a clean pass here does not close F2: maxtok10 now ends
on its budget and gets no successor, so the original corruption path is not re-exercised.
F2 stays open.

### Shutdown reruns

- 4a, SIGTERM during a 512-token stream: PASS - the stream completed in full
  (output 512/512, engine log `output limit`) 7.4 s after the signal, inside the 30 s
  timeout; log: `shutdown requested | draining 1 in-flight | timeout 30 s`,
  `drain complete in 7.4 s`, `engine released in 2.0 s`.
- 4b, second SIGTERM: the engine cancels as designed (`req#1 done | cancelled | output
  355`), but **the client saw `finish_reason=stop` with no 503 error event** - the spec
  said the stream ends with a 503 error event. A client cannot distinguish cancellation
  from a natural EOS. Protocol finding, filed separately.
- 4c, immediate restart: the guard is active on this device (CUDA reports
  `CU_DEVICE_ATTRIBUTE_INTEGRATED = 1`). Three immediate restarts (0.5 s after a graceful
  exit and 0.5 s after a SIGKILL) all came up healthy. The wait line never fired because the
  kernel reclaims the killed process's ~85 GiB in under 200 ms on this box (MemAvailable:
  24.4 GiB while loaded, 109.8 GiB at the first 200 ms sample after the kill, flat after) -
  there was nothing to wait for. The line is only printed when a wait actually grew
  MemAvailable by >256 MiB.

## Reruns after a5581fce (2026-10-01, serve sha256 `4d35a292...`; build ~3.3 min, zero new
warnings; gates: unit tests PASS, `test_serve_options` PASS, real test PASS bitwise 80 s)

Opus landed `a5581fce`: a cancelled generation that ends normally with a Cancelled finish
during shutdown now throws the shutdown error (503), and pipelined decode is opt-in
(`--pipelined-decode`, default off; off = the Engine never launches the early round).

### Shutdown 503 (pwd) - PASS

Double SIGTERM during a 512-token stream: the client now sees the error event
`{"code": "service_unavailable", "message": "server is shutting down", "type":
"server_error"}` (107 tokens delivered, stream ends 2.0 s after the second signal,
`finish=null`); engine log: `shutdown forced by a second signal | cancelling 1 in-flight
request(s)`, drain complete in 2.0 s. Verified on the OpenAI route; the Anthropic and
Responses routes share the same service call and were not exercised separately.

### F1 bisect: the reworked serial path carries the residual

Cold first 256-token request on a fresh serve, original shape config (C4 serve,
`pages 1,152/4,608`, `runtime 4.14 GiB`), with and without `--pipelined-decode`:

| Flag off (serial loop) | Flag on (pipelined) |
|---|---|
| **1172 B `86bd9884`** - the residual value | **1172 B `86bd9884`** - identical |

Pipelining off reproduces the residual, so the cause is in K4b's rework of the serial
path itself (the alternating host output buffers and the output copies moved out of the
graph), not the early-launched round; pipelining adds no further divergence on this
shape. The first-request x fresh-serve condition also proved serve-config-dependent:
the same request on a C1-config serve (`pages 1,152/1,152`, `runtime 2.15 GiB`) decodes
to 1212 B `9f1893c5` in both modes - a third value (K4a reference: 1234 B `847314b2`).
Three serve configs/trees, three first-request values; a correct engine returns 1234 in
all of them.

## Decision runs after a5581fce (2026-10-01)

Two runs to split the remaining causes for the first-request residual (cold first
256-token request, fresh serve, pipelining off):

| Run | Tree | Serve config | CUDA graphs | Result |
|---|---|---|---|---|
| K4a, C1 | `3d3d7a25` + `6d851f69` (fresh worktree build, sha `7bd0c8bb...`) | C1 (`pages 1,152/1,152`, 2.15 GiB) | on | **1212 B `9f1893c5`** - not 1234 |
| K4b, C4, no graph | `a5581fce` (`4d35a292...`) | C4 (`pages 1,152/4,608`, 4.14 GiB) | **off** (`--no-cuda-graph`, 2.81 GiB) | **1172 B `86bd9884`** - the residual |

Readings:

- **The bug predates K4b** (K4a on the C1 config is already wrong: 1212 vs 1234).
  Next step per the plan: bisect back through K3 -> K2 -> K1.
- **Not a graph-profile-install cause**: the K4b residual reproduces with
  `--no-cuda-graph`, byte-identical to the graphed run.
- **K4b's layout change is invisible under the C1 config**: K4a and K4b on the C1
  config are byte-identical (1212 B `9f1893c5`), although K4b allocates the backup
  buffers even with pipelining off. The config-dependence is a property of the
  pre-existing first-request bug, not of K4b's backup allocation.
- The C4-config residual (1172 vs K4a's 1234) remains K4b-specific; with the graphs
  ruled out, K4b's persistent layout growth (backup buffers) is still the leading
  candidate for which stale bytes the C4 layout exposes.

Full first-request matrix so far:

| Tree | C4 config | C1 config |
|---|---|---|
| K4a (`3d3d7a25` + fix) | 1234 B `847314b2` (correct) | 1212 B `9f1893c5` |
| K4b / `a5581fce`, pipelining off | 1172 B `86bd9884` (residual) | 1212 B `9f1893c5` |
| K4b / `a5581fce`, `--pipelined-decode` | 1172 B `86bd9884` | 1212 B `9f1893c5` |
| K4b / `a5581fce`, C4, `--no-cuda-graph` | 1172 B `86bd9884` | - |

## Token-logprobs check (2026-10-01)

The check that decides whether the first-request divergence is a bug or legitimate
numerical variation: the cold first 256-token request with `--token-logprobs`
(`top_logprobs: 2`), one serve per variant. `logprob_probe.py` / `logprob_compare.py`.

| Run | Binary | Output |
|---|---|---|
| K4a C4, original gate build, plain (18:24) | `82f77128...` (no longer exists) | 1234 B `847314b2` |
| K4a C4, rebuilt, plain x3 | `7bd0c8bb...` | 1172 B `86bd9884` (deterministic) |
| K4a C4, rebuilt, `--token-logprobs` | `7bd0c8bb...` | 1172 B `86bd9884` (flag changed nothing) |
| K4a C1, rebuilt, plain and `--token-logprobs` | `7bd0c8bb...` | 1212 B `9f1893c5` (flag changed nothing) |
| K4b C4, plain and `--token-logprobs` | `4d35a292...` | 1172 B `86bd9884` (flag changed nothing) |

The flag changed nothing on any existing binary. The 1234 came only from the original
gate binary (`82f77128...`, 18:24), which no longer exists; the rebuild's source state
is verified exactly `3d3d7a25` + `6d851f69` (single-file diff, matches the fix hunk;
clean startup, no PLE stalls), and its plain C4 output is 1172 in three consecutive
runs (deterministic; C1 -> 1212). The original binary's provenance is unresolvable
from this machine, and it no longer matters: that output is one more side of the same
tie.

**First divergent token (position 28 of 256; context "... Need"):**

| Run | Chosen | Top-2 (logprob) | Gap |
|---|---|---|---|
| K4a C1 (1212 B) | ` provide` | ` provide` -1.0465564727783203, ` likely` -1.0465564727783203 | **0.0 - exact tie, last bit** |
| K4a C4 (1172 B) | ` likely` | ` likely` -0.8636550903320312, ` provide` -1.1136550903320312 | **0.25 nats (exact 2^-2)** |
| K4b C4 (1172 B) | ` likely` | bit-identical to K4a C4 | 0.25 nats |

Readings:

- The 0.25 gap is a near-tie, not a distortion: the rounding that matters happens on
  the BF16 logits before the log-softmax. Logits in the tens are spaced 0.125 (16-32)
  or 0.25 (32-64) apart in BF16, so the C4 gap of exactly 2^-2 is one or two logit
  steps - a pair tied up to rounding. The C1 exact tie (gap 0.0) and the C4
  one-to-two step gap describe the same near-tie.
- K4a and K4b under the 1172 route are **bit-identical at all 256 positions** (same
  tokens, same chosen logprobs, 0 differing positions): no measurable K4b numerical
  difference under these configurations.
- **F1 passes for K4b**: the gate that matters is K4b against K4a on the same config,
  and that holds with pipelining on or off. The C1-versus-C4 difference also exists
  without K4 - a legitimate config-dependent route on a tied token - and the real
  test, the bitwise contract, passes.

## K4 speed gate: `--pipelined-decode` off vs on (2026-10-01)

ABBA, C4 K=1 warm serve (prose 256x3+1, same load as the K4a gate data), current
binary `4d35a292...` (a5581fce), no nsys trace. `k4-speed/run_speed_abba.sh`.

| Arm | Flag | decode tok/s | host exposed ms/round | device wait ms/round | rounds |
|---|---|---:|---:|---:|---:|
| A1 | off | 22.6 | 0.05 | 71.5 | 1939 |
| B1 | on | 22.5 | 0.06 | 73.8 | 1891 |
| B2 | on | 22.4 | 0.07 | 73.7 | 1901 |
| A2 | off | 23.3 | 0.05 | 70.9 | 1901 |

- decode tok/s: A mean 22.95, B mean 22.45. Both B arms sit below both A arms
  (B1 22.5 < A1 22.6, B2 22.4 < A1 22.6), so this is a slight regression, not
  drift; the last arm (A2) is the fastest.
- host exposed ms/round: 0.05-0.07 under both flags. The a5581fce serial rework
  has already removed the host work (K4a gate: 2.15 ms/round under nsys), so
  pipelining has almost nothing left to hide.
- device wait is about 2.5 ms/round higher under B - the second round's launch is
  not free.
- Verdict: pipelined decode does not pay on this workload; off stays the default.
- K4b (off) against the K4a gate: decode 22.6-23.3 vs 19.9 tok/s, host exposed
  about 0.05 vs 2.15 ms/round (K4a under nsys, K4b untraced) - the rework gain is
  **unmeasured**: no untraced pre-rework baseline exists. Only the on-versus-off
  comparison on the same binary is measured, and it shows on is slightly slower.
  (Corrected 2026-10-01 per Opus.)

### Route-naming run (same binary)

`k4-speed/run_routename.sh`. C4 config with the KV page pool shrunk to C1's
pool: serve enforces `--kv-capacity >= --max-context`, so both are set to
18432 -> `pages 288/1,152` (pool 1,152 = the C1 config's pool; the probe
request is ~356 tokens, far below the cap). Cold first request:
**1172 B `86bd9884...`** - the C4 output, not the C1 output
(1212 B `9f1893c5...`). **The split does not track KV page-pool size; it
tracks `max_concurrency`** (state-slot count). Caveat: the PLE file page cache
was warm from the speed run; the KV pool was cold.

## Route diagnostics: the `max_concurrency`-dependent route is in prefill (2026-10-01)

`k4-speed/run_routediag.sh` (four cold serves, `--no-cuda-graph`, binary
`4d35a292...`), `route_probe.py` (cold 256-token probe; `prompt_tokens`=73),
`compare_routediag.py`. Taps: `NINFER_FLASH_NEXT_LOGITS_DIR` (every
decode/verify round) and `NINFER_FLASH_NEXT_STATE_DIR` +
`NINFER_FLASH_NEXT_STATE_FRONTIER=73` (GDN state at the end of prefill). The
split reproduces exactly under `--no-cuda-graph`: C1 1212 B `9f1893c5...`,
C4 1172 B `86bd9884...`.

- GDN state at the prefill frontier (73 tokens): **layer 0 bitwise identical;
  layers 1-35 differ - all 70 conv/recurrent files**. Route (`mtp`), lane (0),
  and ledger identical on both configs; only the physical slot differs (1 vs 2,
  a pool-indexing artifact of the different pool sizes).
- First MTP verify round (serial 0): **identical metadata on both configs**
  (route `verify`, positions [53,54], batch 1, width 2, `kv_table_rows` [0],
  same draft tokens [1596,1144]) but the logit vectors differ at
  247,786/248,320 (position 53) and 247,841/248,320 (position 54), max |delta|
  17.0 / 15.6 against a logit range of about -11 to +25 - a real divergence,
  not a precision-floor difference. All 148/148 common rounds differ
  downstream (capture counts 151 vs 148).

Layer 0's GDN input and committed state are identical, and layer 1's GDN input
is not, so the divergence is introduced inside **layer 0's attention / PLE /
MLP / hyperconnection stage** - the first decode round takes the same route
with the same shapes and input tokens; it inherits a different state.
Concrete candidate (inference, to confirm): the KV page shape differs by
config (C1 1,152 pages x 64 tokens, C4 4,608 pages x 16 tokens, same
73,728-token capacity), so the attention tiling/reduction over the paged KV
differs from the first attention layer on.

Raw per-round logit captures for rounds 1-150 stay on this machine
(`k4-speed/routediag/`); round 0 and the state captures are committed as
the supporting evidence.

## K4a gate data (3d3d7a25 + fix, serve sha256 82f77128... from the K4a verify turn)

- Unit tests: PASS; real test: PASS bitwise (92 s)
- C4 K=1 warm nsys serve (prose 256x3+1, C4, same load as K2/K3):
  - capacity: `runtime 4.13 GiB` (K2: 3.70) -> **+0.43 GiB at C4** (expected +0.45:
    the 113 MB x 4 spare state slots)
  - request log: **host exposed 2.15 ms/round** (K2: 8.93; 1910 rounds, 3136 tokens) -
    the drop is the K3 off-thread PLE gather landing
  - device wait 80.5 ms/round (K2: 69.6) - higher; consistent with a colder PLE page
    cache in this session (per-round host gather now on the stage thread)
  - NVTX: `decode.mtp_round` 533 (K2: 540); `frame_upload` 36 (K2: 38, unchanged within
    load variance); `decode.mtp.submit` 2.07 ms/round with ingress ~0 (K2: 8.31/7.43)
  - K3 check 4: `wait_staged_kernel` 554 launches, 10.7 us mean, 59.1 ms max - no 5 s
    timeouts

## State

- Beads: `ninfer-gb10-04m` (open, P1 - agreed removal of the pipelined-decode
  machinery + K4a spare state slots, and documentation of the prefill route
  found by the diagnostics below; removal by Opus, reruns by twoFour);
  `ninfer-gb10-nb2` (closed - K4 verify complete); `ninfer-gb10-mol` (closed);
  `ninfer-gb10-3w8` (closed, verified); `ninfer-gb10-pwd` (closed, verified);
  `ninfer-gb10-nkw` (closed).
- Open work for Opus: identify which op inside layer 0's attention/PLE/MLP/HC
  stage takes the config-dependent route (leading candidate: KV page shape),
  and document it; then the agreed removal.
- Machine state: code at `a5581fce` + the K4 record commits; build dir holds
  `4d35a292...`; worktree `~/ninfer-gb10-k4a` holds the K4a gate tree
  (`3d3d7a25` + `6d851f69`, binary `7bd0c8bb...`); GPU free.
