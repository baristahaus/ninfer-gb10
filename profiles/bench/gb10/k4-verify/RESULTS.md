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

- Beads: `ninfer-gb10-nb2` (open - K4 verify; F1-F3 filed); `ninfer-gb10-mol` (F1/F2 -
  F1 fixed for 3 of 4 shapes by 97beaaaf, first-request shape residual noted 2026-10-01;
  F2 replays faithful on rerun, stays open); `ninfer-gb10-3w8` (F3 - fixed by 97beaaaf,
  verified on rerun, pending close); `ninfer-gb10-nkw` (closed).
- Machine state: tree at branch tip `b61f9867`, clean; build dir holds that binary
  (`d3b07d5f...`); GPU free.
