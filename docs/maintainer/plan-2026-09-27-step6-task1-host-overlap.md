# Step 6, task 1 — kill the MTP decode round-boundary host gap (flash-next 125B, GB10)

Dated plan for active work; remove when the task is done. Owner: Opus (engine change).
Verification machine: 24 / twofour (GB10), runs the captures and re-bench below.
Supersedes the earlier "PDL vs fusion" framing for step 6's first task (decision +
evidence: `gb10-worklog.md` § Idle-time attribution). PDL port (knoopx `e0a6d18c`)
is explicitly deferred to task 2.

## Problem

Per-decode-round the GPU sits idle while one commit thread does the round-boundary
host work. Measured on GB10 (campaign 3, `profiles/bench/gb10/step3/mtp2.sqlite`,
production config `--spec mtp --draft-tokens 2`, no new capture needed):

- Round-boundary gaps (kernel timeline, `CUPTI_ACTIVITY_KIND_KERNEL` LAG(end)):
  59 gaps >1 ms totaling 233.7 ms = **90.6% of the 258 ms total idle**; median
  ≈1.7 ms; outliers 26.5 ms / 14.2 ms at the MTP verify-inputs boundary. Signature:
  `shortlist_exact_select` / `ple_fold` (last kernels of round N) → host →
  `set_i32_scalar` / `speculative_prepare_verify_inputs` (first kernels of round N+1).
  (The 79.4 ms / 8.4 ms gaps at ~1% and ~50% span are run-boundary pauses between
  the two benchmark runs, not engine work.)
- Host NVTX (same capture): `decode.mtp.submit` **median 1054 µs, max 2028 µs** per
  round (ingress build + PLE staging + graph launch); `decode.mtp.wait`
  (`device.synchronize()`) median 64.9 ms = the round's GPU time;
  `ninfer.host/1|ple.gather` median **9.2 µs warm** (max 72 ms = cold PLE file
  page faults); `ninfer.host/1|ple.hash` median 1.4 µs; `engine.commit_output`
  median 4.5 µs (the output transaction is already cheap).
- Decode is 85% CUDA-graph-captured (40,446 of 47,408 kernels); in-round gaps are
  only 19.7 ms (7.6% of idle). The 194.6 ms of sub-10 µs gaps seen in the mtp0
  (MTP-off) capture are eager-launch latency of the diagnostic config — not this
  task.
- Recoverable: ≈73 ms/run ≈ 2.3 ms/decode round ≈ ~20% of the ~12 ms/token gap vs
  the 43.9 tok/s vLLM DGX Spark reference.

## Root cause (code, flash-next MTP path)

The whole round is synchronous on one commit thread, one stream
(`ProgramImplCore::decode_mtp_batch`, `src/models/qwen3_8_flash_next/impl/runtime/
program_impl.h` ~12083–12293):

1. **Submit** (`decode.mtp.submit`): row validation + `checked_i32` loop
   (:12097–12120, :12139–12172), envelope + `select_graph_profile` /
   `install_graph_profile` (:12128–12137), **PLE staging** —
   `compute_ple_ids` + `gather_ple_fp8` (file-backed 51 GB table) into pinned
   `flash_ple_host` + H2D `lanes × width × 2560 B` (:12173–12201),
   `materialize_sequence_kv` (:12194), then `mtp_decode_batch` → `run_prepared`
   → `cudaGraphLaunch` (:12219, `schedule::mtp_decode_batch` in
   `impl/runtime/mtp_impl.h:287`).
2. **Wait** (`decode.mtp.wait`): `device.synchronize()` (:12226) — full stream sync
   for the whole round.
3. **Commit**: read `mtp_host_egress` (D2H is the *last op of the captured graph*,
   `mtp_impl.h:274–276`), validate, update stats / `pending` / lifecycle
   (:12231–12273), return `BatchedGeneratedRound` to the engine commit path.

Two structural facts bound the fix:

- The next round's ingress depends on this round's egress: `current_drafts` come
  from the host ledger `sequence.mtp_drafts` (filled from the previous egress's
  `next_drafts`), and the PLE tokens are history suffix + those same drafts
  (:12177–12184). The egress only exists when the graph's tail D2H lands — i.e. at
  round end. **The overlap window with the previous round's GPU work is ~0 for
  token-dependent host work.** The fix is (a) shrink the post-egress host path,
  (b) phase B: move the dependency off the host entirely.
- PLE is consumed at layer 0's hyper stage (`text_context_impl.h:1328–1357`), so
  the PLE H2D has no in-round overlap window; it sits at the head of the critical
  path. Warm it is ~11 µs (gather 9 µs + H2D 2 µs) — the cold-fault outliers
  (72 ms gather max, the 26.5 ms gap outliers) are the file-page-fault tail, not
  the steady state.

## Task

**Phase A — shrink the post-egress host path (this task).**

0. **Instrument first.** Add nvtx sub-ranges inside `decode.mtp.submit`
   (per-row ingress loop / PLE staging / `materialize_sequence_kv` /
   `install_graph_profile` / `cudaGraphLaunch`) plus the egress-commit slice
   (sync return → next submit start). One re-capture on 24 (step 3 protocol) →
   identify the dominant block. No fix ships unmeasured.
1. Fix the dominant block(s):
   - `install_graph_profile` (`program_impl.h:663`): if it re-records or
     node-updates the graph every round (frontier-dependent params), make updates
     one-shot per topology-class change; steady state = plain `cudaGraphLaunch`.
   - `materialize_sequence_kv`: if per-round KV reservation does real page work,
     pre-reserve the full per-sequence extent at prefill; per-round = bookkeeping.
   - Per-row ingress loop: hoist the request-invariant fields (sampling,
     rope_delta, kv table rows, state slots); only anchors / frontier / extents /
     drafts / positions change per round.
   - PLE staging: keep on the main path (warm 11 µs — a worker thread is not
     justified). For the cold-fault outliers: instrument (mincore / fault count)
     and, if they persist warm, `madvise(MADV_WILLNEED)` the row page range
     during the previous round's gather. Do not mlock the 51 GB table.
2. Re-capture on 24 and re-bucket.

**Phase B — device-side next-ingress (follow-on, only if A leaves >0.5 ms/round).**

Graph tail writes the next round's ingress fields (next_drafts from the AR step,
extents, positions) directly into a device ingress arena slot; the host enqueues
back-to-back graph launches with no egress read on the critical path; egress /
ledger / public-token commit moves to an async worker (side stream); PLE staging
moves to that worker keyed off the async token readback (PLE consumed at layer 0,
so the worker's H2D + event wait is inserted before the first PLE-consuming
kernel; PLE bytes stay file-backed — no table upload). Boundary collapses to
launch latency. Touches: ingress/egress layouts (`round_state.cpp`), graph tail,
`run_prepared` (`graph_impl.h`), engine commit path (`src/runtime/engine/
engine_core.h`), PLE worker. Numerics unchanged: same tokens, same PLE bytes.

## Constraints

- Keep the 85% graph capture; do not regress decode to eager.
- PLE table stays file-backed (host mmap, `artifact::MappedRange`); pinned
  staging buffers are address-stable under graph capture.
- Same commit structure exists in ordinary decode (`impl/runtime/decode_impl.h`,
  `program_impl.h` ~11990–12012 PLE staging) and the qwen3_5 family
  (`src/models/qwen3_5/program/prefill.cpp:905–928`, `execution/decode.cpp`,
  `speculative/mtp.cpp:80–197`). Phase A targets flash-next MTP + flash-next
  ordinary (the GB10 production paths); the qwen3_5 mirror is optional.
- No new public API. No behavior change to sampling, MTP accept/commit, KV
  ownership, or output semantics.

## Acceptance (all on 24, per phase)

- Re-capture step 3 mtp2 (production K=2, existing script + `--profile-measured`)
  and re-run the gap bucketing + NVTX stats (queries in `gb10-worklog.md` §
  Idle-time attribution).
- Phase A targets: `decode.mtp.submit` median 1054 µs → **<400 µs**; round
  boundary gap median 1.7 ms → **<600 µs**; mtp2 >1 ms gap total 233.7 ms →
  **<120 ms** (the 87.8 ms run-boundary pauses are excluded by construction);
  mtp2 total idle 258 ms → **<160 ms**.
- Phase B targets: boundary gap median **<200 µs**; mtp2 idle **<100 ms**.
- Correctness: fixed-prompt fixed-seed decode, token-for-token diff vs the
  pre-change build must be identical (MTP accept/commit state unchanged);
  ctest + the step 0 real-artifact checks on 24.
- Perf claim: re-run the step 2 baseline bench (K=2, 8K/512; campaign 3 baseline
  44.6 tok/s) and report the measured delta — no projected tok/s in the PR.

## Out of scope (deferred)

- PDL port (knoopx `e0a6d18c`): step 6 task 2. Its target is the residual 19.7 ms
  of in-round gaps + the ~6,962 eager kernels/run (PLE fold/gather, scalar ops,
  verify prep), and it must coexist with graph capture.
- mtp0 eager-launch latency (194.6 ms): diagnostic-config artifact; revisit only
  if the MTP-off config becomes a supported product path.
- SM-count / L2-prefetch tuning of decode stages: decode is bandwidth-saturated
  (step 5 owns launch constants; attribution shows no SM-share headroom in the
  current data).
