# EXL3 follow-up plan and lane split (2026-10-29)

Context: the fused TensorFold decode landed on master (`56abf9e9`, merged with the
remote at `4160d3d2`): decode 9.6→60.4 tok/s at 8K, saturation C=1 at 60.2 tok/s,
bit-identical to the superseded route, conformance oracle revived and green. This
plan covers the follow-up work split between the 5090 lane and the GB10 lane.
Items not listed here — the MoE
tiering packages, GB10 125B bring-up (SP-6), Qwen 4.0 readiness — stay under their
own authorizations and are unaffected.

## Work items and ownership

### 5090 lane (all GPU measurement + serving changes)

**M1. Round attribution probe.** Profile one steady C=1 decode round of the fused
route at 8K and at 256K context (serve's `--gpu-timing`, or the per-call CUDA-event
probe pattern from the October two-pass campaign). Deliverable: a kernel-class
breakdown of the 16.5 ms round (rot, fused matmul, transpose, attention, KV,
per-round schedule overhead) recorded in `docs/performance/qwen3.8-27b.md`,
replacing the current "consistent with attention dominance" inference with
measured numbers. This is the gate for the decode-ceiling decision below.

**M2. EOS-suppression knob + batched saturation rerun.** The C=2/4/8 saturation
waves are refused because the suite's completion contract requires every stream to
reach the 8,192-token limit and at least one stream samples EOS mid-window under
both sampling profiles. The decode engine itself is already batch-capable: T≤8 rows
are the fused register route's domain, so C≤8 needs no kernel work — only the
measurement gate. Implementation depends on the GB10 lane's contract decision (D1 below):
a server-level CLI flag (no protocol change; the bench runs its own server
instance) or a request-level `ignore_eos` extension field (matches the vLLM bench
convention; touches the advertised OpenAI surface, so the schema tests and
`docs/serving.md` update in the same change). Then rerun decode-saturation
C=1..8 and record the batched tok/s-vs-C table. Sub-linear scaling is a recorded
finding, not a blocker.

**M3. Real-tensor check.** The a16 test's argv mode never ran: it needs the
container that `tools/convert/quantization/exl3_real_check.py` writes. Rebuild the
converter environment (`uv venv`; the converter is self-contained — exllamav3 is
reimplemented under `tools/convert/sources/exllamav3.py`, so the deps are torch
2.11.0+cu130 + numpy per the recorded conversion metadata), pack one real 27B
layer, and run the argv mode. Acceptance: relative L2 ≤ 2/256, output guards clean.

**M5. Decode-ceiling work (post-gate).** Owner and target decided by M1's
breakdown: if the projections still dominate, the candidates are per-shape tuning
at T=1 (split-K reduction overhead on the 1024/2048-row slices, tile occupancy);
if attention/KV dominates, that lane is the target (the groupwise route's 84 tok/s
at 8K is the same-engine reference ceiling). Standing caveat before spending:
the 27B EXL3 route's value is the 7.91 GiB footprint, not absolute rate, so M5
competes with the tiering packages and is re-evaluated against them after M1.
Operator-level microbench only after the relevant kernel is identified; the
end-to-end claim needs a battery rerun.

### GB10 lane (contract and coverage decisions)

**D1. EOS-knob contract decision (blocks M2).** Decide the knob's surface:
server-level CLI (no protocol change) vs request-level `ignore_eos` field (an
advertised-protocol extension → schema tests + serving doc in the same change).
Record the decision and its rationale as a section appended to this file, or in
the PR/review thread for M2.

**D2. mxfp8 conformance test revival.** `test_mxfp8_a16.cpp` was orphaned by the
same linear-suite consolidation that dropped the EXL3 registrations; unlike them
it is still unregistered even though the route is real (`mxfp8_dispatch` is wired
in `src/ops/linear/projection/linear.cpp`). Revive it: register in
`tests/ops/linear/tests.cmake`, port to the current DeviceBuffer/execution APIs —
`tests/ops/linear/linear_shape_runner.cpp` (the shared shape harness, restored
with the EXL3 fix) is the pattern and is already linked into the test support
library. Validate on GB10; the 5090 lane runs a confirmation pass (M4)
after it lands. If the file is too stale against current APIs, rewrite it onto
the consolidated cases runner instead of resurrecting dead scaffolding.

**D3. Converter-environment recipe (small, optional).** Record the reproducible
converter environment (interpreter, torch/numpy pins, `uv` invocation) in the
convert docs so the next env loss is a lookup, not archaeology. M3 derives its
env from the recorded conversion metadata either way; this makes it durable.

**D4. SP-7 closure records (after M2 + M3 land).** With the batched scaling
table and the real-tensor check recorded, the EXL3 package's measurement gates
are complete: append the closure status to the strategy/effort-report trail per
the ratified "gates recorded on both sides" cadence.

In flight and unaffected: the upstream notes for quench6k and Neroued, the GB10
golden-check helper.

### 5090 lane (small, after D2)

**M4. mxfp8 confirmation run.** Run the revived mxfp8 test on the 5090 once D2
lands, so both build targets have evidence.

## Dependency graph

- D1 → M2 → (batched table) → D4.
- M1 → M5 (and re-evaluation of M5 against the tiering packages).
- D2 → M4.
- M3 is independent; D3 is independent.
- M1, M3, D2 can all start immediately in parallel.

## Coordination rules

- File ownership, no overlap: 5090 lane touches `src/serve/**`,
  `tools/bench/run_serve_concurrency.py`, `docs/performance/qwen3.8-27b.md`,
  `docs/serving.md` + schema tests (only if request-level), `profiles/bench/**`,
  converter env/execution. GB10 lane touches `tests/ops/linear/**` (mxfp8),
  `docs/maintainer/**`, `tools/convert/**` docs. `tests/ops/linear/tests.cmake`
  is sequential: the EXL3 registrations are already on master; D2 appends.
- Each lane runs its own GPU; no cross-box contention. The 5090 box keeps the
  single-GPU-instance rule for its own runs.
- Merge cadence per the ratified rule: stable points, gates recorded on both
  sides, no rebase.

## Exclusions

Not in this plan and not authorized by it: MoE tiering packages, GB10 125B
bring-up (SP-6), Qwen 4.0 readiness, any 125B banks work (closed by decision),
and any upstream sync (TensorFold stays a pinned vendored revision per its
NOTICE; divergence is a deliberate port, not an obligation).

## Decision record (append here)

(empty — D1 goes here)
