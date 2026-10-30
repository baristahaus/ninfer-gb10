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

### D1 — EOS-suppression knob surface (decided 2026-10-29)

Decision: a server-level CLI flag on `ninfer-serve` (for example `--ignore-eos`), not a
request-level `ignore_eos` field.

Rationale: the decode-saturation suite runs its own server instance and needs every stream on
that instance to decode to the full output budget, so instance scope is exactly the gate's
semantics and no per-request routing is needed. A CLI flag changes no advertised protocol:
`docs/serving.md`, the schema tests and every existing client stay valid, and the change stays
inside the 5090 lane's file ownership. A request-level field remains the better long-term API
if a real client need appears (it matches the vLLM bench convention), and adding it later is
additive; doing it now would expand the advertised contract only to serve a measurement tool.

Constraints for the implementation (M2): the flag suppresses end-of-sequence sampling for all
requests on the instance, the help text and `docs/operations.md` mark it as a
benchmark/measurement knob, and the flag must compose with thinking templates and speculative
decoding the way the ordinary route does, because the suite runs the production profile.

## 5090 lane execution record (2026-10-29, appended after the run)

All four executable M-items are done; commits carry the details, this section is the map.

- **M1 — round attribution** (commit `39d33ec6`, `docs/performance/qwen3.8-27b.md`):
  operator-level microbench of the dispatched unit at every registered shape, weighted by the
  512-call per-round census from the artifact (all 256 EXL3 objects are k2=4, codebook mcg).
  Fused projections 11.39 ms per round (eager upper bound, launch overhead included); the fp8
  KV read floor 0.6/4.8/9.6/19.1 ms at 8K/64K/128K/256K; measured rounds
  16.6/17.7/19.1/21.7 ms. Route is projection-bound through 128K, KV-bandwidth-bound at 256K
  (the projections overlap under the KV stream). No `nsys` on the 5090 box — the attribution
  is the microbench plus the battery's own numbers, not a kernel trace.
- **M2 — `--ignore-eos` + batched saturation** (commits `dd2b0559`, `ac3f92c1`): the knob
  implemented per D1 (schema-driven, help + `docs/operations.md` marked as a benchmark knob);
  the concurrency runner takes a repeatable `--serve-arg` passthrough (use the `=` form:
  `--serve-arg=--ignore-eos`; argparse rejects the spaced form when the value starts with
  `--`). Saturation with the knob: C=1/2/4/8 = 60.30/117.93/225.39/418.82 tok/s, near-linear
  (1.96×/3.74×/6.95×), per-row 87% of solo at C=8, every wave at the output limit.
- **M3 — real-tensor check** (commit `ddf98838`): a second refactor casualty found and fixed —
  the converter's EXL3 writer path (codebook ids, rate validator, bank/dense geometries, the
  EXL3 numeric format, the data-dependent payload sizing) was dead on master; the serving
  artifact was never affected. One real 27B down_proj quantized with the production route
  (k2=4, mcg, the campaign's calibrated Hessian, nmse 0.1208) through the production writer
  into the revived argv mode: relative L2 0.00165 against the composed FP64 GEMM, inside the
  2/256 criterion. Converter environment rebuilt at `~/.venvs/ninfer-convert` (torch
  2.14.1+cu130 — newer than the campaign's recorded 2.11; the container is validated
  numerically regardless).
- **M4 — mxfp8 confirmation**: `ninfer_linear_mxfp8_a16_test` passes on the 5090.
- **M5 — small-slice work**: authorized after M1 and attempted the same day; **not landed**.
  The approach — one fused-parent dispatch per mixer instead of four row-slice dispatches,
  with the parent row bands copied flat into the caller's tensors — is bit-identical to the
  slice route at the operator level: the [14336, 5120] and [16384, 5120] parents were
  verified against the row-sliced route on real artifact weights (zero-diff, all-nonzero,
  finite), and both wrappers pass with real weights and real-scale activations in isolation.
  Under serving graph replay the fused route deterministically decodes garbage (immediate
  EOS), while the un-fused route reproduces the reference output hash exactly. The delta is
  narrowed to the capture-path interplay — the fused route adds capture-time workspace
  allocations and device-to-device memcpy nodes that the row-slice route does not have — but
  the mechanism is not identified. The wrappers are reverted; the parent shapes stay
  registered in the dispatch table as preparation for the reattempt.
  Next step: instrument the capture path itself (ncu on the captured graph, or logging the
  capture-time arena offsets versus the baked memcpy addresses) before reapplying.
- **Pre-existing finding (not from this work): eager decode is broken on pure HEAD.** With
  `--no-cuda-graph`, the same 27B EXL3 serving binary that decodes correctly under graphs
  emits immediate stop tokens (empty content) on the unchanged tree, reproducibly. The
  production profile always runs graphs, so serving is unaffected; the eager path is an
  unvalidated route and a separate bug to investigate.

## M5 capture-path investigation record (2026-10-30, appended after the run)

The fusion was reapplied and re-tested under the full-build-only protocol (the earlier
bisection had used `--target` builds, which are unreliable; every conclusion below is from
full builds). Findings:

1. **The serving breakage reproduces cleanly** with the fusion alone (three consecutive
   immediate-EOS decodes), so it is real, not a build artifact.
2. **The placement copies are not the cause.** Replacing all `cudaMemcpyAsync`/2D placements
   with a capturable elementwise copy kernel changed nothing (after fixing a destination-
   stride bug in the first kernel variant). The copies are bit-correct in isolation.
3. **The capture-time arena allocation is not the cause.** Allocating the fused parent
   outside the op scope (persistent bump, no rewind) changes nothing.
4. **GPU performance counters are now enabled on this box** (the profiling-permission
   modprobe option is set and the driver was reloaded); ncu works.
5. **First ncu census of `tf_exl3_linear_4_1_8` launches during a fused-route serving
   request (300 profiled launches):** the decode rounds show a periodic per-layer block of
   eight launches with the SLICE-era shapes — grids 128×2, 384×2 (n=2048/6144: the
   attention k/v and q/gate slices), 320, 1088×2, 320 (n=5120/17408: the MLP down and
   gate/up) — while the fused-parent shape (grid 896, n=14336) appears only ten times,
   scattered, not per-layer-per-round. For a 64-layer model with the attention fused, the
   parent should launch every full-attention decode round.

The census is the live question for the reattempt: the decode is executing the old
slice-shaped launches, and the fused parent appears far too rarely. Either the captured
graph was recorded before the fused wrappers ran (a capture-path ordering question), or
the fused branch is not being taken for most layers under the capture. Next step: correlate
the ten parent launches with their phase (prefill vs decode rounds) and layer indices —
the model is 64 layers, 48 linear-attention (GDN) + 16 full-attention — and then instrument
the capture to see which code path the recorded graph actually contains. The experiment
edits were reverted; the tree is clean at master.

## M5 record continued (2026-10-30, the marker and replay experiments)

twoFour's engine answer (recorded upstream): the captured graph cannot contain pre-fusion
code — capture is once at startup in `Graphs::prepare`, launches are baked, and the
wrapper's branch is a pure function of fixed inputs. The marker experiment then settled the
census: with fused-vs-sliced prints in the wrapper's decode branch, a serving request shows
**224 `ATTN-FUSED cols=1` launches** (16 full-attention layers × 14 decode rounds — the
fused branch IS baked and running), plus later `cols=2`/`cols=5` captures and the prefill's
`ATTN-SLICED cols=51/52`. The earlier census misread is resolved: the slice-grid launches
were the unfused GDN's four slices (q/k 2048, v/z 6144) and the MLP, not the attention;
the grid-64 shape the attention k/v slices would produce never appears because the
attention really is fused.

With the fused branch proven live, the remaining suspects were eliminated one by one:
the placement copies (kernel-copy variant: no change), the capture-time arena allocation
(persistent-bump variant: no change), and the arena capacity account (the capacity now
also covers `exl3_dense_workspace_bytes(14336, 5120, T)`; no change — and
`MemoryArena::alloc_bytes` throws `bad_alloc` on overflow rather than corrupting, so an
undersized account would fail loudly at capture, not silently at replay).

The decisive new instrument: `tests/ops/linear/test_exl3_dense_a16.cpp` now runs the
**fused single-dispatch [2048, 5120] problem through a real CUDA graph capture,
instantiation, and two replays** against the FP64 reference — the same mechanics the
engine's mixer capture uses — and it passes. The fused dispatch is replay-stable in
isolation. The serving delta is therefore the multi-profile capture interleave: the
markers prove T=2 and T=5 profiles are captured lazily during the first request (not
only at startup), all against one shared plan-sized arena, and the serving replay of the
T=1 profile still decodes garbage while an isolated capture-replay of the same dispatch
is bit-stable. Next step for the reattempt: reproduce the breakage with a multi-profile
capture sequence in the test harness (capture T=1, then T=2/T=5 against the same arena,
then replay T=1), which turns the remaining engine-side question into a seconds-per-cycle
test instead of 4-minute serve cycles. The fusion, markers, and capacity edit are reverted;
the graph-replay test coverage is kept (passing).

## M5 verdict: the profile-swap repro lands engine-side (2026-10-30)

The harness repro twoFour specified — capture T=1, then T=2 and T=5 against the same
arena with `MemoryArena::reset()` between them (the engine's child-swap cycle via
`update_child` + `executable.upload` with `resources.work.reset()`), then replay T=1 —
**reproduces the serving corruption: relative error 1, complete garbage**, in
`test_exl3_dense_a16.cpp` (the block after the single-graph replay check). The same
dispatch replays bit-stably before the swap cycle in the same test. Per the agreed
criterion this is the child-graph update path under a shared arena — engine-side, the
GB10 lane's to fix in `text_graph.cpp` / `src/core` graph RAII.

The repro block is committed as a diagnostic (it prints and does not count a failure)
until the graph-RAII fix lands, at which point the GB10 lane flips it to a hard failure.
M5's fusion reattempt resumes after that fix: the kernel, the dispatch, and the wrapper
are exonerated; the single-graph replay is proven bit-stable; only the swap-cycle
replay-stability remains.

Pre-existing damage outside the 5090 lane, not touched: `docs/operations.md` still documents
`--lm-head-draft`, which no longer exists on master (the GB10 lane's doc; the live flag is
`--proposal-head-rows`/`--proposal-head`).

## Status and campaign split (2026-10-30)

Verified state, by item:

- **D2 (mxfp8 conformance revival): CLOSED.** `test_mxfp8_a16.cpp` is registered in
  `tests/ops/linear/tests.cmake` (landed with `48174f71`/`8c66f403`); built and green on
  GB10 (`ninfer_linear_mxfp8_a16_test`, passed 12.45 s, 2026-10-30). The 5090-side
  confirmation pass (M4 for this test) is all that remains.
- **D3 (converter-environment recipe): CLOSED.** `docs/weight-conversion.md` now carries
  the rebuild recipe plus the verified 2026-10-30 combination (Python 3.11 `uv venv`,
  torch 2.14.1+cu130 aarch64 on GB10, numpy, safetensors, pytest). M3's environment
  prerequisite is satisfied; the recorded 2.11.0+cu130 pin is stale but the local
  exllamav3 reimplementation keeps the route self-contained.
- **Converter route restoration:** the v3 refactors had gutted the converter-side EXL3/QAD
  routes the SP-7 tests bind. Restored to master in `c6b0126b`: four exl3 conversion
  methods, the dense (fused) and bank official recipes, the QAD layouts and mxfp8/fp8
  formats, the I16 dtype, the Flash-Next v2 upgrade route, the SourceInputs campaign
  pass-through, and the data-dependent payload-size declaration (`spec_bytes`). Full
  `tests/convert`: 106 passed, 1 skipped.

New work item:

- **N1. Bank-route real-shape emit proof (GB10 lane).** The bank emit/import routes are
  proven only at synthetic shapes. Add a real-geometry variant of the bank oracle at the
  125B Flash-Next scale (hidden 2560, intermediate 640, 512-expert directory): one genuine
  `quantize_linear` result per role at the real shape, expanded to the full expert count by
  the sign-negation identity the 512-expert test already uses (the full per-expert Viterbi
  campaign stays out of scope — it is compute-bound and exercises nothing the emit path
  depends on). Acceptance: the emit path produces a loadable bank at real [n, k], the tile
  re-verification stays exact, and the C++ bank loader accepts the directory geometry.
  This closes the last synthetic-only gap in the restored route before the EXL3 serving
  work (issue `ninfer-gb10-6qd`) leans on it.

Ownership and lanes:

- GB10 lane (this machine): N1; the D1 decision below; GB10 battery reruns as EXL3 fixes land.
- 5090 lane (Trueno, mid-port — do not double-run): the `ninfer-gb10-6qd` P1 root-cause,
  then M1, M3 (environment ready per D3), M2 after D1, M4/M5 per the original split.
- Upstream: nothing to port for this package. The local `tools/convert/sources/exllamav3.py`
  reimplementation is the working oracle; Spark presets stay parked until a Spark cutover
  is authorized (ratified strategy). The GB10 field report remains an un-sent goodwill
  offer, held until the `ninfer-gb10-6qd` divergence is resolved.

**D1 recorded decision (2026-10-30): server-level CLI flag.** The decode-saturation suite
needs the EOS gate off; the bench runs its own server instance, so a server-level flag
(`--ignore-eos`, no protocol change, no advertised-surface touch) satisfies M2 at the
lowest contract cost. A request-level `ignore_eos` extension stays available if a serving
consumer ever needs per-request control; until then the OpenAI/Anthropic schema tests and
`docs/serving.md` stay untouched.

### Engine-side capture-path answer (2026-10-30, twoFour, from master 8887b550)

Question: can the recorded decode graph contain the pre-fusion wrapper code — capture
ordering relative to wrapper selection, per-layer capture specialization, a cached
first-capture — and is attn_input_proj's decode dispatch reachable other than through the
modified wrapper?

**1. No capture-ordering hazard, no cached first-capture, no per-layer specialization.**
Graphs are captured once at startup (`Graphs::prepare`, `src/models/qwen3_5/program/graphs.cpp`):
each phase's body runs once as warmup, then `definition.capture(device.stream(), body)` records
the same calls, and replay never re-dispatches — kernels and launch shapes are baked at
capture. Each phase graph captures the whole layer loop in one body (batch/token-count are the
only specializations, via `prepare_main(Main, Ordinary)` and the segment profiles). The
wrapper's branch is a pure function of the weight view and `x.ne[1]`, both fixed before
capture. Whatever the wrapper dispatched at capture is what replays forever.

**2. The census's slice-era block is master's decode behavior, not a stale graph.** On
master, every EXL3 decode dispatch is row-sliced:

| route | wrapper (decode branch, `cols <= 8`) | launches per layer-round |
|---|---|---|
| attention | `src/ops/wrapper/attn_input_proj.cpp` EXL3 branch: `exl3_dense_slice` q(6144)/k(1024)/gate(6144)/v(1024), four `exl3_dense_dispatch` calls | 4 tf launches (grids 384x2 x2, small x2) |
| GDN | `src/ops/wrapper/gdn_input_proj_exl3.cpp`: q(2048)/k(2048)/v(6144)/z(6144) slices + four dispatches + 2D placement copies | 4 tf launches (128x2 x2, 384x2 x2) |
| MLP | `src/ops/wrapper/linear_swiglu.cpp` EXL3 branch: gate/up halves (17408 each) of the [34816] object, then down | 2x 1088x2 + 1x 320 |

That is exactly the census's periodic per-layer block. On master **nothing** launches the
[14336] attention parent at decode — the wrapper always slices it.

**3. The attention decode dispatch is NOT reachable through anything but
`src/ops/wrapper/attn_input_proj.cpp`.** `attention_projection`
(`src/models/qwen3_5/execution/attention.cpp`) takes `ops::attn_input_proj` for the single
EXL3 weight; the alternative `linear_split` route has **no EXL3 case** and throws
(`src/ops/linear/split/dispatch.cpp`), so an EXL3 27B artifact cannot route around the
wrapper. GDN has its own wrapper (`gdn_input_proj_exl3.cpp`) — if the fusion was made
there for the attention parent by mistake, it never runs for attention.

**Consequence for the M5 attempt:** if a fused build's captured decode graphs still contain
slice launches, the fused branch was not on the capture path. The fusion must replace the
four `detail::exl3_dense_dispatch` calls inside the wrapper's `cols <= 8` branch (not the
prefill branch, not the dispatch layer). Cheapest verification: a one-line capture-time
marker (log/NVTX) in that branch printing fused-vs-sliced — the serving log then shows what
got baked into Main and Ordinary.

**4. The ten fused-parent launches are predicted to be the eager rounds.** The round(s)
executed outside the captured graphs (the pre-replay round(s) after start, or any
non-capturable fallback) run the live wrapper; if the fusion took effect eagerly but the
capture preceded it — or vice versa — the sample window shows a scattered handful of parent
launches. The planned parent-launch-to-phase correlation should confirm: they should all sit
in one phase boundary, not distributed per round.

**5. One more suspect for the serving breakage itself: workspace capacity, not placement.**
`attention_projection_workspace_bytes` sizes the caller's arena from the per-slice route
(`ops::attn_input_proj_workspace_capacity_bytes(first_profile.n, ...)`). If the fused branch
allocates or touches more than the slice-sum capacity (the TensorFold route's scratch scales
with N: rotated activations, split-K counters, fp32 partials, and the [tokens, N] staging
before the tiled transpose), replay overruns the arena and corrupts adjacent state — a
plausible mechanism for deterministic immediate-EOS under graphs while the kernel stays
bit-identical in isolation. The allocation-outside-the-scope change (finding 3 in the record
above) moves the allocation but does not fix a capacity account computed for the slices.
Check `attn_input_proj_workspace_capacity_bytes` against the fused parent's requirements, and
run one eager fused decode under compute-sanitizer memcheck; a graph-replay overrun needs the
capacity account fixed first.

### Capture-site inventory and the multi-profile interleave (2026-10-30, twoFour, second pass)

Reconciling the marker evidence (cols=2/5 captures observed during the first request) with
the engine source, master's capture sites are exactly:

- `RankProgramImpl::initialize_graphs` (`program/graph_bodies.cpp:9`) → `Graphs::prepare`
  (`program/graphs.cpp:267`): the Main/Ordinary/Verify phase graphs, the Draft/Sample/Accept
  segments, and the mtp-history graphs (graphs.cpp:335, :366). Triggered from
  `create_rank_program` inside the engine's `ProgramInitialize` startup phase
  (`src/runtime/engine/startup.inl:106`) — server startup, before any request.
- `TextGraph::prepare` (`program/text_graph.cpp:108`), called from `Graphs::prepare`:
  **Resident** (no MoE schedule — the dense 27B route): per-profile per-layer mixer child
  graphs all captured at prepare (:154), the main graph with profile-0 children appended
  (:158), then per-profile `update_child` + `executable.upload` swaps (:168) and a final
  swap-back to profile 0 (:172-175). **Offloaded** (MoE schedule present): per-layer captures
  through `schedule->capture_layer(...)` (:216), which may capture lazily under the MoE
  pressure machinery — this is the only lazy capture path in the program.

So "capture is once at startup" needs one refinement: there is no request-time capture path
for a dense (Resident) program; if the fused build's markers show captures inside the first
request, either that build created the rank program at first request (engine-side deferral
or the **two-attempt budget loop** in startup.inl:90 — a failed memory-budget check re-runs
the whole program creation, a second full capture pass), or the served route took the
Offloaded path. The startup observer's phase log discriminates: count
`StartupPhase::CudaGraphPrepare` occurrences.

The harness repro should mirror the Resident interleave exactly — the current isolated test
captures one dispatch, which is why it passes. The engine sequence is:

1. capture the per-layer mixer child graphs for profile 0 (T=1) from the shared arena;
2. capture the main graph with those children appended (baked arena addresses);
3. for each later profile (T=2, T=5, ...): `update_child` swaps the child references and
   `executable.upload` re-uploads — against the **same arena the earlier captures baked**,
   with `resources.work.reset()` rewinding the arena between prepare bodies;
4. swap the children back to profile 0 and re-upload;
5. replay profile 0.

Two concrete hazards fall out of that sequence, either of which the harness repro can hit:
the child-graph uploads against arena memory whose addresses the earlier captures baked
(rewind-and-reuse across profiles), and the final swap-back upload racing the first replay
launch. If the repro reproduces with the child-swap cycle and passes without it, the bug is
the child-graph update path under a shared arena — engine-side, `DecodeGraphDefinition`
/`update_child` ownership in `program/text_graph.cpp` and the graph RAII in `src/core`.
