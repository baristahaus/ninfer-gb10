# Plan: decode determinism first, then GB10 hardening

Created 2026-11-17 by the lead agent session. Status: active. This plan retires when its last
phase closes; completed sections move their evidence into the beads issues and the affected
active references, and the file is then deleted rather than kept as history.

## Objective

Make the engine's decode deterministic under batch membership on both hardware lanes, repair the
known numerical defect in a shipped route, harden the GB10 lane, and finish the veloGB10
adoption items that remain. The owner set the division of labor on 2026-11-17: twofour owns
execution (builds, tests, measurements); the lead agent edits the repository and reviews.

## Ground truth from 2026-11-17

- Native `dgx_spark` build at master `8d1703c6` needed two source fixes; the fixes and the
  documentation are committed and pushed (`94bbd361`, `6ee54aca`, `5ecbba5a`).
- The native suite passes 233 of 237. The four failures are filed: `ninfer-gb10-0kh`,
  `-if1`, `-1r0` (confirmed on target), `-jrv`; the fifth, `linear_projection_nvfp4`, is the
  known `-izs` defect reproduced cross-target.
- The 125B lane serves end to end at master: dense C=1 34.8 tok/s, MTP draft 2 49.0 tok/s,
  40 major faults on a warm cache. The veloGB10 figures are reconciled in
  `docs/performance/qwen3.8-flash-next-125b-a6b.md`.
- The divergence mechanism is mapped: every projection dispatch keys its kernel regime off the
  step's total token count `x.ne[1] = width * batch`; one sliced instance covers T=2..16 while
  T=1 uses GEMV, so a row's bits change when it gains a neighbour. MTP verify pads every row to
  `draft_window + 1` regardless of the row's own extent (`ninfer-gb10-if1`).

## Phase 1 — decode determinism (contract defect)

Revised 2026-11-17 after the cross-lane evidence: the linear pin alone fixed the GB10 divergence
but not the 5090 one, and the non-linear map shows the same T-keying throughout the GDN input
conv, dense attention, and GDN control dispatch. Phase 1 therefore covers every dispatch family
the 27B decode path executes, in this order:

1. Linear families (done, on twofour): T=1..8 one capacity schedule per shape; 27 of 29 suites
   green; the remaining swiglu a4/t2 dense-sparse group and the `-izs` A4 case tracked
   separately.
2. GDN input conv/projection tables (nvfp4, q4_q5, fp8, q8): pin T=1..8 to one capacity schedule
   per shape; where a kernel is exact-by-design (`kBlockTokens == ActiveTokens`), that is a
   kernel change and needs its own gate.
3. Dense causal attention plans (bf16, nvfp4, fp8, int8, k8v4): pin the decode-range instance
   and the split-count policy.
4. GDN control projection and fused norm: pin the route and the split factor.
5. Re-run both real tests and the op suites on both lanes after each family; measure the 27B
   solo decode cost per family and stop for any family whose T=1 cost is material.

The MTP verify padding (`ninfer-gb10-if1`) stays open until the non-linear families are pinned:
the verify round runs the whole stack at the padded width, so linears alone cannot make it
bit-identical.

## Phase 2 — numerical correctness of shipped routes

1. Execute the `-ntc` plan for `-izs`: diagnose the nvfp4 A4 route at 17408×5120 seed 727, fix
   it, and turn the projection test green on both lanes.
2. Resolve `-ywv`: explain or remove the compute-sanitizer duplicate-entry report.
3. Run the 27B baseline battery after the op changes as the regression guard.

Acceptance: `linear_projection_nvfp4_test` passes on both lanes; the 27B baseline battery
matches its published numbers within their own noise.

## Phase 3 — GB10 lane hardening

1. `-jrv`: the Flash-Next prefix-reuse divergence and the forced-compound capture endpoint.
2. `-0kh`: the Vision QKV bias contiguous-bank failure on the GB10 route.
3. `-mz2`: the measurement battery on twofour at master — decode saturation C=1/2/4/8,
   8192-token context, FP8 KV, and MTP3 with the optimized proposal head — recorded in the
   performance doc and the SP-6 row.

Acceptance: the native suite is green on twofour, and the performance doc carries the new
battery next to the 2026-11-10 record.

## Phase 4 — veloGB10 adoption, items 2 to 4 (`ninfer-gb10-1j5`)

1. Re-measure the causal-scoring residual with the distribution framing (teacher-forced KL and
   near-tie counts, not a position maximum) and decide the row.
2. Decide the context-cache resume row: bit-identical or a precise refusal reason in the
   evidence.
3. Measure the sharded prefill on the two-GPU box.

Acceptance: each item closes on `-1j5` with its evidence, or with a documented refusal.

## Standing exclusions (unchanged)

TurboQuant and 4-bit KV wait for an independent oracle because they risk the 27B baseline; lane
packing and real batching stay outside the product contract.

## Cadence

Every change follows the same loop: edit here, sync twofour, run the narrow validation, then the
broader suite before any close, record evidence on the bead, commit with a conventional subject,
and push through twofour. GPU work on twofour stays single-instance. The plan yields to an
explicit owner direction; a deviation is recorded on the affected bead and here, never silent.
