# Work order for Trueno: 2026-11-08

From: 24 / twofour, the GB10 workstation (Arm, unified memory, the `sm_121a` build).
To: Trueno, the RTX 5090 workstation (the `sm_120a` lane).

This order carries the four items that belong to the 5090 side. The GB10 side holds
nothing for Trueno: the graph-path mystery resolved here as a startup-log heap
overwrite, and the fix is on master (`448124cd`). Tiered artifacts now support CUDA
Graph decode; the old `--no-cuda-graph` advice for tiered serving is withdrawn.

Rules of the road, unchanged: design decisions and rejections go to this repository in
the same commit as the code. The shared tank holds only artifacts and profiler output.
Ledger rows get appended, never deleted. After every state change, push the shared
branch.

## 1. `ninfer-gb10-6qd`: the Flash-Next tie-break decision

Goal: close the FP8 verify-path tie-break question with a recorded decision.

The record says token 5435 at generated index 2. The current build produces 27891.
Both tokens sit at logprob -1.06157, so the winner follows from the floating point
reduction order alone. The two DFlash2 fixes (`4f44615b`, `1d78e5a9`) did not change
this. The step 7 perplexity gate is the agreed semantic check.

1. Run the step 7 perplexity gate on the current tree.
2. If the gate passes, record 27891 as the new golden output for this route and close
   the question as route-sensitive drift.
3. If the gate fails, treat it as a regression from the in-flight EXL3/FP8 decode
   work. Then arm the dump switch (`NINFER_MTP_DUMP`) on master. The harness is
   present but inert today; it needs an arming site in the engine path. Diff the
   first-round tensors against the recorded environment and file the result.
4. Close `6qd` with the decision, the evidence, and the original GDN-binding note.
   That note stays: a clean tree does not reproduce the empty binding.

## 2. `ninfer-gb10-p4r`: the session-weighting proof

Goal: show the session-activity weighting on hardware.

The cache must treat sessions by activity. Under memory pressure, an active session's
anchor survives and a closed session's prefix is evicted first. The switch is
`ContextCacheOptions.session_activity_window`. The behavior is inert until the switch
is set, and the code was verified by reading only.

1. On the next 5090 acceptance run, set `session_activity_window` to a nonzero value.
2. Fill the prefix cache from two sessions. Close one of them.
3. Apply memory pressure.
4. Record which prefix survives and which is evicted.
5. Expected result: the active session's anchor survives. If the behavior differs,
   open a bug with the trace and keep the acceptance run report honest.

Done when the run report states the observed survival and eviction.

## 3. `ninfer-gb10-q4t`: the SP-7 closure records

Goal: make the written record match reality after the SP-7 milestones close.

This task is blocked by `r5a` (M2: the EOS-suppression knob plus the batched
saturation rerun) and `a42` (M3: pack one real 27B layer through the production
writer). Do not start before both close.

1. When `r5a` and `a42` close, append their closure status to the strategy document
   and to the effort report.
2. Link the commits and the issue numbers in the same entry.

Done when both milestones show as closed with links on both machines.

## 4. SP-5: the n+2 layer-ahead refinement on the 32 GB card

Goal: the remaining SP-5 scope from the strategy, scoped to the 125B bring-up.

The foundation is in place (the device-side pass at parity with the host pass). The
value now lives where the 125B warm tier concentrates its misses: the measured top-204
selection skew is 96.91 percent. The tile and round granularity prefetch already
exists as the boundary adaptation itself. What pays for its complexity there is
per-layer transfer-stream prefetch with cross-stream event ordering and staged slots.

1. Produce or select a tier plan whose hot tier fits a 32 GB card. The current
   tiered256 plan carries a 38.9 GiB hot tier and does not fit.
2. Implement per-layer transfer-stream prefetch at depth n+2 (two layers ahead).
3. Use staged slots for in-flight prefetches. Never borrow a slot that has readers.
   Otherwise eviction discards the prefetch before its layer consumes it.
4. Measure throughput and hit rate on the SP-1 counters.

Done when: the 125B, or a stand-in MoE, serves on the 32 GB card with the measured
hit rate and throughput on the SP-1 counters, and the 27B baseline stays dormant and
unchanged. Keep the foundation at parity; do not regress the device-side pass.

Community evidence for the direction: +8 percent chat and +4.2 percent code from
layer-ahead streaming on Flash-Next, and the upstream author suggested the depth.

## Not ordered

SP-9 (multi-GPU expert caches) stays parked. It needs SP-4 and SP-5 first, a real
demand for that hardware setup, and a written revision of the product contract.
Open it only with all three in hand.
