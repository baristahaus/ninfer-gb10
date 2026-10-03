# MoE distinct-expert accounting (fcdd4f1a, trace-build route stats)

Run: `ROUTE_STATS=1 CONC=4 DRAFT=1 tools/gb10/round_attribution.sh`, 2026-10-03 03:06-03:08Z,
GB10, tree fcdd4f1a (= 6192117b stages-3 revert + the `NINFER_MOE_ROUTE_STATS` counter).
C4 K=1, MTP draft 1, two batches of 4 x 256 tokens; attribution window 124 rounds.
Round context: moe.nvfp4 36.298 ms/round (back at baseline after the revert),
host wall 66.268, device wait 57.3, 29.4 tok/s. Counter kernel cost: 0.069 ms/round.

## Counter coverage (verified against the trace)

- The counter fires only on the fused grouped path: `grouped = nvfp4 && tokens >= 8`
  (`kGroupedDecodeMinTokens = 8`) and `tokens <= kFusedRouteTokens (8)` — i.e. exactly the
  T=8 decode/verify MoE calls. T=1..7 uses the non-grouped `DecodeRouteWork` path (no
  counter); T>8 grouped calls (prefill etc.) are not counted.
- Trace cross-check (`cuda_gpu_kern_sum`): `route_stats_kernel` = 6321 launches, matching
  the file total. `route_kernel` = 25141 launches: 24896 at T 1-8, 49 at T 9-16, 49 at
  T 113-120, 98 at T 129-136, 49 at T 137-144 (the T>8 shapes are per-request/MTP/prefill
  MoE calls outside the counted stage).
- The 6321 counted calls = 6076 inside the attribution window (48/round target verify +
  1/round predictor, uniform) + 245 T=8 calls outside the window (mean 17.4 distinct;
  concentrated routing; batch-transition/edge rounds — not resolvable further, the file
  has no timestamps). The window's 6076 equals the report's moe.nvfp4 op calls exactly.

## moe.nvfp4 per-kernel table (target.verify, ms/round, 124 rounds)

| Kernel | Launches/round | ms/round | Stage share |
|---|---:|---:|---:|
| nvfp4_w4a4_mma_kernel | 96.0 | 33.072 | 91.1% |
| fp8_a16_sliced_k_mma_kernel | 96.0 | 1.268 | 3.5% |
| bf16_small_t_inner_kernel | 48.0 | 0.794 | 2.2% |
| route_kernel | 48.0 | 0.595 | 1.6% |
| quantize_grouped_kernel | 48.0 | 0.202 | 0.6% |
| gather_quantize_routes_kernel | 48.0 | 0.200 | 0.6% |
| reduce_grouped_kernel | 48.0 | 0.098 | 0.3% |
| route_stats_kernel | 48.0 | 0.069 | 0.2% |

96 nvfp4 launches/round = 48 calls x (up + down). Per call: 689.0 us.

## Distinct experts per call and in-model bandwidth

In-window cluster (6076 calls, T=8, 80 picks = 8 tokens x top-10):
min 41, mean 60.07, max 78; 93.1% of all counted calls have >= 50 distinct.
The 245 outside-window calls: 13-20 distinct (mean 17.38).

- 689.0 us/call / 60.07 distinct = **11.47 us/expert in-model** vs 11.6 us/expert at the
  M2 microbench (95-97% of plain read): 101% of the bench per-expert time.
- Effective bandwidth: 60.07 x 2.765 MB / 689 us = **241 GB/s** (88% of the GB10 273 GB/s
  peak), consistent with the microbench's plain-read efficiency.
- 80 picks -> 60 distinct: 25% of picks are duplicate experts; routed bytes already scale
  with the distinct count, so this dedup is captured in the 2.765 MB/expert model.

**Reading: case 1 of the two outcomes.** ~60 distinct experts per call at essentially
microbench efficiency: the expert kernels are bandwidth-bound in-model, and the M2
microbench reproduces them in-model (nothing in the model slows these kernels). The
33.1 ms/round expert GEMMs are ~60 x 11.5 us. Levers are bytes per call (routing/dedup,
expert quantization, overlap) — not kernel efficiency.

Caveats: the counter does not split target verify from predictor (both T=8; the 124
predictor calls are inside the 60.07 mean); timings include the 0.069 ms/round counter
kernel (attribution only, not a speed result); non-atomic increments may lose rare
concurrent-stream counts.

Data: `moe_route_stats.txt`, `report.{md,json}`, `summary.md`, `load.{json,log}`,
`serve.log`, `requests.jsonl`, `build_trace.log`; `trace.nsys-rep`/`trace.sqlite` stay
on the GB10 workstation, not committed.
