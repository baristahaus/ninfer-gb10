# FP8 stages-3 adoption verification (5912b05b)

Two long-K Flash-Next FP8 decode projections — [320,10240] HyperConnection down and
[2560,6144] mixer output — run with three pipeline stages at T ≤ 16 (5912b05b); every
other shape stays at two. Stage depth changes no output bit.

## Gates (2026-10-03, GB10 workstation, tree 5912b05b)

- `ninfer_linear_fp8_a16_test`: OK
- `ninfer_hyperconnection_test`: OK
- Real Engine test on the FP8-MTP artifact (119 GB multi-volume, via
  `NINFER_QWEN38_FLASH_NEXT_WEIGHTS`): Passed, 77.8 s
- HC bench fused down+SiLU at T=8: 17.52 µs/layer (expectation ~17.5 vs 18.59 at
  fcceba99) — the criterion the run list set for the kernel-level check.

## HC bench (no profiler; `hc_bench.txt`; 2-stage column is fcceba99)

| T | case | 2 stages | 3 stages |
|---|---|---:|---:|
| 8 | fp8 down+SiLU alone | 18.59 µs (71.7%) | **17.52** (76.0%) |
| 16 | fp8 down+SiLU alone | 19.51 (68.3%) | **18.46** (72.2%) |
| 8 | combine_mix (whole op) | 43.03 | 42.13 |
| 16 | combine_mix (whole op) | 45.54 | 44.66 |
| 8 | fp8 up+mix alone (unchanged shape) | 17.93 | 18.17 |
| 16 | fp8 up+mix alone (unchanged shape) | 19.30 | 19.48 |

## C4 K=1 round attribution (tools/gb10/round_attribution.sh; nsys; ms/round)

Three runs back to back under identical conditions: two on 5912b05b and one re-run of
the baseline tree 2c935f31 (the committed `round-attribution/` holds the Oct 2
qsa-verify baseline; this re-run anchors the baseline under today's conditions).
Rounded per-round rows:

| Stage | baseline re-run (2c935f31) | run1 (5912b05b) | run2 (5912b05b) |
|---|---:|---:|---:|
| moe.nvfp4 | 35.743 | 36.422 (+0.679) | 36.522 (+0.779) |
| gdn.record | 13.463 | 13.455 (−0.008) | 13.480 (+0.017) |
| hyper.combine_mix | 5.472 | 5.422 (−0.050) | 5.407 (−0.065) |
| qsa.select | 4.481 | 4.485 (+0.004) | 4.487 (+0.006) |
| GPU work/round | 65.020 | 65.550 (+0.530) | 65.758 (+0.738) |
| host wall/round | 65.702 | 66.238 (+0.536) | 66.454 (+0.752) |
| device wait (request log, ms/round) | 57.4 | 58.0 | 57.4 |
| aggregate tok/s | 29.6 | 29.5 | 29.8 |

**Reading.**

1. The hc.down gain lands in hyper.combine_mix: −0.050/−0.065 against an expected
   −0.048 (48 layers × ~1 µs).
2. The mixer output projection [2560,6144] sits inside the fused GDN record op
   (`flash_next_gdn_replay_record`, `src/ops/linear_attention/flash_next_gdn.cu`,
   `NINFER_PERF_SCOPE("gdn.record")` wraps the output-projection call), so it is
   attributed to gdn.record. gdn.record is flat between arms (−0.008/+0.017): the
   out projection's standalone −1.37 µs/call (HC bench, T=8) does not show in-model.
3. Adverse, and the reason the "no rise elsewhere" criterion fails: the unchanged
   MoE nvfp4 verify kernel runs +1.9–2.2% per round on the new tree in both runs
   (32.613 → 33.279/33.358 ms/round across 96 launches). It dominates the net:
   +0.53/+0.74 ms/round slower overall. The kernel's source, object file (its TU is
   untouched by 5912b05b), grid and launch count are identical; the recorded
   per-call useful-bytes envelope is identical (228.8 MB — a model-derived
   per-envelope value that does not capture expert routing). Per-round acceptance
   counts differ slightly between runs (C4 is not run-to-run bitwise), so routing
   content is not proven identical; the back-to-back baseline re-run is the
   same-conditions comparison the K=3 lesson called for.
4. The baseline tree reproduces the Oct 2 number (35.743 vs 35.531), so the
   baseline is not a fast outlier.
5. No mechanism in the changed code touches the MoE kernel. Leading hypothesis:
   a GPU-wide clock/power/thermal shift from the 96 deeper-stage launches per round
   (50% more shared memory per block, longer prefetch window). Unproven; needs SM
   clock telemetry sampled during serve to settle.

**Net on this data:** the kernel-level gain is real (HC bench −1.07/−1.05 µs at
T=8/16), but the in-model C4 K=1 round is ~0.8–1.1% slower as measured, entirely
from the MoE stage. Keep/revert/split (hc.down only) is the open decision.

Data: `hc_bench.txt`; `attribution-run1/`, `attribution-run2/`,
`attribution-baseline-arm/` (each: summary.md, report.{md,json}, load.json, load.log,
serve.log, requests.jsonl; trace.nsys-rep and trace.sqlite stay on the GB10
workstation, not committed).
