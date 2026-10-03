# GB10 parity: NInfer vs DGPP after the post-K5 batch (parity-2026-10-03)

2026-10-03 04:11–04:36 UTC (24.4 min), tree `05ed1f2a`,
`BLOCK_I_ROOT=profiles/bench/gb10/parity-2026-10-03 PHASES=I9 tools/gb10/block_i.sh`.
Prior scoreboards: `parity-2026-10-k5` (2026-10-02, tree `e0a140e2`),
`parity-2026-10` (2026-10-02, tree `0e8f7516` + uncommitted PLE decode fix),
and `block-i/I9` (original I9, 2026-09-30).

## Protocol

Identical to the original I9 and the prior reruns: one shared load client
(DGPP's `serve_load.py`), five classes (prose, code, json, math, chat),
concurrency C1/C2/C4, 256 output tokens, greedy (temperature 0), thinking off,
3 repeats per class-concurrency (45 phases, 105 requests per arm).
All 420 requests status 200, all finish=length; no rerun legs, no errors
(`I9_EXIT=0`, no nonzero load exits).

- NInfer arms: tree `05ed1f2a`, artifact
  `qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer`, fp8 KV, CUDA graphs on,
  `--max-concurrency 4`, MTP draft K = 1/2/3, port 18087. Drop caches before
  each arm.
- DGPP arm: `dd58d6d3` (same commit as the three prior runs; rerun in this
  campaign, no reused numbers), RadixArk Qwen3.8-Flash-Next-NVFP4 W1 config
  (mtp_depth 1, bf12+bf16 routed experts, fp8 dense, decode_graph on,
  kv_capacity 262144), `DGPP_RESIDENT_CACHE=off`, port 18080, drop caches
  before start.

Since the K5 NInfer arm (`e0a140e2`): the MoE routing and schedule changes
(M1/M2, the only change in the batch with non-bit-identical outputs), the PLE
record change (M3), the QSA split restaging, the HC norm, and the QSA
projection grouping. The route-kernel shared-gate hoist was tried in this
batch and reverted (bit-identical, performance-neutral: `3b257c91` →
`05ed1f2a`).

## Scoreboard

tok/s, repeats averaged. K5 columns are the `parity-2026-10-k5` run.

| class | C | K5 K1 | K5 K2 | K5 K3 | K5 DGPP | new K1 | new K2 | new K3 | new DGPP | dK1% | dK2% | dK3% | dDGPP% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prose | 1 | 42.61 | 44.67 | 43.16 | 45.11 | 45.54 | 48.04 | 46.69 | 45.43 | +6.9 | +7.5 | +8.2 | +0.7 |
| prose | 2 | 62.58 | 61.65 | 60.88 | 63.08 | 68.01 | 67.09 | 67.43 | 63.47 | +8.7 | +8.8 | +10.8 | +0.6 |
| prose | 4 | 89.10 | 89.26 | 86.19 | 85.12 | 99.01 | 99.49 | 98.79 | 85.42 | +11.1 | +11.5 | +14.6 | +0.4 |
| code | 1 | 46.89 | 55.91 | 58.72 | 50.03 | 49.89 | 60.11 | 63.65 | 50.38 | +6.4 | +7.5 | +8.4 | +0.7 |
| code | 2 | 68.87 | 78.83 | 88.00 | 69.64 | 73.00 | 87.78 | 97.59 | 69.90 | +6.0 | +11.4 | +10.9 | +0.4 |
| code | 4 | 98.06 | 109.08 | 114.89 | 94.00 | 107.63 | 122.32 | 130.79 | 93.91 | +9.8 | +12.1 | +13.8 | −0.1 |
| json | 1 | 49.58 | 61.37 | 70.00 | 50.51 | 52.84 | 66.53 | 74.69 | 50.92 | +6.6 | +8.4 | +6.7 | +0.8 |
| json | 2 | 74.31 | 88.43 | 101.24 | 73.25 | 79.15 | 96.75 | 112.46 | 73.23 | +6.5 | +9.4 | +11.1 | −0.0 |
| json | 4 | 100.24 | 116.22 | 124.68 | 96.31 | 112.52 | 131.34 | 142.97 | 96.97 | +12.3 | +13.0 | +14.7 | +0.7 |
| math | 1 | 47.07 | 58.43 | 64.62 | 47.85 | 50.24 | 62.73 | 69.87 | 48.08 | +6.7 | +7.4 | +8.1 | +0.5 |
| math | 2 | 71.06 | 83.20 | 94.82 | 70.87 | 78.26 | 89.21 | 106.14 | 71.41 | +10.1 | +7.2 | +11.9 | +0.8 |
| math | 4 | 97.62 | 111.34 | 113.68 | 92.20 | 112.00 | 125.23 | 130.82 | 93.31 | +14.7 | +12.5 | +15.1 | +1.2 |
| chat | 1 | 41.12 | 43.89 | 40.95 | 41.31 | 43.59 | 47.10 | 44.57 | 41.62 | +6.0 | +7.3 | +8.8 | +0.8 |
| chat | 2 | 61.76 | 65.21 | 65.53 | 62.91 | 66.26 | 71.01 | 74.00 | 63.20 | +7.3 | +8.9 | +12.9 | +0.5 |
| chat | 4 | 92.54 | 93.17 | 91.63 | 87.03 | 104.19 | 105.04 | 103.57 | 87.03 | +12.6 | +12.7 | +13.0 | +0.0 |

All-class means:

| C | K5 K1 | K5 K2 | K5 K3 | K5 DGPP | new K1 | new K2 | new K3 | new DGPP | dK1% | dK2% | dK3% | dDGPP% |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45.45 | 52.85 | 55.49 | 46.96 | 48.42 | 56.90 | 59.89 | 47.29 | +6.5 | +7.7 | +7.9 | +0.7 |
| 2 | 67.72 | 75.46 | 82.10 | 67.95 | 72.94 | 82.37 | 91.53 | 68.24 | +7.7 | +9.2 | +11.5 | +0.4 |
| 4 | 95.51 | 103.81 | 106.20 | 90.93 | 107.07 | 116.68 | 121.39 | 91.33 | +12.1 | +12.4 | +14.3 | +0.4 |

`dK*%` = new vs the K5 NInfer arm at the same K; `dDGPP%` = new DGPP vs its
own K5 run (machine control).

## K=1 against DGPP (like-for-like: DGPP drafts one token deep)

| C | K5 K1 | K5 DGPP | K5 Δ | new K1 | new DGPP | new Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45.45 | 46.96 | −3.2% | 48.42 | 47.29 | +2.4% |
| 2 | 67.72 | 67.95 | −0.3% | 72.94 | 68.24 | +6.9% |
| 4 | 95.51 | 90.93 | +5.1% | 107.07 | 91.33 | +17.2% |

Per class at C1 (new): prose 45.54 vs 45.43 (+0.2%), code 49.89 vs 50.38
(−1.0%), json 52.84 vs 50.92 (+3.8%), math 50.24 vs 48.08 (+4.5%), chat 43.59
vs 41.62 (+4.7%). Per class at C4 (new): prose +15.9%, code +14.6%, json
+16.0%, math +20.0%, chat +19.7%.

## Decode-only rates and acceptance (request log, server side)

Decode tok/s is the per-request server-side decode rate averaged over the 35
requests of each class-concurrency; acceptance is tokens per decode round.

| K | K5 accept | new accept | K5 decode C1/C2/C4 | new decode C1/C2/C4 | decode Δ C1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.802 | 1.803 | 47.1 / 36.5 / 26.9 | 50.5 / 39.5 / 30.2 | +7.2% |
| 2 | 2.427 | 2.425 | 55.3 / 41.2 / 30.0 | 59.6 / 45.2 / 34.1 | +7.8% |
| 3 | 2.922 | 2.917 | 58.5 / 45.1 / 31.6 | 63.6 / 50.9 / 36.0 | +8.7% |

Acceptance is flat across the batch: per-class deltas ≤ 0.006 tok/round,
within run-to-run — the M1/M2 schedule change did not move MTP acceptance in
this protocol. Decode-only is up 7–14% at every K and C (C2 +8.2/+9.7/+12.9%,
C4 +12.3/+13.7/+13.9% for K1/K2/K3), so the batch's gain is per-round
latency, not acceptance.

Per class at C1, decode tok/s (new, K1/K2/K3): prose 46.4/49.0/47.6, code
52.2/63.6/67.6, json 55.0/69.3/80.5, math 53.2/67.3/75.6, chat 45.4/48.9/46.6.
Per class acceptance (new, K1): prose 1.630, code 1.857, json 1.966, math
1.895, chat 1.665.

Device wait (request mean): K1 52.2, K2 61.6, K3 68.8 ms/round.

## C4 per class

New (K1 / K2 / K3 / DGPP), K5 in parentheses:

| class | new K1 | new K2 | new K3 | new DGPP | K5 K1 | K5 K2 | K5 K3 | K5 DGPP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prose | 99.01 | 99.49 | 98.79 | 85.42 | 89.10 | 89.26 | 86.19 | 85.12 |
| code | 107.63 | 122.32 | 130.79 | 93.91 | 98.06 | 109.08 | 114.89 | 94.00 |
| json | 112.52 | 131.34 | 142.97 | 96.97 | 100.24 | 116.22 | 124.68 | 96.31 |
| math | 112.00 | 125.23 | 130.82 | 93.31 | 97.62 | 111.34 | 113.68 | 92.20 |
| chat | 104.19 | 105.04 | 103.57 | 87.03 | 92.54 | 93.17 | 91.63 | 87.03 |

C4 runs four requests per leg and its output is not identical from run to
run; per the K5 rule the evidence rests on the C1 rows and the decode-only
rates (both positive above). The C4 rows are directional and consistent
with them (all positive at +9.8 to +15.1%).

## Notes

- The DGPP rerun is the machine control: at the same commit it moves +0.4 to
  +0.7% on every row against its own K5 run — flat — so the NInfer deltas are
  attributable to the tree, not the box.
- `bench_compare.py` (campaign log) refuses the pair on model-name mismatch
  (`RadixArk/Qwen3.8-Flash-Next-NVFP4` vs `qwen3.8-flash-next-125b-a6b`); the
  tables above are computed from the raw serve_load JSON and request logs.
