# GB10 parity: NInfer vs DGPP after K5 (parity-2026-10-k5)

2026-10-02. Plan run list "Parity scoreboard after K5" (tree `e0a140e2`).
Prior scoreboards: `parity-2026-10` (2026-10-02, tree `0e8f7516` + uncommitted PLE
decode fix) and `block-i/I9` (original I9, 2026-09-30).

## Protocol

Identical to the original I9 and the `parity-2026-10` rerun: one shared load
client (DGPP's `serve_load.py`), five classes (prose, code, json, math, chat),
concurrency C1/C2/C4, 256 output tokens, greedy (temperature 0), thinking off,
3 repeats per class-concurrency (45 phases, 105 requests per arm, all
status 200, finish length).

- NInfer arms: tree `e0a140e2`, artifact `qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer`,
  fp8 KV, CUDA graphs on, `--max-concurrency 4`, MTP draft K = 1/2/3, port 18087.
  Drop caches before each arm.
- DGPP arm: `dd58d6d3` (same commit as both prior runs), RadixArk
  Qwen3.8-Flash-Next-NVFP4 W1 config (mtp_depth 1, bf12+bf16 routed experts,
  fp8 dense, decode_graph on, kv_capacity 262144), `DGPP_RESIDENT_CACHE=off`,
  port 18080, drop caches before start.
- Since the `parity-2026-10` NInfer arm: the PLE page-in (+18% C4 decode),
  the grouped MoE route (+3.1% C4), and K5 (+2.1% C4).

## Scoreboard

tok/s, repeats averaged.

| class | C | 09-30 old | p10 K1 | p10 K2 | p10 K3 | new K1 | new K2 | new K3 | DGPP 09-30 | DGPP p10 | DGPP new |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prose | 1 | 37.52 | 39.82 | 40.70 | 39.69 | 42.61 | 44.67 | 43.16 | 44.89 | 44.64 | 45.11 |
| prose | 2 | 55.21 | 56.97 | 54.02 | 50.70 | 62.58 | 61.65 | 60.88 | 63.04 | 62.41 | 63.08 |
| prose | 4 | 73.85 | 74.60 | 69.73 | 65.00 | 89.10 | 89.26 | 86.19 | 83.87 | 83.86 | 85.12 |
| code | 1 | 42.97 | 44.32 | 52.09 | 54.15 | 46.89 | 55.91 | 58.72 | 49.86 | 49.54 | 50.03 |
| code | 2 | 61.52 | 63.53 | 73.18 | 76.25 | 68.87 | 78.83 | 88.00 | 69.36 | 68.89 | 69.64 |
| code | 4 | 81.76 | 84.91 | 89.17 | 91.91 | 98.06 | 109.08 | 114.89 | 93.36 | 93.06 | 94.00 |
| json | 1 | 46.78 | 47.91 | 58.64 | 68.43 | 49.58 | 61.37 | 70.00 | 50.31 | 50.06 | 50.51 |
| json | 2 | 68.73 | 70.71 | 83.53 | 92.66 | 74.31 | 88.43 | 101.24 | 72.93 | 72.70 | 73.25 |
| json | 4 | 89.74 | 89.38 | 100.96 | 104.46 | 100.24 | 116.22 | 124.68 | 95.13 | 95.75 | 96.31 |
| math | 1 | 43.33 | 44.55 | 55.36 | 59.66 | 47.07 | 58.43 | 64.62 | 47.54 | 47.52 | 47.85 |
| math | 2 | 65.11 | 67.38 | 76.35 | 84.08 | 71.06 | 83.20 | 94.82 | 70.93 | 70.59 | 70.87 |
| math | 4 | 84.73 | 86.36 | 93.09 | 89.57 | 97.62 | 111.34 | 113.68 | 92.71 | 91.18 | 92.20 |
| chat | 1 | 39.24 | 38.22 | 40.94 | 38.68 | 41.12 | 43.89 | 40.95 | 41.05 | 40.86 | 41.31 |
| chat | 2 | 56.39 | 56.89 | 58.47 | 56.53 | 61.76 | 65.21 | 65.53 | 62.86 | 62.29 | 62.91 |
| chat | 4 | 75.76 | 78.60 | 74.08 | 70.79 | 92.54 | 93.17 | 91.63 | 85.54 | 86.15 | 87.03 |

All-class means:

| C | 09-30 old | p10 K1 | p10 K2 | p10 K3 | new K1 | new K2 | new K3 | DGPP 09-30 | DGPP p10 | DGPP new |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41.97 | 42.96 | 49.55 | 52.12 | 45.45 | 52.85 | 55.49 | 46.73 | 46.53 | 46.96 |
| 2 | 61.39 | 63.10 | 69.11 | 72.05 | 67.72 | 75.46 | 82.10 | 67.82 | 67.38 | 67.95 |
| 4 | 81.17 | 82.77 | 85.41 | 84.34 | 95.51 | 103.81 | 106.20 | 90.12 | 90.00 | 90.93 |

(`09-30 old` = original I9 NInfer arm, pre-merge tree, K1 only; `p10` =
`parity-2026-10`; DGPP was rerun in every campaign, `dd58d6d3` throughout.)

vs DGPP new (all-class means): C1 K1 −3.2%, K2 +12.5%, K3 +18.2%; C2 K1 −0.3%,
K2 +11.0%, K3 +20.8%; C4 K1 +5.1%, K2 +14.3%, K3 +16.8%.

Per class at C4, NInfer best K vs DGPP: prose K2 89.26 vs 85.12 (+4.9%), code
K3 114.89 vs 94.00 (+22.2%), json K3 124.68 vs 96.31 (+29.5%), math K3 113.68
vs 92.20 (+23.3%), chat K2 93.17 vs 87.03 (+7.0%). In the original I9 DGPP
led prose and chat at C4 by +28–31%; those gaps have flipped.

## Decode-only and acceptance (request_log_summary, 26,944 tokens per arm)

| arm | decode tok/s | tok/round | device ms/round | host ms/round |
|---|---:|---:|---:|---:|
| 09-30 K1 | 26.4 | 1.80 | 63.6 | 4.52 |
| p10 K1 / K2 / K3 | 26.8 / 28.1 / 28.1 | 1.79 / 2.37 / 2.78 | 66.8 / 84.2 / 99.0 | 0.03 / 0.04 / 0.04 |
| new K1 / K2 / K3 | 31.0 / 34.4 / 35.8 | 1.79 / 2.37 / 2.79 | 57.6 / 68.7 / 77.7 | 0.13 / 0.15 / 0.21 |

Acceptance is flat against `parity-2026-10` (1.79 / 2.37 / 2.79 vs
1.79 / 2.37 / 2.78 tok/round): the PLE fork fix did not move MTP acceptance on
this protocol. Decode-only rose +15.7% (K1), +22.4% (K2), +27.4% (K3) against
`parity-2026-10`, and the gain scales with draft depth; device wait per round
fell 9–21 ms, consistent with the PLE page-in removing the host-wait.

## DGPP engine rate (dgpp-metrics.json, `dd58d6d3`)

| run | engine rate | steps/s | draft acceptance |
|---|---:|---:|---:|
| 09-30 (original I9) | 76.3 tok/s | 18.27 | 81.7% |
| p10 | 75.9 tok/s | 18.17 | 81.6% |
| new | 76.6 tok/s | 18.32 | 81.7% |

Engine rate = tokens_generated / (step_ms/1000); 26,944 tokens, 6,441 decode
steps this run. Stable within 1% at the fixed commit.

## Verdict

- C4 all-class: NInfer leads DGPP by +16.8% at the best K (K3, 106.20 vs
  90.93). Parity at the scoreboard level is reached and exceeded; the decision
  rule's "still >3% behind" branch does not trigger.
- Per class at C4: NInfer leads all five classes at the best K (table above);
  prose remains the closest (+4.9%) and its best K is K2 (deep drafts lose on
  prose, as before: K1 89.10 > K3 86.19).
- vs `parity-2026-10` NInfer: C4 +15.4% (K1), +21.5% (K2), +25.9% (K3);
  C1 +5.8%/+6.6%/+6.4%; C2 +7.3%/+9.0%/+13.9%. The PLE page-in dominates.
- DGPP (fixed at `dd58d6d3`, mtp_depth 1): wall rates stable within 1% across
  all three campaigns; engine rate 75.9–76.6 tok/s.

## Execution notes

- One clean pass; no leg rerun, no nonzero load exit, no server error. All 105
  requests per arm: status 200, finish length, 256 tokens. Each NInfer server
  healthy within 5 s of start; DGPP preflight "0 failed check(s)".
- `compare-k1.txt`: `uncomparable runs: unmatched model` — the two json files
  carry different model strings; pre-existing tooling limitation, identical in
  both prior campaigns. The scoreboard is built from the json files directly.
- Campaign wall 18:02:24–18:28:26Z (26 min): three NInfer arms (serve ~25 s +
  load ~6 min each), DGPP arm (start + load ~7 min).
