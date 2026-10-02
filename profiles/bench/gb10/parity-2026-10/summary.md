# GB10 parity: NInfer vs DGPP (parity-2026-10)

2026-10-02. Block L item 1 of `docs/maintainer/plan-2026-10-gb10.md`.
Old baseline: `profiles/bench/gb10/block-i/I9` (2026-09-30 13:4x-14:2xZ), untouched.

## Protocol

Identical to the original I9 run: one shared load client (`serve_load.py`), five
classes (prose, code, json, math, chat), concurrency C1/C2/C4, 2048-char
prompts, 256 output tokens, greedy, thinking preserved, 3 repeats per
class-concurrency.

- NInfer arms: this fork at `0e8f7516` (working tree; carries the uncommitted
  PLE decode fix), artifact `qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer`,
  fp8 KV, CUDA graphs on, MTP draft K = 1/2/3, port 18087.
- DGPP arm: `dd58d6d3` (the same commit as the old run), RadixArk
  Qwen3.8-Flash-Next-NVFP4 W1 config (mtp_depth 1, bf12+bf16 routed experts,
  fp8 dense, decode_graph on, kv_capacity 262144), `DGPP_RESIDENT_CACHE=off`,
  port 18080 (dgpp-cluster default, same as the old run).
- Old NInfer baseline: pre-merge tree (QSA off, HC unfused, host 4.52
  ms/round, K=1 only).

## Scoreboard

tok/s, all repeats averaged.

| class | C | NInfer old | K1 | K2 | K3 | DGPP old | DGPP new |
|---|---:|---:|---:|---:|---:|---:|---:|
| prose | 1 | 37.52 | 39.82 | 40.70 | 39.69 | 44.89 | 44.64 |
| prose | 2 | 55.21 | 56.97 | 54.02 | 50.70 | 63.04 | 62.41 |
| prose | 4 | 73.85 | 74.60 | 69.73 | 65.00 | 83.87 | 83.86 |
| code | 1 | 42.97 | 44.32 | 52.09 | 54.15 | 49.86 | 49.54 |
| code | 2 | 61.52 | 63.53 | 73.18 | 76.25 | 69.36 | 68.89 |
| code | 4 | 81.76 | 84.91 | 89.17 | 91.91 | 93.36 | 93.06 |
| json | 1 | 46.78 | 47.91 | 58.64 | 68.43 | 50.31 | 50.06 |
| json | 2 | 68.73 | 70.71 | 83.53 | 92.66 | 72.93 | 72.70 |
| json | 4 | 89.74 | 89.38 | 100.96 | 104.46 | 95.13 | 95.75 |
| math | 1 | 43.33 | 44.55 | 55.36 | 59.66 | 47.54 | 47.52 |
| math | 2 | 65.11 | 67.38 | 76.35 | 84.08 | 70.93 | 70.59 |
| math | 4 | 84.73 | 86.36 | 93.09 | 89.57 | 92.71 | 91.18 |
| chat | 1 | 39.24 | 38.22 | 40.94 | 38.68 | 41.05 | 40.86 |
| chat | 2 | 56.39 | 56.89 | 58.47 | 56.53 | 62.86 | 62.29 |
| chat | 4 | 75.76 | 78.60 | 74.08 | 70.79 | 85.54 | 86.15 |

All-class means:

| C | NInfer old | K1 | K2 | K3 | DGPP old | DGPP new |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41.97 | 42.96 | 49.55 | 52.12 | 46.73 | 46.53 |
| 2 | 61.39 | 63.10 | 69.11 | 72.05 | 67.82 | 67.38 |
| 4 | 81.17 | 82.77 | 85.41 | 84.34 | 90.12 | 90.00 |

DGPP new vs old (identical commit): -0.4% / -0.7% / -0.1% at C1/C2/C4 -
the protocol noise floor.

## Acceptance and round anatomy

26,944 tokens per arm, mixed classes, from the `serve_load` request logs:

| arm | decode tok/s | tok/round | device ms/round | host ms/round |
|---|---:|---:|---:|---:|
| old K1 | 26.4 | 1.80 | 63.6 | 4.52 |
| new K1 | 26.8 | 1.79 | 66.8 | 0.03 |
| new K2 | 28.1 | 2.37 | 84.2 | 0.04 |
| new K3 | 28.1 | 2.78 | 99.0 | 0.04 |

## Verdict

- vs the old NInfer (QSA, fused HC, admission re-arm, K1-K3 host removal,
  PLE decode fix): C1 +2.4%, C2 +2.8%, C4 +2.0% (K1 vs K1). Host exposure
  4.52 -> 0.03 ms/round; acceptance flat (1.80 -> 1.79 tok/round).
- vs DGPP (fixed at mtp_depth 1):
  - C1: K1 -8.4%; K2 +6.6%; K3 +12.2%
  - C2: K1 -6.5%; K2 +2.4%; K3 +6.8%
  - C4: K1 -8.1%; K2 -5.3%; K3 -6.4%

  NInfer leads at C1/C2 when it uses deeper drafts. The C4 gap narrows from
  -11% (old) to -8.1% (K1) / -5.3% (K2). NInfer's best K is
  concurrency-dependent: K3 at C1, K2 at C4 - acceptance collapses at depth 3
  under concurrency (C4: K2 85.41 > K3 84.34 > K1 82.77).

## Round attribution (C4, draft 1)

`nsys` on the `NINFER_PERFORMANCE_TRACE` build (`build-trace/`), two batches
of 4 x 256-token greedy requests, `decode.mtp_round` NVTX range. Measured
wall 313.0 ms; GPU busy 307.5 ms; attributed GPU work 90.7%.

Per-round GPU work (76.9 ms/round):

| stage | ms/round | share | byte-model efficiency |
|---|---:|---:|---|
| target.verify moe.nvfp4 | 42.86 | 55.7% | 16-104% (coarse expert-activation envelope) |
| target.verify gdn.record | 11.69 | 15.2% | 89.1% (near roofline) |
| target.verify qsa.select | 6.05 | 7.9% | 45.0% (~2x headroom) |
| target.verify hyper.combine_mix | 5.17 | 6.7% | 52.2% |
| target.verify ple.record | 2.27 | 3.0% | 11.8% |
| predictor (moe, qsa, hyper) | 1.52 | 2.0% | - |
| unattributed | 28.6 ms total | 9.3% | launch gaps, non-captured work |

MoE decode is the dominant cost (3.7x GDN). GDN record is near its byte-model
roofline, so little is left there. QSA select shows the clearest headroom
(2x vs the byte model). Full table: `round-attribution/report.md`.

## RoPE check (Block L item 0)

`tools/gb10/rope_check.sh` on upstream `c1bd1a5e` (the batch-3 shape fix):
PASS with CUDA graphs (4-concurrent C4 MTP load completed, 46.0 tok/s) and
PASS with `--no-cuda-graph` (47.7 tok/s). The old failure modes (graph-on
startup crash; no-graph 3-batch shape error) are gone. Upstream `e5c45144`
still SPLITs output text (separate PR); the fork tree does not carry the shape
test that crashed. Evidence: `profiles/bench/gb10/rope-check/`.

## Execution notes

- 2026-10-02 04:57Z: the first attribution run lost its trace. The load
  completed at 04:19:44Z, but the script's `pkill -f '^build-trace/...'`
  pattern did not match nsys's re-execed absolute-path cmdline; the server
  idled under nsys until the 2400 s job timeout, and the harness's
  process-group kill took the nsys client down mid-finalize; the orphaned
  agent (SIGTERM at 05:03Z) exported nothing. Fallback: match by process name
  (`pkill -TERM -x ninfer-serve`) plus a 15-minute finalize watchdog in
  `round_attribution.sh`; reran at 05:03:47Z, completed in 57 s
  (`build-trace/` warm).
- Raw nsys bulk (`trace.sqlite`, `trace.nsys-rep`) is not committed; the
  parsed records are.
