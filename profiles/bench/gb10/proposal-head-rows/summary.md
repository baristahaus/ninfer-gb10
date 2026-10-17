# Optimized proposal head row truncation (--proposal-head-rows)

2026-10-17. Machine: GB10 workstation (sm_121a, 20-core Arm, 121 GiB unified), idle.
Artifact: `out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer` (FP8 MTP, local).
Build: `build/` at this tree, CUDA 13.0.88.

## What the knob does

`--proposal-head-rows N` (0 = full) truncates the Flash-Next optimized proposal head to its
first N frequency-ranked rows. The shortlist rows are score-ordered, so a prefix is the
top-frequency token set. The draft pass over the head shrinks from 147456 rows to N; target
verification stays exact (full lm_head), so committed tokens are unchanged in distribution.
Multiple of 128, at least 1024; MTP + `--lm-head-draft` only.

Implementation: `SpeculativeOptions.proposal_head_rows` (public) -> `StartupFeatures` ->
125b `model.cpp` prefix row view of the Q4 row-split head -> `TextContext` dynamic
shortlist rows; Q4 dispatch accepts 128-aligned k=2560 shapes (same kernels, dynamic grid).
Workspace still reserves full-head scratch (upper bound, unused tail).

## Battery

`ninfer_bench --kv-dtype fp8 --spec mtp --draft-tokens 2 --lm-head-draft --sampling greedy
--repetitions 3 --proposal-head-rows {0,24576,47104,98304}`, default pp512+tg128, tg128 shown.

| rows  | decode tok/s (mean ± s) | acceptance | rounds |
|-------|------------------------|------------|--------|
| full  | 29.97 ± 0.07           | 0.122      | 309    |
| 24576 | 30.86 ± 0.01           | 0.117      | 309    |
| 47104 | 31.48 ± 0.02           | 0.134      | 303    |
| 98304 | 29.96 ± 0.50           | 0.122      | 309    |

Prefill pp512 unchanged (~1306–1339 tok/s across configs; draft head is not in the prefill path).

## Reading

- Truncation to 47104 rows gains ~5% decode over full with no acceptance loss on this corpus.
- 24576 also slightly ahead; 98304 within noise of full.
- Single synthetic corpus, greedy, one prompt shape: directionally useful, not a campaign.
- Stochastic-sampling outputs shift with truncation because the RNG draw alignment changes
  with round composition; greedy output is the comparable surface.

Raw reports: `rows_full.json`, `rows_24576.json`, `rows_47104.json`, `rows_98304.json`.

## Round capture (--mtp-capture)

2026-10-17 (same day, later). Engine-side per-row MTP round capture: every batched decode round
appends fixed 48-byte records (k=2) — proposal ids, verified committed tokens, accept count, KV
position, lane — from the host-resident pinned ingress/egress mirrors, through a background
writer to disk. No device work; analysis is offline (`tools/mtp_capture_summary.py`).

Smoke results, greedy, rows=47104:

- CLI single request: 72 rows, acceptance 16.3% (matches the engine summary exactly);
  16/118 rejected drafts outside the 47104-row prefix on a chat prompt.
- bench, concurrency 2: 211 rounds, 416 rows, acceptance 13.1%;
  **0/714 rejected drafts outside the 47104-row prefix** on the synthetic corpus — every
  rejection was a within-prefix ranking miss, so 47104 rows cost nothing on that workload.

Unit test `ninfer_mtp_capture_test` verifies the file format byte-for-byte.

## TEB natural-corpus campaign (2026-10-17, later)

Corpus: Terminal-Bench 2.1 instruction texts, tokenized (`teb/teb_corpus.ids`, 21795 tokens,
non-repeating — the stock fixture tiles one text and drafts it nearly perfectly). Bench
pp512+tg128, greedy, 3 reps, concurrency 2, draft window 2, capture on every run.

Decode (tg128):

| rows  | decode tok/s | acceptance |
|-------|--------------|------------|
| full  | 27.44 ± 0.01 | 0.257      |
| 98304 | 27.70 ± 0.02 | 0.257      |
| 47104 | 27.73 ± 0.02 | 0.249      |
| 24576 | **27.85 ± 0.01** | 0.249  |
| 16384 | 27.29 ± 0.01 | 0.231      |
| 12288 | 26.99 ± 0.01 | 0.223      |
| 6144  | 26.62 ± 0.02 | 0.207      |

Sizing from the full-head capture (`teb/teb_cap_full.bin`), per candidate prefix:

| rows  | accepted-outside (loss) | rejected-outside (wasted verify) |
|-------|-------------------------|----------------------------------|
| 6144  | 50 (14.5%)              | 306 (30.5%)                      |
| 12288 | 32                      | 218                              |
| 16384 | 24                      | 186                              |
| 24576 | 8 (2.3%)                | 146                              |
| 47104 | 8 (2.3%)                | 82                               |
| 98304 | 0                       | 42                               |

Metric semantics: the truncated head can only propose inside its own prefix, so the honest
loss signal comes from the FULL-head capture — accepted drafts outside the prefix are
acceptance the truncation cannot reach; rejected drafts outside are verify bandwidth spent
on tokens a truncated head would never propose.

Reading: 24576 rows keeps 97.7% of reachable acceptance and beats full decode on this
corpus; 6144 costs 5.0pp. The tuner (`tools/gb10/tune_proposal_rows.sh`) automates this
analysis: one full-head capture pass, then the loss table over candidate prefixes. The
choice is startup-fixed because CUDA graphs bake the row count; capture → tune → restart
is the adaptive loop.

Raw captures and reports in `teb/`.

## Cross-domain, cross-K campaign (2026-10-17, latest)

Corpora: prose (perplexity-1m streams, 200k tokens), code (TEB solutions/tests, 200k tokens),
TEB instructions (21.8k). K = draft window. Full-head capture per (domain, K), greedy,
concurrency 2, 3 reps. Accepted-drafts-outside-prefix per candidate prefix:

| rows  | teb k2 | prose k2 | prose k3 | code k2 | code k3 | worst loss |
|-------|--------|----------|----------|---------|---------|------------|
| 6144  | 50     | 96       | 96       | 90      | 90      | 42.5%      |
| 12288 | 32     | 88       | 88       | 58      | 58      | 38.9%      |
| 16384 | 24     | 64       | 64       | 50      | 50      | 28.3%      |
| 24576 | 8      | 64       | 64       | 32      | 32      | 28.3%      |
| 47104 | 8      | 16       | 16       | 16      | 16      | 7.1%       |
| 65536 | 0      | 16       | 16       | 0       | 0       | 7.1%       |
| 98304 | 0      | 0        | 0        | 0       | 0       | 0.0%       |

Findings:
- The loss table is K-independent: the same prefix serves K=2 and K=3 (K only scales the
  per-round drafted-token count, not which tokens the head can reach).
- The knee is domain-dependent. TEB instructions sit at 24576; prose needs 98304 for zero
  loss; code 65536.
- Measured decode (K=2, tg128): 98304 rows is fastest on every domain and its measured
  acceptance is at or above full (trajectory noise ±0.01):

| domain | full 147456 | 98304 | 47104 |
|--------|-------------|-------|-------|
| prose  | 24.13 ± 0.01 | **24.88 ± 0.01** | 24.46 ± 0.02 |
| code   | 26.58 ± 0.02 | 27.14 ± 0.02 | 27.20 ± 0.03 |

Decision: **98304 is the best single default** across all tested domains (+3.1% prose,
+2.1% code, +0.9% TEB decode vs full, no acceptance loss). 47104 trades a real 7.1%
worst-case loss for a smaller draft pass and loses to 98304 in decode on prose. Per-domain
tuning remains available through `tools/gb10/tune_proposal_rows.sh` when a workload's
domain is known. The engine default stays at full until this is adopted.

Raw captures and reports in `domaink/`.
