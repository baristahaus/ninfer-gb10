# C4 K=1 A/B results — PR #36 speed decision (2026-09-30)

Protocol: `runbook.md` (verbatim brief + machine mapping). Trees: fix = `a1a43667`
(engine code `eb9e87fa`), base = `ed6525fa`. Artifact: fp8_mtp (30 GB). Workload:
`--max-context 73728 --max-concurrency 4 --kv-dtype fp8 --preserve-thinking --spec mtp
--draft-tokens 1 --lm-head-draft`; DGPP `serve_load.py --concurrency 4 --classes prose
--max-tokens 256`, one throwaway warm-up pass then the measured pass (3 waves, `--warm 0`).
Order: fix, base, base, fix; server restarted per run; page cache warmed first (cache
check below). 4 verify rows + 4 draft rows per round.

## Headline

- **Steady-state decode (the clean `running 4 | batch 4.00` waves 2–3): fix 83.9 tok/s
  vs base 79.7 tok/s → fix is +4.1 tok/s (+5.2%).**
- **QSA attention + selection stage (nsys, same method both trees): fix 0.328 ms/round
  vs base 0.817 ms/round → −60%.**
- The per-class per-run aggregate (median of the 3 measured waves, prefill included):
  fix 84.5 / 82.9, base 78.1 / 80.4.
- The stage drop lands exactly where #36 changes the engine:
  - per-round index selection is skipped in the fix tree: `score_groups_batched` and
    `select_top_groups` go from ~9.6k launches to ~52 (all-visible-keys-in-budget path);
  - `selected_attention_split_fp8` per-launch time drops 0.1400 → 0.0522 ms (−63%,
    split sizing); `reduce_selected_attention_splits` 0.0143 → 0.0066 ms (−54%);
  - the rest of the selection family (mma/hierarchical/order) has identical launch
    counts on both trees.
- Decision per the brief ("if the fix is at parity or better and the QSA stage drops
  substantially: merge #36"): **both conditions hold → merge.**

## Per-run table (measured pass only: 12 requests, 3072 tokens)

Per-class = `serve_load.py` median of the 3 wave aggregates, wall, prefill included.
Rounds / tok-per-round / waits from `request_log_summary.py` (full per-request tables:
`run*/request_summary.md`).

| run | tree | per-class tok/s | waves (tok/s) | rounds | tok/round | device ms/round | host ms/round |
|---|---|---:|---|---:|---:|---:|---:|
| 1 | fix  | **84.5** | 60.4 / 84.5 / 84.7 | 1860 | 1.65 | 64.4 | 2.05 |
| 2 | base | **78.1** | 58.8 / 78.1 / 78.6 | 1920 | 1.60 | 67.1 | 1.89 |
| 3 | base | **80.4** | 59.9 / 81.8 / 80.4 | 1925 | 1.60 | 66.2 | 0.21 |
| 4 | fix  | **82.9** | 59.0 / 82.9 / 83.3 | 1881 | 1.63 | 66.5 | 2.66 |

Tree means (2 runs each): fix 83.7 tok/s (spread 82.9–84.5), base 79.3 tok/s
(spread 78.1–80.4) → **fix +4.4 tok/s (+5.6%) on the per-class metric, +5.2% on the
clean steady-state waves**. Per round: fix 1870.5 (1.64 tok/round) vs base 1922.5
(1.60) → fix completes the same 3072 tokens in **2.7% fewer rounds** with 2.5% more
accepted tokens per round; device wait 65.5 vs 66.7 ms/round (−1.8%).

## Width check

Steady-state waves 2–3 of every run show `running 4 (decode-ready 4) | batch 4.00`
throughout — the passes are valid.

**Flag (per runbook):** the first measured wave of every run shows a ~9 s admission
hold on 2 of the 4 requests: throughput lines `running 2 (decode-ready 2) | waiting 2`
(one window `running 1 | materializing 1` in run 1), and TTFT 8.6–11.0 s for the held
requests vs <0.5 s for the other two (run 1: req 0 8693 ms, req 3 8899 ms; run 2: req 1
8965 ms, req 3 8779 ms; run 3: req 2 8566 ms, req 3 8711 ms; run 4: req 3 11037 ms).
The held requests' prefills are 7–14 tokens (prefix cache hits from the warm-up pass),
so the hold is admission, not prefill work.

This is symmetric across both trees (all four runs, both hold patterns) — a protocol
artifact of re-sending the same 4 prompts (warm-up + 3 waves), not a tree difference.
It is the same admission hold the C-DECODE base stall showed at larger width. The
per-class aggregate is the median wave (wave 2), which ran at full width, so the
headline numbers come from clean passes; wave 1 is excluded from them.

Flagged windows (full logs: `run*/server.log`):
- run1-fix: 19:56:22–19:56:32 (`running 2 | waiting 2`, `running 1 | materializing 1`)
- run2-base: 19:57:46–19:57:56 (`running 2 | waiting 2` × 2)
- run3-base: 19:59:14–19:59:24 (`running 2 | waiting 2`, `running 2`)
- run4-fix: 20:00:39–20:00:49 (`running 3 | waiting 1` × 2, `running 1`), 20:01:04
  (`running 1 | waiting 3`)

## nsys stage pair (attention + QSA selection, at 8 rows)

Capture: `nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none
--cpuctxsw=none`, Block J workload (single `serve_load.py --repeat 3` invocation,
13 requests incl. the 64-token warm request). Per decode step = total GPU time of the
QSA op kernels ÷ engine decode rounds from the request log (1903 fix / 1956 base).
Full kernel tables: `nsys-{fix,base}/stage.txt`.

| kernel (QSA stage) | fix | base |
|---|---|---|
| **stage total** | **624.3 ms / 39,422 launches** | **1597.4 ms / 69,065 launches** |
| **per decode step** | **0.328 ms** | **0.817 ms** |
| selected_attention_split_fp8 | 489.5 ms, 9368, 0.0522 ms/launch | 1342.3 ms, 9589, 0.1400 ms/launch |
| selected_attention_batched_fp8 | 11.0 ms, 286, 0.0385 | 10.7 ms, 286, 0.0373 |
| reduce_selected_attention_splits | 63.5 ms, 9576, 0.0066 | 140.4 ms, 9797, 0.0143 |
| compress_index_groups | 28.8 ms, 9654, 0.0030 | 26.7 ms, 9875, 0.0027 |
| prepare_index_query | 19.8 ms, 9654, 0.0020 | 19.4 ms, 9875, 0.0020 |
| score_groups_batched | 0.3 ms, 52 | 22.7 ms, 9641, 0.0024 |
| select_top_groups | 0.1 ms, 52 | 15.3 ms, 9641, 0.0016 |
| score_groups_mma | 2.4 ms, 156 | 2.3 ms, 156 |
| hierarchical_top_groups | 7.4 ms, 416 | 7.4 ms, 416 |
| order_groups_like_persistent_topk | 1.5 ms, 208 | 10.3 ms, 9789, 0.0010 |

Composition change (fix vs base): the per-round selection kernels
(`score_groups_batched`, `select_top_groups`) are no longer launched per round (52
launches = prefill only) — the all-visible-keys-in-budget path skips index selection;
`order_groups_like_persistent_topk` drops from per-round (9789) to 208 launches; the
split attention keeps its per-round launch count (9368 vs 9589 ≈ 4.9/round) but each
launch is 2.7× faster (split sizing). Net stage: **−60.3%**, launch count −42.9%.

**Method note vs the Block J reference (5.7 ms/step at 8 rows):** Block J's figure is
from its window-based method (stage total over graph-replay windows, `profiles/nsys/
block-j/`), not stage-total-over-engine-rounds as here. The absolute numbers are not
directly comparable; the tree-vs-tree comparison above (identical method, identical
workload, same 2025.3.2 capture flags) is the decision-relevant one.

## Cache check (before the A/B)

Fix tree, warm page cache, the C-DECODE 1-request repeat (K=1): prefill 1.36 s for the
1,882-token request (C-DECODE cold fix: 4.84 s; base warm: 1.28 s). The run-order /
cold-page-cache explanation for the C-DECODE prefill gap holds. Decode 42.1 tok/s
(C-DECODE: fix 41.1 cold / base 43.2). Details: `cache-check/`.

## Stall repro (C-DECODE admission stall, base tree)

Protocol per the runbook: base tree, C-DECODE flags (`--max-concurrency 8 --spec mtp
--draft-tokens 2 --lm-head-draft`). Phase 1: one request (wikitext-00; 317 tokens, 8.5 s).
Phase 2: once it finished, 8 requests at once with the phase-1 prompt (the replay) placed
second. Client timing: `stall-repro/stall.json`; server log: `stall-repro/server.log`.

**The C-DECODE stall signature did not reproduce.** No `running 2 | waiting 6` with no
prefill; admission processed the 8 requests smoothly:

| time (UTC) | width line |
|---|---|
| 20:06:22 | `running 1 (prefill 1) | waiting 7` — first request in prefill (1,024 tok) |
| 20:06:27 | `running 3 (prefill 1, decode-ready 2) | waiting 5`, host 61.0 % |
| 20:06:32 | `running 5 (prefill 1, decode-ready 4) | waiting 3`, host 49.8 % |
| 20:06:37 | `running 7 (decode-ready 7) | waiting 1` |
| 20:06:42+ | steady decode, 5–7 running, batch up to 6.85 |

Prefills were processed serially (~17 s; ~13–14 k tokens; host 50–61 % during that span —
PLE gather for the new prompt text). The replay request (second in the batch, cached from
phase 1) neither blocked nor held admission. All 8 requests completed 48–52 s after the
fire (512-token requests at a 6–7-wide batch); no request was held by a stalled peer.

Interpretation: the specified repro does not trigger the C-DECODE stall on this tree, so
the "state fork that never settles during MTP decode" hypothesis is not confirmed by this
repro. The C-DECODE stall (19:37:09–19:37:19, base, MC=8, draft 2) required a condition
not present here (different admission order, or a fork from the N=4→N=8 batch transition
of the sweep). The ~9 s wave-1 admission hold seen in all four A/B measured passes (both
trees, MC=4, 2 of 4 held, cached-prefill requests — see the width check above) is a
milder, symmetric variant of admission holding that does not bias the A/B.
