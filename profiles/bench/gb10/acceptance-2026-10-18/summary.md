# MTP acceptance battery — FP8 MTP artifact (2026-10-18)

Artifact: `out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer`
(70.5 GiB resident). Command per run: `ninfer <art> --spec mtp --draft-tokens 2
--lm-head-draft --max-new 128 --greedy --no-thinking` (CUDA graphs enabled,
default). GB10, sm_121a, CUDA 13.0.88, driver 580.173.02.

## Results (8 prompts, greedy, 128 new tokens each)

| Prompt | Accept rate | Accept length |
|---|---|---|
| Hamlet plot | 16.7% | 1.33 |
| Transformer attention | 14.9% | 1.30 |
| Merge sorted lists (Python) | 34.7% | 1.69 |
| Water cycle | 12.4% | 1.25 |
| Great Depression | 21.5% | 1.43 |
| Lentil soup | 23.4% | 1.47 |
| FR/DE translation | 22.2% | 1.44 |
| Playing cards | 13.7% | 1.27 |
| **Mean** | **19.9%** | **1.40 tok/round** |

Accepted-by-pos (run 1): 18 drafts at position 1, 5 at position 2 — second
draft position rarely accepted, consistent with the 1.4 mean length.

## Graph-path A/B (same artifact, prompt 1)

`--no-cuda-graph` rerun: output text byte-identical to the graph-enabled run
(426/426 bytes), and the MTP acceptance summary lines are identical (16.7%,
1.33). The captured decode path matches the uncaptured path exactly.

## Notes

- The battery was interrupted after run 1 on the first attempt; runs 2-8 were
  rerun in a second process. All eight completed cleanly.
- Raw per-run files: `run<N>.out`, `run<N>.err`, `run1-nograph.*` in this
  directory.
