# Work order: EXL3 sensitivity measurement on the GB10 (twoFour)

Campaign context: `docs/maintainer/2026-11-02-exl3-sc385-campaign-plan.md`, work items
W2.1–W2.5. This document is the self-contained submission for the GB10 lane; everything
the executing agent needs is here or at the referenced paths.

## Mission

Run the JRamirez-UAB EXL3 "SC pipeline" sensitivity measurement (`sc_measure.py`) for
Qwen3.8-27B against an English sysadmin/agent calibration trace, producing the
per-tensor sensitivity table that feeds the greedy allocation (W2.4). This is the
measurement his pipeline was built for, rerun on our domain: his trace was 196×2048
rows of Spanish sysadmin text; ours is 250×2048 rows of English agent traffic.

Primary compute: the GB10's 121 GiB unified pool — the unquantized BF16 27B (≈54 GiB)
fits natively, which the 5090 cannot do. Expected compute: ~1–2 h.

## Deliverables

1. `sensitivity/sc_measure_results.json` (or his tool's native output) — the per-tensor
   sensitivity table at two noise levels, keyed by the checkpoint's tensor names.
2. The exact command lines used, the tool versions (`pip show exllamav3`), and the run
   log — reproducibility is part of the deliverable.
3. If the trace (below) is regenerated rather than shipped: the trace safetensors plus
   its manifest.

## Prerequisites on the GB10

| Item | Source | Notes |
|---|---|---|
| exllamav3 v1.6.1 (pinned) | `pip install exllamav3==1.6.1` | his v1.5.1 build has a known version skew; the plan pins v1.6.1 |
| his pipeline scripts | `sc_trace.py`, `sc_measure.py`, `sc_optimize.py`, `eval/qbench.py` from his repo/published bundle | "his pipeline as published" is the point — not a port |
| BF16 source checkpoint | `/home/barista/.cache/huggingface/models/Qwen/Qwen3.8-27B` on the 5090 box (18 safetensors shards, ≈54 GiB) | copy via scp/rsync; the config, tokenizer and chat template files in that directory are required too |
| calibration trace | `cal_trace.safetensors`, 250×2048 rows, his `sc_trace.py` format | two options, in order of preference: **(a)** the 5090 box generates it through the resident pool API (W2.1 — the sampling script lives with the requestor) and ships the safetensors; **(b)** the GB10 samples it itself — see "Optional trace sampling" below |
| his public trace (comparison) | `qbench_prompts_gen.json` / his calibration bundle | needed only for the W2.4 head-to-head, not for W2.3 itself |

## W2.3 — the measurement itself

Per his pipeline's published usage:

1. Load the BF16 27B from the copied checkpoint (the unified pool holds it; no
   layer offload needed).
2. Compute the clean-reference forward boundaries over the trace (his tool manages
   the boundary-state cache).
3. For each quantizable linear tensor (attention q/k/v/o, GDN qkv/z/out, MLP gate/up/
   down — per layer-class), inject LDLQ-shaped noise at his two default levels and
   record the KLD against the clean reference.

Exact flags follow his `sc_measure.py --help`; record them. Nothing in this step is
NInfer-specific — if his tool runs on a stock CUDA box with 96+ GiB, it runs here.

## Optional trace sampling on the GB10 (W2.1, if requested)

If the requestor ships `out/qwen3_8_27b_nvfp4.v3.ninfer` (≈19 GiB, from the same
directory on the 5090) instead of a prebuilt trace: serve it

```
ninfer-serve --host 0.0.0.0 --port 8000 --model-id qwen3.8-27b \
  --max-context 262144 --kv-capacity 262144 --max-concurrency 2 \
  --kv-dtype k8v4 --prefill-chunk 1024 --spec mtp --draft-tokens 3 \
  --proposal-head optimized --preserve-thinking --log-level info --no-vision \
  <artifact path>
```

and drive the sampling prompt pack (the TEB scenario pack plus long agent sessions —
the requestor supplies the pack with the submission) through `/v1/chat/completions`,
capturing the continuations into the 250×2048-row `sc_trace.py` format. Thinking mode
on; the deployment profile. The sampling is I/O-bound on generation, so the GB10's
rate is fine.

## What NOT to do

- Do not quantize (W2.5 is a separate submission after W2.4's recipe lands).
- Do not modify the pipeline scripts; version skew gets reported, not patched.
- Do not run the 5090-lane fallback for W2.3 — the GB10 IS the sanctioned lane.

## Acceptance

- The per-tensor table covers every quantizable linear in the text stack (36 layers ×
  the class list, plus the embedding/head exclusions his tool documents).
- Two noise levels, KLD per tensor per level, with the run log and versions.
- The requestor's W2.4 step (CPU, runs anywhere) consumes the table unchanged.
