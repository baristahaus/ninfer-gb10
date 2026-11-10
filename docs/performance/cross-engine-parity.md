# Cross-engine parity

Two engines, one box, one artifact, one flag set. This page records a measured comparison of
NInfer against capy-flash (formerly ninfer-tp2) and the limits of that comparison. The harness is
[`bench/parity/measure_parity.py`](../../bench/parity/measure_parity.py); the protocol is in
[`bench/parity/README.md`](../../bench/parity/README.md).

This is a review measurement, not a product benchmark of either engine.

## Conditions

| | |
|---|---|
| box | trueno, one RTX 5090 32 GB, idle |
| artifact | `/nfs/ninfer-gb10/qwen3_8_27b_v3_plain/qwen3_8_27b_v3-plain.ninfer`, 21,492,938,224 bytes, the same file for both engines |
| engines | NInfer `master` at `9b6d1f17` (`rtx5090-release`); capy-flash v0.9.0 (`release`) |
| flags | `--host 127.0.0.1 --model-id parity --max-context 8192 --kv-capacity 8192 --max-concurrency 1 --no-thinking` |
| workload | 5 fixed prompts, greedy (`temperature 0`), `max_tokens 256`, streamed; one warmup request per prompt discarded, then 3 measured waves of 5 |
| measured | 15 requests per engine, 3429 and 3507 generated tokens |

Both engines loaded the same artifact bytes. That is a result in itself: the shared v3 layouts read
on both, so the comparison needs no requantization step.

## Result

| | NInfer | capy-flash v0.9.0 | delta |
|---|---|---|---|
| decode, median | 72.01 tok/s | 73.77 tok/s | capy +2.4 % |
| decode, standard deviation | 0.172 | 0.139 | |
| decode, range | 71.72 to 72.20 | 73.48 to 74.02 | non-overlapping |
| TTFT, median | 52.1 ms | 52.1 ms | identical |
| empty responses | 0 / 15 | 0 / 15 | |

The decode distributions do not overlap, so the 2.4 % difference is larger than run-to-run noise.
It is also small: the two engines are within a fortieth of each other on decode, which is what to
expect when both are bound by the same DRAM bandwidth on the same silicon. Time to first token is
identical.

## What this does not establish

The scope is narrow, and a parity number invites over-reading:

- tp 1 only. The two-device lane is not measured.
- One artifact. This is a dense 27B artifact. A 4-bit MoE artifact such as Flash-Next 125B
  exercises different kernels.
- No speculation. Both engines ran without MTP or DFlash2, so the drafter paths are unmeasured.
- Short prompts. A TTFT of about 52 ms is dominated by request setup, not by prefill. This workload
  says nothing about prefill throughput on long contexts.
- Decode is the memory-bound regime. Near-parity there is expected, and it is not evidence that the
  compute-bound paths (prefill, MoE, attention) are equally matched.

The next measurements that would discriminate are prefill on a long prompt and decode under MTP.

## A methodological failure worth recording

The first run of this campaign produced 3001 ms TTFT for NInfer against 1467 ms for capy-flash,
which looked like a two-times prefill regression. It was not. The client counted only
`delta.content`. A thinking model streams `reasoning_content` first, so the first-token timestamp
was really the first content token, after all the reasoning. With `max_tokens 256` a thinking model
often never reaches content at all: capy-flash returned zero tokens for 4 of 5 prompts, so only 3 of
its 15 requests produced anything.

The fix counts a token from any non-empty delta channel, forces `--no-thinking` on both engines, and
refuses to report a comparison when either side produced empty responses. The refusal is the
important part: a parity harness that silently averages over missing data is worse than no harness.

The numbers above are from the corrected protocol.
