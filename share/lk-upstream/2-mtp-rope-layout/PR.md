**Title:** fix(flash-next): decide MTP RoPE layout by element count, not a batch-3 shape test

**Target:** `lkarlslund/ninfer6000` `master` (`8f574ee4`).
**Head:** `baristahaus/ninfer-gb10` branch `upstream/ninfer6000/mtp-rope-layout` (one commit).
Independent of the other PRs in this set.

## Problem and scope

Related Issue: none

`TextContext::mtp_forward_flash_next` treats its RoPE positions as three-axis MRoPE when
`rope_positions.ne[2] == 3 || rope_positions.ne[1] == 3`. Batched MTP decode passes text positions
as `[width, batch]`, so whenever the MTP batch has exactly three rows the second test matches.
The predictor then views `width × 3` positions as `[width, 3, 3]` and throws
`view element count mismatch: requested 3·N, available N`. In practice:

- **CUDA graphs on, `--max-concurrency` ≥ 3 with MTP:** startup fails while capturing the batch-3
  graph. With draft 2 the message is "requested 27, available 9".
- **`--no-cuda-graph`:** decode fails with HTTP 500 as soon as a compacted batch has three rows.
  With draft 1 the message is "requested 18, available 6".

Single-request MTP, and batches of one, two or four rows, are unaffected.

## Implementation

The element count decides the layout:
- three values per token is MRoPE (`[T,3]` or `[width,batch,3]`);
- one value per token is text positions, expanded as before;
- anything else is rejected with `invalid_argument` instead of being viewed.

## Verification

- **On `master` (`8f574ee4`), NVIDIA GB10:** reproduced at startup with `--max-concurrency 4` and
  MTP, for every draft count: "view element count mismatch: requested 36, available 12" (K=3),
  27/9 (K=2), 18/6 (K=1). With this commit the same server prepares its graphs in 2.5 s and served
  a full C1/C2/C4 load run (105 requests, five prompt classes) at K=1 and K=3, and with adaptive
  drafts.
- The condition depends only on the decode batch reaching three rows, so it also applies on
  RTX PRO 6000 at `--max-concurrency` 3 or more. It has not run there.
