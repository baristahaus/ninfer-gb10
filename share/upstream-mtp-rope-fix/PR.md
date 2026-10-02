**Title:** fix(flash-next): decide MTP RoPE layout by element count, not a batch-3 shape test

**Target:** lkarlsund's NInfer fork, `master` at `2aa87467`.
**Head:** `baristahaus/ninfer-gb10` branch `upstream/flash-next-mtp-rope-layout` (`c1bd1a5e`).
Independent of the PLE fork fix.

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

- **Downstream fork (`baristahaus/ninfer-gb10`, NVIDIA GB10, `sm_121a`):** the same change has been
  in service since 2026-09-29. Before it, C3, C4 and C8 MTP serving failed at startup. After it,
  C4 MTP decode under 4×256-token concurrent load runs, both with graphs and with
  `--no-cuda-graph`.
- **On `master` (`2aa87467`):** reproduced on GB10. C4, MTP draft 1, `--no-cuda-graph`, four
  concurrent requests: all four fail with HTTP 500 ("requested 18, available 6"), while a single
  request succeeds. This commit is only compile-checked on `master` (`sm_121a`, 125B runtime TU).
- **Not verified:**
  - the rebased commit run end-to-end on `master`;
  - RTX PRO 6000.

  The failing condition depends only on batch size, so it should reproduce there too.
