**Title:** fix(ops): size the QSA workspace for every call up to its row count

**Target:** `lkarlslund/ninfer6000` `master` (`8f574ee4`).
**Head:** `baristahaus/ninfer-gb10` branch `upstream/ninfer6000/qsa-workspace` (one commit).
Independent of the other PRs in this set.

## Problem and scope

Related Issue: none

`flash_next_qsa_workspace_capacity_bytes(tokens, max_context)` adds the hierarchical top-k
scratch and the split-attention partials only when `tokens <= 16`. A workspace planned for a
longer call still serves calls of at most 16 rows, which take exactly that decode route. Two
examples are a prefill chunk's short tail and a decode step. So the returned capacity does not
cover every call it is planned for.

For a 16-row call, the old formula's own value at 16 rows is larger than its value at `tokens`
rows over these ranges (MiB):

| `max_context` | 16-row call needs | `tokens` with less capacity than that |
|---|---:|---|
| 4,096 | 25.7 | 17–288 (chunks 128 and 256) |
| 16,384 | 26.1 | 17–258 (chunks 128 and 256) |
| 73,728 | 27.8 | 17–178 (chunk 128) |
| 262,144 | 33.6 | 17–100 |

**This is a latent accounting fault on `master`. No failure has been observed there.** Whether a
short call actually runs out depends on the slack the shared arena gets from other plans; the
decode plan usually provides enough. In our downstream fork it surfaced as a `bad_alloc`
through an additional dense attention route that reuses prefill-sized workspaces for ≤16-row
calls.

## Implementation

Both decode-route terms are sized for `min(tokens, 16)` rows, for every `tokens`. The header
documents the contract: capacity for any call of 1..`tokens` rows.

Cost: the largest planned workspace (8192 rows × 262,144 context) grows by 28.2 MiB, from
2,879,492,096 to 2,909,048,832 bytes. The QSA op test's pinned full-context value is updated to
match.

## Verification

- Applies to `master` unchanged. On NVIDIA GB10 (`sm_121a`, with the other fixes from this set
  stacked) `test_flash_next_qsa` and both Flash-Next real-artifact tests pass.
- The same change has run downstream since 2026-09-30, covered by that fork's QSA op tests and
  serving runs on GB10.
- The table above was computed from the capacity formula, not measured. Not run on RTX PRO 6000.
