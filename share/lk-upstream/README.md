# Pull requests for lkarlslund/ninfer6000

Five independent PRs against `lkarlslund/ninfer6000` `master` (`8f574ee4`, 2026-10-07), the GitHub
parent of this fork. Each folder holds the PR text (`PR.md`) and its `git format-patch` series. Each
series applies to `master` on its own, and all five stack without conflicts in the order below.

| # | PR | Why he wants it | Verified on his `master` |
|---|---|---|---|
| 1 | `fix(flash-next)`: read PLE history from the fork source during prefill | output changes with `--max-concurrency` after a context-cache capture (his code still updates the destination slot in place) | compile only; the fix was measured downstream |
| 2 | `fix(flash-next)`: MTP RoPE layout by element count | MTP with `--max-concurrency` 3 or more fails at startup | yes, GB10: the full load campaign ran on it |
| 3 | `fix(flash-next)`: reasoning effort `none` with disabled thinking | every OpenAI request with `"reasoning_effort": "none"` gets HTTP 400 | compile only |
| 4 | `fix(ops)`: QSA workspace for every call up to its row count | latent capacity under-count for short calls | compile only |
| 5 | GB10 (`sm_121a`) support: build gate, MemAvailable sizing, SM-count grids, fused HyperConnection fallback | his engine on a DGX Spark | first three commits yes (built, ctest, full campaign); the fourth compile only |

Dropped from the September drafts:
- the PLE page-in overlap, because his own `f202d3f2` already starts every row's page read ahead;
- the MTP RoPE and QSA drafts on `2aa87467`, now rebased here as PRs 2 and 4. The PLE fork fix
  needed a conflict resolution: his op now takes FP8 or BF16 rows (`gathered`,
  `embedding_from_table`), so the fix splits the history into source and destination around that.

## Validation run before sending (GB10, one GPU job)

Validate on all five stacked, on a clean clone of his `master`:

```bash
git clone https://github.com/lkarlslund/ninfer6000 lk-pr && cd lk-pr && git checkout 8f574ee4
for d in 1-ple-fork-source 2-mtp-rope-layout 3-reasoning-effort-none 4-qsa-workspace 5-gb10-support; do
  git am /path/to/ninfer-gb10/share/lk-upstream/$d/*.patch
done
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121a
cmake --build build -j
NINFER_QWEN38_FLASH_NEXT_WEIGHTS=<v3_fork entry> ctest --test-dir build --output-on-failure
```

Expected:
- `hyperconnection` now passes (PR 5's fourth commit);
- `flash_next_ple` passes all four cases: FP8 and BF16, in place and forked (PR 1);
- `qwen3_8_flash_next_frontend` passes the new reasoning-effort-none case (PR 3);
- `flash_next_qsa` passes (PR 4);
- the two Flash-Next real-artifact tests pass;
- the remaining Qwen3.5 artifact gaps are unchanged.

Then one server with `--max-concurrency 4 --spec mtp --draft-tokens 3`, plus the flags of
`profiles/bench/gb10/lk-parity-2026-10-10/parity_arm.sh`:
- a chat request with `"reasoning_effort": "none"` returns 200 and starts in content (PR 3);
- the PLE fork probe from PR 1: the same 73-token system + user prompt, greedy, 256 tokens, cold,
  `--max-concurrency 1` and `4` on two fresh servers, with identical output required.

## Sending

Pushing the five head branches (`upstream/ninfer6000/*` in `baristahaus/ninfer-gb10`) and opening
the PRs are external actions: they need the owner's go-ahead. The branches are built from these
folders with `git am` on `8f574ee4`, one branch per folder. The September `upstream/flash-next-*`
branches stay as they are; nothing here rewrites them.
