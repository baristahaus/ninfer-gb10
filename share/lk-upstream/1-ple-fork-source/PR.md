**Title:** fix(flash-next): read PLE history from the fork source during prefill

**Target:** `lkarlslund/ninfer6000` `master` (`8f574ee4`).
**Head:** `baristahaus/ninfer-gb10` branch `upstream/ninfer6000/ple-fork-source` (one commit).
Independent of the other PRs in this set.

## Problem and scope

Related Issue: none

Sometimes a Flash-Next prompt runs past a context-cache capture point, for example a chat-turn
boundary with private or shared prefix capture enabled. Prefill then continues as a StateImage
`DeviceFork`, with distinct source (checkpoint) and destination slots. The GDN layers read the
source slot and write the destination slot. Prefill PLE instead updated the **destination** slot
in place:

```cpp
Tensor ple_state = ple_state_->slice(2, linear_state_destination_slot_, 1).view({10240, 9});
ops::flash_next_ple(hyper, gathered, source.ple, ple_state, ple_output, work_, stream, bf16_gemm_);
```

Nothing on the fork path copies PLE history into the destination: `begin_fork` copies only the
DFlash local cache. So layer 1, the only PLE layer, started from whatever history that slot last
held, not from the checkpoint's nine columns. The effects:

- Output depends on which slot the pool hands out and what that slot last held. With an identical
  prompt and greedy sampling, output therefore changes with `--max-concurrency`.
- Layer 0's GDN state at the prefill frontier is bitwise identical across configurations. Every
  GDN layer from 1 onward differs.
- Single-turn prompts with no capture point never fork, so they are unaffected. This is why the
  real-artifact goldens did not catch it.
- It is independent of MTP: the split reproduces with speculation off.

Decode and MTP verify are unaffected. They already use `flash_next_ple_batch_update` and
`flash_next_ple_replay_record`, which read the source slot and write the destination slot.

## Implementation

`flash_next_ple` follows the same source/destination contract as the GDN op and the batched PLE
transition:

- **Op** (`include/ninfer/ops/flash_next_ple.h`, `src/ops/launcher/flash_next_ple.cu`): it takes
  `source_state` and `destination_state`. They may be the same tensor, for an in-place
  continuation. Distinct tensors must not overlap; the launcher rejects overlap. Each thread of the
  dilated-convolution kernel owns whole channels and reads the source history before writing the
  destination, so aliasing stays correct. Arithmetic is unchanged.
- **Schedule** (`text_context_impl.h`): prefill passes the source and destination slots, as the GDN
  layers already do.
- **Docs** (`qwen3.8-flash-next-125b-a6b-model.md`): it states that a fork-continued execution
  reads every persistent component from the source and never reads the destination's prior
  content.

An alternative was to copy the PLE slot in `begin_fork`, as is done for the DFlash local cache.
I rejected it: it adds a copy on every fork, and it leaves prefill PLE as the one state consumer
that ignores the selectors.

## Verification

**Unit test.** `test_flash_next_ple` now runs the in-place case and a fork case for both FP8 and
BF16 tables. The fork case fills the destination with unrelated values and requires:
- the output and the destination history match the independent FP64 oracle;
- the source history is left bit-for-bit unchanged.

**Rebase.** On `master` the op takes `gathered` (FP8 or BF16 rows) and builds its embedding with
`embedding_from_table`; the fix keeps both and only splits the history into source and destination.

**On `master`** (NVIDIA GB10, `sm_121a`, with the other fixes from this set stacked, the NVFP4
artifact with BF16 dense projections):
- `test_flash_next_ple` passes all four cases, and both Flash-Next real-artifact tests pass;
- the 73-token probe prompt below (greedy, 256 tokens, cold, two fresh servers with CUDA graphs and
  MTP K=3) gives identical output at `--max-concurrency` 1 and 4.

**Measured downstream** (`baristahaus/ninfer-gb10`, NVIDIA GB10, `sm_121a`, CUDA 13.0, the
`nvfp4_fp8_mtp` artifact). Probe: `ninfer-serve --max-context 73728 --kv-dtype fp8
--preserve-thinking`, a 73-token system + user chat prompt, greedy, 256 output tokens, cold
first request, `--max-concurrency 1` vs `4`:

| Build | MTP draft 1 | MTP off |
|---|---|---|
| before | C1 ≠ C4 | C1 ≠ C4 |
| after | C1 == C4 | C1 == C4 |

- The fixed output differs from both old outputs, as expected when each old configuration read
  different stale history.
- Real-artifact goldens: bitwise unchanged (the canonical prompt does not fork).
- C4 MTP decode speed: unchanged within run-to-run variation; the fix only changes which slot one
  prefill launch reads.

**Not verified:** RTX PRO 6000; multimodal prefill through the fork path (same schedule code, not
exercised); the unpatched C1 vs C4 comparison on `master` itself (the "before" row is from the
downstream fork, which had the same in-place destination update).
