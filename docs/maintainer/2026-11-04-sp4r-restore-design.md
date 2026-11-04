# SP-4-R restore design: tiering after the family convergence

Status: **design for requestor review — no code landed.** The workplan's premise
("re-apply the dropped wiring onto the current tree") does not survive contact
with master `3d2de369`: the MoE execution was converged onto the shared family
core and the tiered path did not survive it. This doc states what was verified,
the two restore targets, their designs, and what each needs.

## What was verified (all checked on master `3d2de369`)

1. **The runtime commits are ancestors of master** (`4953f2e3` SP-4 runtime,
   `5e76877f` phase 2b, `e7d73e0c` SP-5 device-side pass). Nothing deleted them
   by diff; later rewrites orphaned the wiring.
2. **The surviving surface**: `include/ninfer/ops/sparse_moe.h` carries the full
   tier contract (`tiered_hot_experts`, `expert_activations`, `expert_residency`,
   the warm twins); `src/ops/wrapper/sparse_moe.cpp` still implements the tiered
   split; the sparse_moe kernels still arm the counters.
3. **The qwen3_5 (35B stand-in) execution no longer uses that route.** Its
   `FfnParameters` is now `std::variant<DenseParameters, moe::Parameters>` — the
   shared family-core MoE parameters (`src/models/common/moe/execution.h`:
   `ExpertBankDeviceView resident`, `ops::MoeProfile`, divisors, subset). The
   old `execution/ffn.cpp` is **not in `execution_sources.cmake`** — a dead
   pre-convergence leftover that cannot compile against the current header.
4. **`moe::Parameters`/`ExpertBankDeviceView` have no tiering surface**: no warm
   twins, no counter buffer, no residency, no tier boundary. The converged path
   is untiered by construction.
5. **The Flash-Next lane still runs the tiering-capable Op**: `text_context_impl.h`
   and the 125b variant call `ops::sparse_moe` with `ops::SparseMoeWeights`
   (fields null → the untiered direct path).
6. **No converter route emits tiered artifacts** (`tools/convert/` has no tiered
   warm-twin writer), and no tiered artifact exists on the tank or locally.

## Target A — the converged qwen3_5 path (where the SP-4 measurements lived)

Restore tiering as a first-class surface of the shared core, per the SP-8
architecture (families shrink; the core carries the math):

1. `ExpertBankDeviceView` gains the tier state: the pinned-host warm mini-bank
   view, `tiered_hot_experts`, the counter buffer, and the residency table — or
   `moe::Parameters` gains a sibling `TieredExperts` member; the layout comes
   from `WeightBankLayout`/`ExpertMatrixEntryLayout` (the new geometry machinery
   is richer than what the old kernel's `ExpertCacheBankParams` encoded).
2. The shared MoE execution arms the counters and consults the residency in the
   routing stage (the old kernels' arming pattern, re-expressed on the new
   kernel family).
3. `expert_cache_adapt.{h,cu}` ports to the new geometry (the hysteresis and
   copy logic are unchanged; the plane offsets come from the layout planner).
4. The converter writes tiered artifacts again (the warm mini-banks + warm-rank
   table + tier metadata), and a tiered 35B artifact is converted for
   verification.
5. Acceptance per the workplan: zero-cost counters (the 27B dense baseline
   within noise), residency populating under MoE load, `bench --moe-profile`,
   the moe contract tests, bit-identical generation at every tail size.

Effort: several days — a design task, not a re-apply. The converter and the
artifact are on the critical path for any measurement.

## Target B — the Flash-Next 125B lane (smaller; serves the SP-6 lane)

The Op side already supports tiering; what is missing is the binding and the
runtime:

1. The Flash-Next MoE binding populates the warm twins + tier metadata when the
   artifact carries them (the converter work is shared with target A).
2. `parameters`-equivalent state (the Flash-Next program's runtime chain) gains
   `ExpertCacheState` and the boundary passes (prefill chunk / every eighth
   decode round), ported from `e7d73e0c`.
3. A tiered 125B artifact (or a tiered Flash-Next MoE variant) for measurement.

Effort: roughly a day after the converter exists. This lane is where the
strategy's tiered-125B goal actually lives (SP-6 measured the FP8/MTP form
fully resident; a tiered form is what produces the SP-5-R miss profile).

## Decision: Target B — the Flash-Next 125B lane

Decided 2026-11-04. Rationale, each measurable:

1. **Strategy value**: SP-4/5/6 all aim at the 125B; the 35B was always a
   stand-in. A tiered 125B is also the only form that produces the SP-5-R miss
   profile (its gate is closed on the fully-resident FP8/MTP form).
2. **~~The Op surface is already in the production path~~ CORRECTED (2026-11-04, blocker):**
   the 125B's production MoE route is `ops::flash_next_moe` over
   `ops::FlashNextExpertBank` (`include/ninfer/ops/flash_next_moe.h`) — that
   structure has **no tier surface** (`codes/scales/divisors/qtype/experts/rows/
   columns` only). The `SparseMoeWeights` arm in the Flash-Next variant is the
   **EXL3 route**, not the NVFP4 bank route. Restoring tiering on this lane
   means adding the tier state to `FlashNextExpertBank`, the counter arming and
   residency reads to the `flash_next_moe` kernels, and the adaptation binding
   to the NVFP4 plane geometry — the **same class of work** as the converged
   core, not a populate-existing-fields job. The lane choice therefore rests on
   strategy value and the verifier artifact, not on effort.
3. **Effort — CORRECTED**: both lanes are the same class now (Op tier surface +
   kernels + loader/binding + runtime + a tiered artifact). The differentiators
   are the strategy lane and which kernel family carries the tier state.
4. **Convergence direction**: adding tiering to `moe::Parameters` now would
   shape the family core around a stand-in lane (SP-8 shrinks families); the
   125B's tiering requirements (per-expert widths, PLE interaction) are exactly
   the "variable-extent change" the strategy already schedules — design it once,
   there, with real requirements.

## Target B execution plan (converter first; nothing measures without it)

1. **Op tier surface**: `FlashNextExpertBank` gains the warm-twin pointers, the
   tier boundary, the counter buffer, and the residency table; the
   `flash_next_moe` kernels arm the counters and read the residency in the
   routing stage (the old sparse_moe kernels' arming pattern, re-expressed for
   the NVFP4 block-scale route).
2. **Adaptation binding**: `expert_cache_adapt` (already on master) binds to the
   NVFP4 plane geometry — the per-expert code/high/scale extents of the 512-way
   banks come from the bank's block-scale layout, which the kernel owns today.
3. **Converter**: the tiered-artifact writer — per-expert warm mini-banks
   (pinned-host twins), the warm-rank table, and the tier metadata for the
   Flash-Next 512-way MoE; a conversion-policy knob (hot/warm split per bank).
   Reuses the existing Flash-Next 125b converter route (`tools/convert/
   qwen3_8_flash_next_125b_a6b/`) as the base.
4. **Loader/binding**: the Flash-Next loader applies the tier plan
   (`parse_tier_plan` → `set_tier`), and the model assembly populates the new
   tier fields from the binder's tiered placement.
5. **Runtime**: the Flash-Next program's parameters chain gains
   `ExpertCacheState` + the boundary passes (prefill chunk / every eighth decode
   round), ported from `e7d73e0c` — the device-side pass is stream-ordered and
   must stay outside the captured CUDA Graph regions; verify against the graph
   frontiers.
6. **Artifact**: a tiered FP8 125B conversion. **Disk note**: ~65 GB free
   locally — the artifact goes to the tank or the local cache is cleaned first.
7. **Measure**: the SP-5-R gate re-runs (the miss profile, the warm-hit rate,
   the promotion/eviction behavior), the zero-cost counter check, and the
   bit-identical-generation check at every tail size.

Effort: several working sessions — the same class as target A; the lane choice
rests on strategy value (the tiered 125B is the strategy's goal and the SP-5-R
verifier), not on cost.

## Immediate cleanups available regardless of decision

- Delete the dead `src/models/qwen3_5/execution/ffn.cpp` (not in any sources
  list; cannot compile against the current header).
- `pgrep`-style process matching in the SP-6 probe is already fixed; no other
  action pending.
