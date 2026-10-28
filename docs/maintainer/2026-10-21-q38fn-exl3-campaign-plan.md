# Plan: the q38fn (Qwen3.8 Flash-Next 125B-A6B) EXL3 banks campaign (twoFour)

Status: closed by decision (2026-10-23). The 125B will be served directly from the
existing FP8 MTP artifact — no re-encode. Rationale: the model's quality is the original
authors' guarantee when its bytes are loaded faithfully; the engine's obligation is loading
fidelity (validated by the existing oracles and serving battery), not the quality of a lossy
repack we would then have to validate ourselves. The EXL3 banks route remains the documented
option if a denser resident posture is ever needed (the 27B dense route stays accepted; the
35B bank route stays smoke-verified). The shape-readiness sweep below was completed and
verified by twoFour (commit e9e35a67, q38fn-bank-shapes) and is retained as the standing
preparation for any future banks campaign.

## 0. Decision context

The resident FP8 MTP Flash-Next artifact is 70.5 GiB. The 121B routed parameters dominate it.
Converting the routed banks to uncalibrated EXL3 (banks-only: dense projections keep the
artifact's existing FP8 formats) drops the resident posture to roughly 33 GiB at 3 bpw or ~25 GiB
at 2 bpw — the largest single tiering-posture lever on the GB10 lane. Uncalibrated banks need no
Hessians, so the campaign never requires the full 250 GB model in memory; the converter streams
one layer's routed tensors (~5 GB) at a time from the BF16 source on local disk.

The 35B-A3B bring-up validated the route end to end (assembly, smoke serving, prefill and decode)
but produced no quality verdict: the full-corpus acceptance run was stopped at 19/496 windows
(early overall ~5.9 against the FP8 control's 4.55) by decision. The 125B go/no-go therefore
rests on the quality-per-byte evidence this campaign's acceptance must produce first on the 35B
(see §5 step 1) — or on a deliberate decision to skip it.

## 1. What the 35B bring-up established (reusable facts)

- **Layout**: gate and up are separate complete banks (contract §2.2); down is its own bank. The
  packed gate_up alternative was executed and reverted — it cannot feed the
  `exl3_routed_gate`/`exl3_routed_up` kernel route.
- **Engine path**: banks artifacts serve after the 2026-10-21 reconciliation (layout-aware tier
  rank stride, EXL3 workspace sizing, removal of the single-token wrapper restriction). Prefill
  runs as looped per-token decode — numerically identical, adequate for evaluation, not for
  serving throughput.
- **Campaign mechanics**: per-layer shard streaming with file-resume; a finished projection's
  layer file carries over to a later campaign when its rate is unchanged. Per-layer wall clock on
  the RTX 5090 box (CUDA 13.4.92): ~2800 s per 35B layer (separate gate/up linears), ~2230 s
  packed. The packed rate is not the supported layout; use ~2800 s/layer as the separate-layout
  basis and scale by parameters per layer.
- **Commit trail**: `f2ca42b6` (packed, reverted), `b0fc2af3` (revert), `e7960dc5` (engine
  reconciliation), `bf3e0551` (repo strip-down). The converter-side bank quantize route with its
  tests is in `tools/convert` (quantization/exl3_convert.py, methods.py, recipe.py,
  official_recipes.py, tests/convert/test_exl3_bank_quantize.py).

## 2. Code readiness: the 125B shape sweep (blocking)

The sparse-MoE bank route hardcodes the 35B/27B contract shapes as compile-time constants. The
125B Flash-Next MoE is 512 experts, hidden 2560(?), routed intermediate 1280(?), down
[2560(?), 640(?)]: every constant below must become shape-derived or template-specialized before
the campaign's artifact can bind. No semantics change; each site is a shape admission change.

| Site | Constants | Change |
|---|---|---|
| `src/ops/wrapper/sparse_moe.cpp` | `kExperts=256`, `kIntermediate=512`, `kHidden=2048`, `kSharedGateRows`, `kExpertRows=1024` and their uses in `validate_weights`, `require_exl3_bank` callers | carry the shapes from the weights (`exl3_experts`, `exl3_rows_per_expert`, `n`, `k`) and admit the 125B geometry explicitly |
| `src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu` | `kHidden/kExperts/kIntermediate` constexpr in the decode kernels and their tiling (`kGroups = kIntermediate / Codec::kGroupK`, `kRowBlocks = kHidden / Rows`) | template the kernels on the bank geometry (the 35B and 125B instantiations) or derive the tile counts from runtime arguments; verify the warp-uniform gather sites stay correct |
| `src/ops/wrapper/sparse_moe.cpp` EXL3 branch | `exl3_experts == 256` and the 35B bank-shape check (`rows_per_expert == 512/2048`) | admit the 125B expert count and shapes; keep the 35B contract admissions for regression |
| `src/models/qwen3_8_flash_next` binding route | the 125B's MoE binding via the family runtime | wire the experts' parameters to the bank layout through the Flash-Next family's own load path (the qwen3_5 route is not reused, per the family boundary) |
| tier-rank stride | `src/models/qwen3_5/execution/parameters.cpp` layout-aware stride (EXL3: `gate.shape[0]`) | the Flash-Next family's equivalent check, if it carries one, gets the same layout-aware form |

Verification for this sweep: the existing bank quantize test extended with a second synthetic
geometry (a 512-expert MoE), plus one smoke assembly of a small 125B-shaped artifact. No GPU
quality claims from the sweep itself.

## 3. Campaign procedure

1. **Rates**: banks-only, one bank-wide rate first (k2=6, ~3.5 bpw on the routed experts) to
   mirror the 35B bring-up; a mixed-rate map is a follow-up once per-expert-rate acceptance
   evidence exists. Dense projections keep the artifact's existing FP8 formats; no Hessians are
   captured (uncalibrated route, contract §6.2.1).
2. **Streaming**: per layer, load the layer's routed tensors from the BF16 source shards
   (~5 GB resident), quantize 512 packed-equivalent gate/up linears + 512 down linears as
   separate banks, write `layer_NN.pt`, discard. Resume per layer file.
3. **Cost projection**: the 125B's routed parameters per layer are ~2.5B vs the 35B's ~0.66B
   (48 layers × 512 experts × (packed [1280·2? rows × 2560] + down [2560, 640])). At the measured
   ~1.07 s per million routed parameters, ~8400-8700 s/layer × 48 ≈ **4.7-5.0 days** unattended
   on the 5090 box. The GB10 box's streaming rate is unmeasured; measure one layer before
   committing the box.
4. **Assembly and smoke**: the Flash-Next artifact recipe binds the banks through the family's
   packing groups; smoke with a short generation, then a serving pass.
5. **Single-instance discipline**: no second device-allocating run while the resident server or
   the campaign is in flight.

## 4. Deliverable

`out/qwen3_8_flash_next_125b_a6b_exl3_banks.ninfer` (+ conversion report): the Flash-Next
artifact with EXL3 bank experts and unchanged dense projections, serving decode and (looped)
prefill on the GB10 lane, with its resident posture measured (target ≤ ~35 GiB at k2=6).

## 5. Acceptance

1. **Precondition (35B anchor)**: a completed 35B full-corpus acceptance of the same uncalibrated
   route — the stopped run — OR an explicit decision to proceed without a quality anchor. The
   acceptance table: per-corpus and overall perplexity against the FP8 control at the artifact's
   byte cost, plus the worst-window spread.
2. **125B acceptance**: the same table against the resident FP8 MTP artifact, run on the GB10
   box; decode throughput at the production token shapes; a serving smoke battery. The route is
   accepted if the quality-per-byte trade is at least as good as the 27B dense route's
   (5.18 vs 4.55 at 7.92 vs 26.5 GiB) and the serving posture target is met; otherwise the mixed-
   rate map design work is the next decision, not a silent acceptance.
3. **Numerical verification**: the converter-side exact-oracle tile checks (contract §6.1 step 4)
   on the emitted banks; no new oracle is introduced by this campaign.

## 6. Merge-up boundary for this plan

Upstream (Neroued/ninfer, 81c8ce09..64d10fbf, 77 commits) moved to a Qwen4-exp experiment lane,
an ops-family reorganization, and CPU-offload infrastructure. None of that carries this plan's
semantics: the banks quantize route, the tier-rank fix, and the decode kernels are fork-side work.
Upstream's sanctioned-port candidate found in the survey: the length-bounded JSON string mask
optimization (0a8530d4) — deferred: it is a 387-line rewrite across the diverged vendored xgrammar
and must not land without its test suite running. Rejected for the fork: the Qwen4-exp family
(speculative, outside the family-readiness contract), the ops-family reorganization (the fork's
op ownership model differs), and the CPU-offload/MoE-offload lane (outside the product contract).
