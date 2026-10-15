# Plan: native quantization routes for alternate checkpoint setups (QAD-class)

Status: proposal for effort evaluation. Branch `qad-native-quant-plan`.
Tracking: beads `ninfer-gb10-<id>` (created with this plan).
Origin of the request: `/nfs` checkpoint
`edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD` (93 GB) — a Qwen3.8
Flash-Next 125B-A6B whose quantization mix differs from our closed recipe. The
user decision governing this plan: **no dequant-to-BF16 fallbacks; alternate
quant setups are handled natively** (converter packs, artifact stores, kernels
compute in the source-declared format).

## 0. Principle and the pattern this plan establishes

Our Flash-Next 125B converter is closed-contract: it already ingests
already-quantized NVFP4 expert banks without requantizing
(`tools/convert/qwen3_8_flash_next_125b_a6b/convert.py:106-140,227-286`) and
every non-expert tensor must arrive BF16 (`_validate_source` is a closed set,
`convert.py:172-198`). The QAD family breaks that assumption in three places.
Instead of widening the closed set ad hoc, each new quant format gets the same
five-part treatment the NVFP4 bank already has:

1. a storage-layout spec (`docs/maintainer/storage-layouts.md`, `tensor-formats.md`);
2. a converter route with source-signature validation and post-write payload verification;
3. a binder route with hard shape checks (`load_plan.cpp`);
4. an exact independent FP32 oracle;
5. a kernel route with a measured bandwidth/efficiency record.

Future checkpoints that mix these formats then become inventory data, not new
machinery.

## 1. Target checkpoint facts (verified from the shards, 2026-10-15)

| Component | Format on disk (verified dtypes) | Our artifact today |
|---|---|---|
| Routed experts (48 layers) | NVFP4 W4A4: `weight` U8 `[n,k/2]` + `weight_scale` F8_E4M3 `[n,k/16]` + `weight_scale_2` F32 + `input_scale` F32 | same — **native ingestion already exists** |
| PLE n-gram tables | **NVFP4 group-16**: `weight` U8 `[2500012, 80]` (=160 el/row) + `weight_scale` F8_E4M3 `[2500012, 10]`, 11 shards (~2.5M rows — much smaller than our table) | FP8 shards (~51 GB-class table) |
| Attention, GDN, QSA-indexer, shared-expert projections | **MXFP8**: `weight` F8_E4M3 `[N,K]` + `weight_scale` U8 UE8M0 `[N,K/32]` | BF16 |
| PLE projections/conv, routers, hyper-connections, norms, embeddings, lm_head | BF16/F32 | BF16 — matches |
| MTP drafter trunk (fc, pre-norms, hyper-connections, self_attn + indexer, mlp.gate, shared_expert) | BF16 (names match our `_mtp_specs()` inventory exactly) | BF16 — matches |
| MTP routed experts (1 layer, 512 experts) | **NVFP4 W4A16**: `weight` U8 `[640,1280]` per proj + E4M3 scales + F32 `weight_scale_2`; **no `input_scale`** | BF16 dense (fp8mtp pass optionally re-emits A4 banks) |
| Draft output head | none (tied to 248,320 `lm_head`) | none — converter builds the 147,456-row shortlist head from `lm_head` (unchanged machinery) |

Architecture is declared `Qwen4ExpForConditionalGeneration` / `qwen4_exp` with
tensors prefixed `model.language_model.`; config carries
`text_config.mtp_num_hidden_layers: 1` (our validator's requirement, under
`text_config`); the MTP expert banks live in the separate
`hybrid-main-0000{1,2}.safetensors` shards. `mtp.layers.48` in
`hf_quant_config.json` is a stray alias tag (no such tensors). The SGLang
recipe's `SPARKTALK_FLASH_NEXT_DRAFT_VOCAB=ko64k` is a runtime draft-vocab
truncation, not a checkpoint shape — out of scope (§5).

## 2. Work item A — MXFP8 dense-leaf route (shared experts, attention, GDN, indexer)

This is the largest item and also a performance opportunity: MXFP8 stores
attention-path weights at 1 byte vs our BF16's 2, on the projection-heavy
prefill path (step 3 attribution: hyper.combine_mix 22%, qsa.select 21% of
prefill GPU time; gdn.prefill 35%).

- **A1 storage spec.** New rank-2 layout
  `dense_block_scale_k32_mxfp8_v1`: E4M3 code plane `N*K` bytes + UE8M0 scale
  plane `N*(K/32)` bytes along K; no global scale word (pow-2 scales carry
  their own exponent). Spec'd in `tensor-formats.md` (numeric definition) and
  `storage-layouts.md` (payload layout); codec next to
  `tools/artifact/codecs/nvfp4.py`.
- **A2 converter.** Source signature: `weight` F8_E4M3 + `weight_scale` U8
  `[N,K/32]`; direct pack (no requantization); closed-set validation extended
  to accept the MXFP8 pairs per parameter; post-write reopen+verify follows
  `convert.py:397-399`.
- **A3 binder.** `projection()` (`load_plan.cpp:41-55`) accepts the new format
  alongside BF16 / `fp8_e4m3fn_row_bf16`; Use roles unchanged.
- **A4 kernels.** Two-step decision:
  1. Probe tcgen05 MXFP8 MMA availability on `sm_121a` (0.5 d; unknown — O2).
  2. Fallback (bounded): in-kernel dequant — E4M3→BF16 LUT × 2^(UE8M0 e) —
     into the existing BF16 HMMA projection paths
     (`src/ops/linear/bf16/flash_next/*`, GDN `in_proj` sites). Dequant
     *in the kernel* is not the forbidden conversion-side dequant; the stored
     format and the oracle stay MXFP8.
  Sites: `flash_next_project_query_gate` (packed q+gate), k/v projections,
  GDN `in_proj_a/b/qkv/z` + `out_proj`, QSA indexer `index_qk_proj`,
  shared-expert gate/up/down (shared expert sits inside the MoE wrapper —
  route selection question O5).
- **A5 oracle.** Independent FP32 reference: exact `E4M3 × 2^e` products
  accumulated in FP32; production routes compared at BF16-appropriate
  tolerance. Definition recorded next to `nvfp4_maxabs`
  (`tensor-formats.md:104-109` pattern).

## 3. Work item B — W4A16 NVFP4 MTP expert banks

- **B1 format.** Reuse `expert_block_scale_k16_m128x4_v1` (code plane, swizzled
  E4M3 scale plane, E FP32 weight divisors — the existing bank layout). The
  difference is the *Use role*: today `expert_bank()` requires `AllowA4` +
  exactly one `activation_divisor` auxiliary (`load_plan.cpp:57-72`); add
  `AllowA16` — bank without activation divisor, activations consumed BF16.
- **B2 converter.** MTP bank assembly mirroring the main-layer path
  (`convert.py:227-286`): per-expert ModelOpt triple → fused
  `gate_up`/`down` banks, `weight_scale_2` → per-expert weight divisors,
  `_matching_pair` byte-equality check for gate/up. No `input_scale`
  handling — that is the W4A16 contract.
- **B3 runtime.** W4A16 variant of the MTP MoE decode route: gather top-10
  experts, in-kernel dequant `E2M1 × E4M3 × divisor → BF16`, BF16 MMA.
  Route selection extends `bind_mtp_moe`'s either/or (`load_plan.cpp:137-156`)
  and the sparse-MoE wrapper variant flag (`src/ops/wrapper/sparse_moe.cpp`).
  Perf expectation: drafter expert GEMMs are small-M and memory-bound; W4A16
  banks are ¼ the BF16 traffic. The existing A4 kernel remains available if
  later calibration justifies A4 for the drafter (deferred, O4 — the source
  declares A16; inventing an activation scale would change semantics).
- **B4 oracle.** FP32 `code × scale × divisor` products; same independent
  reference discipline as every NVFP4 bank.

## 4. Work item C — source adapter for the QAD checkpoint family

- **C1 inventory.** Name mapping: strip `model.language_model.`, keep `mtp.*`
  as-is (names already match `_mtp_specs()`); multi-source read across
  `model-*.safetensors` + `hybrid-main-*.safetensors`; accept
  `Qwen4ExpForConditionalGeneration`; read `mtp_num_hidden_layers` from
  `text_config`.
- **C2 signatures.** MXFP8 pairs (A2), W4A16 banks (B2), PLE per E's findings;
  closed-set validation updated so *unexpected* tensors still fail loudly.
- **C3 recipe.** `qwen3_8_flash_next_125b_a6b_qad_native-v1`, own output
  basename, conversion report with the usual payload verification.
- **C4 shortlist head.** Unchanged: rebuilt from this checkpoint's `lm_head`
  (`convert.py:372-376`, `draft_head.py`); vocab pins (248,320/248,077) hold.

## 5. Work item D — validation and evidence

1. Unit oracles for A and B at real shapes (CUDA-math row of the test table).
2. Converter reopen+verify on the emitted artifact.
3. Real-artifact battery on GB10: step 0; step 7a (first PPL record for this
   model — the 5090's fp8mtp PPL gates do not transfer); MTP acceptance
   compared against the checkpoint's SGLang reference (35.8–39.1 tok/s at 64k,
   mtp_steps 3, `runtime-qualification.json`).
4. Serving smoke incl. vision; MXFP8-attention A/B against the BF16-attention
   artifact for prefill/decode attribution (step 3 harness, now working on
   GB10).
5. Abliteration sanity: refusal-direction weights are inert at the engine
   level; the tokenizer/template pins are unchanged (248,077 tokens).

## 6. Work item E — PLE native route (verify-first)

The QAD PLE tables are NVFP4 group-16 (codes U8, E4M3 scales) and much smaller
than ours (~2.5M rows). Our PLE ingestion expects FP8 shards. First 0.5 d:
verify the exact row/element geometry (160 el/row at group 16 — matches the
shard shapes) and what the runtime PLE stage assumes about table size (the
binder carries table geometry, so a smaller table should bind — verify). Then
size the gather-path work: NVFP4 rows dequant-in-gather vs a new gather
kernel. Risk-bearing only if the runtime pins the FP8 row format in the file
page-in path (`flash_next_ple_stage.cu`).

## 7. Effort estimate (single owner, rough; for evaluation)

| Item | Content | Estimate | Risk |
|---|---|---|---|
| A1–A3 | spec + codec + converter + binder | 2–3 d | low |
| A4 | kernel probe + route (tcgen05 or SIMT fallback) + tuning | 4–6 d | **medium: tcgen05 MXFP8 on sm_121a unverified (O2)**; SIMT fallback bounds it |
| A5 + D1 | oracles + tests | 1–2 d | low |
| B1–B3 | bank route + binder role + W4A16 kernel | 3–4 d | low-medium |
| B4 + D1 | oracle + tests | 1 d | low |
| C1–C4 | source adapter + recipe | 2–3 d | low (mechanical, wide validation surface) |
| D2–D4 | GPU evidence campaign | 2–3 d | low |
| E | PLE verify + (conditionally) gather route | 0.5 d + TBD | medium (O1 geometry verified; runtime pinning unknown) |
| **Total** | | **~3–4 weeks** | dominated by A4 |

Sequencing: C first (unblocks everything on real data), then B (small,
independent), then A (largest), with D woven in per item. B alone (with C)
already yields a servable artifact if A is deferred — attention would then be
the only piece needing the BF16 conversion path, which the user has ruled out;
so A is on the critical path for this checkpoint, and B/C can land first for
the MTP-only consumers.

## 8. Open questions

- **O1** PLE runtime pinning: does the PLE stage/file page-in path assume the
  FP8 row format, or is the format carried per artifact? (verify first, §6)
- **O2** tcgen05 MXFP8 MMA availability on `sm_121a` (probe, A4).
- **O3** UE8M0 exponent bias in the ModelOpt export (read a `weight_scale`
  row from a shard during C2; likely bias-127).
- **O4** A4-for-drafter calibration: only if W4A16 decode cost is visible in
  profiles; would need real calibration data and would deviate from the
  source-declared contract.
- **O5** shared-expert MXFP8: joins the routed-expert wrapper's dense side or
  a separate projection route (wrapper topology decision during A3).

## 9. Explicitly out of scope

- Conversion-side dequant-to-BF16 (user decision, 2026-10-15).
- Reduced draft vocabulary (the ko64k runtime trick); draft/verify/accept
  stays in the 248,320 domain; our Optimized 147,456-row shortlist head
  already occupies that design niche.
- Yarn-4 1M context (their qualification configuration) — separate runtime
  feature.
- DFlash/DFlash2 components (absent in this checkpoint family).
- Abliteration evaluation as a quality matter (weights-only change; out of
  engine scope).
