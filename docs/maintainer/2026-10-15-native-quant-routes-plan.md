# Plan: native quantization routes for alternate checkpoint setups (QAD-class)

Status: evaluated and approved for execution (2026-10-15); this revision folds the
closed E1 verification and its binding consequence in (E1 closed 2026-10-15).
Branch `qad-native-quant-plan`. Corrections are grounded in code reads plus compile
probes on the 5090 box (CUDA 13.4.92); shard-side facts remain the GB10-side
header reads recorded in section 1 (the `/nfs` checkpoint is not mounted
on the 5090 box), with the PLE row superseded by E1's verified geometry.
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

One limit surfaced by the evaluation — the QAD PLE as a *geometry* divergence, a
different row space, gated behind a verification-first treatment (E1) — is closed
and overturned: the QAD PLE occupies the same 320,001,536-row space as ours (128
shards × 2,500,012 rows = exactly `kPleRows`), with lineage-specific trained
values. It is a format route after all — NVFP4 group-16 native ingestion of the
same row space — and the five-part pattern below covers the QAD family without
exception. Section 6 records the verification and its binding consequence; no item
in this plan is gated by an unknown row space.

## 1. Target checkpoint facts (verified from the shards, 2026-10-15)

| Component | Format on disk (verified dtypes) | Our artifact today |
|---|---|---|
| Routed experts (48 layers) | NVFP4 W4A4: `weight` U8 `[n,k/2]` + `weight_scale` F8_E4M3 `[n,k/16]` + `weight_scale_2` F32 + `input_scale` F32 | same — **native ingestion already exists** |
| PLE n-gram tables | **NVFP4 group-16**: `weight` U8 `[2500012, 80]` (=160 el/row) + `weight_scale` F8_E4M3 `[2500012, 10]`; **128 shards × 2,500,012 rows = 320,001,536 rows — the same row space and shard split as our source (E1-verified, §6)**; ~28.8 GB; values retrained during QAD distillation (E1); no global `scale_2`; three corrupt metadata tensors covered by the converter's `_PLE_METADATA` ignore-set | FP8 shards (51.2 GB table; 320,001,536 rows × 160 B; the RadixArk source's own values) |
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
- **A4 kernels.** O2 is resolved by the evaluation: the MXFP8 MMA exists on both
  lanes, via the instruction family the repo already uses — not tcgen05.
  - The Blackwell MMA library is `mma.sync` (`src/ops/common/mma.cuh`: BF16
    `m16n8k16`, `kind::f8f6f4` FP8, and the shipped `mxf4nvf4` block-scale NVFP4
    wrapper). tcgen05 is not used in-tree and is unsupported on `sm_120a`/`sm_121a`
    (compile probe, CUDA 13.4.92: `tcgen05.fence` rejected on both targets).
  - `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.
    row.col.f32.e4m3.e4m3.f32.ue8m0` compiles clean for `sm_120a` and `sm_121a`
    (CUDA 13.4.92, 5090 box, 2026-10-15 probe). Route: native MXFP8 block-scale MMA
    as the primary path — expected ~2x the BF16 MMA ceiling on compute-bound prefill
    GEMMs plus halved weight traffic.
  - Keep a 0.5 d GB10 throughput microbench: GB10 runs CUDA 13.0.88 and the probe
    box 13.4.92 (the kind set only grew between 13.0 and 13.4) — confirm
    throughput, not existence.
  - Fallback (bounded, kept as the A/B baseline): in-kernel dequant —
    E4M3→BF16 LUT × 2^(UE8M0 e) — into the existing BF16 HMMA projection paths
    (`src/ops/linear/bf16/flash_next/*`, GDN `in_proj` sites). Dequant *in the
    kernel* is not the forbidden conversion-side dequant; the stored format and
    the oracle stay MXFP8.
  Perf caveat: SP-6 measured prefill kernels at 7–33% of modeled minimums
  (qsa.select 7%, gdn.prefill 15%). MXFP8 halves MMA work and weight traffic but
  does not fix that inefficiency; the realized prefill win is bounded by each
  site's MMA/traffic share, and the step-3 A/B measures it.
  Sites: `flash_next_project_query_gate` (packed q+gate), k/v projections,
  GDN `in_proj_a/b/qkv/z` + `out_proj`, QSA indexer `index_qk_proj`,
  shared-expert gate/up/down. O5 is resolved: the shared expert is bound through
  the dense `projection()` route (`load_plan.cpp:106-108`, leaves
  {640,2560}/{2560,640}) — A3's format-set extension plus one more A4 kernel
  site; no wrapper-topology decision exists.
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
  Route selection extends `bind_mtp_moe`'s either/or (`load_plan.cpp:137-156`).
  Corrected pointer (evaluation): the Flash-Next MTP MoE is `ops::flash_next_moe`
  under `src/ops/sparse_moe/flash_next/` — the W4A16 variant sits beside the
  existing W4A4 path (`flash_next_nvfp4_w4a4.cuh`) and the BF16 GEMM machinery
  there. `src/ops/wrapper/sparse_moe.cpp` is the Qwen3.5 family's MoE wrapper:
  wrong family, no touch.
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
- **C2 signatures.** MXFP8 pairs (A2), W4A16 banks (B2), PLE per E1's closed
  findings (128 NVFP4 group-16 shards, no `scale_2`, corrupt metadata via the
  existing ignore-set);
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
   mtp_steps 3, `runtime-qualification.json`). These PPL and acceptance records
   are also the acceptance evidence for the PLE binding's residual assumption —
   hash-scheme invariance across the QAD retraining (§6) — which cannot be
   value-validated because the values are lineage-specific.
4. Serving smoke incl. vision; MXFP8-attention A/B against the BF16-attention
   artifact for prefill/decode attribution (step 3 harness, now working on
   GB10).
5. Abliteration sanity: refusal-direction weights are inert at the engine
   level; the tokenizer/template pins are unchanged (248,077 tokens).

## 6. Work item E — PLE native route (E1 closed 2026-10-15: a format route, same row space)

The evaluation had re-scoped this item as a row-space question and the plan's only
correctness-blocking unknown. E1 closed it: the re-scoping is overturned — the row
space is the same — and the values are retrained, not remapped, so no conversion-side
remap exists and the route is native ingestion of the source's own table.

Runtime facts (code, verified): the PLE geometry is NOT carried by the binder — it
is compile-time constants of the family runtime. `ple_table.h`: `kPleRows =
320,001,536`, `kPleHeadWidth = 160` bytes FP8, `kPleHeads = 16`. `ple_table.cpp`:
a 16-head pyramid of prime-sized ~20M-row tables (`kSizes = 20000003, 20000023,
...` with running `kOffsets`), hashed by fixed `kMultipliers` (8 bigram + 8
trigram heads). `gather_ple_fp8` throws unless the table is exactly
320,001,536 × 160 bytes (51,200,245,760) — the byte-format assert is the one
runtime constant E2c must make recipe-dependent; `kPleRows` and the hash constants
are unchanged.

Shard facts (E1-verified 2026-10-15, CPU-only, on the QAD layer-1 PLE): 128 NVFP4
shards × 2,500,012 rows, width 160, E2M1 low-nibble-first + per-16 E4M3 scales —
the community sidecar convention (`primitive-ai/Qwen3.8-Flash-Next-PLE-quant`);
128 × 2,500,012 = 320,001,536 = exactly `kPleRows` — the same row space and the
same shard split as our RadixArk source: no compaction, no pruning, no second fixed
dimension. QAD omits the global `scale_2`; its three metadata tensors are corrupt
and safely ignored (the converter's `_PLE_METADATA` ignore-set already covers
them). At ~90 B/row (codes + scales) the table is ~28.8 GB NVFP4 vs our 51.2 GB
FP8.

- **E1 — row-space/mapping verification (closed 2026-10-15).** Values: three
  tests against 128 rows gathered from our artifact at our SGLang-qualified hash
  IDs — flat shard concat (global = s×2,500,012 + i) and per-head padded blocks
  (h×20,000,096 + intra-head row) both at cosine ≈ 0 in both nibble orders; a
  full-table NN scan (128 queries × 320,001,536 QAD rows, 390 s) found best
  +0.41 / median +0.28 — the noise floor (a random 160-dim query peaks ~+0.30
  within one shard, ~+0.5 expected over the full table; NVFP4 reconstruction of
  a matching vector would be ≥0.99). The scan searched every row, so it also
  excludes any same-values-different-hash hypothesis. Conclusion: the QAD table
  is retrained during QAD distillation (consistent with the LIL card) and
  contains the Qwen-lineage vectors under no mapping, in no order.
- **E1 binding consequence (provenance, resolved from the repo records the same
  day).** The QAD-native artifact binds the QAD source's OWN retrained table —
  native NVFP4 group-16 ingestion through the same flat shard concat the FP8
  route already uses (`convert.py:_ple_payload` validates each shard's
  signature and concatenates the 128 in order; no reindexing exists to do, and
  none is needed). Our current artifacts need nothing: fp8mtp is single-source
  coherent — its PLE is the RadixArk source's own layer-1 table, byte-preserved
  through both conversion stages (convert.py emits the shards as-is; dense_fp8
  copies the PLE table byte for byte), and its dense weights are the same
  source's, so no shipped or planned artifact mixes lineages, and the QAD PLE
  cannot be sourced from fp8mtp's table. The hash scheme
  (`kSizes`/`kOffsets`/`kMultipliers`) is architecture-constant, not a trained
  parameter, so bucket indexing carries across the QAD retraining; that
  invariance cannot be value-validated (the values are lineage-specific) and is
  accepted behaviorally through D3's PPL and acceptance records.
- **E2 — NVFP4 group-16 PLE format route (five parts, like A and B; after C).**
  - E2a storage spec: the NVFP4 group-16 PLE layout — E2M1 codes (low-nibble
    first) + per-16 E4M3 scales along the 160-element row, no global scale word
    (QAD omits `scale_2`); spec'd in `tensor-formats.md` and `storage-layouts.md`
    beside the MXFP8 spec (A1).
  - E2b converter: source-signature validation (`weight` U8 `[2500012, 80]` +
    `weight_scale` F8_E4M3 `[2500012, 10]` per shard, 128 shards, corrupt
    metadata via the existing ignore-set) and the flat shard concat into the one
    table payload — mirroring the FP8 PLE route structurally.
  - E2c binder: the table byte-length/format assert becomes recipe-dependent
    (FP8 160 B/row vs NVFP4 ~90 B/row); `kPleRows` and the hash constants are
    unchanged.
  - E2d kernels: NVFP4 rows dequant-in-gather vs a new gather kernel; the
    staged FP8 contract (`kPleEmbeddingDim` bytes/token in `ple_stage.h/.cpp`)
    and the first PLE consumer's format decide where the decode lives.
  - E2e oracle: the NVFP4 group-16 decode (E2M1 × per-16 E4M3) at real row
    shapes — the same independent-oracle discipline as every NVFP4 bank (B4).
- Side effect to record in D: the QAD PLE resolves at ~28.8 GB NVFP4 (not the
  ~225 MB the earlier shard read implied) vs 51.2 GB FP8 — faulted-row bytes
  roughly halve, so the SP-6 cold-collapse case (169–171 → 2.14 GB/s, 114.9
  µs/faulted row) persists at reduced scale rather than evaporating.
- Estimate: E1 is closed (0.5 d, benign); E2 is bounded like B (2–3 d) — spec
  and converter/binder are small, the gather/consumer kernel carries the
  effort, and no gating unknown remains in this plan.

## 7. Effort estimate (single owner, rough; for evaluation)

| Item | Content | Estimate | Risk |
|---|---|---|---|
| A1–A3 | spec + codec + converter + binder | 2–3 d | low |
| A4 | kernel route (native mma.sync MXFP8, O2 resolved) + tuning + GB10 throughput microbench | 4–6 d | low-medium: instruction compile-verified on both lanes; SP-6 kernel inefficiency bounds the realized win |
| A5 + D1 | oracles + tests | 1–2 d | low |
| B1–B3 | bank route + binder role + W4A16 kernel | 3–4 d | low-medium |
| B4 + D1 | oracle + tests | 1 d | low |
| C1–C4 | source adapter + recipe | 2–3 d | low (mechanical, wide validation surface) |
| D2–D4 | GPU evidence campaign | 2–3 d | low |
| E1 | PLE row-space/mapping verification | closed 2026-10-15 (0.5 d) | resolved benignly: same row space, retrained values, no remap (§6) |
| E2 | NVFP4 group-16 PLE format route (E2a–E2e, §6) | 2–3 d | low-medium: mirrors the existing NVFP4 bank pattern; the gather/consumer kernel carries the effort |
| **Total** | | **~3–4 weeks** | dominated by A4; E1 is closed and no gating unknown remains |

Sequencing: E1 is closed, so C first (unblocks everything on real data), then
B (small, independent), then A (largest), with E2's format route landing beside
A (its spec and oracle follow the same five-part pattern) and D woven in per
item. Run O3's scale read before drafting A1 —
a 15-minute shard read that decides whether "no global scale word" survives;
don't spec the layout first.

Nothing serves until A lands: attention/GDN/indexer/shared-expert weights are
MXFP8 and the no-dequant rule leaves no BF16 path, so A is the critical path.
B and C land first as validated increments (the MTP drafter and the converter),
not as a servable artifact.

Lane note: the artifact and kernels serve both lanes, and the 5090 (`sm_120a`)
is the default build target — the MXFP8 compile probe covers both `sm_120a` and
`sm_121a`, and each evidence row in D records which lane it came from. E1
resolved benignly but the PLE is ~28.8 GB NVFP4 (not ~225 MB), so the artifact
shrinks from ~126 GB toward ~95–105 GB (MXFP8 dense leaves at −4.15 GB, the
halved PLE at −22.4 GB, W4A16 MTP banks at −3.6 GB; order estimate pending C),
which still changes the 5090 tiered picture materially (headroom against the
73.7 GiB SP-7 tiered load ceiling).

## 8. Open questions

- **O1** (resolved 2026-10-15, E1) the QAD PLE row-space/mapping question is
  closed: the QAD export is the full 320,001,536-row space in 128 shards of
  2,500,012 rows — the same space and shard split as our source — with values
  retrained during QAD distillation and no remap of the Qwen-lineage vectors
  under any mapping. The binding is native ingestion of the source's own table
  (§6). The residual assumption — hash-scheme invariance across the retraining
  — is accepted behaviorally through D3's PPL and acceptance records.
- **O2** (resolved) MXFP8 MMA exists on both lanes via `mma.sync
  .kind::mxf8f6f4.block_scale ... ue8m0` (compile-verified `sm_120a` +
  `sm_121a`, CUDA 13.4.92); tcgen05 is unsupported on both and is not the
  repo's MMA family. The GB10 throughput microbench remains (A4).
- **O3** UE8M0 exponent bias in the ModelOpt export, and whether the MXFP8
  pairs carry any global scale word (read a `weight_scale` row from a shard;
  likely bias-127). Runs before drafting A1, not during C2.
- **O4** A4-for-drafter calibration: only if W4A16 decode cost is visible in
  profiles; would need real calibration data and would deviate from the
  source-declared contract.
- **O5** (resolved) the shared expert is bound through the dense `projection()`
  route (`load_plan.cpp:106-108`); A3's format-set extension plus one more A4
  kernel site. No wrapper-topology decision.

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
