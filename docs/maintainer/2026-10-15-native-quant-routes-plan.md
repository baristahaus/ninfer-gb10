# Plan: native quantization routes for alternate checkpoint setups (QAD-class)

Status: C, A1-A3, B1-B2 and E2a-E2c are executed (2026-10-15: the converter, binder
routes and both storage specs landed as a274d9aa/8145f705/449edc04; the conversion run
completed and verified — see section 10); A4, B3 and E2d (the kernel routes) remain,
and nothing serves until they land. Earlier: evaluated and approved 2026-10-15, with
E1/O3 and the first-hand shard facts folded in.
Addition 2026-10-16: the artifact serves on GB10 (the MXFP8 parent routes landed on
this branch and the B3 blocking NaN was fixed — beads `ninfer-gb10-e86`, closed), and
section 11 adds work item F: MTP draft realignment by distillation, with the measured
misalignment evidence that motivates it.
Branch `qad-native-quant-plan`. Corrections are grounded in code reads plus compile
probes on the 5090 box (CUDA 13.4.92); the `/nfs` checkpoint is now mounted on
the 5090 box (verified 2026-10-15), so the PLE facts below are first-hand reads
from the shards and both model cards.
Tracking: beads `ninfer-gb10-<id>` (created with this plan).

Status reconciliation and remaining-effort evaluation (2026-10-28, master at
074a6aa5): the 2026-10-27 work-line integration carried the kernel routes to
`master` — A4's MXFP8 tiled-MMA and linear routes with the GDN gating fix
(8c66f403, `src/ops/softmax_attention/kernels/causal/mxfp8_tiled_mma.cuh`,
`src/ops/linear/mxfp8/`), B3's W4A16 MTP bank route
(`src/ops/sparse_moe/flash_next/flash_next_nvfp4_w4a16.cuh`) and the QAD
converter module (`tools/convert/qwen3_8_flash_next_125b_a6b/qad_convert.py`);
the verified QAD-native artifact sits in `out/` (98.6 GiB, §9 record). A4 and B3
are therefore closed on `master`; the stale status line above names them as
remaining. What actually remains on `master`:

- E2c binder: `bind_ple_table` still refuses everything but the FP8 table
  ("PLE requires a complete raw FP8 table",
  `src/models/qwen3_8_flash_next_125b_a6b/impl/load_plan.cpp`); the recipe-
  dependent NVFP4 ~90 B/row route exists on `5090-test` (449edc04) and is a
  mechanical port. 0.5–1 d, low.
- E2d kernels: the PLE consumer contract
  (`src/models/qwen3_8_flash_next/impl/ple_stage.h`) stages kPleEmbeddingDim
  FP8 bytes per token; the NVFP4 decode-in-gather vs new-gather-kernel decision
  is still open exactly as §6 states it. 1.5–2.5 d, low-medium.
- E2e oracle: the NVFP4 group-16 decode at real row shapes. 0.5–1 d, low.
- D serving evidence on the QAD-native artifact (perplexity, acceptance, TEB
  battery on GB10). 1–2 d, low.
- F (MTP draft realignment by distillation, §11): not started; ~1 week,
  dominated by the training runs, and it should be re-scoped only after the
  native route serves so the misalignment is measured on the served artifact.

Remaining to a servable QAD-native artifact: **~3.5–6.5 working days**, with E2d
the only nontrivial design decision left; no gating unknown. F is a separate
follow-on decision (~1 week) and stays gated on that serving milestone.

Origin of the request: `/nfs` checkpoint
`edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD` (93 GB) — a Qwen3.8
Flash-Next 125B-A6B of a different training lineage (the LIL QAD student,
abliterated and ModelOpt-requantized) whose quantization mix differs from our
closed recipe. Its weights were trained in the source-declared formats (the
student's forward pass used the quantized weights), which is the strongest case
for serving them natively. The
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
| PLE n-gram tables | **NVFP4 group-16**: `weight` U8 `[2500012, 80]` (=160 el/row) + `weight_scale` F8_E4M3 `[2500012, 10]` + a global `weight_scale_2` F32 scalar **present with a real value** (≈3.3242362e-05 — the decode includes it); **128 shards × 2,500,012 rows = 320,001,536 rows — the same row space and shard split as our source (first-hand, §6)**, spread across 17 of the 41 model files; ~28.8 GB; values retrained during QAD distillation (E1); the three metadata tensors carry the vendor hash constants, bit-identical to the runtime's (§6) | FP8 shards (51.2 GB table; 320,001,536 rows × 160 B; the vendor's FP8 table via the RadixArk source) |
| Attention, GDN, QSA-indexer, shared-expert projections | **MXFP8**: `weight` F8_E4M3 `[N,K]` + `weight_scale` U8 UE8M0 `[N,K/32]`; no `scale_2`, no `input_scale`; UE8M0 bias-127 (O3 resolved, §8) | BF16 |
| PLE projections/conv, routers, hyper-connections, norms, embeddings, lm_head | BF16/F32 — **except the 97 main-model `hc_norm` and PLE norm vectors, which are F32 in this source** (the MTP trunk's are BF16) | BF16; the artifact keeps the vendor BF16 contract for the norms with round-to-nearest conversion of the F32 sources — a real requant of those 97 vectors, pinned in the closed set, not a verbatim copy |
| MTP drafter trunk (fc, pre-norms, hyper-connections, self_attn + indexer, mlp.gate, shared_expert) | BF16 (names match our `_mtp_specs()` inventory exactly) | BF16 — matches |
| MTP routed experts (1 layer, 512 experts) | **NVFP4 W4A16**: `weight` U8 `[640,1280]` per proj + E4M3 scales + F32 `weight_scale_2`; **no `input_scale`** | BF16 dense (fp8mtp pass optionally re-emits A4 banks) |
| Draft output head | none (tied to 248,320 `lm_head`) | none — converter builds the 147,456-row shortlist head from `lm_head` (unchanged machinery) |

Architecture is declared `Qwen4ExpForConditionalGeneration` / `qwen4_exp`
(config-verified) with tensors prefixed `model.language_model.`; config
carries `text_config.mtp_num_hidden_layers: 1` (our validator's requirement,
under `text_config`) and `text_config.vocab_size: 248,320` (the C4 pin). The
file split is verified (2026-10-15): `hybrid-main-00001` carries the MTP W4A16
banks only (4,608 tensors: 512 experts × 3 projections × {weight,
weight_scale, weight_scale_2}); `hybrid-main-00002` carries all 240
attention/GDN MXFP8 pairs (36 GDN layers' `in_proj_a/b/qkv/z` + `out_proj`;
12 QSA layers' `q/k/v/o` + `indexer.index_qk_proj`); the shared-expert MXFP8
pairs (48 × 3), the NVFP4 W4A4 routed banks, the PLE, and the BF16/F32 bulk
live in the `model-*` files. `mtp.layers.48` in `hf_quant_config.json` is a
stray alias tag (no such tensors). The SGLang recipe's
`SPARKTALK_FLASH_NEXT_DRAFT_VOCAB=ko64k` is a runtime draft-vocab truncation,
not a checkpoint shape — out of scope (§5).

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
- **C2 signatures.** MXFP8 pairs (A2), W4A16 banks (B2), PLE per the verified
  shard facts (128 NVFP4 group-16 shard pairs + the global `weight_scale_2`
  scalar; hash-constant metadata via the existing ignore-set);
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

Shard facts (first-hand reads 2026-10-15, from the now-mounted `/nfs` checkpoint):
128 NVFP4 shards × 2,500,012 rows, width 160, E2M1 low-nibble-first + per-16 E4M3
scales — the community sidecar convention
(`primitive-ai/Qwen3.8-Flash-Next-PLE-quant`); 128 × 2,500,012 = 320,001,536 =
exactly `kPleRows` — the same row space and the same shard split as our RadixArk
source: no compaction, no pruning, no second fixed dimension. A global
`weight_scale_2` F32 scalar is present with a real value (3.324236240587197e-05),
so the decode is code × block scale × global scale — the standard NVFP4 pattern;
the earlier "QAD omits the global scale_2" reading is corrected. The three
metadata tensors are not corrupt: they carry the vendor hash constants,
bit-identical to the runtime's hardcoded values (`layer_multipliers` =
`kMultipliers`, `ngram_heads_offsets` = `kOffsets`,
`ngram_heads_vocab_sizes` = `kSizes`, verified against `ple_table.cpp`), and the
converter's `_PLE_METADATA` ignore-set stays correct because the runtime
hardcodes those constants, not because the tensors are corrupt. At ~90 B/row
(codes + scales) the table is ~28.8 GB NVFP4 vs our 51.2 GB FP8.

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
  cannot be sourced from fp8mtp's table. The lineage is card-verified on both
  sides: the RadixArk source is a post-training quantization of the vendor model
  whose PLE is the FP8 table from Qwen's own
  `Qwen/Qwen3.8-Flash-Next-FP8` revision (128 shards F8_E4M3 + per-table
  scalar), and the QAD family is the LIL QAD student — trained against the BF16
  teacher with quantized weights in its forward pass, its PLE trained in that
  distillation per the card's "What's quantized" table — which Huihui then
  abliterated and ModelOpt-requantized; that lineage split is exactly why E1
  found no shared values. The hash scheme
  (`kSizes`/`kOffsets`/`kMultipliers`) is architecture-constant, not a trained
  parameter, and the export's own metadata tensors carry exactly those
  constants bit-identically, so bucket indexing carries across the QAD
  retraining by direct evidence; D3's PPL and acceptance records remain the
  behavioral corroboration.
- **E2 — NVFP4 group-16 PLE format route (five parts, like A and B; after C).**
  - E2a storage spec: the NVFP4 group-16 PLE layout — E2M1 codes (low-nibble
    first) + per-16 E4M3 scales along the 160-element row + the global
    `weight_scale_2` F32 scalar (present in the export at ≈3.3242362e-05; the
    decode is code × block scale × global scale); spec'd in
    `tensor-formats.md` and `storage-layouts.md` beside the MXFP8 spec (A1).
  - E2b converter: source-signature validation (`weight` U8 `[2500012, 80]` +
    `weight_scale` F8_E4M3 `[2500012, 10]` per shard, 128 shards, the global
    `weight_scale_2` scalar, hash-constant metadata via the existing ignore-set)
    and the flat shard concat into the one table payload — mirroring the FP8
    PLE route structurally.
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

Sequencing: E1 and O3 are closed, so A1 and E2a's specs draft first (one docs
pass, both layouts unblocked), then C (unblocks everything on real data), then
B (small, independent), then A (largest), with E2's packer landing beside B
(both small, both fully fact-verified) and D woven in per item.

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
- **O3** (resolved 2026-10-15, first-hand) UE8M0 bias-127 confirmed by
  magnitude sanity: `layers.0.linear_attn.in_proj_qkv.weight_scale` (in
  `hybrid-main-00002`) carries bytes 113–117, decoding to exponents
  2^-14..2^-10, and the sampled max |E4M3 code| 448 × 2^-10 = 0.4375 is a sane
  projection magnitude. No MXFP8 pair carries a global scale word or an
  `input_scale` (the F32 inventory holds only the routed experts'
  `scale_2`/`input_scale`, the MTP banks' `scale_2`, the PLE's `scale_2`, and
  the norms) — A1's `dense_block_scale_k32_mxfp8_v1` spec stands as drafted,
  and A1 is unblocked.
- **O4** A4-for-drafter calibration: only if W4A16 decode cost is visible in
  profiles; would need real calibration data and would deviate from the
  source-declared contract.
- **O5** (resolved) the shared expert is bound through the dense `projection()`
  route (`load_plan.cpp:106-108`); A3's format-set extension plus one more A4
  kernel site. No wrapper-topology decision.

## 9. Conversion run record (executed 2026-10-15, the QAD-native artifact)

The tooling (a274d9aa formats/layouts/codecs/AllowA16/recipe/oracles, 8145f705 binder
routes, 449edc04 the NVFP4 PLE binding and package-boundary refusal) produced
`out/qwen3_8_flash_next_125b_a6b_qad_native.ninfer` — 1,634 objects, 105,891,776,256
bytes (98.6 GiB), weights id `qad_native`, converted in 2,069 s. Gates: the closed-set
validation passed with no unexpected or missing tensors; per-tensor signatures were
verified against the real 43-file checkpoint before writing; the post-write reopen and
payload verification (D2) matched all 1,634 payloads by full SHA-256; D1's 68 oracle
tests passed (independent float64 E4M3×2^(e−127) and NVFP4 decodes at real shapes); the
C++ binder smoke validated all objects and bound the full text+MTP+vision plan (1,625
device objects, 76.9 GB device capacity, PLE file-backed); the vendor regression held
(fp8mtp load-plan unchanged, the focused Flash-Next ctest set green). Payload
exactness: the PLE object is 320,001,536×90+4 bytes exactly; the MTP bank planes match
the bank geometry to the byte on the AllowA16 + weight_divisor contract.

Deviations from the guidance's letter, recorded: the run used a dedicated
`qad_convert` module rather than extending the vendor converter (the vendor route and
its closed contract stay untouched — zero regression surface — at the cost of a
duplicated packing surface to maintain); the run executed on CPU (the venv torch is
CPU-only and pick_device fell back; encoder determinism means the verification is
unaffected); and the 97 F32 main-model norm vectors were rounded to the vendor BF16
contract (section 1's corrected row) rather than kept F32.

Serving remains blocked at the package boundary by design: `model.cpp` refuses the
NVFP4 PLE with a named error until A4 (MXFP8 MMA kernels + the GB10 microbench), B3
(the W4A16 MTP MoE route) and E2d (the NVFP4 PLE gather) land — the plan's "nothing
serves until A".

## 10. Explicitly out of scope

- Conversion-side dequant-to-BF16 (user decision, 2026-10-15).
- Reduced draft vocabulary (the ko64k runtime trick); draft/verify/accept
  stays in the 248,320 domain; our Optimized 147,456-row shortlist head
  already occupies that design niche.
- Yarn-4 1M context (their qualification configuration) — separate runtime
  feature.
- DFlash/DFlash2 components (absent in this checkpoint family).
- Abliteration evaluation as a quality matter (weights-only change; out of
  engine scope).

## 11. Work item F — MTP draft realignment by distillation (added 2026-10-16)

### F0. The measured problem

The QAD-native artifact's MTP acceptance is 0.458 accepted/drafted (step 7a phase 2,
`profiles/bench/gb10/step7a-qadfix/`) against the vendor BF16 artifact's 0.623 on the
same corpus-continuation protocol — 33.3 → 26.6 tok/s. The 2026-10-16 research cycle
(beads notes `mtp-acceptance-research-*`, `mtp-improvement-cycle-*`) decomposed it:

- The QAD source's MTP module is **byte-identical** to the original checkpoint's
  (source `bf16-audit.json`: 144 tensors changed by the abliteration —
  `linear_attn.out_proj` ×36, `experts.down_proj` ×48, `shared_expert.down_proj`
  ×48, `self_attn.o_proj` ×12 — MTP 0/31). The artifact therefore already carries
  the original draft stack (BF16 adapters, NVFP4 expert banks). Swapping any
  "original BF16 module" in is a no-op.
- The draft-target misalignment is **not** the recipe: the margin-conditional
  acceptance curve is monotone for the vendor (verify margin ≥ 8 → 0.91–1.00
  accept) but flat for QAD (margin ≥ 8 → 0.54): the draft confidently contradicts
  even unambiguous target top-1s. The draft weights are unchanged, so this is the
  draft mis-reading the *shifted* hidden states its unchanged `fc_hidden`/
  `fc_embedding` adapters now receive.
- **Drift ceiling** (teacher-forced greedy agreement between the original target —
  the RadixArk vendor artifact — and the abliterated target over 261,167
  quick-corpus positions): **0.894**. Both targets pick the corpus token as top-1
  at the same rate (0.7113 / 0.7112), so the 10.6% disagreement is directional
  drift, not quality. Caveat: each side carries its own quantization noise
  (NVFP4 experts / MXFP8 leaves), which slightly inflates the measured drift.
- The draft's position-0 acceptance (0.585) sits **~31 pp below** the 0.894
  frozen-original ceiling. Only ~10.6 pp of the gap is irreducible; ~31 pp is the
  trainable part. Recovering it takes tokens/round from ~1.92 toward ~2.4 and
  serving throughput up ~20%.
- **Correction sub-option closed by measurement (2026-10-16 15:20Z).** Paired
  final-layer hidden states were dumped for both targets on the quick corpus
  (261,167 aligned positions, temporary `NINFER_SCORE_HIDDEN_DUMP` tap, removed
  after use). The original-minus-abliterated delta has norm ~49.9 against hidden
  norms ~174 (≈29% relative), but it is **not** a systematic shift: the mean
  vector carries 2.4% of the delta energy, rank-256 of the centered delta
  carries 35%, and a full-rank ridge linear map of the state explains ~13%.
  The difference is high-rank, per-token computational divergence of the 144
  changed weight matrices — an affine or low-rank correction at the MTP bridge
  cannot capture it. This closes the "affine/low-rank input correction" route
  and leaves distillation (learning the mapping nonlinearly through the draft
  stack) as the only engine-compatible lever. The paired dumps remain on the
  shared tank for the trainer's stage (a) warm-up data.

### F1. Objective

Fine-tune the QAD artifact's draft stack against the **frozen abliterated target**
so the draft's next-token distribution matches the target's on the state
distribution serving actually produces. The abliteration itself is preserved: the
144 changed tensors, the embeddings, the `lm_head`, the PLE and the target's whole
forward path stay frozen. Only the draft-side modules train.

### F2. What trains and what stays frozen

| Module | Params (approx) | Trains | Rationale |
|---|---|---|---|
| `mtp.pre_fc_norm_*`, `mtp.fc_embedding`, `mtp.fc_hidden` | ~26 M | yes | the input adapters that mis-map the shifted hidden states |
| `mtp.layers.0` attention + indexer + hyper/mixers + gate + shared expert | ~60 M | yes | the draft's transformer body |
| `mtp.layers.0.mlp.experts` (routed banks) | ~2.8 B | **no** | quantized weights have no training path; the vendor achieves 0.73/0.51 with NVFP4 draft banks, so they are not the bottleneck; unfreezing would need BF16 master weights (~11 GB) and a requantize pass |
| target (all 48 layers, PLE, embeddings, `lm_head`) | — | **no** | the alignment target and the product model; touching it changes PPL and the abliteration |

Trainable set ≈ 90 M params: AdamW states ≈ 1.4 GB, gradients 0.2 GB — comfortably
inside the memory envelope alongside the frozen 70 GB target.

### F3. Loss and data

- **Loss**: forward-KL between the draft logits and the frozen target's
  next-token distribution, on target-scored positions (standard MTP distillation).
  Greedy-CE on the target argmax is a cheaper fallback but optimizes the wrong
  thing near ties; KL also fixes the confident-contradiction shape measured above.
- **Distribution**: the loss must be computed on the states the serving loop
  produces. Two stages: (a) teacher-forced warm-up on corpus continuation text
  (the perplexity-1m corpus is the right base distribution — it is what phase 2
  measures); (b) on-policy rounds: generate with the current draft+target pair,
  score the target on the generated sequences, add them to the mix. Stage (b)
  matters because acceptance is measured on the engine's own committed streams.
- **Scale**: 50–200 M tokens is the first budget. The throughput reference is the
  measured scoring rate (732 tok/s at 4096/2048): the target forward dominates, so
  a 100 M-token pass is roughly 2 days of single-GPU time if run at scoring-like
  throughput; batch the target forward and the draft forward in one pass to avoid
  a second sweep.

### F4. Procedure

1. Extract the training interface from the engine: the draft consumes the target's
   last-layer hidden state plus the terminal token's embedding through
   `mtp.pre_fc_norm_*`/`fc_*` (see `mtp_impl.h`,
   `mtp_bridge_and_propose`/`mtp_forward_ar_step`). A standalone training script
   can reuse the engine for target forwards (the public `score_tokens` route is
   too narrow; the trainer belongs to the checkpoint lane and may use the source
   framework directly — the `/nfs` checkpoint loads under the model's own
   tooling).
   **Revised 2026-10-16 after the feasibility probe (beads `ninfer-gb10-6wy`):**
   the source-framework shortcut is not available — no HF modeling code ships for
   this architecture, and SGLang's `qwen4_exp` draft is runtime-integrated (QSA
   indexer metadata from the model_runner; its full server also hits a loader
   bug on this checkpoint). The trainer's draft forward is therefore built in
   plain torch from the model doc's exact math, with SGLang's `qwen4_exp_mtp.py`
   as a cross-reference for the composition (fc fusion: pre-normed embedding +
   `fc_hidden(pre_fc_norm_hidden(hidden))` per HC slice; `num_hidden_layers=1`,
   `is_nextn`) and the engine + the paired `/nfs` dumps as the oracle. The
   module constructs standalone on GB10 (3.89 B params; recipe in the issue).
   Estimate for this step: **3–5 d**, and the routed expert banks are
   dequantized ModelOpt-NVFP4 → BF16 (~5 GB: `[512,1280,2560]` + `[512,2560,640]`).
2. Train stage (a), then stage (b); checkpoint the ~90 M trainable params.
3. Re-export the trained `mtp.*` tensors into a source checkpoint directory in
   the QAD source layout (`model.language_model`-stripped, `mtp.*` as-is —
   the converter's closed-set validation in work item C1 accepts exactly these).
4. Re-run `qad_convert` (no code change: `mtp.*` is in the BF16 ignore set) into
   a new artifact (`..._qad_native_mtp2.ninfer`, recipe
   `qwen3_8_flash_next_125b_a6b_qad_native-v2`).
5. Validate per F5, then promote.

### F5. Validation gates (the fixed battery)

1. **Step 7a phase 2 acceptance** (the primary gate): position-0 acceptance
   ≥ 0.80 (ceiling 0.894), overall accepted/drafted ≥ 0.55, throughput ≥ the
   vendor's 33 tok/s. Record per-position and margin-conditional curves as in the
   2026-10-16 research note, so the remaining gap can be attributed.
2. **Perplexity unchanged**: the target is frozen, so the phase 1 record must
   reproduce within measurement noise (the 2026-10-16 record: 3.9649 / 3.7762).
   A moved PPL means the target was touched — fail.
3. **Normal tests**: the focused Flash-Next ctest set green; the real engine test
   passes with the plain-decode canonical golden unchanged (plain decode never
   reads the draft) and the QAD MTP golden re-recorded per the documented
   near-tie mechanism if the verify tie-breaks move.
4. **TEB --short** (the second-order contextual validator): ≥ the recorded
   baseline (97/100, 15/15, safety gate clean). Full hardmode stays as the
   existing extended-set record.
5. **Vendor regression**: the BF16 artifact's real test and battery records
   unchanged.

### F6. Risks and honest limits

- The 0.894 ceiling is teacher-forced on corpus contexts; on the engine's own
  on-policy streams the achievable number may differ in either direction. The
  phase-2 gate, not the ceiling, decides promotion.
- Distillation can overfit the corpus mix and lose the chat-shaped robustness the
  TEB --short gate watches for; keep the stage (b) on-policy mix and the TEB gate.
- The routed expert banks stay frozen, so a residual misalignment through expert
  selection may cap the recovery below the ceiling; if phase 2 stalls below
  0.70, escalate to BF16 master weights for the banks (a separate decision —
  ~11 GB + requantize).
- Training runs on the checkpoint lane; the engine repo's role ends at
  ingesting the re-exported `mtp.*` tensors (step 4) — no engine or converter
  code change is expected.

### F7. Effort estimate

| Step | Content | Estimate | Risk |
|---|---|---|---|
| F4.1 | trainer interface + target forward harness | 2–3 d | medium: checkpoint-lane tooling, outside this repo's build |
| F4.2 | stage (a)+(b) training runs | 2–4 d wall | medium: single-GPU throughput-bound; resumable |
| F4.3–4.4 | re-export + conversion + new artifact | 0.5 d | low: existing closed pipeline |
| F5 | the fixed battery + records | 1 d | low |
| **Total** | | **~1 week** | dominated by the training runs |
