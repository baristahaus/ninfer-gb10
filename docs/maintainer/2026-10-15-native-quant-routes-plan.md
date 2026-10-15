# Plan: native quantization routes for alternate checkpoint setups (QAD-class)

Status: evaluated and approved for execution (2026-10-15); this revision folds the
evaluation corrections in. Branch `qad-native-quant-plan`. Corrections are grounded in
code reads plus compile probes on the 5090 box (CUDA 13.4.92); shard-side facts remain
the GB10-side header reads recorded in section 1 (the `/nfs` checkpoint is not mounted
on the 5090 box).
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

One limit surfaced by the evaluation: the pattern covers *format* divergence. The
QAD PLE additionally presents *geometry* divergence — a different row space, not a
different encoding of the same rows — and that is not a format route. It gets its
own verification-first treatment (section 6, E1), which runs before everything else
because its outcome can gate artifact viability.

## 1. Target checkpoint facts (verified from the shards, 2026-10-15)

| Component | Format on disk (verified dtypes) | Our artifact today |
|---|---|---|
| Routed experts (48 layers) | NVFP4 W4A4: `weight` U8 `[n,k/2]` + `weight_scale` F8_E4M3 `[n,k/16]` + `weight_scale_2` F32 + `input_scale` F32 | same — **native ingestion already exists** |
| PLE n-gram tables | **NVFP4 group-16**: `weight` U8 `[2500012, 80]` (=160 el/row) + `weight_scale` F8_E4M3 `[2500012, 10]`, 11 shards (~2.5M rows — much smaller than our table; structurally different row space, see §6/E1) | FP8 shards (~51 GB-class table; 320,001,536 rows × 160 B) |
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

## 6. Work item E — PLE native route (re-scoped by the evaluation: geometry, not format)

The evaluation's headline correction: this is not a gather-format question, it is a
row-space question, and it is the plan's only correctness-blocking unknown.

Runtime facts (code, verified): the PLE geometry is NOT carried by the binder — it
is compile-time constants of the family runtime. `ple_table.h`: `kPleRows =
320,001,536`, `kPleHeadWidth = 160` bytes FP8, `kPleHeads = 16`. `ple_table.cpp`:
a 16-head pyramid of prime-sized ~20M-row tables (`kSizes = 20000003, 20000023,
...` with running `kOffsets`), hashed by fixed `kMultipliers` (8 bigram + 8
trigram heads). `gather_ple_fp8` throws unless the table is exactly
320,001,536 × 160 bytes (51,200,245,760) — a smaller table cannot bind.

Shard facts (GB10-side header reads): 11 shards × 2,500,012 rows × 160 elements
NVFP4 group-16 ≈ 225 MB total. Per-row *width* matches ours (160 elements). The
row *space* is ~12–128x smaller on any reading of the shard counts, and the table
count differs (11 vs 16). A format swap cannot explain that: the QAD PLE is a
structurally different pyramid — pruned to used rows, re-indexed, or re-configured.

- **E1 — row-space/mapping verification (runs FIRST, before C).** Object of
  study: the QAD packer's inference-side hash/compaction scheme (the same SGLang
  patch already referenced for the draft-vocab finding), plus the shards' real
  per-shard row counts. Outcomes:
  (a) a discoverable row space ⇒ E2 sized as below, plus a second fixed dimension
      set in the 125B package (`kSizes`/`kOffsets`/`kMultipliers`/`kPleRows` +
      the gather assert become recipe-dependent — real work the original estimate
      did not contain);
  (b) the mapping is unrecoverable from the export ⇒ the PLE is blocked pending
      either the packer's compaction artifact or a user decision on our own
      NVFP4 requant of the standard 320M-row table — a new quantization with its
      own qualification, not native ingestion.
- **E2 — gather/consumer kernel work (only after E1).** NVFP4 rows
  dequant-in-gather vs a new gather kernel; the staged FP8 contract
  (`kPleEmbeddingDim` bytes/token in `ple_stage.h/.cpp`) and the first PLE
  consumer's format decide where the decode lives.
- Side effect to record in D: if the PLE resolves at ~225 MB, the SP-6
  cold-collapse case (169–171 → 2.14 GB/s, 114.9 µs/faulted row) largely
  evaporates for this artifact — the demand-paging pressure profile changes
  character.
- Estimate fork: 0.5 d for E1; E2 bounded after E1; the blocked fork can gate
  artifact viability and is the plan's largest estimate risk.

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
| E1 | PLE row-space/mapping verification (packer scheme + shard row counts) | 0.5 d | **can gate artifact viability** (blocked fork is packer-dependent) |
| E2 | PLE gather/consumer kernel work (after E1) | TBD after E1 | low-medium once the row space is known |
| **Total** | | **~3–4 weeks if E1 resolves benignly** | dominated by A4; the E1 blocked fork is the timeline risk |

Sequencing (corrected): E1 first (its outcome gates artifact viability), then
C (unblocks everything on real data), then B (small, independent), then A
(largest), with D woven in per item. Run O3's scale read before drafting A1 —
a 15-minute shard read that decides whether "no global scale word" survives;
don't spec the layout first.

Nothing serves until A lands: attention/GDN/indexer/shared-expert weights are
MXFP8 and the no-dequant rule leaves no BF16 path, so A is the critical path.
B and C land first as validated increments (the MTP drafter and the converter),
not as a servable artifact.

Lane note: the artifact and kernels serve both lanes, and the 5090 (`sm_120a`)
is the default build target — the MXFP8 compile probe covers both `sm_120a` and
`sm_121a`, and each evidence row in D records which lane it came from. If E1
resolves benignly, the artifact shrinks from ~126 GB toward ~70–75 GB (MXFP8
dense leaves + a ~225 MB PLE; order estimate pending C), which materially
changes the 5090 tiered picture (headroom against the 73.7 GiB SP-7 tiered
load ceiling).

## 8. Open questions

- **O1** (re-scoped) the QAD packer's PLE row-space/compaction scheme — the
  mapping from the vendor pyramid (16 prime-sized ~20M-row tables, hashed into
  a 320,001,536-row space) to the QAD shards (11 × ~2.5M rows). Resolved
  negative on the original premise: the runtime pins the FP8 row format AND the
  exact table byte length as compile-time constants (`gather_ple_fp8` asserts
  320,001,536 × 160). The open part is the packer's mapping (E1, §6).
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
