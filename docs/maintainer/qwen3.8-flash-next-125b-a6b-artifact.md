# Qwen3.8 Flash-Next 125B-A6B artifact reference

This reference defines the two registered storage profiles for Qwen3.8 Flash-Next 125B-A6B: the
checkpoint-word profile written by the converter and the dense-FP8 profile derived from it. Generic
framing, layouts, and numeric formats remain governed by `artifact-container.md`,
`storage-layouts.md`, and `tensor-formats.md`; model mathematics are defined in
[`qwen3.8-flash-next-125b-a6b-model.md`](qwen3.8-flash-next-125b-a6b-model.md).

## Architecture and representation

The v3 Text component selects `Qwen3_8FlashNextForCausalLM` with
`model_type=qwen3_8_flash_next_text`, hidden size 2560, 48 layers, vocabulary 248320 and 512 experts.
The package validates this fixed configuration and the selected logical bindings. `metadata.name`
is descriptive; filenames and former v2 `model_id`/`weights_id` fields do not select execution.
The full recipe emits Text, MTP, Vision, the indexed proposal head and six frontend resources.
Only enabled components and their dependencies are bound and materialized.

## Inventory and formats

The closed inventory contains 1,627 tensors and six raw resources. Main routed expert banks use
`nvfp4` with `expert_block_scale_k16_m128x4_v1`; their FP32 input divisors are per-expert `activation_divisor` auxiliaries on the
`AllowA4` Use. Other projections consume A16 and require a declared activation policy. MTP expert banks, projections, HyperConnection weights, norms, embeddings, output head,
and shared experts retain BF16. GDN control vectors and NVFP4 divisors use FP32. Vision retains the
existing Q4/Q5/Q6 groupwise and W8 merger profiles.

The 320,001,536-by-160 PLE embedding is one contiguous FP8 E4M3FN tensor plus a BF16 multiplier.
It is the artifact's only file-mapped tensor. The reader validates its descriptor and payload
extent, then exposes read-only mappings spanning the v3 payload shards. A gathered row may cross a shard boundary. Demand paging and the operating-system file
cache own residency; the generic materializer does not allocate host or device storage for the
complete 51.2 GB table. Prompt preparation gathers only the sixteen rows selected for each token.

The FP8 profile (recipe `qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp-v3`) gives the MTP layer the
formats of a main layer. Its projections that every decode or draft step reads in full are
`fp8_e4m3fn_row_bf16` with `row_scale_v1`: in the 48 text layers and the MTP layer, the GDN
query/key/value, output-gate and output projections, the QSA packed query/gate and output
projections, the HyperConnection down and up projections and the shared-expert projections; and
the text and MTP final mixers and the output head. Each keeps its object and binding, except each
shared expert's gate and up projections: one `[1280,2560]` parent,
`...mlp.shared_expert.gate_up_proj.weight`, stacks the gate rows then the up rows, and the two
parameters bind its first and second halves, so the fused SwiGLU consumes the whole parent. Their
A16Only Uses are unchanged: activations stay BF16 and only the weights are quantized.

The MTP routed experts become NVFP4 banks in the main layers' representation: parameters
`mtp.layers.0.mlp.experts.gate_up` `[512,1280,2560]` and `.down` `[512,2560,640]` replace the
checkpoint's BF16 `experts.gate_up_proj` and `experts.down_proj`, with `nvfp4` codes quantized by
`nvfp4_maxabs` from the checkpoint words (`tensor-formats.md` 2.4) and AllowA4 Uses whose
`activation_divisor` auxiliaries, `..._input_divisors`, give every expert the smallest divisor
of the same role over the 48 main layers' calibrated banks (their largest input range). The
conversion report records that divisor with each main layer's minimum and median.

The router, norms, the MTP layer's `[2560,2560]` embedding and hidden projections (no main layer
has them) and the small projections (QSA key/value/indexer, GDN a/b, PLE) keep the
checkpoint-word representation. The profile has 49 fewer shared-expert objects and two more
divisor objects than the checkpoint-word profile. It stores about 4.2 GB less for the text
projections (8.3 GB of BF16 becomes 4.2 GB of FP8, the same amount off every decoded token's reads)
and about 3.7 GB less for the MTP layer (its 5.03 GB BF16 banks become 1.42 GB of NVFP4; a draft
step reads its ten experts, about 98 MB of BF16 before and 28 MB after, and about half the bytes of
its FP8 dense leaves).

The binder takes each of these leaves, and the MTP bank pair, in the representation the artifact
records, so an artifact with BF16 projections or BF16 MTP banks loads through the same routes.

The upgraded artifact occupies 134,755,956,216 bytes across five files capped at 32 GB each. Its format allocation is 1,249 BF16, 168 FP32,
one FP8 table, 96 NVFP4 expert banks, 55 Q4, 54 Q5, one Q6, one INT32 map, and two Q8 tensors. The six embedded
resources are `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`,
`generation_config.json`, `preprocessor_config.json`, and `video_preprocessor_config.json` under
the `frontend/` namespace.

## Conversion

The converter accepts only the closed `RadixArk/Qwen3.8-Flash-Next-NVFP4` checkpoint allocation.
It preserves BF16 and FP8 words, transposes channel-wise convolution kernels into the target layout,
and rearranges ModelOpt expert-major NVFP4 codes and scales into NInfer's bank layout without
dequantizing or requantizing them. It concatenates the 128 PLE shards directly into the one table
payload and writes a conversion report beside the artifact.

```bash
python3 -m tools.convert.qwen3_8_flash_next_125b_a6b.convert \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --out out/qwen3_8_flash_next_125b_a6b_nvfp4.ninfer \
  --device cuda
```

The output basename is fixed. Conversion rejects missing, unexpected, incorrectly shaped, or
incorrectly typed source tensors, mismatched paired gate/up scales, invalid divisors, incompatible
model configuration, and incomplete frontend resources.

The FP8 profile is derived from a checkpoint-word artifact, whose BF16 words are the
checkpoint's:

```bash
python3 -m tools.convert.qwen3_8_flash_next_125b_a6b.dense_fp8 \
  --source out/qwen3_8_flash_next_125b_a6b_nvfp4.ninfer \
  --out out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer
```

Each selected matrix is quantized with `fp8_row_maxabs` rounding: one BF16 row multiplier, the
round-to-nearest-even BF16 value of the row's maximum magnitude divided by 448, and E4M3FN codes
rounded to nearest even. Rows are quantized independently, so a packed gate/up parent holds
exactly the codes and multipliers of its two matrices. Every other object is copied byte for
byte. The re-encoder rejects a selected parameter that is not one whole contiguous BF16 object of
its exact shape, or whose object another binding or auxiliary also references, and a packed
parent id that already names an object. The report beside the output records, per written
matrix, the relative RMS and maximum absolute error of the represented weights against the BF16
words. Each MTP expert is quantized with `nvfp4_maxabs` from its BF16 words, experts
independently (one weight divisor each); the report records each bank's error the same way and the
chosen activation divisors.

## Upgrade an existing v2 artifact

```bash
python3 tools/upgrade_ninfer_v2_to_v3.py \
  out/candidate-wide-q4/qwen3_8_flash_next_125b_a6b_nvfp4.ninfer \
  out/v3/qwen3_8_flash_next_125b_a6b_nvfp4.ninfer
```

The one-time upgrader preserves all encoded weight bytes and Flash-Next's registered chat template.
It writes v3 components, logical bindings, Uses and shard framing. Keep the entry file and all
`.part-NNNN` companions together. The original v2 file remains available to the previous binary;
the v3 Engine accepts only v3 artifacts.

## Runtime binding

Text alone selects 1,260 device objects. MTP adds 31, Vision adds 333, and the optimized proposal
adds two. Frontend resources are owned host bytes; PLE is a read-only mapped range whose lifetime
is owned by the loaded model. PLE mappings are not counted as GPU uploads or materializer staging.
The C++ loader checks logical shapes, formats, expert layout, activation permissions, selected
component targets and indexed-proposal geometry before building the Program. The projections of
the FP8-projection profile accept either BF16 or `fp8_e4m3fn_row_bf16`, as the artifact records
it, wherever they occur (the MTP layer included); an FP8 shared-expert gate/up pair must be the two
halves of one parent. Every other leaf has one declared format. The consuming Ops take the FP8
words directly: A16 FP8 linear routes for the exact problems, LinearSwiGLU's Flash-Next profile
for the packed shared-expert parent, fused FP8 decode forms for the HyperConnection down
projection and the QSA query/gate split, and an FP8 exact-head rescoring for MTP drafting.

Flash-Next retains its registered template renderer and does not support `--chat-template`
overrides. Text, Vision, MTP, prefix reuse and concurrency still use the public Engine route.
