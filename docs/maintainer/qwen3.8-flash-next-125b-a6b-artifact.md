# Qwen3.8 Flash-Next 125B-A6B artifact reference

This reference defines the registered NVFP4 storage profiles for Qwen3.8 Flash-Next 125B-A6B. Generic
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

The FP8-PLE inventory contains 1,627 tensors and six raw resources. Main routed expert banks use
`nvfp4` with `expert_block_scale_k16_m128x4_v1`; their FP32 input divisors are per-expert `activation_divisor` auxiliaries on the
`AllowA4` Use. Other projections consume A16 and require a declared activation policy. MTP expert banks, projections, HyperConnection weights, norms, embeddings, output head,
and shared experts retain BF16. GDN control vectors and NVFP4 divisors use FP32. Vision retains the
existing Q4/Q5/Q6 groupwise and W8 merger profiles.

The 320,001,536-by-160 PLE embedding is one contiguous FP8 E4M3FN tensor plus a BF16 multiplier
for the RadixArk checkpoint. Swift 1.5 retains the same table as unscaled BF16 and has no multiplier;
its artifact contains 1,626 tensors. The format determines the PLE gather stride, decode staging,
and whether the Op dequantizes or consumes the gathered BF16 words directly.
It is the artifact's only file-mapped tensor. The reader validates its descriptor and payload
extent, then exposes read-only mappings spanning the v3 payload shards. A gathered row may cross a shard boundary. Demand paging and the operating-system file
cache own residency; the generic materializer does not allocate host or device storage for the
complete 51.2 GB FP8 or 102.4 GB BF16 table. Prompt preparation gathers only the sixteen rows selected for each token.
The package maps PLE with random-access advice to suppress sequential readahead. Each Program
owns sixteen host gather workers; prefill distributes token ranges across them and joins all
reads before upload. Small decode gathers remain serial. This preserves token/head ordering
and exact stored words while allowing cold page faults to overlap.

The upgraded artifact occupies 134,755,956,216 bytes across five files capped at 32 GB each. Its format allocation is 1,249 BF16, 168 FP32,
one FP8 table, 96 NVFP4 expert banks, 55 Q4, 54 Q5, one Q6, one INT32 map, and two Q8 tensors. The six embedded
resources are `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`,
`generation_config.json`, `preprocessor_config.json`, and `video_preprocessor_config.json` under
the `frontend/` namespace.

## Conversion

The converter accepts the closed `RadixArk/Qwen3.8-Flash-Next-NVFP4` and
`ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4` checkpoint allocations.
It preserves BF16 and FP8 words, transposes channel-wise convolution kernels into the target layout,
and rearranges ModelOpt expert-major NVFP4 codes and scales into NInfer's bank layout without
dequantizing or requantizing them. It concatenates the 128 PLE shards directly into the one table
payload and writes a conversion report beside the artifact.
The writer disables Btrfs compression before writing any shard that contains PLE. Compressed
extents otherwise amplify each small random row read into a much larger disk read and
decompression operation. This is a filesystem storage policy; artifact bytes and model
precision are unchanged. Other filesystems retain their normal output behavior.

```bash
python3 -m tools.convert.qwen3_8_flash_next_125b_a6b.convert \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --out out/qwen3_8_flash_next_125b_a6b_nvfp4.ninfer \
  --device cuda
```

The output basename is fixed for each source profile. Conversion rejects missing, unexpected, incorrectly shaped, or
incorrectly typed source tensors, mismatched paired gate/up scales, invalid divisors, incompatible
model configuration, and incomplete frontend resources.

Swift 1.5's source is an experts-only ModelOpt NVFP4 checkpoint with BF16 PLE shards. Convert it
with the explicit source profile; the BF16 PLE words are concatenated without quantization:

```bash
python3 -m tools.convert.qwen3_8_flash_next_125b_a6b.convert \
  --model /storage/ai/huggingface/main/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4 \
  --source-profile swift \
  --out out/v3/swift_1_5_qwen3_8_flash_next_nvfp4.ninfer \
  --device cpu
```

`--ple-format fp8_e4m3fn` instead re-encodes Swift's BF16 PLE table as FP8 E4M3 with one BF16
table scale (the table max-abs over 448, rounded up so no value saturates; round-to-nearest codes).
The table halves to 51.2 GB, which a 128 GB host can keep resident in the page cache; the output
basename gains `_fp8ple`. On the oracle fixtures the re-encoded table moves generated-token
log-probabilities by 0.005-0.074 on average with perplexity changes of -1.0% to +0.5%, and MTP3
acceptance is unchanged (2.06 tokens per round).

`--projection-format fp8_e4m3fn_row_bf16` stores 301 main-model matrices as weight-only row-scaled
FP8 E4M3 (per-row BF16 max-abs scale, round-to-nearest): QSA q/k/v/o, GDN in_proj_qkv/in_proj_z/
out_proj, each MoE layer's router and shared-expert gate/up projections, and the output head. MTP,
indexer, GDN control, HyperConnection, shared-expert down (K = 640, no qualified FP8 route) and routed
expert weights keep their formats, and activations stay BF16. The binder takes each matrix's stored
format; linear dispatches to the FP8 GEMV, K-split and A16 GEMM routes, the shared-expert decode
falls back from its fused BF16 kernel to two linears and SiLU-mul, and the proposal-head rerank
decodes FP8 head rows. The output basename gains `_fp8proj`.

With FP8 PLE as well, Swift 1.5 decode on an RTX PRO 6000 rises from 114.5 to 146.5 tok/s (MTP0,
512 context) and from 118.2 to 153.4 (8K); MTP3 with the optimized proposal head reaches 193.8 and
348.1 tok/s. Against the BF16 artifact, oracle generated-token |dlogp| is 0.006-0.089 on average,
perplexity changes -2.2% to +2.5%, and MTP3 acceptance is unchanged or higher.

Keep the entry file and all `.part-NNNN` companions together. The runtime selects PLE storage
from the artifact's table descriptor, not the checkpoint name.

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

Text alone selects 1,260 device objects for FP8 PLE or 1,259 for BF16 PLE because the latter has
no scale tensor. MTP adds 31, Vision adds 333, and the optimized proposal
adds two. Frontend resources are owned host bytes; PLE is a read-only mapped range whose lifetime
is owned by the loaded model. PLE mappings are not counted as GPU uploads or materializer staging.
The C++ loader checks logical shapes, formats, expert layout, activation permissions, selected
component targets and indexed-proposal geometry before building the Program.

Flash-Next retains its registered template renderer and does not support `--chat-template`
overrides. Text, Vision, MTP, prefix reuse and concurrency still use the public Engine route.
