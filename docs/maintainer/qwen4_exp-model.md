# Qwen4Exp model and weights

This document maintains the Qwen4Exp model structure, the Flash Next instance, the quantization recipe and the persistent weight layout.
Current execution capability, resource ownership and state recovery appear in [Execution and state](qwen4_exp-execution.md).
Conversion commands appear in [Weight conversion](../weight-conversion.md). The generic numeric representations, byte layouts and file contracts are maintained by
[Numeric formats](tensor-formats.md), [Storage layouts](storage-layouts.md) and
[V3 container](artifact-container.md) respectively.

## 1. Architecture and instance

Qwen3.8-Flash-Next uses `Qwen4ExpForCausalLM / qwen4_exp_text`.
The Converter accepts the source `Qwen4ExpForConditionalGeneration` or `Qwen4ExpForCausalLM`
and uniformly outputs this architecture. Vision and native MTP are optional components whose target is `text`.
Qwen3.8-27B uses a different Qwen3.5 architecture, and the two do not share a Text layer implementation just because their names are similar.

The table below holds the instance values of the Flash Next checkpoint. Model execution reads the layer order and the geometry
from the config. Binding describes the actual weights, Parameters/Ops decide the executable representation and the shape, and a recipe name takes no part in execution registration.

| Parameter | Flash Next |
|---|---|
| Text hidden / vocabulary | 2560 / 248320; embedding and head do not tie weights |
| Main | 48 layers; the zero-based layer index `l%4==3` is QSA, 12 layers in total, and the other 36 layers are GDN |
| HC streams / lowrank | 4 / 320; residual width 10240 |
| GDN heads / dimension | Q/K 16 heads, V 48 heads, dimension 128 for both; each Q/K group serves three consecutive V heads |
| GDN QKV / Z / convolution | 10240 / 6144; 4 taps, dilation=1 |
| QSA Q / KV | 24 / 2 heads, head dimension=256 |
| QSA indexer | 4 query heads, 1 key head, dimension=128 |
| QSA compression / selection | four tokens per block; at most 512 complete blocks, plus at most three visible tokens of the incomplete block |
| Routed experts | 512 per layer, top-10, intermediate=640 |
| Shared expert | one SwiGLU per layer, intermediate=640, independent scalar sigmoid gate |
| PLE position | before the attention HC of zero-based layer 1; the source `ple_layer_ids=[2]` normalizes to `[1]` |
| PLE heads / embedding | 2-gram and 3-gram, 8 heads each; 16 rows x 160 dimensions, concatenated into 2560 dimensions |
| PLE table | 128 `[2500012,160]` shards, 320001536 rows in total |
| PLE convolution | 10240 channels, 4 taps, dilation=3, nine columns of history |
| MTP | one independent HC/QSA/MoE layer with no PLE; it shares the Main embedding, shares the Main head by default, and can use an independent quantized draft head |
| Vision blocks | 27 layers, hidden=1152, 16 heads x 72, MLP intermediate=4304 |
| Vision patch / merger | patch `[3,2,16,16]`; 2x2 merge, 4608→4608→2560 |
| RoPE | theta=10000000; Main and indexer both rotate the first 64 dimensions; interleaved MRoPE sections `[11,11,10]` |
| Context / norm epsilon / EOS | 262144 / `1e-6` / 248044 |

The Adapter normalizes the source `full_attention` to `indexed_attention` and writes math configurations such as
`norm_topk_prob=true` into the artifact. The field and source-parameter mapping is maintained by
[the adapter](../../tools/convert/qwen4_exp.py), and [config](../../src/models/qwen4_exp/config.cpp) explains the native configuration.

## 2. Model computation and numeric boundaries

### 2.1 Main and HC

```text
token embedding, replaced by Vision features at the visual positions
  → copied into four HC residual streams
  → each layer: attention HC → GDN/QSA → HC combine
                 MoE HC       → MoE     → HC combine
                 (the zero-based layer 1 entry adds PLE first)
  → final HC norm/mixer → 2560-dimension sample hidden → LM head
```

The four streams are the feature axis, and GDN/QSA/MoE still consume one 2560-dimension block input.
HC performs an offset RMSNorm independently on each stream. The low-rank down/up produces a per-feature sigmoid mixing, and
the block injection uses four `2*sigmoid(logit/4)` coefficients. Combine adds the same block output back to each stream by
these coefficients and forms the public BF16 residual. The next norm consumes these actual BF16 words.
The final mixer produces no injection, and no additional final RMSNorm follows it.

The [HC contract](../../include/ninfer/ops/gated_residual.h) maintains the complete formula.
The model keeps two end features: the 2560-dimension output of the final mixer, and the
10240-dimension residual before its norm/mixer. MTP conditioning uses the latter and cannot derive it back from the former.

### 2.2 GDN, QSA and MoE

GDN projects the HC input into QKVZ and AB. Q/K/V pass through a causal convolution and SiLU, then through Q/K L2
normalization and the gated delta-rule. Z bypasses the convolution. The recurrent state is
FP32 `S[value_dim,key_dim]`. The output uses a head-wise RMSNorm with a direct gamma,
multiplies by `sigmoid(Z)` and then projects back to 2560 dimensions. The convolution SiLU and the output sigmoid are two different nonlinearities.
The computation contract appears in [GDN](../../include/ninfer/ops/gated_delta_net.h),
[controls](../../include/ninfer/ops/gdn_control.h) and
[gated RMSNorm](../../include/ninfer/ops/rmsnorm.h).

QSA keeps the complete main K/V history, and the 4:1 compression applies only to the index key. Each complete block first averages four columns of raw index-K
and rounds them to BF16, then applies an offset norm and the block-start-coordinate RoPE. The sum of the non-negative dots of the four index heads
decides the shared TopK. All 24 query heads consume the same selected block set. Sparse GQA also reads the visible tokens of the current
incomplete block, and the output multiplies by an elementwise sigmoid gate before entering the O projection.
The index score does not join the main attention logits. The complete formula and the KV codec boundary appear in
[the QSA contract](../../include/ninfer/ops/qsa.h).

MoE uses normalized top-k softmax routing, and shared does not join top-k:

```text
expert_e(x) = W_down_e * (SiLU(W_gate_e*x) * (W_up_e*x))
routed(x)   = sum_selected_e(prob_e * expert_e(x))
shared(x)   = sigmoid(W_shared_score*x) * shared_ffn(x)
moe(x)      = routed(x) + shared(x)
```

The Shared FFN also uses SwiGLU. MoE outputs one 2560-dimension block, and HC injects it back into the residual.
The private staging of the projections, the nonlinearities, the route coefficients and the per-expert reduction adds no public BF16
rounding boundary. The [MoE contract](../../include/ninfer/ops/moe.h) maintains the semantic entry and the activation policy.

### 2.3 MTP and Vision

The initial features of MTP are:

```text
e = W_fc_embedding * OffsetNorm_2560(token_embedding)
v = OffsetNorm_10240(multistream_hidden).reshape(4,2560)
h_init[s] = W_fc_hidden * v[s] + e
```

The hidden norm reduces along the complete 10240 dimensions, and then the same fc_hidden acts on the four streams independently.
The first step uses the Main conditioning, and later draft steps use the pre-final-mixer
residual of the previous MTP step. The QSA/MoE layer and the final HC mixer of MTP produce the next set of conditioning and sample.
Draft uses the Main head by default and can also use the independent quantized head of section 3.3. The target still uses the Main head.

Vision flattens the Conv3d patch into 1536 dimensions in C/T/H/W order. The blocks use non-causal attention,
LayerNorm and a `gelu_pytorch_tanh` MLP. The merger performs a LayerNorm before the 2x2 merge and
uses GELU between the two FCs. This instance has no deepstack. The encoder reuses
[the common Qwen Vision](../../src/models/common/qwen/vision/), and the Text layer consumes only its 2560-dimension output.

## 3. Flash Next quantization recipe

`qwen3_8_flash_next` uses the BF16 main checkpoint to provide the config, the resources, the Main non-routed
weights, Vision and PLE. `--source nvidia=PATH` provides the complete
`nvidia/Qwen3.8-Flash-Next-NVFP4` checkpoint and imports the Main routed experts and the MTP-only parameters.
The source and the conversion method are written into `.conversion.json`. The complete selection is maintained by
[the official recipe](../../tools/convert/official_recipes.py).

| Part | Persistent representation | Compute permission or consumption |
|---|---|---|
| Main routed gate/up/down | NVIDIA NVFP4, G16 E4M3FN scale, one FP32 divisor per expert matrix | `AllowA4` |
| Main GDN QKVZ/output, QSA Q/gate/output | BF16 source converted to Q8 G32/FP16 scale | `A16Only` |
| HC, QSA K/V/indexer, PLE projections, router/shared FFN | original BF16 words | `A16Only` |
| GDN AB, convolution, norm, other norm/bias | original BF16 words | the mathematics of each Op |
| GDN A_log / dt_bias | the source BF16 exactly expanded to FP32, `[48]` each | keeps the original meaning |
| Token embedding / Main LM head | two independent BF16 objects | lookup / A16 head; MTP shares by default |
| Optimized MTP head, optional | BF16 Main head source converted to Q4 G64/FP16 scale, complete vocabulary | `A16Only` |
| PLE table | BF16 source converted to `nf4_g16_e4m3fn` | exactly decoded into the BF16 lookup output |
| MTP routed experts | NVIDIA E4M3FN codes, 128x128 block BF16 multiplier | `AllowA8` |
| MTP other unique parameters, all of Vision | original BF16 words | projections `A16Only` |
| PLE hash controls | the original 35 INT64 | lossless Resource |

The GPU MoE already has NVFP4 A4, block-FP8 A8/G128 and small-batch A16 paths, and
the private shape policy of each semantic entry selects the actual computation. The CPU Main experts use BF16 input and
FP32 intermediate computation and do not quantize the activations. A permission allows a higher-precision path and does not require every call to quantize to the lowest width.

Q8 uses `grouped_absmax`, grouping every 32 values along K. The merge and the row permutation determine the final row order first,
and then the encoding happens. Packing, scale swizzle and convolution transpose happen at conversion time, and the loader binds the original bytes.
The dtype of the KV, the recurrent state and the workspace is independent of the weight recipe.

### 3.1 Norm parameters

| Role | Persistent parameter | Reduction and affine |
|---|---|---|
| HC, PLE grouped RMSNorm | BF16 offset `[10240]`, native `[2560,4]` | each stream reduces independently, gain=`1+w` |
| Main/MTP QSA Q/K, indexer Q/K | BF16 offset `[256]` / `[128]` | per-head reduction; heads of the same role share the parameter |
| MTP pre-FC embedding / hidden | BF16 offset `[2560]` / `[10240]` | reduces over the whole corresponding vector |
| GDN output gated RMSNorm | BF16 gamma `[128]` | direct gain, shared by all V heads |
| Vision LayerNorm | BF16 weight/bias `[1152]` | direct affine; the merger norm sits before the merge |

The offset keeps the source words, and `1+w` is not pre-rounded into a BF16 gain. The HC `/4` is not folded into the weights.
The parameter dtype and the reduction precision are different conventions.

### 3.2 NVIDIA encoded import

The `weight_scale_2` and `input_scale` of the Main source are positive finite FP32 multipliers.
The [ModelOpt adapter](../../tools/convert/sources/modelopt_nvfp4.py) uses:

```text
source weight = E2M1(code) * E4M3FN(scale) * weight_scale_2
d_w = RN_FP32(1 / weight_scale_2)
d_a = RN_FP32(1 / input_scale)
stored weight = E2M1(code) * E4M3FN(scale) / d_w
```

The codes/scales words are kept and then rearranged into the target layout. The FP32 reciprocal rounding is a metadata transformation and
does not promise that every decoded value is bit-identical to the source. The recipe requires the two multiplier
words of gate/up in the same expert to be equal. It also requires the GU input multiplier of all experts in one layer to be equal and the down input multiplier
to be equal. Each expert still holds its own weight divisor.

The activation divisors are stored as FP32 `[48,2]`, and the elements `2*l` and `2*l+1` are referenced by the GU/down Use
respectively. They are interpreted separately from the weight divisor. The A4 input is quantized by G16, and the product is encoded directly from FP32,
adding no BF16 intermediate seam.

MTP `fp8_e4m3fn_block128x128_bf16` rebuilds as E4M3FN code x BF16 block scale.
Gate/up each keep the original scale, and the 640-row boundary aligns with the 128-row block. A8 dynamically quantizes by K128
groups per token, and the scale is FP32. Each K128 dot is scaled separately and then accumulated, with no static activation divisor.

### 3.3 Optimized MTP head

At conversion time `--proposal` generates an independent `proposal/head` from the Main BF16 head source, using
`grouped_absmax`, `q4_g64_fp16` and `row_split_k128_v1`. It keeps all
`[248320,2560]` physical rows and the original vocabulary ID order. It does not trim the vocabulary and does not create `proposal/token_ids`.
The directory declares `text.proposal.domain="full"`. The policy of the `mtp/final_hidden` Use is `A16Only`.

The shared binding of the Main BF16 head and `mtp/output_head` stays. When startup selects optimized,
MTP binds `proposal/head` instead, and the target acceptance and the ordinary decode continue to use the Main head.
Both heads use the same vocabulary TP partition: under TP2 each rank holds the local `[124160,2560]` part,
following the global token IDs and the candidate merging, with no token remap needed.

This is an additional draft representation, and disabling the optimized head does not load its objects.
Startup options and the execution flow appear in [Execution and state](qwen4_exp-execution.md#5-mtp-generation-and-adoption).

## 4. Merged matrices and persistent layout

A logical matrix is written `W[N,K]`, where N is the output row and K is the input feature, and both ranges are half-open.
Projections with the same input, representation and compatible metadata merge along N, and each Binding still references the logical role.
Native Tensor dimension 0 is contiguous; `M=T*B`, `column=t+T*b`.

| Parent | Main / MTP count | Logical shape | Format | Output row order |
|---|---:|---|---|---|
| GDN QKVZ | 36 / 0 | `[16384,2560]` | Q8 | Q 2048, K 2048, V 6144, Z 6144 |
| GDN AB | 36 / 0 | `[96,2560]` | BF16 | a 48, b 48 |
| GDN output | 36 / 0 | `[2560,6144]` | Q8 | original row order |
| Main QSA Q/gate | 12 / 0 | `[12288,2560]` | Q8 | Q 6144, gate 6144 |
| Main QSA KV/index | 12 / 0 | `[1664,2560]` | BF16 | K 512, V 512, index-Q 512, index-K 128 |
| Main QSA output | 12 / 0 | `[2560,6144]` | Q8 | original row order |
| MTP QSA input | 0 / 1 | `[13952,2560]` | BF16 | K 512, V 512, index-K 128, Q 6144, gate 6144, index-Q 512 |
| MTP QSA output | 0 / 1 | `[2560,6144]` | BF16 | original row order |
| Block HC down/inject | 96 / 2 | `[324,10240]` | BF16 | down 320, injection 4 |
| Final HC down | 1 / 1 | `[320,10240]` | BF16 | all down |
| HC up | 97 / 3 | `[10240,320]` | BF16 | stream-major, `row=s*2560+d` |
| PLE KV | 1 / 0 | `[12800,2560]` | BF16 | four-stream key 10240, shared value 2560 |
| MoE input | 48 / 1 | `[1793,2560]` | BF16 | router 512, shared score 1, shared gate 640, shared up 640 |
| Shared down | 48 / 1 | `[2560,640]` | BF16 | original row order |
| Routed GU bank | 48 / 1 | `[512,1280,2560]` | Main NVFP4 / MTP block-FP8 | expert-major, gate 640 and up 640 inside each expert |
| Routed down bank | 48 / 1 | `[512,2560,640]` | Main NVFP4 / MTP block-FP8 | expert-major |
| MTP fc_embedding / fc_hidden | 0 / 1 each | `[2560,2560]` | BF16 | independent weights; they handle M / 4M columns respectively |
| Main LM head | one, shared by MTP by default | `[248320,2560]` | BF16 | vocabulary order |
| Optimized MTP head | 0 / optional 1 | `[248320,2560]` | Q4 | complete vocabulary order |

The two input parents of Main QSA have different precision, and the six MTP projection segments merge into one BF16 parent.
The K/V/index-K of MTP sit in the first 1152 rows and can directly form a contiguous `[1152,2560]` weight view.
The complete projection and the three-segment projection share the same weights. TP2 assembles the local parent in the same role order:
K 256, V 256, index-K 128, Q 3072, gate 3072, index-Q 512.
The complete matrix is `[7296,2560]`, and the contiguous prefix is `[640,2560]`.
The MoE control is the first 513 rows and the shared GU is the last 1280 rows, and both work as zero-copy views.
The persistent merge does not force a single projection call; LinearSplit can directly produce independent contiguous activation segments.

### 4.1 Row permutation and convolution

The source q_proj interleaves Q/gate by head, and the conversion de-interleaves it:

```text
Q[h*256+d,:]    = source_q_proj[h*512+d,:]
Gate[h*256+d,:] = source_q_proj[h*512+256+d,:]
```

The convolutions of GDN and PLE both convert to channel-contiguous:

```text
source [C,1,4] → artifact/Binding [4,C] → native Tensor [C,4]
stored[j*C+c] = source[c*4+j], C=10240
```

Both kinds of transformation keep the source BF16 words, and the convolution tap time order does not change. The computation interprets the PLE dilation=3.
Norm, bias and convolution are not requantized as Linear weights.

### 4.2 Vision and expert banks

| Vision parameter | Count | Persistent shape |
|---|---:|---|
| Patch projection | 1 | `[1152,1536]` |
| QKV / bias | 27 | `[3456,1152]` / `[3456]`, Q/K/V in that order |
| Attention output | 27 | `[1152,1152]` |
| FC1 / FC2 | 27 each | `[4304,1152]` / `[1152,4304]` |
| Merger FC1 / FC2 | 1 each | `[4608,4608]` / `[2560,4608]` |
| Position embedding | 1 | `[2304,1152]` |

The routed experts of each layer have two physical parents, GU and down. The logical gate/up are
`[512,640,2560]` each and down is `[512,2560,640]`. Binding references the two halves of GU
and the whole down matrix by expert ID. The Binding range counts logical elements in C-order and does not include the scale/divisor bytes.

Main uses `batched_block_scale_k16_m128x4_v1`, and the bank stores all codes,
the swizzled scales and the FP32 divisors `[512]` in that order. The GU/down payloads are
943720448 / 471861248 B respectively. The scale swizzle uses the local row index inside the expert.
MTP uses `block_scale_m128_k128_v1`, with BF16 scales
`[512,10,20]` / `[512,20,5]` after the codes, and the payloads are 1677926400 / 838963200 B respectively.

The recipe places the resources/auxiliary values, the Main fixed weights, Vision, MTP, the Main experts and the PLE shards.
The Main experts go by layer→GU/down, and PLE goes by shard ID. An ordinary object aligns to 256 B and PLE aligns to 4096 B.
An object can continue across V3 volumes, and the directory maps the actual file range, so the parameter identity cannot be derived from the file name.

## 5. PLE table, hash and lookup exit

The 128 NF4 parents of PLE keep the original shard boundaries and their own divisors.
`stored_scale_mse` performs a bounded two-pass conversion on each shard. The error computation, the scale search and the tie rule appear in
[the conversion numeric contract](tensor-formats.md#10-the-stored-scale-mse-conversion-method).

`row_paged_g16_4k_v1` uses 90 B per row: 80 B of packed codes and 10 B of scales.
Each 4 KiB data page holds 45 rows, and the last page holds 37 rows. Each shard payload is 227557380 B.
A global row g belongs to `shard=g/2500012` and `row=g%2500012`, and a hash head does not correspond to a fixed shard.
The exact padding, the divisor address and the encoding appear in [the layout contract](storage-layouts.md#11-row_paged_g16_4k_v1).

`text.resources.ple_hash_controls` is a 280 B `raw_bytes_v1` Resource, in this order:
three layer multipliers, sixteen head vocab sizes and sixteen head offsets,
all as source little-endian signed INT64 words. The control values are copied losslessly and the seed/prime is not regenerated.

The current position raw ID is x0, and the previous and the second previous are x1 and x2. A negative position pads EOS.
If x1=EOS, set x2=EOS for this query only. x0=EOS still consumes the earlier history:

```text
u2 = word64(x0*multiplier[0]) XOR word64(x1*multiplier[1])
u3 = u2 XOR word64(x2*multiplier[2])
g[h] = positive_mod(signed64(h<8 ? u2 : u3), head_vocab_size[h])
       + head_offset[h]
```

The multiplication keeps the modulo-2^64 words and then interprets them as signed64 to take the positive modulus. The output concatenates
the eight 2-gram heads and the eight 3-gram heads. The segmentation and the incremental append use the real raw-ID tail. Vision features do not replace the IDs,
and the EOS rule does not clear the PLE convolution history. The implementation appears in [hash](../../src/models/qwen4_exp/ple/hash.cpp).

The lookup exit is exactly:

```text
LUT[shard,scale_word,code] =
    BF16_RNE(codebook[code] * E4M3FN(scale_word) / divisor[shard])
```

It performs BF16 RNE directly on the exact represented value and cannot be replaced by an unproven FP32→BF16 double rounding.
The valid range of the scale words is `0x00..0x7e`, and a zero result keeps the sign of the codebook.
The immutable LUT of each shard takes 4096 B, 512 KiB in total. The row decode writes 160 BF16 words,
and sixteen rows form `E[2560,T,B]` in head order. The generic implementation appears in
[NF4 Host numeric helpers](../../src/core/nf4.h), and the execution document covers the asynchronous preparation and the lifetime.

## 6. Weight capacity and residency

Below are the physical weights and control values of this recipe. They do not include the frontend resources, the alignment between objects,
the directory or the file framing. They are not the total memory occupancy at startup.

| Part | Bytes | GiB | Current placement |
|---|---:|---:|---|
| Main non-routed-expert, non-PLE-table weights | 7418143232 | 6.908684 | GPU |
| Main routed experts | 67947921408 | 63.281433 | GPU resident, or a registered Host backing plus the RankProgram GPU cache |
| PLE NF4 table | 29127344640 | 27.126954 | V3 on-disk in-place backing |
| Vision | 897862112 | 0.836199 | GPU when enabled |
| MTP experts | 2516889600 | 2.344036 | GPU when enabled |
| MTP other unique weights | 181136896 | 0.168697 | GPU when enabled |
| Activation divisors and PLE hash controls | 664 | — | read-only control values |
| Total without the optimized head | **108089298552** | **100.666004** | — |
| Optimized MTP head, optional | 337715200 | 0.314522 | GPU when enabled; half per rank under TP2 |
| Total with the optimized head | **108427013752** | **100.980526** | — |

The Text-only weights and control values are about 97.317072 GiB. In offload mode the fixed GPU weights of all components are about
10.257616 GiB, plus the Main expert cache. In resident mode the GPU weights are about 73.539049 GiB.
Both modes charge the StateImage, the KV, the workspace, the Graph and the staging separately.
A disabled Vision/MTP does not materialize, and the shared embedding/head is not charged twice.
Enabling the optimized head adds about 322.07 MiB on top of the GPU weights above, and
about 161.04 MiB per rank under TP2.

The Host backing of offload mode includes the experts already cached to the GPU, and an eviction needs no write-back. Resident
mode binds the GPU bank directly and keeps no Host expert copy. PLE startup reads only the directory, the control values and the divisors, and does not load the whole table.
ModelDefinition holds the configuration, the representation description and the Host control values. ModelShard holds the local immutable weights and the PLE file backing.
Session manages the logical state and the context directory, and RankProgram holds the mutable caches and the physical execution resources.
The complete resource budget appears in [Execution and state](qwen4_exp-execution.md).

## 7. Mathematical sources

The source structure references the pinned versions of
[the Transformers model](https://github.com/huggingface/transformers/blob/8073e6dcaea9aa4b42acf2d92ec72148786bbaeb/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py),
[the configuration](https://github.com/huggingface/transformers/blob/8073e6dcaea9aa4b42acf2d92ec72148786bbaeb/src/transformers/models/qwen4_exp/configuration_qwen4_exp.py) and
[vLLM MTP](https://github.com/vllm-project/vllm/blob/6fddc9b21e763bfc947cc01d3ce815179aac1dc6/vllm/models/qwen4_exp/nvidia/mtp.py).
[The NVIDIA source note](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/blob/fc694b54fb0174e0913e6adf86691ef85a4ead47/README.md)
records the quantization source. The production Op contracts maintain the official numeric boundaries of NInfer. The upstream TP, the private casts and
the kernel packing do not automatically become this project's mathematical or persistent contract.
