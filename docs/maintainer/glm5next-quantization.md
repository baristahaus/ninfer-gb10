# GLM5Next quantization and numerical representation

This document maintains the source weight representation, the selected quantization scope, the exact
encoding, the compute precision and the observable rounding boundaries of GLM-5.3-Flash over the long
term. [Model semantics](glm5next-model.md) maintains the complete mathematics and the logical weight
roles. These are fixed target policies. They do not mean that the converter, the Ops or the model
integration are complete. The [design index](../designs/glm5next/README.md) holds the implementation
status.

[Number formats](tensor-formats.md) and [storage layouts](storage-layouts.md) maintain the general
encoding and the byte layout respectively. An existing format is referenced directly. This document
adds the GLM format selection, the source import and the explicit encoding rules that still have to
be implemented. The physical parent, the page, the record and the workspace are not defined again
here.

## 1. Weight sources and quantization scope

The source path on both Spark hosts is:

```text
/home/neroued/models/GLM-5.3-Flash/nvidia-nvfp4
```

This checkpoint is the only weight starting point. It adds quantization in exactly two places, the
large KDA projections and the MTP routed experts:

| Part | Target persistent representation | Compute permission or kept content |
|---|---|---|
| Main routed experts | the original NVIDIA NVFP4 | `AllowA4`; keep the meaning of the codes, the G16 E4M3 scales and the original calibration parameters |
| Main first three dense MLP layers | the original NVIDIA NVFP4 | `AllowA4`, input/weight scales independent of the routed experts |
| Q/K/V/O of the 34 KDA layers | `q8_g32_fp16`, converted from the source BF16 | `A16Only`; quantize these four large projections only |
| KDA beta, the two low-rank gate projections, output norm | the original BF16 | projection `A16Only`, no extra quantization |
| KDA convolution, A_log, dt_bias | the original FP32 | keep the source words and the control meaning |
| Main/MTP MLA and Indexer weights | the original BF16 | the projection stays A16; the Indexer head coefficients, the dot-product accumulation and the score reduction stay FP32 |
| Main/MTP shared experts | the original BF16 | `A16Only` |
| Main/MTP router | the original BF16 weight, FP32 correction bias | FP32 logits and the original route mathematics |
| mHC, Text/MTP norms | the original BF16 | section 5 holds the FP32 precision of the mHC coefficients, the norm and the mixing, and the BF16 residual boundary; the other norms follow the model mathematics and section 4 |
| MTP routed experts | `fp8_e4m3fn_block128x128_bf16`, converted from the source BF16 | `AllowA8` |
| MTP stem, embedding, LM head, all Vision | the original BF16 | keep the source weight, projection `A16Only` |

Section 3 covers the cache and the compute policy and the exact codec. It is separate from the static
weight quantization of the table. Main MLA Q_b/O and the shared experts take no extra quantization
scope.

## 2. Static weight encoding and import

### 2.1 NVIDIA NVFP4 import

The source packs two E2M1 codes along K into one U8. Each K16 group has one E4M3 scale, each matrix
has one FP32 `weight_scale_2`, and each input has one FP32 `input_scale`. Both scalars are
multipliers:

```text
w_source = E2M1(code) * E4M3(scale) * weight_scale_2
d_w = RN_FP32(1 / weight_scale_2)
d_a = RN_FP32(1 / input_scale)
w_ninfer = E2M1(code) * E4M3(scale) / d_w
```

When the NInfer divisor format is used, the codes and scales can be rearranged without changing the
value. The FP32 reciprocal itself has rounding, so the transform is not bitwise identity. `d_a` is
activation metadata of the use site and must not be interpreted together with the weight divisor. The
K16 quantization of the A4 input and the activation exit of the clamped SwiGLU follow the matching Op
contract.

All 42×288 Main experts of this source checkpoint were checked. The gate/up `weight_scale_2` is the
same within one expert. The gate/up `input_scale` is the same for all experts of a layer, and the down
`input_scale` is likewise the same for all of them. The two scalars of the gate/up in the first three
dense MLP layers agree as well. The GU can therefore merge along the output rows without changing
these calibration values.

### 2.2 KDA Q8 and MTP block-FP8

KDA Q/K/V/O use one FP16 scale for each 32 values along K, and the conversion uses the existing
`grouped_absmax` numerical rule. Each logical weight takes 1 B of code plus `2/32` B of scale. The
input activation stays A16. The quantized weight enters the agreed KDA formula and does not change the
type of the convolution, the gate or the recurrent state.

The GU/down of MTP uses 128×128 block E4M3FN codes and a BF16 multiplier:

```text
w_hat[e,n,k] = FP32(E4M3FN(code[e,n,k]) * BF16(scale[e,n/128,k/128]))
```

The gate/up/down of each expert is quantized separately, and a scale does not cross an expert or a
matrix boundary. When the GU merges along N, the 2048-row boundary aligns with the 128-row blocks.
The weight has no static activation divisor. The allowed A8 path quantizes the activation dynamically
in K128 groups per token and uses the matching activation scales.

The conversion method from BF16 to block-FP8 is `fp8_block_maxabs`. It uses the numerical rule of the
[existing row FP8 encoder](../../tools/convert/quantization/fp8_row.py) and changes the max-abs
reduction scope to each independent 128×128 block. The source BF16 words expand exactly to FP32. An
all-zero block writes a positive-zero BF16 scale and all-positive-zero E4M3FN codes. A non-zero block
is encoded in this order:

```text
amax       = max(abs(block_fp32))
raw_scale  = RN_FP32(amax / 448)
scale_word = RN_BF16(raw_scale)
if scale_word == 0x0000:
    scale_word = 0x0001                    # smallest positive BF16 subnormal
scale32    = exact_bf16_to_fp32(scale_word)
inv_scale  = RN_FP32(1 / scale32)
normalized = RN_FP32(weight_fp32 * inv_scale)
code       = RN_E4M3FN(clamp(normalized, -448, 448))
```

Every `RN` is round-to-nearest, ties-to-even. The codes are encoded with the FP32 reciprocal of the
**actually stored BF16 scale**. Keep the rounding order of computing the reciprocal first and then
multiplying the weight. A zero value inside a non-zero block keeps its sign. The conversion reports an
error when the source contains NaN or Inf, when a non-zero block has a non-positive or non-finite
scale, or when the normalized result is non-finite. Neither the streaming read batch nor a later GU
merge changes the reduction domain of the block or the encoding result.

This method uses the source weight only and needs no activation calibration data. The runtime dynamic
A8 quantization keeps the K128 convention above. The meaning of the encoded representation and of the
permission does not change with the implementation progress of the converter.

## 3. MLA/Indexer cache and compute policy

| Part | Persistent representation | Compute |
|---|---|---|
| MLA latent | BF16 or E4M3FN/G128/FP32 scales; the original basis | BF16 operands, FP32 accumulation; the softmax exponent, the statistics and the blocked reduction in FP32 |
| Indexer pooled keys | fixed BF16, no scale and no Hadamard | BF16 Q/K, FP32 accumulation; the head coefficients and the score reduction in FP32 |
| Indexer raw tail/records | BF16 K/gate, without APE | pooling softmax and weighted sum in FP32, the pool exit in BF16 |

Main/MTP and both ranks use the same MLA storage profile at startup and do not switch during a run.
The 16-bit operand of this profile is BF16. It introduces no FP16, no FP8 query and no FP8 dot-product
branch. FP8 changes the latent storage only. The format choice grants no permission for lower-precision
computation.

The Indexer keeps a BF16 cache and BF16 computation. Only when a later measurement shows a bottleneck
and the user explicitly decides again may an FP8 cache with BF16 computation be considered. That
condition also does not automatically authorize an FP8 query or an FP8 dot product. An execution route
must not change these precision policies by T/B, by memory pressure or by a device fact.

### 3.1 The two storage profiles

- `BFloat16`: one BF16 `c[512]` per token, 1024 B, no scale.
- `Fp8E4m3fnGroup128F32`: 512 E4M3FN codes and 4 FP32 scales, 528 B/token.

Both profiles sit in the original latent basis with no Hadamard and no other rotation. There is only
one latent, shared by the K and V roles. Do not store the expanded K/V separately. Do not apply the
BF16-K/FP16-V interpretation of the existing GQA or the D256 cache codec.

FP8 splits each token into four groups of 128 consecutive values. Four tokens are the loading unit,
not the quantization group. A scale is not shared across tokens, and an old latent is not re-encoded
when a new token arrives.

### 3.2 Encoding

The input is the FP32 x that the latent RMSNorm actually produces. The BF16 profile writes `RN_BF16(x)`
directly. The FP8 profile encodes the FP32 norm result directly and adds no intermediate BF16
rounding.

For each 128-wide group:

```text
a = max_i(abs(x[i]))

if a == 0:
    scale   = FP32(+0)
    code[i] = E4M3FN(+0)
else:
    scale   = max(RN_FP32(a / 448), 2^-126)
    inv     = RN_FP32(1 / scale)
    z[i]    = RN_FP32(x[i] * inv)
    code[i] = E4M3FN_RNE(clamp(z[i], -448, 448))
```

Every RN is round-to-nearest, ties-to-even. The scale is the actually stored FP32 multiplier and is
not rounded to a power of two. An all-zero group normalizes to positive zero. A signed zero in a
non-zero group is kept as the encoding above says. Run the FP32 steps as listed. Do not replace them
with another division or scale rounding rule that changes the code words. Do not change the required
behaviour at the smallest values through flush-to-zero.

The valid input, the intermediate values that are needed and the computed result must be finite.
NaN and Inf are not in the valid compute domain and must not be hidden by satfinite. The caller
guarantees the validity of the device content. Do not add a Host readback scan.

### 3.3 Consumption

```text
decoded[i] = RN_FP32(FP32(E4M3FN(code[i])) * scale[group(i)])
operand[i] = RN_BF16(decoded[i])
```

The decode runs on chip, and the QK and the latent weighted accumulation use the same BF16 operand.
There is no globally expanded BF16 cache. A newly written token must also pass the encode/decode
boundary of the selected profile. Do not bypass the FP8 precision with the unencoded latent of the
current window. Fusion can remove a memory round trip. It cannot remove this representation boundary.

The exact rule of the storage codec does not require different legal implementations of the complete
projection/norm to agree bit for bit. The exact codec acceptance applies to a given FP32 input. The
complete prepare is verified separately against an independent mathematical oracle.

### 3.4 Differences from the upstream Indexer representation

The local vLLM reference path applies a normalized Hadamard-128 to the query and to the pooled key,
then encodes with E4M3FN per vector and uses an FP32 scale rounded to a power of two. NInfer does not
adopt this Indexer representation. The direct Transformers reference path has no such codec either.
Explain a backend intermediate rounding and the NInfer policy separately. Do not treat the output of
one backend as an independent mathematical oracle. This difference is independent of the NVIDIA static
weight quantization.

## 4. Projection, control values and the latent attention boundary

A norm uses the epsilon and the direct gamma that the model specifies. The Text/Vision RMSNorm is
`1e-5`, the KDA Q/K L2 norm and the Indexer LayerNorm are `1e-6`, and the reduction and norm
arithmetic is FP32. Do not use the offset-gamma of Qwen4Exp. Every source FP32 convolution, control
and bias keeps its words.

### 4.1 KDA and FFN

The KDA input keeps the BF16 raw forget/beta logits, and the inside generates the actual
log-decay/beta from the FP32 A_log/dt_bias. Do not turn the activated FP32 gate into an external
projection input. The recurrent state that runs and that is published is FP32, and the convolution
history is BF16. The exact record and replay of the actual gate words of the round belong to the
record/fold contract and do not change the compute inputs above.

The Router uses FP32 logits, sigmoid scores, route coefficients and the original correction bias
semantics. Section 1 gives the activation permission of NVFP4, Q8 and block-FP8. Quantization must not
move the mathematical position of the SwiGLU clamp, of the route normalization or of the shared
contribution. The source weight import and the GU merge must keep the quantization domain and the
calibration meaning of gate and up separately, and must not re-select a scale across experts. The
implementation design defines the concrete parent and the local TP/EP geometry.

### 4.2 MLA/Indexer input preparation

The preparation projection of the same input uses BF16 operands and FP32 accumulation. The
representation boundary of its five results is:

| Logical result | Numerical boundary |
|---|---|
| Q_a | outputs the BF16 q_r after the FP32 RMSNorm |
| KV_a | enters the section 3 encoder directly after the FP32 RMSNorm, without a BF16 conversion first |
| Index K | outputs the BF16 raw K after the FP32 affine LayerNorm |
| Pool gate | the projection result becomes the BF16 raw gate; APE is not added yet |
| Index head coefficients | kept in FP32 with no BF16 rounding; the public raw coefficient excludes `1/sqrt(32)` |

The RMSNorm is `x*gamma/sqrt(mean(x*x)+1e-5)` and the LayerNorm is
`(x-mean(x))*gamma/sqrt(mean((x-mean(x))^2)+1e-6)+bias`. The epsilon is interpreted as an FP32
constant. The preparation projection has no extra BF16 landing point before the norm. The pooling, the
tail and the records consume the same actual BF16 raw K/gate. Even when all four tokens are inside the
current call, do not bypass this boundary. APE is added in FP32 during pooling, each channel runs a
softmax and a weighted sum along the four tokens, and the pool exit becomes BF16 with no extra norm,
rotation or quantization.

### 4.3 Attention in the latent domain

This model keeps the BF16 U_K/U_V factors and does not merge them offline with Q_b/O into a larger
absorption matrix. QK, the weighted accumulation and the factor projection keep the BF16/F32 compute
constraints. The softmax max, sum and exp and the blocked merge are FP32. q_latent/u are private
intermediates. They are not forced into global materialization, and no extra BF16 landing point is
added to reproduce a split call. Keeping an FP32 intermediate to fuse the computation is allowed. Do
not lower a matrix operand to FP8. The public output is the BF16 heads after the complete U_V mapping.
The complete mathematics is in section 4.1 of the model document.

cache_operand is produced by section 3.3, and a BF16 store uses the stored value directly. The current
token of FP8 also passes the actual encoder/decoder and does not take a bypass through an unencoded
temporary latent. The attention scale is fixed at 1/16 and does not become `1/sqrt(512)` because the
latent is 512 wide. The O projection consumes the BF16 heads. The consumer interface defines the write
domain and the exact-zero rule of an empty selection set.

## 5. mHC numerical boundaries

### 5.1 The current fixed profile

| Item | Convention |
|---|---|
| `D / S` | 4096 / 4 |
| `eps_coeff / eps_branch / eps_final` | `1e-5` |
| `eps_pre / eps_sinkhorn` | `1e-6` |
| `sinkhorn_iters` | 20 |
| post multiplier | 2 |
| `X / X_next / embedding / u / y / hidden` | BF16 |
| source and resident `F / base / scale / gamma / gamma_final` | BF16, keeping the source words |
| `F*X`, the sum of squares, collapse and residual mixing | FP32 accumulation and computation |
| logits, sigmoid, softmax, Sinkhorn, norm | FP32 |
| `q / C` kept across branches | FP32 |

The geometry and the epsilon in the configuration belong to the model semantics. The table is the
selected profile, and these parameters are not tuning knobs. Different token counts and private
execution routes keep the same precision policy and add no FP8 residual, no low-bit mHC weight and no
low-precision dynamic coefficient branch.

The original `F` and `X` are BF16, so `F*X` can use BF16 products and FP32 accumulation, and `r` is
then applied in FP32. That does not require the whole `F` to be resident and expanded to FP32. A small
BF16 parameter is read and then takes part in the arithmetic in FP32. A BF16 representation in the
checkpoint and FP32 computation at execution are different concepts.

### 5.2 BF16 residual boundaries that must stay

The BF16 residual of each post is the actual input of the next site. A fused `post_pre` must first
form these BF16 values, and then compute the projection, the full-width RMS statistic and the collapse
of the next site.

```text
X_next_math → RN_BF16 → X_next
                         ├→ next F projection / full-width RMS
                         └→ next pre collapse / branch RMSNorm
```

The conversion here can happen in registers or in shared memory and needs no extra device-memory
round trip. It must not write the BF16 `X_next` outward while using the unrounded `X_next_math` to
generate the next group of coefficients. `RN_BF16` is round-to-nearest, ties-to-even. This contract
does not fix the reduction association order or the execution decomposition.

### 5.3 No added internal low-precision rounding

The following values are not independent BF16 exits: the full-width normalized vector, the 24-row
logits, `p`, the collapse `v`, a separate `C^T X` or `q*y`, and the final four-stream mean. They keep
computing at the FP32 precision of this profile.

In particular:

- Do not convert to BF16 between `collapse` and `branch RMSNorm`.
- Do not convert to BF16 between the final `mean` and `final RMSNorm`.
- The BF16 residual is produced only after `C^T X` and `q*y` are added. Do not write each to BF16 and
  then add.
- The public `q/C` are FP32 and are not lowered to BF16 before the mixing.

This contract does not require bitwise equality with any reference backend. In the checked vLLM
revision `00b7847c8036b667742b4efb21aab1de51fd4721`, the
[independent pre](https://github.com/vllm-project/vllm/blob/00b7847c8036b667742b4efb21aab1de51fd4721/vllm/model_executor/kernels/mhc/torch.py)
reads the BF16 residual, while one
[mhc_fused_tilelang path](https://github.com/vllm-project/vllm/blob/00b7847c8036b667742b4efb21aab1de51fd4721/vllm/model_executor/kernels/mhc/tilelang_kernels.py)
uses the unrounded `new_r` to generate the next projection and the sum of squares while it writes the
BF16 residual. Its pre/norm fusion path also has a BF16 staging point. These implementation details
must not become the basis for different NInfer routes to use different mathematical inputs.

The reference implementation therefore serves to check the model mathematics and the parameter
mapping. NInfer uses the representation boundaries of this section and an independent mathematical
oracle for acceptance. The fusion tests must cover both keeping the representation boundary and
removing the extra internal rounding.

The initial four-stream residual copies the BF16 words of the embedding exactly, including a signed
zero. The final post keeps the BF16 residual boundary in the same way. The mean and the final norm
then complete in FP32 and convert to BF16 only at the final hidden exit. The q/C across branches are
the actual FP32 exit, and the later mixing consumes them directly.

## 6. Weight capacity

The source capacity was counted tensor by tensor from the headers of the 33 safetensors files of the
checked checkpoint, 147661 tensors in total. It agrees with `total_size=204419110596` B in the source
index. The model and quantization configurations of the two source hosts are the same. The table below
is the tensor payload of the whole model in GiB. It excludes the file headers, the object alignment,
the runtime, the shard replication and any extra model weight. The target column is an estimate
computed from the selected formats. It is not the actual artifact size yet.

| Part | NVIDIA source | Estimate after the selected representation |
|---|---:|---:|
| Main routed experts | 159.469020 | 159.469020 |
| Main dense MLP | 0.237305 | 0.237305 |
| Main KDA | 8.729325 | 4.744950 |
| Main MLA + Indexer | 2.559384 | 2.559384 |
| Main shared experts | 1.968750 | 1.968750 |
| Main router | 0.092330 | 0.092330 |
| Main mHC + norm | 0.066617 | 0.066617 |
| Embedding + LM head | 2.363281 | 2.363281 |
| MTP routed experts | 13.500000 | 6.750824 |
| MTP other exclusive weights | 0.344283 | 0.344283 |
| Vision | 1.049837 | 1.049837 |
| **Total** | **190.380132** | **179.646581** |

The original payload of the KDA Q/K/V/O is exactly 8.5 GiB, and 4.515625 GiB after Q8-G32, which is a
reduction of 3.984375 GiB. The MTP experts hold 7247757312 weights in total, and the block-FP8 codes
plus BF16 scales take 7248642048 B, which is a reduction of 6.749176 GiB. The two together reduce the
total by about **10.733551 GiB**.

When MTP is not enabled, deduct its experts and its other exclusive weights. Do not deduct the
embedding/head twice. When Vision is not enabled, deduct the independent Vision weights. The weight
capacity of each rank on the two hosts also depends on the actual shard and replication plan. Do not
divide the total by two as the final resident budget.

The wording about quantized shared experts in the NVIDIA model card does not match the actual
configuration, and the "MTP not exported" note in the linked recipe also does not match this local
checkpoint. This document uses the actual tensors: shared is BF16, the complete MTP weight exists, and
the original experts are BF16.

## 7. Sources and general contracts

The source checkpoint, the configuration and the model implementation entries are in
[model sources](glm5next-model.md#8-sources). The NVFP4 source representation also refers to the
[NVIDIA recipe](https://github.com/NVIDIA/Model-Optimizer/blob/4956213d670c382385d9bc43e17379b9fc064e50/modelopt_recipes/models/zai-org/GLM-5.3-Flash/ptq/nvfp4_experts_dense_mlp-kv_fp8_cast.yaml).
The tensors that actually exist and the capacity follow the check of the source files.

The grouped_absmax of Q8, the NVFP4 divisor and the existing block-FP8 represented values follow the
[general number formats](tensor-formats.md). When the actual format already has a definition, do not
add a GLM-specific interpretation. After code or the general format takes over the encoding rules that
still have to be implemented, reference them directly. The model format selection stays for the long
term. This reference document does not claim the model load, the conversion command or the artifact
state. See the design index and the actual converter.
