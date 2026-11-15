# GLM5Next model semantics

This document maintains the architecture, the complete computation, the logical weight roles and the
historical dependencies of GLM-5.3-Flash over the long term.
[Quantization and numerical representation](glm5next-quantization.md) maintains the selected
representation, the exact encoding, the compute precision and the explicit rounding boundaries. The
mathematics in this document runs under those boundaries.

The model semantics and the selected representation do not represent a capability that exists today.
The [GLM5Next design index](../designs/glm5next/README.md) holds the implementation status and the
task entries in one place. This document does not define the two-host placement or the execution
interfaces again.

## 1. Architecture and instances

GLM-5.3-Flash uses `Glm5NextForConditionalGeneration / glm5_next`, which contains
`glm5_next_text` and `glm5_next_vision`. The table below is the actual configuration of the current
checkpoint. These values belong to the model instance. Later execution must read the config and must
not treat them as architecture constants.

| Parameter | GLM-5.3-Flash |
|---|---|
| Text hidden / vocabulary | 4096 / 154880; the embedding and the LM head do not share weights |
| Main layer order | 45 layers; the zero-based layer numbers `3,7,11,15,19,23,27,31,35,39,43` are sparse MLA and the other 34 layers are KDA |
| MLP layer order | zero-based 0–2 are dense MLP, 3–44 are MoE, 42 MoE layers in total |
| mHC | 4 residual streams, complete residual width 16384; one independent parameter group for attention and one for FFN in each layer |
| mHC mapping / Sinkhorn | `[24,16384]` mapping, 24 base, 3 scale; 20 column normalizations, complete mathematics in section 2 |
| KDA | 64 heads × 128; Q/K/V are 8192 wide each; four-tap depthwise convolution |
| KDA controls | the forget and output gates each have a 128-wide low-rank bottleneck; beta is one scalar per head; the forget log-decay lower bound is -5 |
| MLA | 64 heads, Q/K are 256 wide per head, V is 256 wide per head |
| MLA latent | Q bottleneck=1536, KV latent=512; `qk_rope_head_dim=0`, `mla_use_nope=true` |
| Indexer | 32 query heads × 128, one 128-wide index key; Key LayerNorm |
| Kpool4 / selection | per-channel weighted compression of four index keys; select at most 512 complete pools, expand to 2048 tokens, and add at most three tail tokens |
| Main indexer sharing | `indexer_types` is `full` everywhere, and each MLA layer produces its own selection |
| Dense MLP | intermediate=12288 |
| Routed experts | 288 in each MoE layer, top-8, intermediate=2048 |
| Shared expert | one in each MoE layer, intermediate=2048, added directly to the routed output |
| Router | FP32 logits, sigmoid scoring, `noaux_tc` correction bias, the selected coefficients normalized and then multiplied by 2.5; `n_group=topk_group=1` |
| SwiGLU | gate upper bound 10, up limited to [-10,10]; the clamp runs before SiLU/product |
| MTP | one independent sparse MLA plus MoE layer, source layer number 45; ordinary residual, no mHC; shares the Main embedding and head |
| Vision | 24 layers, hidden=1024, 16 heads × 64; MLP intermediate=4096 |
| Vision patch / merge | patch `[3,2,14,14]`; 2×2 spatial downsample to 4096 wide; merger intermediate=10240 |
| Context | 1048576 tokens |
| Epsilon | Text/Vision RMSNorm `1e-5`; KDA Q/K L2 norm, Indexer LayerNorm and mHC pre/Sinkhorn `1e-6` |
| EOS / padding | EOS `[154820,154827,154829]`, padding=154820 |

Neither the text MLA nor the Indexer has a RoPE component in this instance. Vision still has its own
two-dimensional spatial RoPE. The layer order of each part selects KDA, MLA or MLP. The last Main
layer is KDA and MTP is MLA.

This document uses the logical matrix `W[N,K]`, where N is the output row and K is the input feature.
The vectors in the formulas are column vectors. T is the input column count of each request, B is the
batch and `M=T*B`. These logical axes do not specify a private CUDA kernel layout.

## 2. Main and the mHC residual

```text
token embedding, replaced by Vision features at the media placeholder positions
  → copy each 4096-wide vector into four residual streams
  → each layer: attention mHC pre → RMSNorm → KDA / sparse MLA → mHC post
                 FFN mHC pre       → RMSNorm → dense MLP / MoE → mHC post
  → mean over the four streams
  → final RMSNorm → 4096-wide hidden → LM head / MTP conditioning
```

The attention and the FFN of each layer have independent parameters, so the 45 layers hold 90 mHC
sites in total. The token and batch axes are omitted below, and the computation is independent for
each token. `X[i,d]` is the four-stream residual, `i,j=0..3`, `d=0..4095`, `D=4096`. i is the old
stream and j is the new stream. `eps_coeff=eps_branch=eps_final=1e-5`,
`eps_pre=eps_sinkhorn=1e-6`, `sinkhorn_iters=20`, and the post multiplier is 2.

### 2.1 Full-width RMS and the 24-row projection

The current site holds the logical matrix `F[24,16384]`, `base[24]`, `scale[3]` and the `gamma[4096]`
of that branch. Let:

```text
r = rsqrt(sum_{i,d}(X[i,d]^2) / (4*D) + eps_coeff)
z[a] = r * sum_{i,d}(F[a,i*D+d] * X[i,d])
```

The meaning of the 24 rows is fixed:

| Row range | Meaning |
|---|---|
| `[0,4)` | pre logits |
| `[4,8)` | post logits |
| `[8,24)` | mixing logits, row number `8+4*i+j` |

```text
p[i]   = sigmoid(scale[0]*z[i]       + base[i])       + eps_pre
q[j]   = 2*sigmoid(scale[1]*z[4+j]   + base[4+j])
A[i,j] = scale[2]*z[8+4*i+j]         + base[8+4*i+j]
```

`p` is the read gate and `q` is the write gate. `p` is no longer normalized across streams and is not
divided by 4. A saturated sigmoid endpoint stays as the numerical result gives it. This model has no
extra temperature division.

This full-width RMS serves coefficient generation only. The branch collapse still reads the original
X.

### 2.2 The fixed Sinkhorn sequence

```text
C = softmax_row(A) + eps_sinkhorn
C[i,j] /= sum_i(C[i,j]) + eps_sinkhorn

repeat sinkhorn_iters-1 times:
    C[i,j] /= sum_j(C[i,j]) + eps_sinkhorn
    C[i,j] /= sum_i(C[i,j]) + eps_sinkhorn
```

The denominator of each step comes from the complete row or column at the start of that step. Do not
modify elements while accumulating the remaining denominator. The initial softmax runs by row, and
the numerical implementation uses the stable form that subtracts the row maximum.

The current `sinkhorn_iters=20`, so the sequence is an initial row softmax, one column normalization,
and then 19 rounds of "row normalization, column normalization". That is 20 column normalizations and
19 explicit row normalizations in total. Every denominator adds epsilon. Do not replace it with 20
full row-and-column rounds, with one softmax, with a sigmoid initialization or with tolerance-based
early stopping.

This finite iteration produces an approximate doubly-stochastic matrix. The row and column sums of
`C` are not guaranteed to be exactly 1. Later computation must not rely on a strict
doubly-stochastic identity to omit the mixing or the final mean. For the conceptual source, see
[mHC paper §4](https://arxiv.org/html/2512.24880). This section is the authority for the actual
algorithm.

### 2.3 Four-stream collapse and branch norm

```text
v[d] = sum_i(p[i] * X[i,d])
u[d] = gamma[d] * v[d] * rsqrt(sum_d(v[d]^2) / D + eps_branch)
y = Branch(u)
```

`Branch` is KDA, sparse MLA, dense MLP or MoE, selected by the layer order. The 4096-wide weighted
branch norm and the 16384-wide unweighted coefficient norm are two different reduction domains.
`gamma` is a direct gain. It is not `1+gamma`.

`u` has already passed the attention/FFN input norm, so the later branch does not run that norm
again.

### 2.4 Residual mixing and branch injection

```text
X_next[j,d] = sum_i(C[i,j] * X[i,d]) + q[j] * y[d]
```

The matrix direction is fixed as `C[old,new]`, which is `C^T X + q*y`. `q/C` come from the same `X`
as before the current branch ran, and are used directly after the branch returns. Do not recompute
the gate or Sinkhorn.

`y` is the complete output of the whole branch. The
[numerical policy](glm5next-quantization.md#5-mhc-numerical-boundaries) defines the BF16
representation boundary of X_next and the actual value that the next site consumes.

### 2.5 Initialization and the final Main exit

Initialization copies the same embedding into four streams:

```text
X[i,d] = embedding[d], i=0..3
```

The media feature replacement completes before this. Four streams that start equal do not change the
complete coefficient mathematics of the current site.

The last site first completes the residual update of section 2.4, and then:

```text
mean[d] = sum_j(X_next[j,d]) / 4
hidden[d] = gamma_final[d] * mean[d]
            * rsqrt(sum_d(mean[d]^2) / D + eps_final)
```

`hidden` serves both the Main head and the MTP conditioning. Do not run the final norm again before
those two consumers. The numerical document holds the precision boundaries of the four-stream mean,
the final norm and the residual in one place.

### 2.6 Differences from the Qwen4Exp GR

| Stage | GLM5Next mHC | Qwen4Exp GR |
|---|---|---|
| residual | 4×4096 | 4×2560 |
| normalization before coefficient generation | all 16384 wide, no affine RMSNorm | each stream independently does a weighted RMSNorm |
| dynamic mapping | one `[24,16384]` projection | down/up and injection projections with lowrank=320 |
| read gate | one scalar per stream, 4 in total | one per stream per channel |
| branch input | weighted collapse of the original four streams, then a 4096-wide RMSNorm | per-channel gating of the normalized four streams, then divide by 4 |
| norm gain | direct gamma | offset gain `1+weight` |
| branch output injection | one `2*sigmoid` gate per stream | also one per stream |
| residual self-transition | a dynamic 4×4 matrix mixes across streams | each stream keeps its own residual |
| final Main exit | four-stream mean, then final RMSNorm | a learned final GR mix |
| native MTP | ordinary residual | four-stream HC |

[Qwen technical report §2.2](https://arxiv.org/html/2608.30320#S2.SS2) reports that GR and dynamic
mHC are close in capability at its ablation scale, and attributes the efficiency gain of removing the
cross-stream mixing to reading less of the complete residual. That is not a conclusion that the
projection FLOPs are fewer at any model geometry, and it does not change the GLM mixing mathematics.

## 3. KDA

### 3.1 Projection, convolution and gate

The KDA of each layer generates the following from the 4096-wide block input `x_t`:

| Source weight role | Logical shape | Effect |
|---|---|---|
| `q_proj / k_proj / v_proj` | `[8192,4096]` each | 64×128-wide Q/K/V |
| `q_conv1d / k_conv1d / v_conv1d` | `[8192,1,4]` each | four-tap causal depthwise convolution, FP32 weights, no bias |
| `b_proj` | `[64,4096]` | the beta logit of each head |
| `f_a_proj / f_b_proj` | `[128,4096]` / `[8192,128]` | the forget logit of each key channel |
| `A_log / dt_bias` | `[64]` / `[8192]` | FP32 decay controls |
| `g_a_proj / g_b_proj` | `[128,4096]` / `[8192,128]` | the sigmoid gate logit of each output channel |
| `o_norm` | `[128]` | the direct RMSNorm gamma shared by all heads |
| `o_proj` | `[4096,8192]` | back to the block hidden |

Q/K/V pass through the convolution and SiLU. The gate branch bypasses the convolution. In source tap
order the four taps are `t-3,t-2,t-1,t`, and a short history is zero-padded. Keep the **pre-convolution
projected values** of the most recent three columns.

For each head, compute the following in FP32:

```text
q = conv_silu(Q) / sqrt(sum(conv_silu(Q)^2) + 1e-6)
k = conv_silu(K) / sqrt(sum(conv_silu(K)^2) + 1e-6)
v = conv_silu(V)

f = W_f_b * (W_f_a * x)
g[h,k] = -5 * sigmoid(exp(A_log[h]) * (f[h,k] + dt_bias[h,k]))
beta[h] = sigmoid((W_b*x)[h])
z = W_g_b * (W_g_a * x)
```

There is no extra non-linearity between the two low-rank projections. The forget gate is a bounded
log-decay independent for each key channel. beta is one scalar per head. `g` and the output gate `z`
are different control values.

### 3.2 Recurrence and output

Let `S[v,k]` be the FP32 state of one head. The complete recurrence for each token is:

```text
S_decay[v,k] = S_prev[v,k] * exp(g[k])
delta[v]    = beta * (v[v] - sum_k(S_decay[v,k] * k[k]))
S_next[v,k] = S_decay[v,k] + delta[v] * k[k]
o[v]        = sum_k(S_next[v,k] * q[k]) / sqrt(128)
```

The decay applies to the key axis. delta uses the **already decayed** state, and the output uses the
updated state. Prefill, per-token decode and speculative verify must all keep this mathematical
transition.

The output does a 128-wide RMSNorm per head, multiplies by the direct gamma, and then multiplies by
`sigmoid(z)`:

```text
r[h,:] = gamma * o[h,:] / sqrt(mean(o[h,:]^2)+1e-5) * sigmoid(z[h,:])
y = W_o * flatten(r)
```

The logical history of each layer consists of the recurrent state and the pre-convolution Q/K/V of the
most recent three columns. When an accepted prefix is restored, both must be restored together.
Shortening the token history or the MLA cache alone cannot undo the KDA recurrence. Section 7 covers
the state composition, and the design index covers the implementation design of the record and the
fold.

## 4. Sparse MLA and the Kpool4 Indexer

This section defines the model mathematics, the grouping and the causal visibility. For the cache
representation, the codec and the compute boundaries, see
[quantization and numerical representation](glm5next-quantization.md#3-mlaindexer-cache-and-compute-policy).

### 4.1 MLA mathematics and the latent cache

The 11 MLA layers of Main and the one MLA layer of MTP use the following geometry:

| Weight | Logical shape | Output |
|---|---|---|
| `q_a_proj` | `[1536,4096]` | the Q bottleneck; a 1536-wide RMSNorm follows |
| `q_b_proj` | `[16384,1536]` | 64 Q heads of 256 |
| `kv_a_proj_with_mqa` | `[512,4096]` | one 512-wide latent; a 512-wide RMSNorm follows |
| `kv_b_proj` | `[32768,512]` | per head, 256-wide K then 256-wide V |
| `o_proj` | `[4096,16384]` | 64 attention outputs of 256 combined back to hidden |

Write `q_r=RMSNorm(W_qa*x)` and `c=RMSNorm(W_kva*x)`, with epsilon `1e-5` at both places. The
`W_kvb` rows of each head are read as `U_K[h]` and `U_V[h]`, each `[256,512]`. After the Indexer gives
the token set `J_t` that the current query can see:

```text
q_h = W_qb[h] * q_r
k_jh = U_K[h] * c_j
v_jh = U_V[h] * c_j
a_thj = softmax_{j in J_t}(dot(q_h,k_jh) / 16)
y_h = sum_{j in J_t}(a_thj * v_jh)
y = W_o * concat_heads(y_h)
```

Storing the complete K/V expansion is not a mathematical requirement. The equivalent latent-domain
computation is:

```text
q_latent_h = transpose(U_K[h]) * q_h
a_thj = softmax_{j in J_t}(dot(q_latent_h,c_j) / 16)
u_h = sum_{j in J_t}(a_thj * c_j)
y_h = U_V[h] * u_h
```

Note that the scale is still `1/sqrt(256)=1/16`. The 512 is the width of the cache latent and does not
change the attention scale used in training. Only one `c[512]` per token needs caching, shared by the
K/V computation of all heads, and there is no extra RoPE cache. The equations above are real-valued
equivalences. The finite-precision latent-domain computation is verified against the mapping and the
rounding boundaries of the numerical document, and algebraic equivalence does not imply bitwise
equality.

Sparse selection reduces the token set read on each call. The historical latent is still stored per
token for a later query to select. Each layer has its own latent and indexer cache.

### 4.2 Indexer input and learned pooling

The Indexer consumes the same block input `x` and the normalized Q bottleneck `q_r` of MLA:

| Weight | Logical shape | Effect |
|---|---|---|
| `wq_b` | `[4096,1536]` | 32 index queries of 128 |
| `wk` | `[128,4096]` | one raw index key |
| `k_norm.weight / bias` | `[128]` each | a LayerNorm with affine, epsilon=`1e-6` |
| `weights_proj` | `[32,4096]` | 32 coefficients that weight the index heads |
| `index_kpool_compress_gate` | `[128,4096]` | the pooling logit of each token and each channel |
| `index_kpool_compress_ape` | `[4,128]` | the learned bias of the four positions inside a pool |

```text
q_index[h,:] = reshape(W_index_q*q_r)[h,:]
k_raw = LayerNorm(W_index_k*x, gamma, bias, eps=1e-6)
p_raw = W_pool_gate*x
w[h]  = (W_head_weight*x)[h] / sqrt(32)
```

`w` is a signed linear projection result with no sigmoid and no softmax. For the complete pool p that
four consecutive valid tokens form, normalize along the four token positions for each channel d:

```text
P[r,d] = softmax_{r=0..3}(p_raw[4p+r,d] + APE[r,d])
k_pool[p,d] = sum_{r=0..3}(P[r,d] * k_raw[4p+r,d])
```

Each channel has its own four weights. The compressed object is the index key. The main MLA latent
does not pass through this step. Grouping starts at the first valid token of the request, and padding
must not change the pool phase.

### 4.3 Scoring, selection and tail

```text
score[t,p] = sum_h(w[t,h] * ReLU(dot(q_index[t,h],k_pool[p]) / sqrt(128)))
```

A pool takes part in the TopK only when its last token is visible to the current query. Select at most
512 complete pools, where each pool stands for four original tokens, and then add the 0–3 visible tail
tokens of the current unfinished pool. The effective budget is therefore at most 2051 tokens, and all
64 MLA heads share this set. The newest complete pool is not forced into the selection. Selecting the
complete pools and adding the unfinished tail are two steps. The Indexer score serves selection only
and does not enter the main attention logits.

A short context can select everything and omit the scoring and the TopK, but it must still maintain
the compressed keys and the tail that a later call needs. Each MLA layer of the main model scores
independently, and the current configuration shares no selection across layers.

The incremental state consists of the compressed keys of the complete pools and the raw K/gate of the
unfinished pool. A restore to a boundary other than four tokens must restore the tail as well. Prefill
chunk, decode, verify and restore keep the same grouping phase and causal visibility, and the
grouping phase depends only on the number of valid tokens already processed.

## 5. Dense MLP and MoE

The SwiGLU of every Text, MTP and Vision block includes the clamp:

```text
a = min(W_gate*x + b_gate, 10)
b = clamp(W_up*x + b_up, -10, 10)
ffn(x) = W_down * (SiLU(a) * b) + b_down
```

Text/MTP has no projection bias. The Vision blocks do have bias. The gate has an upper bound only and
up has both bounds. Do not move the clamp after the SiLU/product.

The GU of each of the first three dense MLP layers is `[12288,4096]` and down is `[4096,12288]`. The
GU of each routed/shared expert in the later 42 layers is `[2048,4096]` and down is `[4096,2048]`.

The MoE routing is defined as:

```text
logit_e = FP32_Dot(W_router[e,:], x)
s_e = sigmoid(logit_e)
choice_e = s_e + correction_bias_e
E_selected = Top8(choice)
p_e = 2.5 * s_e / (sum_{j in E_selected}(s_j) + 1e-20)
moe(x) = sum_{e in E_selected}(p_e * expert_e(x)) + shared_ffn(x)
```

The correction bias affects the selection only. The final coefficients use the sigmoid scores without
the bias. This instance has one expert group, so the group filter removes no candidate. The shared
expert is added directly, does not take part in the Top8, and has no independent shared sigmoid gate
like the Qwen4Exp one.

The Router source weight is BF16 and the correction bias is FP32. The config explicitly requires FP32
router logits. The numerical document covers weight quantization and activation permission. The
representation choice does not change the mathematical position of the clamp, of the route
coefficient or of the shared contribution.

## 6. MTP, Vision and input/output

### 6.1 Native MTP

The local NVIDIA checkpoint actually contains `model.language_model.layers.45.*`, and its routed
expert source weight is BF16. MTP reuses the Main token embedding and LM head but has its own
`enorm`, `hnorm`, `eh_proj`, MLA/Indexer, MoE and `shared_head.norm`.

```text
e = RMSNorm_enorm(token_embedding)
h = RMSNorm_hnorm(previous_hidden)
x = W_eh * concat(e,h)                         # [4096,8192]
r = x + sparse_MLA(RMSNorm_input(x))
u = r + MoE(RMSNorm_post_attention(r))
next_hidden = RMSNorm_shared_head(u)
draft_logits = shared_Main_LM_head(next_hidden)
```

The conditioning that Main provides is the 4096-wide hidden after the final RMSNorm. A later draft
step uses the `next_hidden` of the previous step. MTP has no four-stream residual and does not need
the 16384-wide mHC activation of Main. vLLM zeroes the MTP embedding input at position 0 and then runs
`enorm`. The other positions run by the correspondence between the input token and the previous
hidden.

The configuration `index_share_for_mtp_iteration=true`: the first step of a draft round computes the
selection, and the later steps reuse the complete pools selected at the first step and the frozen
visibility boundary, including the tail as it stood at the first step. They do not append a new tail
automatically for a later position. A reuse step still maintains its own latent, pool cache and tail,
and can omit the index query, the scoring and the TopK. The selection is a temporary choice of the
round and does not enter the checkpoint. The Main and MTP histories are independent.

The canonical MTP history updates from the real Main conditioning. A recursive draft hidden does not
become a real Main hidden because its token was accepted. Recompute the accepted part that needs
rebuilding from the real Main conditioning. Do not just shorten the draft cache and adopt it. The
hidden of the draft layer after its own final norm serves both the next-step conditioning and the LM
head, and the norm does not run again before the head.

### 6.2 Vision

```text
image/video preprocessing, split into patches by grid
  → Conv3d patch projection: 1176 → 1024, with bias
  → 24 layers: RMSNorm → QKV → Q/K head RMSNorm → 2D RoPE
                    → non-causal attention → O projection → residual add
           RMSNorm → clamped SwiGLU → residual add
  → 1024-wide post RMSNorm
  → Conv2d downsample per 2×2 patches: 4×1024 → 4096, with bias
  → merger projection 4096→4096 → LayerNorm → GELU
  → clamped SwiGLU: 4096→10240→4096
  → replace the media placeholder embedding in the Text input
```

Vision has 16 heads of 64. The QKV projection is `[3072,1024]` and the O projection is `[1024,1024]`.
The gate/up of the block FFN is `[4096,1024]` and down is `[1024,4096]`. The Q/K head norm gamma is
`[64]` each. The patch weight source shape is `[1024,3,2,14,14]` and the downsample is
`[4096,1024,2,2]`. The merger GU is `[10240,4096]` each and down is `[4096,10240]`. These merger
projections have no bias. The LayerNorm has weight/bias.

The block norm, the Q/K norm and the post norm are all direct-gamma RMSNorm with epsilon=`1e-5`. The
merger uses an ordinary LayerNorm with epsilon=`1e-5`, followed by a non-approximate GELU. The spatial
RoPE theta=10000. The input has no independent learned absolute-position embedding. The encoder
sequence is separated by media grid, and the media token count is `T_grid*H_grid*W_grid/4`.

The source processor uses patch=14, temporal patch=2 and merge=2. The image token range is 16–8000.
The video configuration allows at most 240000 at fps=2. The RGB normalization mean is
`[0.48145466,0.4578275,0.40821073]` and std is `[0.26862954,0.26130258,0.27577711]`. These are the
source preprocessing settings. Local vLLM has its own runtime media budget limits. They do not change
the Text maximum context and do not automatically become an NInfer product limit.

### 6.3 Tokenizer, thinking and tools

The tokenizer, `chat_template.jinja`, generation and processor configuration of the source checkpoint
together define the input/output resources. The template starts with `[gMASK]<sop>` and uses the GLM
system/user/assistant/observation role markers. Thinking uses `<think>…</think>`. A tool call uses
the `<tool_call>`, `<arg_key>` and `<arg_value>` structure, and a tool result is expressed with
`<tool_response>`. The image/video markers, the expanded token count and the Vision feature order must
agree.

The Frontend owns the correspondence between the template, the token boundaries and the
thinking/tool parsing. The model computation consumes the prepared input and does not own the input
protocol or the media acquisition.

## 7. Logical state and historical dependencies

| Component | Logical content | Relation to the accumulated context |
|---|---|---|
| Main/MTP MLA | one 512-wide latent per layer per token | grows with the history; Main and MTP are independent |
| Indexer complete pools | one 128-wide pooled key per 4 valid tokens | grows with the complete pool count and is tied to the matching MLA history |
| Indexer unfinished tail | the raw K and the raw gate of the current pool, at most 3 columns each | a fixed-length frontier state per request per layer; APE is not added in advance |
| KDA recurrent | 64 `[value=128,key=128]` states per layer | accumulates the whole adopted prefix, and the size does not grow with length |
| KDA convolution | the pre-convolution values of the most recent 3 columns of each of the 8192 Q/K/V channels per layer | a fixed-length history per request, ordered oldest to newest |
| MTP conditioning | the 4096-wide hidden after the final Main norm; the recursive draft consumes the previous MTP hidden | distinguished by the real Main/MTP frontier and not interchangeable |
| mHC residual, dynamic coefficients | the four-stream activation and the coefficients of the current token along the network depth | does not accumulate history and is not a request checkpoint field |
| MTP selection, verify records | temporary results of the current round or of an accepted prefix restore | does not become long-term history or a checkpoint |

A checkpoint must describe the kept prefix together with its KDA state/conv, the Indexer tail and the
MTP conditioning it needs. Truncating the growing history alone cannot restore these accumulated
states. The valid column count of the Indexer tail is decided by the number of processed valid tokens
modulo 4. MLA and the Indexer must restore to a consistent visible frontier.

NInfer manages the content that grows with the history separately from the accumulated per-request
state. The implementation design defines the physical pool, the slot, the record plane and the commit
interface, and does not change the logical dependencies of the table above. The two-host sharding and
the owner do not change the meaning of the model state either. The private writable history of each
request is independent, and the context management handles prefix sharing and restore explicitly.

## 8. Sources

- Local NVIDIA checkpoint: `/home/neroued/models/GLM-5.3-Flash/nvidia-nvfp4` on both Spark hosts;
  the config, the quant config, the processor/tokenizer/template, every tensor header and the NVFP4
  scalar metadata were checked. The
  [official configuration](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/blob/main/config.json)
  and the
  [NVIDIA recipe](https://github.com/NVIDIA/Model-Optimizer/blob/4956213d670c382385d9bc43e17379b9fc064e50/modelopt_recipes/models/zai-org/GLM-5.3-Flash/ptq/nvfp4_experts_dense_mlp-kv_fp8_cast.yaml)
  provide the source representation description. The
  [numerical document](glm5next-quantization.md) holds the actual representation and the capacity.
- [Transformers 5.16.1 model definition](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/glm5_next/modeling_glm5_next.py):
  the direct mathematical definition of mHC, KDA, sparse MLA, learned pooling, clamped SwiGLU and
  Vision.
- The
  [model body](https://github.com/vllm-project/vllm/blob/00b7847c8036b667742b4efb21aab1de51fd4721/vllm/models/glm5next/common/model.py),
  [KDA](https://github.com/vllm-project/vllm/blob/00b7847c8036b667742b4efb21aab1de51fd4721/vllm/models/glm5next/common/kda.py),
  [MLA / Indexer](https://github.com/vllm-project/vllm/blob/00b7847c8036b667742b4efb21aab1de51fd4721/vllm/models/glm5next/common/attention.py),
  [Kpool codec](https://github.com/vllm-project/vllm/blob/00b7847c8036b667742b4efb21aab1de51fd4721/vllm/models/glm5next/nvidia/ops/kpool_compress.py),
  [MTP](https://github.com/vllm-project/vllm/blob/00b7847c8036b667742b4efb21aab1de51fd4721/vllm/models/glm5next/common/mtp.py)
  and
  [Vision](https://github.com/vllm-project/vllm/blob/00b7847c8036b667742b4efb21aab1de51fd4721/vllm/models/glm5next/common/multimodal.py)
  of local vLLM revision `00b7847c8036b667742b4efb21aab1de51fd4721`: checked against the actual
  weight mapping, the latent execution, the incremental state, the Indexer FP8 encoding and the
  native MTP handoff.
