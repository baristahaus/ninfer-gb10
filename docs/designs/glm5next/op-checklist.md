# GLM5Next: Spark TP2 / EP2 base Op checklist

Nature: an existing base execution task. The P/F/M/R/E numbers, the check state and the
implementation scope stay unchanged during the document pass. The
[design index](README.md) holds the related design status and the reading entry.

Goal: prepare local GPU operators for GLM-5.3-Flash on two DGX Spark hosts, and complete the Linear,
stateless projection fusion, dense/shared MLP and Main/MTP MoE support and tuning that this checklist
already fixed. [Model semantics](../../maintainer/glm5next-model.md) covers the architecture and the
mathematics. The [numerical policy](../../maintainer/glm5next-quantization.md) covers quantization
and precision. The [parallel design](parallel.md) fixes the split: on one group of two Spark ranks,
MLA/Indexer DP2, KDA/dense/shared/head TP2 and routed experts EP2. This round prepares only the
TP2/EP2 local geometry below. It adds no complete single-rank profile and no RTX 5090 profile for
these operators. The complete-heads DP contract of MLA/Indexer is in
[Kpool Sparse MLA](kpool-sparse-mla.md). Its profiles and implementation are issued separately and
are not appended automatically by this checklist.

This checklist follows the form of `.50` `docs/qwen4-exp-spark-tp2-ep2-op-checklist.md`. The
execution baseline is the Qwen TP2/EP2 local operator result that `.50` `/home/neroued/ninfer`
already completed, in particular its indexed expert bank and its local subset computation. An
unchecked item means this round has not finished it. It does not mean the existing implementation
lacks support. First confirm and measure an existing suitable implementation, and then decide whether
it needs a change.

**This base checklist prepares only the pure projections and the stateless split of KDA.**
[KDA ReplaySSM](replayssm-kda.md) already fixed the internal gate, the per-round record/fold and the
accepted-prefix state contract. Its implementation is issued as a separate task.
[Kpool Sparse MLA](kpool-sparse-mla.md) and [mHC](mhc.md) also have fixed complete contracts, and
neither implementation enters this round. mHC replicates the weights and the four-stream residual and
updates the residual only after it has the complete branch output. That does not change the local
geometry or the execution entries of this checklist. Do not start the converter, a full weight
conversion, the model integration, communication or an engine refactor.

## 1. Local split, merging and implementation order

### 1.1 The local geometry adopted in this round

| Part | TP2/EP2 split | Computation per rank |
|---|---|---|
| TP/EP block hidden in this checklist | replicates the 4096-wide input | the same group of tokens, T/B not halved |
| KDA projection | split by complete heads into 32 + 32 | Q/K/V are 4096 wide each; beta 32; f_a/g_a 128 wide each, complete; f_b/g_b 4096 wide each |
| Main first three dense MLP layers | intermediate 12288→6144 + 6144 | local clamped SwiGLU, down outputs a 4096-wide local contribution |
| Main/MTP shared expert | intermediate 2048→1024 + 1024 | local clamped SwiGLU, the output is added directly to this rank's MoE contribution |
| Main/MTP routed experts | EP2, no TP inside an expert | 288 global experts, 144 complete experts actually resident per rank, global top-8 |
| Router | replicated | the complete 288-wide logits, the correction bias and the same global Top8 |
| Main/MTP LM head | TP2 on the vocabulary/N axis | `[77440,4096]` per rank, Main and MTP share the weight |

MLA/Indexer computes only the requests of its owner, and the replication convention of the table
above does not apply to its input scope. A later model execution design decides the result handoff
with the TP/EP part of this table. This DP decision does not change the local weight shape of P1–P9.

The gate/up of dense/shared split along the intermediate rows separately, and down splits along the K
range of the same intermediate. Their per-channel clamp/SiLU/product does not cross a shard, so the
sum of the two local outputs corresponds to the complete FFN. Shared is two intermediate shards of one
FFN. Do not compute one complete shared on each side.

The ownership of the routed experts is given by an explicit global ID to local payload mapping, and
the local set size is 144. The local Op does not fix a contiguous range, an odd/even assignment or a
load-balancing policy. A later parallel plan provides them.

### 1.2 Common conventions

- Write the weight as `W[N,K]` and the input as `[K,M]`, with `M=T*B` and `column=t+T*b`. Native Tensor
  dimension 0 is contiguous. The stateless operators below can flatten over M. Do not replace the KDA
  sequence and state semantics with M.
- The public input and output of an ordinary Linear, split, MLP and local MoE are BF16. The down/O
  that splits along K and the EP local sum also output BF16. The internal logits and the public
  coefficients of the Router use FP32.
- The local FFN/MoE output uses Write semantics. The Main mHC or the MTP residual consumes it only
  after the cross-rank sum. Do not add one complete residual on each side.
- **Complete the pure `linear()` of the matching shape first, and then extend the split, the
  clamp/SiLU/product and the complete FFN.** Each dependency chain advances on its own. Verify a fused
  Op against the complete mathematics. The BF16 output of a pure Linear does not become a mandatory
  internal rounding boundary for the fusion, and the fusion does not require calling the public
  Linear first and then chaining the post-processing.
- New semantics such as the clamp and the route coefficients must be expressed explicitly in the
  public profile/parameters and in the contract. Do not swap the mathematics of the original operator
  by a model name or a shape. Follow the matching family, and Spark has its own dispatch. The existing
  path without a clamp and the behaviour already supported by the other models stay valid.
- The view, alignment and non-overlap requirements of the weight, of the output and of the
  caller-owned workspace follow the family they belong to. The capacity query covers the complete call
  and the M range of the request. There is no hidden device allocation, no transfer and no Host
  readback, and CUDA Graph capture and replay are supported. A local Op takes no rank, no world size
  and no communicator.

Dependencies:

```text
P1 → F1: QKV split                    P3 → F2: beta / f_a / g_a split
                                      └→ R1: FP32 logits / sigmoid / global Top8
P5 → F3: NVFP4 clamped GU ┐
P6 ─────────────────────┴→ M1: complete dense MLP local contribution
P7 → F4: BF16 clamped GU ┐
P8 ─────────────────────┴→ M2: complete shared MLP local contribution
R1 + E1 + M2 → E3: Main complete local MoE
R1 + E2 + M2 → E4: MTP complete local MoE
P2 / P4 / P9 complete independently; E1 / E2 complete the contraction and the fusion inside the MoE family
```

## 2. Pure Linear: nine profiles

The pure Linear stage writes only one contiguous BF16 output. It contains no split, no norm, no
convolution, no clamp and no routing.

| Number | Weight representation / permission | Local W[N,K] | Use |
|---|---|---|---|
| P1 | Q8-G32/FP16, A16Only | `[12288,4096]` | KDA merged QKV: `[Q4096;K4096;V4096]` |
| P2 | Q8-G32/FP16, A16Only | `[4096,4096]` | KDA O, the local sum split along the input heads/K |
| P3 | BF16, A16Only | `[288,4096]` | KDA merged `beta32;f_a128;g_a128` |
| P4 | BF16, A16Only | `[4096,128]` | KDA f_b and g_b, two independent projections sharing one profile |
| P5 | NVIDIA NVFP4, AllowA4 | `[12288,4096]` | dense MLP local GU: `[gate6144;up6144]` |
| P6 | NVIDIA NVFP4, AllowA4 | `[4096,6144]` | dense MLP local down |
| P7 | BF16, A16Only | `[2048,4096]` | Main/MTP shared local GU: `[gate1024;up1024]` |
| P8 | BF16, A16Only | `[4096,1024]` | Main/MTP shared local down |
| P9 | BF16, A16Only | `[77440,4096]` | Main/MTP LM head vocabulary shard |

- [ ] **P1**: Q8 `[12288,4096]`.
- [ ] **P2**: Q8 `[4096,4096]`.
- [ ] **P3**: BF16 `[288,4096]`.
- [ ] **P4**: BF16 `[4096,128]`; confirm the role reuse with different f_b/g_b weights and their own inputs.
- [ ] **P5**: NVFP4 `[12288,4096]`, AllowA4.
- [ ] **P6**: NVFP4 `[4096,6144]`, AllowA4.
- [ ] **P7**: BF16 `[2048,4096]`.
- [ ] **P8**: BF16 `[4096,1024]`.
- [ ] **P9**: BF16 `[77440,4096]`.

Slice and merge conventions:

| Object | Local row/column range |
|---|---|
| KDA heads | the two sides take the original heads `[0,32)` and `[32,64)`; each head keeps all 128 channels |
| P1 | take the matching 4096 rows of the source Q, K and V separately, then arrange them as Q/K/V; do not split the whole QKV parent evenly |
| P2 | N=4096 is kept, and K takes the same 4096 columns as the local KDA output heads |
| P3 | beta takes the local 32 rows; f_a/g_a each replicate the complete 128 rows, in the order `[beta;f_a;g_a]` |
| P4 | f_b/g_b each take the local 4096 output rows and K=128 is kept; f_a and g_a are different inputs, so f_b/g_b must not be treated as one simple concatenation of a same-input projection |
| P5/P6 | each side takes 6144 rows of gate/up and the matching 6144 columns of down |
| P7/P8 | each side takes 1024 rows of gate/up and the matching 1024 columns of down |
| P9 | the two sides are the physical vocabulary `[0,77440)` and `[77440,154880)`; a later output layer handles the vocabulary validity and the token ID decision |

The K split point of Q8 stays G32-aligned and the K split point of NVFP4 stays G16-aligned. Keep the
meaning of the source codes/scales and of the weight/input divisors, and rearrange the storage for the
local matrix. Do not re-quantize or re-calibrate because of the shard. The GU/down of P5/P6 each use
the activation divisor of the matching source dense MLP.

The numerical geometry of P3 is exactly the same as the router's `[288,4096]`, but its BF16 output must
not serve as the logits input of R1. R1 requires complete FP32 logits. It can share the contraction
basis, and it must not share the wrong exit.

## 3. Stateless projection fusion

### 3.1 LinearSplit

- [ ] **F1 — KDA QKV split, depends on P1.** Q8 `[12288,4096]`, writing three independent contiguous
  BF16 outputs Q/K/V of `[4096,M]` each. What is output here is the **pre-convolution projected value**.
  Do not run SiLU, normalization, convolution or a history update.
- [ ] **F2 — KDA control split, depends on P3.** BF16 `[288,4096]`, writing
  `beta_raw[32,M]`, `f_a[128,M]` and `g_a[128,M]` directly. Beta is not yet sigmoid, and f_a/g_a each
  still have a second projection after this. This item fuses no KDA gate or recurrence semantics.

Follow the segmented mathematics and the complete capacity/launch contract of
[LinearSplit](../../../include/ninfer/ops/linear_split.h). The tuning goal is for the contraction to
write the contiguous segments directly. The final implementation must not materialize the complete
output separately and then copy it. The stateless split decision does not promise that a future KDA
state fusion must consume this group of intermediate buffers.

### 3.2 Merged GU plus clamped SwiGLU

The non-linearity that this round adds explicitly is:

```text
gate = sum_k(decode(W_gate)[i,k] * x[k,m])
up   = sum_k(decode(W_up)[i,k]   * x[k,m])
product[i,m] = SiLU(min(gate, limit)) * clamp(up, -limit, limit)
limit = 10
```

`limit` is an explicit positive finite semantic parameter, and this model takes 10. The gate clamps
the upper bound only and up clamps both bounds. The clamp runs before the SiLU/product. This parameter
enters the matching SwiGLU/MLP/MoE profile at the same time. An existing computation without a clamp
keeps its original meaning. Express it through a family-consistent interface. Do not introduce a
checkpoint-specific execution entry.

- [ ] **F3 — NVFP4 GU plus clamped SwiGLU, depends on P5.** Input BF16 `[4096,M]`, GU NVFP4
  `[12288,4096]`, AllowA4, output contiguous BF16 `[6144,M]`. Use the original activation divisor of
  the GU. One gate/up position pair produces one product.
- [ ] **F4 — BF16 GU plus clamped SwiGLU, depends on P7.** Input BF16 `[4096,M]`, GU BF16
  `[2048,4096]`, A16Only, output contiguous BF16 `[1024,M]`.

The existing [Linear SwiGLU](../../../include/ninfer/ops/linear_swiglu.h) has only the formula without
a clamp, and its current format entry does not include BF16. F3/F4 must complete the public contract,
the format support, the capacity, the dispatch and the measurement together. The fused computation
does not require rounding gate/up to BF16 first and then running the non-linearity.

## 4. The local contribution of the complete dense/shared MLP

The complete MLP mathematics here is `y=W_down*product`, and product uses the definition in section
3.2. The public input is BF16 `[4096,M]`, the GU/down weights and the activation
permission/metadata. The output is contiguous BF16 `[4096,M]` with Write semantics and no routes, no
gate coefficient and no residual input.

- [ ] **M1 — Main dense MLP, depends on P5/P6/F3.** Local I=6144, GU/down both NVIDIA NVFP4,
  AllowA4. The complete call covers the input quantization, the GU, the clamp/SiLU/product, the down
  input quantization, down and the output write. Keep the original activation divisor of each of
  GU/down.
- [ ] **M2 — Main/MTP shared MLP, depends on P7/P8/F4.** Local I=1024, GU/down both BF16, A16Only.
  The output is the local shared contribution and the coefficient is 1 in the mathematics. The entry
  expresses an unscaled shared FFN directly. It does not require the caller to fake a router row or an
  all-1 coefficient tensor.

F3/F4 are independent GU-product operators and their public output is BF16. M1/M2 are complete FFN
semantics and their internal product adds no mandatory BF16 rounding boundary. In particular, the A4
down of M1 can quantize the internal product directly. The independent stage helps implementation and
localization, but the acceptance is against the mathematics and the timing of the complete MLP.

These computations enter the existing SwiGLU/MoE compute basis and reuse a suitable contraction.
Whether to use a persistent kernel, staged kernels or shared scratch is an implementation choice. Do
not require one CUDA kernel to do all of it.

## 5. GLM sigmoid plus correction-bias routing

- [ ] **R1 — complete global Top8 routing, complete the pure projection preparation of P3 first.**
  Every rank uses the same BF16 input `[4096,M]`, the BF16 router weight `[288,4096]` and the FP32
  correction bias `[288]`, and outputs contiguous I32 IDs `[8,M]` and FP32 coefficients `[8,M]`.

The complete mathematics of this item is fixed as:

```text
logit[e,m] = FP32_Dot(router_weight[e,:], x[:,m])
s[e,m] = sigmoid(logit[e,m])
choice[e,m] = s[e,m] + correction_bias[e]
ids[:,m] = Top8(choice[:,m])
coefficient[j,m] = 2.5 * s[ids[j,m],m]
                   / (sum_{i=0..7}(s[ids[i,m],m]) + 1e-20)
```

The IDs are in descending choice order, and an equal value puts the smaller global expert ID first.
The correction bias serves selection only and does not enter the final coefficients. Top-k=8, scaling
factor=2.5 and normalization epsilon=1e-20 are passed in or recorded explicitly as semantic
configuration. This instance has one expert group, and this round does not extend grouped routing.

The Router weight keeps the original BF16 words. The logits are computed and consumed directly in
FP32 and do not pass through the BF16 output of an ordinary Linear. Every rank replicates the same
global routing and uses the same numerical path and tie-break. Do not first split the experts into 144
and then select a Top8 on each side, and do not output a shared sigmoid coefficient.

The softmax plus extra shared logit formula of the existing `topk_softmax_route` does not apply. This
item must provide an explicit sigmoid/correction routing semantic and can reuse the public TopK and the
projection basis. The whole entry contains the projection, the sigmoid, the TopK, the coefficient
normalization and the write, and the timing covers the complete call.

## 6. Main/MTP EP2 and the complete local MoE

### 6.1 The local expert contribution of known routes

| Object | Requirement per rank |
|---|---|
| Global E / top-k / local E | 288 / 8 / 144; the local hit count of each token can be 0..8 |
| Bank | only the payload of the local 144 experts, keeping the explicit global-ID mapping and the local subset |
| Single expert GU / down | `[4096,4096]` / `[4096,2048]`; the gate/up of the GU is 2048 rows each |
| Input | contiguous BF16 `[4096,M]` |
| Routes | global I32 IDs `[8,M]`, FP32 coefficients `[8,M]`, consumed directly without re-normalization |
| Output | contiguous BF16 `[4096,M]` local contribution, Write; writes zero when there is no local hit |
| Activation | the clamped SwiGLU of section 3.2, limit=10 |

```text
y_routed_local[:,m] = sum_{j: ids[j,m] belongs to local bank}
                       coefficient[j,m] * expert_ids[j,m](x[:,m])
```

- [ ] **E1 — Main NVFP4 EP contribution.** AllowA4; the G16 E4M3 scales of GU/down, the weight
  divisor of each expert and the activation divisor of each layer's GU/down keep their original
  meaning. Extend the local H=4096, I=2048, global E=288, top-k=8 geometry and the clamp mathematics,
  and complete the tuning for a small M and for a grouped large M.
- [ ] **E2 — MTP block-FP8 EP contribution.** AllowA8; 128×128 BF16 block scales, the activation is
  quantized dynamically in K128 groups per token, and there is no static activation divisor. The other
  geometry, the local-ID mapping and the clamp mathematics are the same as E1.

Reuse the indexed bank, the `ExpertSubset` and the local contribution mechanism that `.50` already
completed. Do not allocate 288 complete payloads just to make testing easy. Do not read the address or
the weight of an expert that is not owned, and do not change the top-k to 4. The local subset filter
does not change the source coefficients, and the meaning of the coefficient 2.5 stays in the output of
R1.

The expert banks are not part of the nine profiles of the ordinary Linear. Inside the MoE family,
first complete the correctness of the local matrix contraction, and then add the clamp, the route
weighting and the reduction. Do not add a second ordinary `linear()` interface for the expert bank. A
test can construct a legal encoded weight and does not depend on the full GLM converter.

### 6.2 The complete local fusion of Router plus EP plus shared

The complete local MoE consumes these explicit data: the BF16 block input, the R1 router weight and
bias, the global routing configuration, the local routed bank/subset, the TP2 shared GU/down, the
clamp parameter and the matching activation permissions/divisors. It writes only BF16 `[4096,M]`
outward:

```text
y_local = routed_contribution(R1(x), local_bank, x)
          + shared_intermediate_shard(x)
```

- [ ] **E3 — Main complete local MoE, depends on R1/E1/M2.** NVFP4 routed plus BF16 shared, global
  top-8, local E=144, shared I=1024, outputting the complete contribution of this rank.
- [ ] **E4 — MTP complete local MoE, depends on R1/E2/M2.** Block-FP8 routed plus BF16 shared, sharing
  the semantic entry and the parameter organisation with E3; the format and the permission decide the
  internal implementation.

The routes and the staged contribution can be private intermediates in the caller workspace and are
not forced outward. The fused output is limited to one BF16 exit by the complete local formula. A zero
routed hit still computes the local shared. Shared counts once. It is not multiplied by the routed
scaling factor and not by an extra sigmoid.

The two ranks' `y_local` must be added to make the complete MoE. The Main mHC and the MTP residual
update both come after this communication boundary, and E3/E4 do not carry the residual, the
communication or the state commit. The fusion goal is to reduce the repeated input read, the
allocation and the intermediate contribution write, and to allow a sensible organisation of the
route/group preparation and the shared computation. Do not force the whole flow into one kernel.

## 7. Tuning scope and how it completes

Follow the [Op development rules](../../maintainer/op-development.md), the
[Linear tuning method](../../maintainer/linear-tuning.md) and the existing
[tests](../../../tests/README.md) and [bench](../../../bench/README.md). First check the fixture, the
independent mathematical reference, the precision policy and the traffic model of the item, and then
implement and tune. Acceptance is per complete call of the public Op, and the capacity query, the
dispatch, the tests and the bench cover the new geometry together.

### 7.1 Projections and MLP

| Scope | Workload and target |
|---|---|
| small M | every integer `M=1..128`; the focus is 1, 4, 8, and keep 6 as well; reduce the M=1 latency, and reuse the weight and the effective parallelism more as M grows |
| concurrency/spec | ordinary generation and a single-step proposal are `M=B`; the representative verify widths are `M=4B, 6B, 8B, 16B` with B=1..8, and the actual draft policy is not fixed |
| middle range | add points at M=256 and near the actual route switch; do not scan every point up to 4096 |
| large M | **512, 1024, 2048, 4096** are all independent anchors; report the total latency, the cost per column, the bandwidth and the actual arithmetic throughput reference |

Observe the startup and reduction cost of the narrow output of P3 and of the narrow K of P4
separately. P9 includes the complete logits write. The Q8 of P1/P2 is an A16 weight-decode path and
must not use the INT8/NVFP4 Tensor Core peak. `AllowA4` for NVFP4 does not mean that every M uses A4.
The actual production dispatch decides the compute reference.

The useful FLOPs of a pure Linear are `2*N*K*M`, and the weight bytes follow the actual codes/scales.
The BF16 input and output are `2*K*M` and `2*N*M`. F1/F2 include every split output. F3/F4 include the
complete exit of the clamp/product. M1/M2 include both contractions, the necessary quantization and
the workspace read and write. Do not use the GU kernel time in place of the complete MLP result.

### 7.2 Routing and EP/MoE

R1 uses the same M range as the projections. Observe the FP32 logits, the TopK and the coefficient
write separately. E1–E4 cover `M=1..128` and the large-M anchors above, with a representative zero
hit, all 8 hits, a mixed two-sided case, a repeated hot spot across tokens and a more spread
distribution. Do not run the Cartesian product of every distribution at every M.

The actual local token-expert count `P=sum_m(local_hits[m])` decides the useful contraction FLOPs of
the routed path: `6*4096*2048*P`. Compute the de-duplicated weight payload by the expert set that is
actually accessed, and keep the token-expert work at the same time. Do not infer the payload from a
fixed global top-k/2. The encoded payload of a single expert GU plus down is about NVFP4 **13.5 MiB**
and block-FP8 **24.00293 MiB**, plus the NVFP4 scalar metadata. E3/E4 also count the complete router
and the shared TP2 shard, and shared does not disappear when local routed hits=0.

For a small M the focus is the weight read, the startup and the cost of zero or few hits. For a large
M the focus is the group reuse, the quantization preparation, the reduction and the workspace traffic.
Report the cold and warm cache results of the Graph separately, and keep the submission cost of a
representative Eager point. Reference the roofline with the Spark bandwidth and the actual execution
precision. Record the median, P95, the important switch points, the worst change and the limits that
remain. Do not read a logical GB/s as a DRAM counter.

### 7.3 Mathematical and regression checks

- Verify the pure Linear, the split, the GU product, the complete MLP, the routes, the EP contribution
  and the complete local MoE each against its independent mathematical reference. Do not assemble the
  fusion reference from already rounded stage outputs.
- The clamp covers the gate upper bound and both up bounds. The route covers that the correction bias
  changes the selection but not the coefficient source, the tie-break and the normalization. The
  precision criterion follows the public semantics/permission. Do not switch the acceptance criterion
  by which internal A16/A8/A4 path is actually taken.
- EP covers non-contiguous global IDs, the actual 144-entry bank, 0/8 local hits, the shared term
  added only once and the caller workspace range. Two local mathematical contributions can form the
  complete reference in a single-host test without a communication implementation.
- Check the existing profiles without a clamp or without a softmax that the modified shared basis
  actually affects. Do not expand into a model regression or a whole-machine benchmark unrelated to
  this task. An existing capability that passes this round's check can stay and does not need a
  rewrite.

After an item completes, update this checklist and link a short verification and performance result.
State a new capability and an improvement to an existing optimised path separately. Do not claim a
local Op result as a two-host model throughput gain.

## 8. Deferred in this round

The following work is not given to this round's execution agent to implement or tune:

- **The whole KDA computation chain and the state interface**: the convolution, the Q/K normalization,
  the bounded gate, the recurrence, the batch update, the output gated norm, and KDA fusions such as
  projection plus conv/gate. ReplaySSM and the accepted-prefix state management follow the
  [agreed design](replayssm-kda.md), and their implementation and tuning are issued as a separate task
  and are not extended automatically by this document. The dispatch of the experimental general entry
  is not a production decode constraint.
- **MLA and the Indexer**: [Kpool Sparse MLA](kpool-sparse-mla.md) already fixed the interface, the
  precision, the codec/page layout, the projection merge, the two prepare exits, the scoring/TopK, the
  frozen selection and the tail commit for the complete heads of Main/MTP. The Indexer keeps a BF16
  cache and BF16 computation. MLA can select a BF16/FP8 cache, and the computation stays BF16/F32.
  Implementation and tuning are issued separately and are not extended into this round automatically
  because the contract is fixed.
- **mHC**: the [fixed contract](mhc.md) covers the complete mathematics, the five entries, the BF16
  residual and the FP32 coefficients, the collapse/norm and the adjacent post/pre fusion, the final
  mean/norm and the two-host replicated placement. Implementation and tuning are issued separately and
  do not join this round automatically because the interface is fixed.
- **MTP stem, embedding, Vision**: the local placement and the operator combination are not fixed in
  this round, so their Linear/fusion profiles are not added yet. The preparation of the MTP MoE and of
  the shared LM head is already in the earlier sections.
- The candidate merge of sharded logits, the sampling, the MTP acceptance and the scoring. P9 in this
  round produces local Linear logits only.
- The full quantization/converter, the model binding and load, the communication/compute-communication
  fusion, the Engine/Program integration, the cross-rank state and resource planning, and the complete
  model performance measurement.

These parts gain an implementation requirement only after the user explicitly issues an execution
task. If an item depends on work outside the scope above during execution, stop the extension on that
dependency and report the concrete boundary, and continue with the other independent items.

## 9. Where the checklist goes after completion

When an item completes, keep the entry to its implementation, test and performance evidence. After
every item completes and the evidence belongs to the formal contract, the tests or the benchmark,
delete this checklist and let Git keep the history. Do not add an execution item because a design is
fixed, and do not let a document pass trigger a remote task issue.
