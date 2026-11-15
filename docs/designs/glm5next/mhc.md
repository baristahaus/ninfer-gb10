# mHC implementation design

Nature: a four-stream residual Op interface and fusion design that still has to land. For the
decisions, the implementation state and the task scope, see the [design index](README.md). For the
complete mathematics, the row order and the differences from the Qwen GR, see
[model mHC](../../maintainer/glm5next-model.md#2-main-and-the-mhc-residual). For precision and the
rounding boundary, see
[mHC numerical policy](../../maintainer/glm5next-quantization.md#5-mhc-numerical-boundaries).

This document covers the five complete entries, the native layout, the alias, the workspace, fusion
at adjacent sites and acceptance. The first geometry is D=4096 and S=4. `mhc` keeps a different
compute contract from the existing
[gated residual](../../../include/ninfer/ops/gated_residual.h). It can reuse a private mechanism. It
must not change the existing GR mathematics through a model name or a shape.

## 1. Weights and tensor layout

Native Tensor dimension zero is contiguous. `T` is the physical token width for each request, `B` is
the batch, `M=T*B` and `m=t+T*b`. The mathematics computes each column independently. It does not mix
tokens or requests.

| Object | Native layout | Meaning |
|---|---|---|
| `residual / updated_residual` | BF16 `[4096,4,T,B]` | hidden axis contiguous, then the stream |
| `embedding / block_input / block_output / hidden` | BF16 `[4096,T,B]` | the complete block hidden |
| `post` | FP32 `[4,T,B]` | the activated `q` |
| `mix` | FP32 `[4_new,4_old,T,B]` | `mix[j,i,t,b]=C_math[i,j,t,b]` |
| `F` | BF16, logical `[24,16384]`, native `[16384,24]` | the K axis is contiguous for each output row |
| `base / scale` | BF16 `[24]` / `[3]` | matches the [model coefficient row order](../../maintainer/glm5next-model.md#21-full-width-rms-and-the-24-row-projection) |
| `gamma / gamma_final` | BF16 `[4096]` | the direct gain |

The key offsets are given as element counts:

```text
residual(d,i,t,b) = d + D*(i + 4*m)
block(d,t,b)      = d + D*m
post(j,t,b)       = j + 4*m
mix(j,i,t,b)      = j + 4*i + 16*m
F(a,k)           = k + 16384*a, k=i*D+d
```

The order of source `F` rows `[8,24)` is `8+4*i+j`. That matches the contiguous `j` axis of the
native mix. The Converter imports the values by logical role. It does not quantize. It does not merge
the F of the different attention/FFN inputs into one common projection. It does not repack at
runtime.

The same layout does not mean that the GR and the mHC formulas are the same. The existing GR contract
and weight binding stay independent.

## 2. The five complete Ops

The signatures below are semantic notation. The inputs come before the semicolon and the outputs
after it. Each compute entry also takes the caller workspace and a `CudaExecutionView`.
`W={F,base,scale,gamma}` means the bound weights of one site. `P` means the semantic parameters of
the [numerical policy](../../maintainer/glm5next-quantization.md#51-the-current-fixed-profile). The
actual C++ declaration must express all these inputs and outputs in full. It must not expose kernel
tiling, a private stage or the model schedule.

```text
mhc_expand_pre(embedding, W, P;
               residual, block_input, post, mix)

mhc_pre(residual, W, P;
        block_input, post, mix)

mhc_post(residual, block_output, post, mix;
         updated_residual)

mhc_post_pre(residual, block_output, post, mix, next_W, next_P;
             updated_residual, next_block_input, next_post, next_mix)

mhc_post_mean_norm(residual, block_output, post, mix, gamma_final, eps_final;
                   hidden)
```

### 2.1 Entry effects

| Entry | Complete effect |
|---|---|
| `mhc_expand_pre` | copies the embedding exactly into four streams, then generates the normalized branch input and `q/C` from the current W |
| `mhc_pre` | consumes the complete residual and runs coefficient generation, Sinkhorn, collapse and the branch norm |
| `mhc_post` | consumes the complete branch output and the old `q/C` and runs the mixing and the BF16 residual update |
| `mhc_post_pre` | after the current post, runs the complete pre of the next site on the actual BF16 new residual |
| `mhc_post_mean_norm` | the last post, the BF16 residual boundary, the four-stream mean and the final norm; it outputs only hidden |

`mhc_pre` does not output `p`. It uses `p` only inside this call. `post/mix` are what is kept across
branches, until the matching post has consumed them. The public post takes coefficients that are
already activated or iterated and applies them by value. It does not run sigmoid again. It does not
run Sinkhorn again. It does not require the row and column sums of C to be exactly 1.

`post_pre` keeps the old and new W and the coefficients explicitly apart. The old coefficients can
only finish the post of the old site. Only the new residual and `next_W` can generate the next group
of coefficients. The choice of the adjacent site belongs to the model call order.

### 2.2 Valid domain, writes and alias

- `T` is any positive physical width, limited by the range that views and storage can represent.
  `B=1..8`. There is no separate decode/verify/prefill mathematical mode. A call with no valid column
  does not invoke this Op.
- Every tensor is contiguous and the base address is aligned to at least 16 bytes. The shape must
  match section 1. The weights and the mathematical inputs use valid model values. The Op does not
  read a device coefficient to validate against the Host.
- Every entry writes all of its output columns in full. The Op takes no valid length and no padding
  mask. The caller gives valid finite input for the physical padding columns and manages the valid
  columns at the later consumer. mHC publishes no historical state.
- For `mhc_post` and `mhc_post_pre`, `updated_residual` can be the exact same view as `residual` or
  fully disjoint. Partial overlap is forbidden. An in-place execution must still compute every new
  value from the complete old four streams.
- Apart from that special case, the outputs do not overlap any input, any weight or any other output.
  The old and new `post/mix` buffers must not alias. A residual modified in place must not overlap
  another read-only input either.
- Read-only inputs can share backing when the semantics and the shape allow it. Every input not
  declared mutable stays unchanged.
- The residual of `mhc_expand_pre` is an exact BF16 word copy. The other mathematical outputs are
  accepted at the common mathematical precision. They are not promised to be bitwise identical
  across a fused and a separate execution.

### 2.3 Execution resources

Every entry uses a caller-owned, call-scoped workspace that does not overlap any operand. The
matching capacity query reads only Host metadata. It supports an address-free profile before
startup. It covers every legal T/B in the queried range and the private stages and alignment needs of
the complete call. It must not report only the need of one internal kernel. A route that needs no
workspace can report zero. It must not use that to narrow the public support domain.

An entry submits to the given execution stream and supports Graph capture and replay that read the
current content at a fixed address. It allocates no hidden device memory, does no Host readback,
starts no communication and owns no request or long-term state. The precision policy, the
mathematical parameters and the alias rules do not change with a private route. The kernel count, the
tiles, the reduction decomposition and the scheduling thresholds belong to the implementation. They
are not a new public mode.

## 3. Layer order and fusion organisation

```text
embedding / replaced media features
  → mhc_expand_pre(attention 0)
  → attention 0
  → mhc_post_pre(attention 0 → FFN 0)
  → FFN 0
  → mhc_post_pre(FFN 0 → attention 1)
  → ...
  → the last FFN
  → mhc_post_mean_norm
  → Main hidden → LM head / MTP conditioning
```

The 90 sites correspond to one entry preparation, 89 adjacent `post_pre` calls and one final close.
This is the logical Op organisation of the complete Main computation. It does not promise that one Op
launches one kernel. The separate `pre/post` support calls that need an independent boundary and
their mathematical verification. The public fusion describes the complete combination directly and
does not require those separate Ops to be called first.

The optimisation follows these data dependencies:

1. The full-width sum of squares and the 24-row projection share the residual read. They do not
   generate a 16384-wide normalized tensor. They do not generate FP32 logits through the BF16 output
   of an ordinary Linear.
2. The small gate, the 4×4 softmax and all Sinkhorn steps complete on chip in sequence. This avoids a
   device-memory round trip every round.
3. After the post forms the BF16 new residual, feed it directly to the next projection, statistic and
   collapse where possible. At the same time keep the residual that the next branch still has to read
   after it returns.
4. Fuse collapse with its norm. Do not materialize a separate low-precision merge vector. Fuse the
   mean and the norm at the end in the same way.
5. Merge the initial copy with the first group of pre. At the end, form the final BF16 four streams on
   chip as the contract requires, without writing them back.

For a small M, evaluate a fused GEMV or a blocked reduction. For a large M, evaluate a narrow-output
GEMM with BF16 operands and FP32 accumulation, plus the post-processing. Route selection must be
verified against the complete Op. Do not turn a single Sinkhorn step or a GEMM partial into a public
Op the model can call.

There is no kernel result and no whole-chain performance result yet. The coefficient projection of
one site at one token is 393216 MAC. The collapse is 16384 MAC. The residual matrix mixing is 65536
MAC. The injection adds another 16384 multiplications. These are geometry counts only. They exclude
the norm, the non-linearities and the iteration. They do not represent actual latency or a fusion
gain.

## 4. Complete branch output and execution resources

The local Op always consumes the complete BF16 y. After the local KDA/FFN contribution reduction and
the MLA owner output gathering finish, the two ranks each run mHC. The
[parallel design](parallel.md#1-the-first-parallel-split) maintains the replicated placement and the
communication handoff. This family takes no rank and no communicator.

ModelShard owns the immutable weights. Parameters borrows the bindings. The RankProgram frame owns
the residual, the block input/output, the coefficients, the workspace and the Graph. The Session
decides the request and column mapping. For the buffer accounting, see
[execution resources](parallel.md#4-mhc-execution-resources). The mHC activation is not historical
state. It does not enter the two persistent pools. MTP keeps only the final Main hidden that it
actually needs.

## 5. Acceptance and performance evidence

### 5.1 The independent mathematical oracle

Every entry computes the complete formula with an independent FP64 oracle that reads the values the
public inputs actually represent. Do not build the reference by chaining production Ops or by
reproducing the GPU staging. The fixed Sinkhorn count, the matrix direction, the epsilon position and
the residual representation boundary must be written into the oracle directly.

The reference at the mathematical exit keeps the high-precision result for the error comparison. The
BF16 residual boundary declared in `post_pre` and in the final fusion is expressed separately. The
other private and public final outputs get no extra rounding step just to match a route. Besides the
complete fused formula check, compute the next pre reference independently from the BF16 updated
residual that actually returns, and check that the next stage really consumes that representation. Do
not look only at the error of the new residual and ignore where the coefficients came from.

Use an exact check for the initial copy, the unmodified inputs and the out-of-range region. Use the
error criterion of the same public semantics for the floating-point computation. Do not require
different reduction routes to agree bit for bit. Fix the error criterion from the public exit and the
mathematics before the implementation evaluation. Do not loosen it in reverse from a candidate
output. vLLM parity and the model output provide supplementary evidence only.

### 5.2 Failure types that must be covered

| Scenario | Problem to detect |
|---|---|
| Clearly asymmetric C, different old-stream inputs | a transposed `C[old,new]` |
| Large magnitude difference across the four streams | grouped RMS used instead of full-width RMS, or collapse reading normalized X |
| Non-trivial pre/gamma | dividing by 4 too many times, normalizing pre, using an offset gamma |
| Zero input, extreme logits, saturated gate | wrong epsilon position, softmax overflow, wrong clipping |
| Unbalanced mixing logits | wrong iteration count or row/column order, wrong reliance on strict doubly-stochasticity |
| Residual near a BF16 rounding boundary | fused pre using an unrounded new residual |
| Clear rounding difference in collapse/mean | an extra BF16 intermediate boundary inserted |
| Final post + mean + norm | the last injection/mixing skipped, the norm missing or applied twice |
| Multiple tokens, multiple batches, column reorder | streams crossed, columns crossed, requests mixed |
| Exact in-place and disjoint output | the old four streams overwritten before they are read, partial overlap not rejected |
| Graph replay with changed content at a bound address | stale input or coefficients cached, hidden allocation, Host synchronization |

Cover the five public entries and every reachable private route. T includes 1, both sides of a route
boundary and representative internal values. B covers representative values in 1..8 and the upper
bound. Verify the fused route and the separate route each against the mathematical oracle, not only
against each other. Later cross-rank integration must also check that the complete y is consumed
exactly once and that the input words and row order agree on both sides.

### 5.3 Performance scope

Use the separate pre/post combination on Spark with the same geometry, precision and mathematics as
the fusion control. Cover small-batch decode, multi-token verify and large-window prefill. Compare the
complete `pre`, `post_pre`, the final close and the combined chain. State the hardware, the toolchain,
T/B, the Graph/cold condition, the workspace, the distribution and the worst change.

A speedup of Sinkhorn alone or of a GEMM alone does not replace the result of the whole Op. A local Op
result must not be claimed as a two-host model end-to-end gain. A cross-rank performance evaluation
must count the output reduction/gathering, the pack/unpack and the handoff to the later consumer.
Today there is only structural and capacity derivation. Nothing is implemented and no performance
comparison has run.

## 6. Takeover and retirement conditions

The mathematical and numerical sources are maintained in the long-term model and numerical documents
respectively. The adjacent post/pre organisation follows the
[GLM call order](https://github.com/vllm-project/vllm/blob/00b7847c8036b667742b4efb21aab1de51fd4721/vllm/models/glm5next/common/model.py)
of local vLLM revision `00b7847c8036b667742b4efb21aab1de51fd4721`. Handle a backend rounding
difference under the numerical policy. It is not directly an NInfer oracle.

After the five entries and their tests land, the actual Op headers and the tests take over the formula
combination, the alias, the write domain and the precision acceptance. After the model integration,
the execution code and the current execution note take over the layer order and the resources. The
unimplemented parts stay in this document. After everything is taken over, delete the design. Do not
keep another specification of a to-be-implemented interface in the maintainer directory.
