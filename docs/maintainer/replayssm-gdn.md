# ReplaySSM: raw-input replay of GDN speculative state

This document discusses ReplaySSM for Gated DeltaNet (GDN) in short-window speculative decoding:
target verify keeps the existing per-token recurrence, but it does not store the complete recurrent
state of each verify position; it records only the raw inputs that drive the state transitions. Once the final accept length is known, it replays the accepted prefix
from the committed checkpoint in order and obtains the next round's state.

The key question here is not only whether the record can rebuild the state. GDN recurrence admits several algebraically equivalent forms,
but a different form changes the normalization, the reduction, the multiply-add association, the Tensor Core precision and the cast boundary. The state is a persistent
value across rounds; even a small rebuild error enters the next round and keeps propagating. The core requirement of this technique is therefore:

> A replay fold must execute the same finite-precision state transition as the verify recurrence, not merely compute an
> equivalent formula in the real domain.

The document covers, in order, the mathematical definition of the state and the record, accepted-prefix replay, the sources of floating-point drift, the conditions for a closed-loop
bitwise clone, the causal-conv history, and the space and compute characteristics of the current Qwen3.5 model instances.

The implementation and the contract of Record/Fold are in [`gdn_replay.h`](../../include/ninfer/ops/gdn_replay.h) and
[`record.cpp`](../../src/ops/linear_attention/gated_delta_net/replay/record.cpp),
[`fold.cpp`](../../src/ops/linear_attention/gated_delta_net/replay/fold.cpp). The model configuration decides the layer and head
counts, and the RankProgram reserves the record capacity for the enabled MTP, DFlash or DFlash2 windows.

---

## 1. The problem: speculative verify needs a selectable state prefix

### 1.1 The size of the GDN state

For each GDN layer and value head, the recurrent state is an FP32 matrix

\[
S\in\mathbb{R}^{V\times K}.
\]

The current instances use \(K=V=128\). The state of one value head holds 16,384 FP32 elements, that is 64 KiB.
Multiplied by all GDN layers and value heads, one complete recurrent state image is:

| Model | GDN layers | value heads | One recurrent state |
|---|---:|---:|---:|
| Qwen3.6/3.8-27B | 48 | 48 | 144 MiB |
| Qwen3.6-35B-A3B | 30 | 32 | 60 MiB |

### 1.2 Snapshot baseline

Let one target verify process \(T\) inputs and obtain in order from the committed state \(S_0\)

\[
S_1,S_2,\ldots,S_T.
\]

In the end only one prefix \(S_m\) is committed. The direct way to support rollback is to store every state:

~~~text
S0 ── input 1 ──> S1 ── input 2 ──> ... ── input T ──> ST
                    │                    │                  │
                    └──────── store the complete state trajectory ──┘

final commit length = m: select Sm
~~~

This writes one \(V\times K\) FP32 matrix for each verify position. Each added window column adds the capacity and the write traffic of one complete
state.

### 1.3 Raw-input ReplaySSM

Raw-input ReplaySSM keeps only:

\[
\text{committed checkpoint }S_0
\quad+\quad
R_1,R_2,\ldots,R_T,
\]

where \(R_t\) is the compact record of the \(t\)-th state transition. Verify still produces all outputs, but it does not write
\(S_1,\ldots,S_T\) to persistent storage. Once the final \(m\) is known, it computes

\[
S_m=F_{\mathrm{fp}}\left(
F_{\mathrm{fp}}\left(
\cdots F_{\mathrm{fp}}(S_0,R_1),R_2
\right)\cdots,R_m
\right),
\]

and publishes only this one state.

Here \(F_{\mathrm{fp}}\) deliberately denotes the actual finite-precision transition, not an abstract real-number formula. Later sections explain why this distinction
decides whether the reconstructed state can align with the verify trajectory.

---

## 2. GDN recurrence and the raw transition record

### 2.1 The logical order that the finite-precision path follows

The layer and value-head subscripts are omitted below. Let

\[
q_t^{raw},k_t^{raw}\in\mathbb{R}^{K},
\qquad
v_t\in\mathbb{R}^{V}.
\]

The query and the key first take an L2 normalization:

\[
\bar q_t=
\frac{q_t^{raw}}
{\sqrt{\sum_i(q_{t,i}^{raw})^2+\epsilon}},
\qquad
\bar k_t=
\frac{k_t^{raw}}
{\sqrt{\sum_i(k_{t,i}^{raw})^2+\epsilon}}.
\]

Let

\[
\alpha_t=\exp(g_t).
\]

One GDN transition can be written in the following order:

\[
S_t^{decay}=\alpha_tS_{t-1},
\]

\[
r_t=S_t^{decay}\bar k_t,
\]

\[
u_t=\beta_t(v_t-r_t),
\]

\[
S_t=S_t^{decay}+u_t\bar k_t^{\mathsf T},
\]

\[
y_t=c\,S_t\bar q_t,
\qquad
c=\frac{1}{\sqrt{128}}.
\]

Merging them gives the usual compact form:

\[
u_t=\beta_t
\left(v_t-\alpha_tS_{t-1}\bar k_t\right),
\]

\[
S_t=\alpha_tS_{t-1}+u_t\bar k_t^{\mathsf T}.
\]

The two formula sets are identical in the real domain; the first set also shows the operation boundaries that must stay consistent in the actual transition:
decay the state first, then read the correction from the decayed state, and finally do the rank-one update.

### 2.2 Grouped q/k heads

If \(H_v\) value heads share \(H_{qk}\) q/k heads, let

\[
G=H_v/H_{qk},
\qquad
h_q=\lfloor h/G\rfloor.
\]

Value head \(h\) uses q/k head \(h_q\). Each value head has its own \(S\), \(v\), \(g\) and
\(\beta\), and one group shares the raw \(q/k\).

### 2.3 The sufficient inputs of a state transition

Given \(S_{t-1}\), computing \(S_t\) needs only

\[
R_t^{raw}=
\left(k_t^{raw},v_t,g_t,\beta_t\right).
\]

The fields act as follows:

- \(k_t^{raw}\): re-executes the same normalization, state read and rank-one write as verify;
- \(v_t\): recomputes the corrected value;
- \(g_t\): forms the decay through the same \(\exp\) path;
- \(\beta_t\): forms the correction;
- \(q_t\): used only for this round's output readout, changes no state, and therefore does not enter the replay record.

For the usual Qwen3.6 numeric boundaries, raw \(k/v\) are BF16 represented values, \(g/\beta\) are FP32
represented values, and the checkpoint is FP32. The raw record is a lossless side copy of these actual transition inputs,
not a fresh derivation from the hidden state or the upstream projection.

---

## 3. Accepted-prefix replay

### 3.1 Verify input, output and commit length

Let the speculative drafter give \(D\) draft tokens. Target verify processes

\[
T=D+1
\]

inputs:

~~~text
input 1       = the anchor not yet processed at the start of the round
input i + 1   = draft i, 1 <= i <= D
~~~

If the first \(A\) drafts match the target, the target licenses

\[
p=A+1
\]

tokens: the \(A\) accepted drafts plus one correction/bonus token.

The inputs that must advance the state are exactly the first \(p\) verify inputs, that is the old anchor plus the first \(A\) accepted
drafts. The correction/bonus produced last has not yet run as an input; it becomes the next round's anchor.

The final output boundary may keep only the first \(m\) of the licensed outputs, so the commit length of the state is

\[
0\le m\le p.
\]

For example, with \(D=5,A=3\), \(p=4\). The four candidate state transitions are the old anchor plus the first three accepted
drafts, and the four outputs are the three accepted drafts plus the correction. If the final \(m=2\), only the old anchor plus the
first accepted draft is replayed; the second output is the new pending anchor.

### 3.2 Verify: produce outputs and raw records

Verify starts from the committed checkpoint and runs the original recurrence in order:

~~~text
S_verify <- S0

for t = 1 .. T:
    q <- normalize(q_raw[t])
    k <- normalize(k_raw[t])
    alpha <- exp(g[t])
    S_verify <- alpha * S_verify
    u <- beta[t] * (v[t] - S_verify * k)
    S_verify <- S_verify + outer(u, k)
    y[t] <- scale * S_verify * q
    record[t] <- (k_raw[t], v[t], g[t], beta[t])
~~~

Here \(S_{verify}\) is the transient trajectory that produces the current window outputs. After verify ends, it is not
published as the committed state; the persistent checkpoint \(S_0\) stays unchanged.

### 3.3 Fold: replay only the accepted prefix

Once the final \(m\) is known, the fold starts from the same \(S_0\):

~~~text
S_fold <- S0

for t = 1 .. m:
    k <- normalize(record[t].k_raw)
    alpha <- exp(record[t].g)
    S_fold <- alpha * S_fold
    u <- record[t].beta * (record[t].v - S_fold * k)
    S_fold <- S_fold + outer(u, k)

publish S_fold
~~~

At \(m=0\) the state does not change at all. The rejected suffix \(R_{m+1:T}\) is never read by the fold.

One head still has at most \(m\) sequential transitions, but different layers, value heads and batch rows are independent.
The parallelism of Qwen3.6 comes from its 30/48 GDN layers, its 32/48 value heads and several active rows, while a single-row
token loop has only 6 or 16 iterations.

### 3.4 Correctness of the accepted prefix

Write the finite-precision transition that verify uses as \(F_{\mathrm{fp}}\):

\[
S_t^{verify}=
F_{\mathrm{fp}}(S_{t-1}^{verify},R_t),
\qquad
S_0^{verify}=S_0.
\]

If the fold uses exactly the same \(F_{\mathrm{fp}}\):

\[
S_t^{fold}=
F_{\mathrm{fp}}(S_{t-1}^{fold},R_t),
\qquad
S_0^{fold}=S_0,
\]

then induction on \(t\) gives directly:

\[
S_t^{fold}=S_t^{verify},
\qquad 0\le t\le m.
\]

Therefore

\[
S_m^{fold}=S_m^{verify}
\]

can be a bit-exact conclusion in finite precision, not merely an equivalence in the real domain. It holds on the premise that both paths call the same
deterministic floating-point transition and consume the same record bits.

The comparison is against the recorded prefix of the same physical verify block. It does not require a second, shorter verify, a different
prefill chunking, or another numeric implementation to produce the same inputs, logits or state. The computation before the record already fixed this round's inputs;
the fold must faithfully commit the trajectory that these inputs form.

---

## 4. The numerical core: why algebraic equivalence still produces state drift

### 4.1 The compute path in finite precision

For the same inputs, one GDN transition can be written in the real domain as

\[
S_t=
\alpha_tS_{t-1}
+\beta_t\left(v_t-\alpha_tS_{t-1}\bar k_t\right)
\bar k_t^{\mathsf T},
\]

It can also be written as

\[
S_t=
\alpha_tS_{t-1}
\left(I-\beta_t\bar k_t\bar k_t^{\mathsf T}\right)
+\beta_tv_t\bar k_t^{\mathsf T}.
\]

The two expressions define the same real-number map, but the floating-point state is the result of a concrete evaluation program. It depends on:

- the reduction tree, the sqrt and the division of the raw key normalization;
- the \(\exp\) path from \(g_t\) to \(\alpha_t\);
- the accumulation order and the operand precision of \(S\bar k\);
- the order of the state decay, the correction and the rank-one update;
- the FMA contraction, the tile decomposition and the state-store boundary.

Algebraic equivalence does not by itself make the represented states equal. Once replay changes any of these steps, it defines a new finite-precision
transition; a single-step difference enters the committed state and becomes an input of the next round's recurrence.

### 4.2 The raw record and the replay path

Raw record

\[
R_t=(k_t^{raw},v_t,g_t,\beta_t)
\]

fixes the represented inputs of the transition. The complete replay state is also decided by the finite-precision map \(F_{\mathrm{fp}}\):

\[
S_t=F_{\mathrm{fp}}(S_{t-1},R_t).
\]

If the fold actually runs another evaluation program \(\widetilde F_{\mathrm{fp}}\), then even when the two correspond to the same formula in the real domain,

\[
\widetilde F_{\mathrm{fp}}(S,R)
\ne
F_{\mathrm{fp}}(S,R)
\]

it can still hold. The raw record fixes the inputs identity; verbatim replay fixes the transition identity.

### 4.3 Verbatim closed-loop replay

The record and the compute duties of closed-loop replay are:

| Item | Requirement |
|---|---|
| raw \(v\) | store the represented bits that the verify recurrence actually consumes |
| raw \(k\) | store the represented bits before normalization |
| \(g\) | store the FP32 log-decay gate that verify already obtained; do not recompute it upstream |
| \(\beta\) | store the FP32 correction gate that verify already obtained |
| normalization | the same epsilon, reduction tree, sqrt/division form |
| decay | the same order of \(\exp\) and state multiply |
| correction | recompute \(u\) from the replay state at the current position |
| rank-one update | the same operand precision, FMA contraction and tile decomposition |
| state store | the same FP32 represented boundary |

Closed loop means that every corrected value is computed on the spot from the current replay state, the raw \(v/k\) and the gate bits.
This makes every fold step enter the same transition as verify, and it makes the represented state of the previous step the direct input of the next
correction.

SGLang observed this numeric boundary during the Kimi K3 bring-up: an early fold recomputed the gate from another compute path, and although the
outputs still looked normal at the time, the recurrent state had already begun to drift. After it recorded the gate values that verify produced and reproduced the recurrence
in the fold, the state reached bit-identical equality with the recurrent baseline. The SGLang GDN exact fold likewise aligned the
state tile, the division-form L2 normalization and the operation order of the verify branch.

### 4.4 How state drift propagates

First consider two trajectories that use the same represented inputs and the same real-number GDN formula, but differ in the starting state

\[
\Delta S_t=\widetilde S_t-S_t.
\]

The recurrence gives

\[
\Delta S_{t+1}
=\alpha_{t+1}\Delta S_t
\left(I-\beta_{t+1}\bar k_{t+1}\bar k_{t+1}^{\mathsf T}\right).
\]

If the reconstructed transition also introduces a local floating-point deviation \(E_{t+1}\), the error model becomes

\[
\Delta S_{t+1}
=\alpha_{t+1}\Delta S_t
\left(I-\beta_{t+1}\bar k_{t+1}\bar k_{t+1}^{\mathsf T}\right)
+E_{t+1}.
\]

The corresponding state readout difference is

\[
\Delta y_{t+1}=c\,\Delta S_{t+1}\bar q_{t+1}.
\]

The outputs that the verify trajectory already produced in this round are not changed retroactively by the fold. The real problem appears in the next round:
the attention/KV, the hidden and the published outputs come from the verify trajectory, while the GDN checkpoint comes from another numeric
trajectory. Even when a single-round output cast hides the difference for a time, the persistent state still takes part in the correction and the readout
over and over in later tokens.

The decay \(\alpha<1\) can attenuate part of the old error, but

\[
I-\beta\bar k\bar k^{\mathsf T}
\]

is a direction-dependent update, and every round can inject a new \(E_t\). Therefore the claim that the gate attenuates the error cannot replace direct verification of the committed
state.

### 4.5 Two levels of correctness

ReplaySSM state reconstruction should distinguish two criteria:

1. **Mathematical correctness**

   Starting from the represented BF16/FP32 inputs and the FP32 \(S_0\), against an independent FP32/FP64 GDN oracle,
   the output and the final state meet the specified error.

2. **Finite-precision clone**

   For the same \(S_0\), the same raw record bits and the same accepted prefix, the FP32 state after the fold equals the corresponding trajectory of
   the same physical verify block bit for bit on every element.

When verifying a bitwise clone, the final text or the BF16 output parity is not enough. Direct evidence should cover:

- the exact bit copy of the raw \(k/v\) and \(g/\beta\) record;
- the committed checkpoint does not change at all during verify;
- the exact comparison of the final state at \(m=0,1,T\) and intermediate \(m\);
- a long chain of replays with different accepted-length sequences;
- very small key norms, extreme gate values and several q/k-to-value-head groups;
- the committed state still stays unchanged after the rejected suffix is rewritten.

If the fold and verify take different arithmetic paths, the result should state a numerical tolerance against the independent oracle, and it must not be called a
bitwise clone.

---

## 5. The finite-window state of the causal convolution

A GDN block usually carries a causal depthwise convolution before the recurrence. If the convolution width is \(W\), the persistent
history holds only the last \(W-1\) projection columns:

\[
H_0=[p_{-(W-2)},\ldots,p_0].
\]

Verify produces

\[
p_1,p_2,\ldots,p_T.
\]

When the final commit length is \(m\), the correct history is

\[
H_m=
\operatorname{tail}_{W-1}
\left(H_0\mathbin\Vert[p_1,\ldots,p_m]\right).
\]

So each verify position needs to record only one represented projection history column, instead of storing the whole
\(W-1\)-column window:

- \(m=0\): the history does not change;
- \(0<m<W-1\): keep some old columns and append the accepted columns;
- \(m\ge W-1\): use the last \(W-1\) columns of the accepted prefix.

As long as the record is the same BF16 represented column that the baseline would write to the history, this commit is an exact gather,
with no recurrent reduction and no floating-point reassociation. Qwen3.6 uses \(W=4\), so one record column is
\(1/3\) of a three-column snapshot.

---

## 6. Space and compute characteristics

### 6.1 General formulas

Let:

- \(L_g\): the GDN layer count;
- \(H_q\): the q/k head count;
- \(H_v\): the value head count;
- \(K,V\): the key/value dimension;
- \(C_p\): the causal-conv projection channels;
- \(W-1\): the conv history columns;
- \(T\): the verify window.

The byte size of one FP32 recurrent state is

\[
R=4L_gH_vVK.
\]

The byte size of one BF16 conv history is

\[
Q=2L_gC_p(W-1).
\]

If raw \(k/v\) are BF16 and \(g/\beta\) are FP32, the GDN record per token is

\[
P_{gdn}=
2L_g(H_qK+H_vV)+8L_gH_v.
\]

The conv column record per token is

\[
P_{conv}=2L_gC_p.
\]

The complete replay log of length \(T\) is

\[
P_{record}(T)=T(P_{gdn}+P_{conv}).
\]

The window-related capacity of the snapshot trajectory is

\[
T(R+Q),
\]

and that of raw ReplaySSM is

\[
T(P_{gdn}+P_{conv}).
\]

### 6.2 Current Qwen instance sizes

| Model | \(L_g\) | \(H_q\) | \(H_v\) | \(K/V\) | \(C_p\) | \(W\) |
|---|---:|---:|---:|---:|---:|---:|
| 27B | 48 | 16 | 48 | 128/128 | 10,240 | 4 |
| 35B-A3B | 30 | 16 | 32 | 128/128 | 8,192 | 4 |

The corresponding per-token state/record sizes are:

| Model | recurrent image | conv history | raw GDN record | conv record | record total |
|---|---:|---:|---:|---:|---:|
| 27B | 144.000 MiB | 2.8125 MiB | 0.767578 MiB | 0.9375 MiB | 1.705078 MiB |
| 35B-A3B | 60.000 MiB | 1.40625 MiB | 0.358887 MiB | 0.46875 MiB | 0.827637 MiB |

The ratio of one raw record to one recurrent+conv snapshot is:

| Model | snapshot/position | raw record/position | Size ratio |
|---|---:|---:|---:|
| 27B | 146.8125 MiB | 1.705078 MiB | about 86.1× |
| 35B-A3B | 61.40625 MiB | 0.827637 MiB | about 74.2× |

The record capacity of typical verify windows is:

| Model and window | raw GDN records | conv records | Total |
|---|---:|---:|---:|
| 27B, \(T=6\) | 4.605469 MiB | 5.625000 MiB | 10.230469 MiB |
| 27B, \(T=16\) | 12.281250 MiB | 15.000000 MiB | 27.281250 MiB |
| 35B-A3B, \(T=6\) | 2.153320 MiB | 2.812500 MiB | 4.965820 MiB |
| 35B-A3B, \(T=16\) | 5.742188 MiB | 7.500000 MiB | 13.242188 MiB |

### 6.3 Compute shape

Raw-input replay keeps the serial recurrence of verify and adds at most \(m\) transitions at commit:

| Item | Snapshot baseline | Raw-input replay |
|---|---|---|
| verify state read | one checkpoint | one checkpoint |
| verify full-state writes | \(T\) copies | 0 |
| verify record writes | 0 | \(T\) small records |
| commit work | select a snapshot | replay \(m\) transitions, write one state |
| rollback | select the matching snapshot | read only the accepted record prefix |
| persistent numerical path | verify recurrence | closed-loop clone of the verify recurrence |

The token dimension of this case is very short: MTP reaches at most \(T=6\), and DFlash/DFlash2 at most \(T=16\). Along one layer, value head
and batch row, the fold has \(m\) sequential transitions; different layers, heads and batch rows are independent of each other.
The overall compute shape is therefore a large number of mutually independent short recurrences. The fold work grows linearly with the accepted length \(m\),
the record traffic grows linearly with the verify length \(T\), and the end writes only one committed state.

The technique is first of all a capacity and state-traffic optimization. Verify writes \(T\) fewer large states at the cost of writing small records and doing one more short prefix fold
after the accept. End-to-end latency depends on the state traffic, the accept length and the available
layer/head/batch parallelism, and it cannot be derived directly from the space compression ratio.

---

## 7. Core conclusions

The state representation of GDN speculative ReplaySSM is

\[
\text{one committed checkpoint}
+
\text{one short raw transition log}.
\]

Its correctness depends on these invariants:

1. Verify produces the outputs and the raw records from the committed checkpoint, but it does not modify the checkpoint;
2. the record stores the raw pre-normalization \(k\), the raw \(v\) and the \(g/\beta\) bits that verify itself produced;
3. the final commit length also defines the record prefix to replay;
4. the rejected suffix is not read by the fold;
5. the fold recomputes every corrected value in closed loop;
6. the fold and verify use the same normalization, gate, reduction, operation order and state-store boundary;
7. the committed state is compared directly with the corresponding state prefix of the same physical verify block.

The raw inputs decide what can be replayed; the verbatim recurrence decides whether the replay gives the same finite-precision state.
The first solves the snapshot capacity, and the second stops cross-round state drift; together they form short-window GDN ReplaySSM.

---

## References

- [SGLang and Miles Add Day-0 Support for Kimi K3](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support): raw-input replay, stored gates and a bit-identical state fold.
- [SGLang GDN exact fold](https://github.com/sgl-project/sglang/blob/bc285b2064c0373227cfb6ada77a37e7b8c43510/python/sglang/kernels/ops/attention/fla/gdn_replayssm_spec_fold.py): a closed-loop fold aligned with the verify recurrence.
- [ReplaySSM: Cache SSM Inputs, Not State](https://tridao.me/blog/2026/replayssm/): the basic idea of caching SSM transition inputs and folding the state when needed.
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464): the mathematical source of GDN and the gated delta-rule.
