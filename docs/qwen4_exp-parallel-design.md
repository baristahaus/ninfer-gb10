# Qwen4Exp Dual-Spark Parallel Strategy (D0)

Status: the first-version TP2/EP2 parallel strategy has converged; it is not implemented yet. This document maintains the placement, computation and state handoff of the first Qwen4Exp TP2/EP2 model plan; the single-column follow-up optimizations are not part of the first-version implementation requirements. The execution framework and lifecycle are defined by
[the distributed execution design](distributed-execution-design.md).
The model mathematics and storage representation follow [model and weights](maintainer/qwen4_exp-model.md),
[execution and state](maintainer/qwen4_exp-execution.md) and the Op contracts.

The target is two homogeneous DGX Sparks, with one GPU rank per machine. It uses the existing Flash Next quantized artifacts: Main routed experts are NVFP4 / AllowA4, and MTP routed experts are block-FP8 / AllowA8; the other projections keep their respective A16 contracts, and the KV representation is independent. Spark uses GPU model computation and Device State/KV.

## 1. Confirmed First-Version Strategy

| Part | First-version choice | Implementation dependency |
|---|---|---|
| GDN / sparse GQA | TP2 by head, the O projection produces a local contribution | Merged parameter slicing, convolution reorder, local state and Op geometry |
| Residual / HC | All M use the replicated path | Full four-stream residual, combine/norm after block reduction |
| Indexer | Four query heads, computation, cache and raw tail are all replicated | Shared decision profile, full append and consistent selection |
| Router / shared | Router replicated, shared intermediate TP2 | Global route decoupled from the local bank, independent shared geometry |
| Routed experts | EP2, 256 complete experts per layer, static mapping fixed at startup | Main interleaves by the existing per-layer priority; MTP splits by global ID parity |
| Embedding / head | Embedding replicated, head TP2 by vocabulary | Candidate exchange, constraints, penalty, MTP acceptance and scoring |
| PLE | Sharded by table row, lookup in advance; later projections and convolution replicated | About 13.56 GiB table per machine, merged lookup output, overlapped with the preceding Main computation |
| MTP | QSA TP2 / experts EP2, shares the sharded head; the two stem FCs replicated | Device-side draft handoff, Main verify and history alignment |
| Vision | Encoder weights replicated, independent encoding units assigned to different ranks | Encoding boundary, feature order, cache and multimedia input distribution |
| State | Large state sharded with the head, small public fields replicated | Common frontier for ordinary, verify, checkpoint/restore |

`T` is the number of physical input columns per row, `B` is the compact batch row count, and `M=T*B`. The layout and execution strategy consume the actual M, the per-row effective length and the context; they do not choose the dispatch by the prefill/decode/verify name.
This round reviews the mathematics, interfaces, loading, communication, state and capacity from the source and the existing single-machine measurements; the two-machine latency has not been measured yet.
The first version does not enable HC token/lowrank TP, Indexer query/history division, MTP FC TP or single-owner MTP; the candidate analysis in the later sections is kept as the basis for later optimization and does not block the current interfaces and loading plan.
PLE table sharding is in section 7; the complete computation and Graph organization are in section 10.

## 2. Local Geometry and Loading

Matrix shapes use `[N,K]`. The table below uses the plan with the fully replicated Indexer:

| Role | Geometry per rank |
|---|---|
| GDN heads | Q/K 8 each, V 24, dimension=128; each Q/K corresponds to three consecutive V heads |
| GDN QKVZ | `[8192,2560]`, Q 1024, K 1024, V 3072, Z 3072 |
| GDN AB / output | `[48,2560]` / `[2560,3072]` |
| GDN convolution | four taps, each with the 5120 channels of the local Q/K/V |
| QSA heads | Q 12, KV 1, dimension=256 |
| Main QSA Q/gate | `[6144,2560]`, Q/gate 3072 each |
| Main QSA KV/index | `[1152,2560]`, K/V 256 each, index-Q 512, index-K 128 |
| Main/MTP QSA output | `[2560,3072]`, keeps its original representation |
| MTP QSA joint input | `[7296,2560]`, keeps the full index-Q/index-K; assembled per [persistent layout](maintainer/qwen4_exp-model.md#4-merged-matrices-and-persistent-layout), K/V/index-K share a contiguous prefix |
| Router / shared score | `[512,2560]` / `[1,2560]`, replicated |
| Shared GU / down | `[640,2560]` / `[2560,320]` |
| Merged router/score/shared GU | `[1153,2560]` if a single projection is kept |
| Routed experts | 256 complete experts; each GU `[1280,2560]`, down `[2560,640]` |
| LM head | `[124160,2560]`; the global vocabulary ID ranges do not overlap |

Config keeps the global geometry; ParallelPlan produces the local parameters, the state schema and the communication description. Merged matrices are sliced by logical role and then a local parent is built; the convolution tap/channel row order is rebuilt with the local Q/K/V.
The Q8 output K split keeps the G32 encoding groups; expert EP keeps the codes/scales/divisor of the complete expert.
Loading materializes the local encodings and stable views directly; it does not load the complete GPU model first and then slice it, and it does not requantize at execution time.

The mathematical closure of this part is settled: GDN normalization, convolution and recurrence all complete inside a full head, and the 8 Q/K heads are distributed together with the corresponding 24 V heads; A_log, dt_bias, AB and state use the same V-head range. Each QSA KV head originally serves 12 Q heads, and TP2 splits exactly these two groups.
The in-head RMSNorm, RoPE, KV Hadamard and storage quantization keep the original dimension and introduce no reduction communication.

What needs to change is the real local parameters and execution geometry. The current GDN/QSA calls use the global Config, and the QSA Op contract still states 24/2 heads; slicing the weights alone cannot complete the integration. LocalParameters, the workspace query, prepare, attention, snapshot/record/fold and the State/KV schema must all consume the derived geometry.
The GDN QKVZ, AB and QSA joint parents cannot be cut in half by the total N directly; take each role separately and then concatenate them in the original role order. This round follows the work order of supporting the pure Linear geometry first and then connecting the split/fused outputs.

vLLM's QSA also uses Q/KV head TP and RowParallelLinear output, and it replicates KV when TP is greater than the KV head count. This practice supports the head division of this plan; its input encoding and private fusion cannot replace NInfer's handling of Q8 grouping, BF16 output and local state.

## 3. Residual and HC

In the first version each rank holds the complete `[2560,4,M]` residual, and HC norm, down/injection, up and mix run repeatedly. GDN/QSA and MoE each produce a `[2560,M]` local contribution, and combine/norm runs after the reduction completes. A normal Main layer corresponds to two main reduction boundaries, and the shared contribution joins the MoE boundary.

vLLM's HC down/up/injection are also replicated; its Qwen4Exp explicitly rejects `use_sequence_parallel_moe`. Therefore the HC token split for large M needs a complete new execution path, and the sequence-parallel switch of other models cannot be applied to this model directly.

The candidate is mathematically feasible: block local contribution → ReduceScatter token rows → complete single-token residual combine/norm/HC mix → AllGather `[2560,M]` input → the next TP/EP computation segment.
The four streams of each token stay in the same owner, and the residual and injection keep the same token partition across consecutive layers. GDN/QSA/experts still consume all tokens; only the local HC processes its own part.
This does not split the GDN time sequence and does not change the expert division.

Under TP2 the main RS/AG byte volume is comparable to an AR of the same size, but the call count changes from one to two. The HC for small M mainly reads the full weights; splitting by token does not reduce the weights each machine reads, so the benefit is very small; only large M has the value of sharing the computation.
The complete variant also has to solve three handoffs:

- PLE uses a nine-column history across tokens. It can keep the replicated residual before PLE and enter the token partition after PLE, avoiding a time halo or a full residual gather for this single layer.
- The final HC can continue per token, but Main's sample/conditioning must gather by the actual consumer. Ordinary generation takes only the last column; MTP history needs its paired Main conditioning column and cannot take only the last column.
- The valid columns of `[T,B]`, padding, token owner and output row order are fixed by the same variant; a flat token slice cannot be reinterpreted as a sequence. The necessary pack/unpack and communication buffers count toward the cost.

The other candidate splits by lowrank=320 into 160+160: local down/SiLU → up's 10240-dimension local sum → reduction → sigmoid/mix; the four-row injection is replicated. A single rank reads about 6.25 MiB less weight, but adds `10240*M` of communication and a new HC segmentation interface. This item is kept as a candidate to be tested and is not the first-version default layout.
`g` was originally an internal quantity of HC; its communication representation must satisfy the complete HC numerical contract and cannot directly use the BF16 rounding boundary of an ordinary Linear output.

The low-rank split cannot do sigmoid on the two machines separately and then add. Splitting by residual feature/stream also works, but it adds a reduction between norm, down or the final stream mix; its complete communication chain matters more than halving one Linear. There is currently no evidence that it beats the replicated path or the token partition for large M.

The Spark Graph/cold microbenchmarks from the existing migration stage give three-run GPU median ranges: HC mix at M=1 is about 56–58 μs, at M=64 about 58–59 μs,
and at M=4096 about 1.13–1.14 ms. Against the repository's sustained-read reference of 257.42 GB/s, a read volume of 6.25 MiB corresponds to
about 25 μs; this is an order-of-magnitude basis for judging the sharding potential, not a distributed benefit measurement.

## 4. QSA Indexer

Each rank replicates four index query heads, one raw index key, the compressed cache and the raw tail.
The main attention Q/K/V is split by head. At the maximum context, the Main 12-layer BF16 index cache
is about 192 MiB/rank/request.

The base path recomputes the complete Indexer and uses it directly for the local sparse GQA. vLLM also replicates
the index-Q/K projections and the index cache. What must be consistent here is the selection set and the cache content:
the precision tolerance of a common Op does not by itself guarantee that two different implementations select the same critical candidates.
The replicated region on the two homogeneous Sparks uses the same input, T/B, envelope, computation path and global-ID
tie-breaking; the current score kernel generates each score independently, and TopK selects by the full score/ID order,
so it does not depend on the completion order of floating-point atomic accumulation. Other sharded Ops still keep their own private dispatch.
When different devices or different numerical paths are used in the future, the plan must change to produce the selection once and distribute the common selection explicitly.

The large-M candidate divides by query, each machine still reads the complete index cache, and then the final block IDs/counts are exchanged.
Each query has at most 512 I32 IDs; the complete IDs at M=4096 are 8 MiB. The switch also depends on the context length;
the full-selection path for short contexts does not bear the repeated scoring cost.

The existing `qsa_prepare` appends the main KV, the compressed index key and the raw tail, and generates all Q.
It must still process the complete input window; the query division should happen after preparation. The current Indexer requires a contiguous
`[128,4,T,B]` input and a `[512,T,B]` output, so half a T across a batch cannot be treated as a contiguous Tensor directly.
An explicit query work domain/output view must be added, or the local queries and metadata must be packed explicitly; the cost is counted too.
Each query has only one selection producer, and both sides consume the complete result after the gather.

The second half of the prefill queries sees more history, so splitting by query count in half does not guarantee load balance. The assignment must at least
consider the number of visible blocks and the full-selection interval; with multiple requests, the per-row context difference also matters. Whether to divide should compare
`complete Indexer time` with `slower shard time + exposed exchange/layout cost`.
Reducing queries also reduces the reuse of the K tiles, so one cannot assume that halving the computation halves the time.

Splitting by query heads requires merging the complete score before TopK: sum after the ReLU of every head,
not select each head's local TopK first. This option also does not reduce the key cache scan volume.
Splitting by history blocks does reduce the scan: each shard computes the complete four-head score, exchanges the local Top512 and merges;
with the same score/ID order, the union of the local Top512 contains the global Top512, but the score and ID must be transferred,
while the current interface returns only ID/count. It also needs the page table, append and checkpoint design for the history shards, so it is not included in the first version.
vLLM returns expanded token IDs; NInfer continues to exchange compressed block IDs and does not introduce this output expansion.

## 5. MoE

The two machines replicate the Router and produce the global top-10 and the normalized coefficients. The local computation covers only the owned global expert IDs and does not renormalize by the local subset. The shared FFN intermediate is split into 320+320, the independent scalar gate is replicated, and the local shared contribution joins the routed contribution before the unified reduction.

The local bank of each layer contains 256 complete experts, and the mapping from global ID to local bank is independent of the storage order.
The first version's Main uses the existing per-layer priority, and the expert at sorted position j belongs to `j%2`; MTP uses global ID parity. Both keep the same count and generate a fixed mapping at load time. The priority is only an ordering and cannot replace exact frequency or co-activation data; the optimization target is the actual completion time of the slower rank. For small M, look at the touched weights and the load skew; for large M, also look at the per-expert token count and the grouped GEMM efficiency.

The first version replicates the token input, computes the local expert subset on each machine and then merges the output; no token AllToAll is needed.
vLLM's `FusedMoEParallelConfig` also separates these two things: TP2+EP sets routed TP to 1 and EP to 2;
only DP/PCP or sequence-parallel tokens further trigger its AllToAll path. The shared MLP TP
is still independent of the routed expert EP.

NInfer's existing capabilities and gaps are as follows:

| Layer | Existing basis | Work this parallel integration needs |
|---|---|---|
| Route | Global 512, top-10, independent FP32 coefficients | Both machines replicate the same control computation; do not renormalize by local experts |
| Device bank | indexed offsets, independent divisor, subset mask | Map 512 logical addresses to 256 local physical slots; remote entries are not read |
| Local computation | `routed_swiglu` supports subset and also shared accumulation | Keep the global route profile; filter local active assignments before reading weights |
| Shared | Public profile already independent | validation, kernel, workspace still fixed at 640; need full 320 support |
| Loading | affine bank description and whole-parent binding of the original artifact | Move encoded planes per expert and generate local slots and global-ID offsets directly |
| Complete fused entry | Assumes all routed experts are readable | Keep the single-machine entry; the EP fused path must explicitly include the subset and the independent shared geometry |

Therefore the first version can keep the logical domain `bank.count=512`, with a physical capacity of only 256 experts.
Reusing the existing expert slot codes/scales organization lets the indexed offsets point to the local storage;
the divisor table is very small and can continue to be addressed by global ID. An arbitrary mapping does not require rewriting the expert GEMM,
but the complete GPU parent must no longer be bound through `bind_banks` to implement "sharding".

The route/group counts in the complete fused path must also be built from the local subset. Replacing only the bank pointer or zeroing at the
final output would still let the intermediate grouped jobs access remote weights. The shared=320 gate/up, down,
activation scratch and the final accumulation must change together; the entry check cannot simply be relaxed.
The local shared and routed should be merged inside the existing accumulation where possible, and then a single BF16 local contribution is output and participates in the AR;
they must not each be rounded to BF16 separately and then treated as the existing fused semantics.

The global Router is a discrete decision region that needs consistency, with the same principle as the Indexer. If the merged projection is kept as
`[1153,2560]`, the router/score segment on both machines must use the same computation path, and only the shared segment is their own weight.

EP does not guarantee that each step computes five experts. The ten experts of a single query may lean to one side; the same count with interleaved
global-priority placement is only a reasonable initial value and cannot guarantee per-request load balance. For large M, each expert still receives
all tokens originally routed to it, so the token count of a single expert GEMM does not automatically halve with EP2.
This helps keep the grouped GEMM efficiency, but the actual speedup is decided by the busier rank.

## 6. Embedding, Head and Output Decision

The embedding is fully replicated, about 1.184 GiB per machine; lookup reads only the needed rows. The LM head is split along the vocabulary rows,
about 0.592 GiB per machine, and Main/MTP continue to share the same physical head.

The cost of replicating the embedding is clear: each machine holds about 0.592 GiB more than TP2, in exchange for per-token lookup without extra
communication. vLLM's vocab-parallel embedding uses zero-filled local results and a reduction; under our capacity conditions,
replicating the embedding remains the reasonable choice, and the head reduces the large matrix read per step through the vocabulary split.

Each rank first applies the mask and penalty to the local logits and then computes the local Top-K. The current sampling keeps at most the
global Top-20; with the same score/ID order, the Top-20 of each shard necessarily covers the global Top-20.
min-p/top-p, normalization and RNG run only after the candidates are merged. This preserves the current sampling algorithm exactly; one cannot compute top-p or softmax locally first and then merge. The candidates use an FP32 adjusted score and an I32 global ID,
at most 160 B per rank per column.

Greedy needs only one candidate per machine. The FP32 score and the inverted global ID can be encoded into a unique total-order U64 key,
and one MAX AllReduce lets both machines obtain the same argmax directly; a mask with no candidate uses an independent invalid key, and the
signed-zero/tie rules follow the existing selector. In this way MTP does not need to gather and then broadcast the ID at each step.
The stochastic path gathers the candidates, and the output rank decides on the GPU and distributes the small result.
A single vocabulary shard is allowed to have no legal candidate; the dead-end is decided by the global vocabulary.

The current `sample`/acceptance interface consumes the complete logits and includes counting effects, so it cannot be called once on each machine
to implement the above protocol. A mathematical interface for local candidate extraction and global decision must be split out, and the single-machine fused path kept;
the existing kernel already has partial TopK and merge, which can serve as the computation basis. The physical vocabulary of the weights is 248320 rows;
the actual `token_domain` comes from the model resource, and the non-public IDs at the end of the shard must be excluded.

MTP verify column j's penalty stacks the previous j draft IDs, because that column is consumed only after that prefix is accepted.
The candidates of each column can be produced in parallel, and the subsequent acceptance runs in order. Global ID, tie-breaking, seed,
logical position and SamplePurpose keep the existing semantics. The canonical token counts can follow the vocabulary shards;
candidate construction reads only them and the round-local overlay, and only the final adoption command adds the output IDs hit in this shard.
The permitted output and the prefix actually adopted by the Frontend remain separate; unpublished correction/bonus cannot be counted into the persistent state.

CausalScoring first reduces the global max, then reduces the shifted exp sum, and takes the logit of the target owner,
to obtain the complete-vocabulary log-probability; it does not use the Top-20 approximation. It still needs only scalar statistics rather than the complete logits.
vLLM already has distributed argmax/TopK methods, but the regular logits path can still gather the full vocabulary; its shortcut methods
do not automatically provide NInfer's penalty, constraint and speculative acceptance contracts.

In the existing Spark Graph/cold migration measurements, the complete BF16 head at M=1/8 has a GPU median of about
4.7–4.8 ms. The vocabulary split shrinks the head computation and read range on each machine; the real benefit after candidate communication needs a two-machine execution measurement.

## 7. PLE

Ownership is assigned by the existing table shard: the two ranks each hold 64 complete NF4 shards, about
13.56 GiB/rank; rank 0 holds shards 0–63 and rank 1 holds 64–127, and each logical row has only one
owner. The metadata/hash parameters of the original 128 shards are very small and can be replicated. The storage ownership of the table is fixed at load time, and this dual-machine plan does not keep an execution variant with the complete replicated table.

vLLM's `Qwen4ExpPLEEmbedding` inherits the vocab-parallel embedding and shards the table rows through the ETP group
and merges the lookup output. NInfer can adopt the same placement principle while keeping its own NF4
encoding, 4 KiB row-page, per-shard divisor and exact BF16 LUT, without converting to its BF16/FP8 table.
Allocating the complete original shard also avoids splitting pages, modifying the divisor and extra reordering.

The GPU flow of each round is:

```text
Known token IDs + the executed raw-ID tail
  → generate 16 global row numbers with the existing INT64 wrap/XOR/signed modulo semantics
  → local owner lookup and exact decode; remote rows are written as zero
  → merge into the complete [2560,M] BF16 lookup
  → replicated PLE projection / gate / convolution
```

The lookup output should keep the BF16 words, including signed zero. Ordinary floating-point addition of zero cannot claim per-word
invariance; each row has 160 dimensions, an even number, so BF16 can be packed pairwise into 32-bit words, each word has only one
nonzero owner, and an integer SUM merge is interpreted as BF16, or an equivalent per-owner gather can be used.
The logical semantics of this merge is to materialize the exclusive rows, not to approximate the sum of multiple numerical contributions.

The communication payload is `5120*M` B: 5 KiB at M=1 and 20 MiB at M=4096, and Main does it only once per round.
The lookup does not depend on the residual and can start after the token IDs are determined, using the time between the embedding and the whole layer 0
up to the PLE consumption in layer 1. It can be placed before the layer-0 reduction of the same communication sequence,
and overlap with the GPU computation; if it is not finished, it can still delay the following reduction, so the communication time cannot be counted as zero in advance.
The MTP body has no PLE; once the Main verify IDs are all present, the whole-window lookup can start, and the draft IDs no longer need to be moved to the Host for PLE. The ordinary output, constraints and the Frontend's Host handoff still follow their original responsibilities.

GPU hash/lookup is a new capability; the existing code has only CPU hash, SSD reads and Host LUT decode.
The existing `make_nf4_bf16_lut` can generate a small LUT for each shard at load time and upload it; the kernel
consumes the original row encoding and does not expand the whole BF16 table at startup. PLE projection/gate/convolution
and the nine-column history are replicated, keeping the existing timing, normalization and BF16 boundaries.

Split the Main/MTP experts and the PLE table in half, and keep the other weights replicated for now, about 54.29 GiB/rank;
other TP slices would reduce the capacity further.
This is the weight upper bound derived from the per-component byte counts of the existing artifact, excluding State/KV, workspace, Graph,
staging, Frontend and system headroom. GPU and CPU allocations share the machine's physical memory, and the loading stage must also
keep bounded staging and no longer keep a complete writable Host weight copy.

## 8. MTP and Vision

The MTP body uses QSA TP2 / experts EP2, with the Indexer and HC replicated, and shares Main's sharded head.
The body of each draft step has two main reduction boundaries, attention and MoE. The single-owner body plus the distributed head is
a control candidate and requires comparing the saved communication with the added single-machine computation. That candidate needs the owner to hold the complete MTP
experts and the complete two-head MTP KV, and then distribute the sample hidden to the head ranks; it is not just turning off two ARs.
The base path continues to share Main's local QSA/EP interfaces.

The two `[2560,2560]` FCs of the stem are replicated initially; fc_embedding processes M columns and fc_hidden processes 4M columns.
The candidate split divides both by output feature, combines the residual locally and then does one AllGather,
then the complete norm/HC. Its comparison with the replicated path should count the original BF16 output and the residual rounding boundary.

vLLM uses ColumnParallelLinear for both FCs with `gather_output=True`, and each gathers its own output.
If NInfer splits them, both branches should use the same output feature range, keep their own BF16 FC output,
do a local broadcast-add and round into the residual, then gather `[2560,4,M]` only once and run the complete norm.
This requires supporting an addition output of the local features; the existing HC norm with a fixed H cannot be used directly for half of H.
The two FCs total about 25 MiB; for small M, halving the weight read volume converts to about 51 μs against the sustained-read reference bandwidth; this is an order-of-magnitude basis for comparison with
one gather, not a measured benefit or upper bound. In practice, measure the FC, the addition, pack, communication and norm together.

Draft steps depend serially on the previous step's ID and pre-final residual. After the global argmax, the device side
enters the next step directly and adds no coordination-side round trip per step. vLLM's option to reuse the step-0 Indexer
selection for later drafts is another computation choice; this plan keeps the existing model semantics of re-selecting at each step.
MTP history setup runs only the stem, attention HC, QSA prepare/tail; it must not be misconnected as a complete draft forward.

The Vision weights are replicated, about 0.836 GiB/rank. The first version uses the existing `VisionItemView/Control` as the division unit,
each machine runs the complete encoder, and finally distributes the merger features to the Text ranks. vLLM's Qwen4Exp
can also use the data-parallel Qwen Vision tower through `mm_encoder_tp_mode=data`.
NInfer needs to extend the existing per-item encoding scheduling to the owner mapping, not modify the encoder mathematics.

The input patches are sent only to the encoding owner, and the final features keep the original item, MRoPE and scatter column mapping.
The feature references are kept until Main and MTP history finish consuming them. A video or a large image cannot be split at arbitrary patches;
further splitting must be based on the actual attention segment. A single indivisible item is still encoded on one machine,
and this first-version choice does not improve its encoding time. The workspace peak of the two-machine parallel encoding also needs separate planning.

## 9. State and the Complete Execution Flow

| State | Placement |
|---|---|
| GDN recurrent / convolution | Sharded with the local heads; recurrent is about 54 MiB/rank/Main image |
| Main/MTP main KV | One KV head per rank, a common logical page group |
| Index cache / raw tail / block coordinates | Replicated |
| PLE history | Replicated, 180 KiB/image |
| Sample hidden / conditioning | Keeps the complete rows needed for the continuation |
| Frontier, input identity, accepted length | Decided on the coordination side and adopted by every rank |

The head sharding changes only the physical schema, not the prefix matching algorithm or the logical token coverage. One logical page group
holds the main KV head shard and the complete index cache on the two machines; the same group ID does not require the same address.
The capacity is planned jointly from each owner's available resources and the per-group byte count; the page count of rank 0 alone cannot represent the whole group.

With BF16 main KV at 262144 context, Main is about 3 GiB/request per rank, plus about 192 MiB of index
cache; MTP main KV adds about 256 MiB and 16 MiB of index. The actual capacity uses the schema of the selected codec.
The GDN 54 MiB is the local recurrent byte count of one image; source/destination, checkpoint and
verify records cannot be counted only once. The Indexer tail and PLE history, even when replicated, must be saved, restored and committed together with the same
source/destination and accepted length.

The three flows must close in D0:

1. Ordinary: common token/position → local lookup/HC → TP mixer/EP MoE and reduction →
   sharded head/candidate gather → output decision → adopt the local state at the common frontier.
2. MTP: shared conditioning → distributed draft steps → Main verify records → per-column candidates and
   acceptance → Frontend adopts the prefix → local GDN fold, QSA/PLE tail and MTP history alignment.
3. Restore: one prefix match obtains the same checkpoint manifest → each owner restores its own State/KV
   → necessary Replay → continuation; the replicated fields and the sharded fields must belong to the same logical execution state.

The Main canonical frontier is F, and the MTP frontier is `max(F-1,0)`; the PLE raw-ID tail corresponds to the executed input,
not to output that has been predicted but not yet adopted. After adopting a permitted outputs, each rank's GDN fold, QSA/PLE
extract, sample/conditioning selection and MTP history alignment advance these frontiers together. The GPU lookup
borrows only the tail's device input/mirror and cannot turn the whole draft window into the canonical tail in advance.

Prefix availability is tied by one global manifest to the valid shards of all necessary owners. Each machine does not hit independently or
evict a shard that is still globally referenced independently; restore follows the existing reference, COW and Replay semantics.
Model mathematics decides the schema and the frontier, ContextDirectory manages the logical references, and RankProgram manages the local backing.
A new sharding scheme needs its own layout identity; this round's checkpoints are reused within the same fixed ParallelPlan and
do not introduce the ability to convert saved state online to another TP/EP placement.

With four machines, the two KV heads of QSA need to be replicated in the corresponding ranks, and the Q heads can continue to be split into six per machine.
ParallelPlan describes this replication explicitly; one cannot assume that all State/KV fields are divided equally by the rank count.

## 10. Computation Organization and Graph Assembly

### 10.1 Native Computation Boundaries

Model computation uses one native enqueue flow; eager executes it directly and Graph capture uses the same flow.
The basic device segments are Main, the draft chain, the output decision and state adoption; each segment continues to call the existing mathematical Ops.
The layer organization and the Graph launch granularity are separate: the model keeps clear layer/Op boundaries, and Spark replay
does not launch layer by layer from the Host.

Main consumes the prepared PLE lookup Tensor `E[2560,T,B]` uniformly. Whether the table resides on disk, in Host memory or on the
Device is the producer's choice; it does not enter Main's mathematical interface and does not decide whether Main captures the complete graph.
The producer provides a stable E view, this round's readiness dependency and the consumption lifetime; the producer in this dual-machine plan is the sharded GPU
lookup/merge and can be captured together with Main. The existing Host producer can also feed the same complete Main graph,
without implementing the GPU lookup first. The whole-round combination that includes the producer and Main's own capture scope are handled separately.

vLLM separates the `CudagraphDispatcher` mode/shape selection from the `CUDAGraphWrapper` capture and replay.
This plan follows that division: Session/the native plan selects the execution combination, RankProgram binds the local
views and the Graph; the Op continues to own the mathematics and the private shape policy and does not decide the model-level graph split.

The main chain of each Main layer is:

```text
Complete residual / normalized
  → HC mix
  → local GDN or QSA (including local State/KV writes)
  → the BF16 local contribution of the output projection
  → AllReduce
  → HC combine / norm / mix
  → global router + local routed subset + shared TP local contribution
  → AllReduce
  → HC combine / the next entry norm
```

At the end of layer 0, materialize the BF16 residual first, then run PLE at the layer 1 entry; PLE itself produces the next set of
normalized. The last layer outputs the pre-final residual for MTP conditioning, and the final HC
produces the sample hidden, after which the local vocabulary head runs. The reductions above sit at the actual numerical dependencies.

### 10.2 PLE Bypass and Stream Organization

RankProgram holds the ordered main computation stream, the PLE preparation stream, the communication stream, and the copy
streams needed for input/result transfer. The caller passes stable execution views; each Op does not create a stream.
In this plan the TP, EP, PLE and head communication participants are the same and can share one physical communicator;
the model roles are still expressed independently. The communication stream carries a fixed common order and does not create an extra communicator for PLE.

```text
                         ┌─ hash / local NF4 lookup ─ PLE words reduction ─ projection ─┐
Input IDs / raw-ID tail ──┤                                                       ├─ PLE gate/conv/norm
                         └─ embedding / HC ─ layer 0 (two reductions) ─ BF16 residual ───┘
                                                                               ↓
                                                                        layer 1 to the last layer
```

The PLE projection depends only on the lookup output and can also run early; the gate only starts depending on the Main residual.
The storage producer's output is still E. The projection can run early inside the Main graph to form the stable key/value parent required by
`PleProjectionView`, and at the insertion layer the existing
`ple_gate → ple_conv_add_norm` and the corresponding record/state actions run; this is scheduling inside the model and
does not change the E interface or the intermediate BF16 boundary.

The communication order is the PLE words reduction, the layer 0 mixer reduction, the layer 0 MoE reduction and the following layer reductions.
As soon as the PLE result merge finishes, the main chain can continue to use the communication stream without waiting for the bypass projection.
The main chain waits for the preparation result only at the PLE consumption point. Cross-stream data dependencies are expressed with events, and at capture time all bypasses
finally reconnect to the main stream; this is the same-graph cross-stream capture method supported by CUDA.
See [CUDA cross-stream capture](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#cross-stream-dependencies-and-events).

The early projection provides an overlap opportunity, but the two computation branches share the SM and memory bandwidth. The actual execution plan can
run the projection in parallel or wait for the main chain to arrive after the lookup merge; choose by the complete stage time; both
are in-graph dependencies and add no Host handoff. The PLE merge sits before the first-layer reduction and still blocks that
reduction when unfinished, so the whole bypass time cannot be treated as hidden.

The bypass storage is planned separately from the main computation scratch. The lookup is `5120*M` B and the projection is
`25600*M` B; at M=4096 they are 20 MiB and 100 MiB, and each scratch is counted separately.
The projection lives from the start of the early computation to the end of the PLE consumption; it cannot use the current single-stream MemoryArena scope reuse rule
and overwrite layer 0 while it runs. The other non-overlapping stages continue to share the main scratch.

### 10.3 External Boundaries of the Graph

Organize the device computation, the output permission and the final adoption separately. The GPU result is still the candidate state of the current frame;
only after the Frontend determines the final output prefix does the Session issue commit/discard.

| Path | Computation and decision Graph | Later adoption |
|---|---|---|
| Ordinary, decision inputs ready | Main (including the PLE bypass) + local head + distributed decision, as one graph | A separate adoption action |
| Ordinary, mask not ready | The Compute graph runs Main + local head; the Decide graph runs after the mask arrives | A separate adoption action |
| MTP, Host decision inputs without draft dependency | draft chain + Main verify + local head + acceptance, as one graph | Finish graph/device segment |
| MTP, constrained mask depends on drafts | The Compute graph runs the draft chain + verify + local head; the Decide graph waits for the mask | Finish graph/device segment |
| Prefill / Replay | Runs the same native flow at the actual T; keep eager first, a fixed complete chunk can be an independent capture profile | Includes the needed MTP history setup |
| Vision | Each owner runs a complete independent item; the features are distributed and then used by Text | Features kept until Main/MTP's last consumer |

When the decision inputs are ready, there is no need to force the segmented launch. When external input must be awaited, the Compute end point is placed after the
local head so that the large matrix computation finishes first; Decide consumes only the retained local logits, mask, counts
and configuration. The two combinations share the native enqueue function, and the graph count and frame binding count toward the preparation budget.

Prefill cannot pad columns arbitrarily just to reuse the maximum-T graph: the existing sequence GDN entry advances its real T
columns, and the tail padding also affects the recurrent state. The tail block uses eager with the actual geometry or a prepared profile.

### 10.4 MTP Whole Round and Constraint Overlap

The draft limit K is fixed at startup; each round selects the actual depth S (1..K) by a common budget, uses the corresponding finite
chain variant, and unrolls S native draft steps inside the graph. Each step includes the MTP stem, QSA TP, EP MoE,
the local head and the global argmax, and the next step consumes the device ID and the pre-final conditioning directly.
The valid draft count of each row is controlled by the device extents, keeping the semantics of the first-step bridge; later rows with no actual work
do not write valid state, and each rank fulfills the common communication contract of the selected variant. Main verify still uses the
fixed K+1 physical columns and the per-row effective length and does not change the communication shape because a row's remaining budget is smaller.
The graph does not return to the coordination side after each draft step to obtain the next ID, and it does not compute the extra drafts that the whole batch does not need just to reuse the maximum-depth graph.

After the draft chain completes, assemble `[anchor,drafts...]` once; Main verify's PLE does one lookup merge over the whole window
and does not follow the single-machine Host producer's incremental D2H/preparation flow.

The constrained path can keep the following fork inside one Compute graph:

```text
draft chain ─┬─ prepare verify IDs ─ Main verify + local head ─ keep local logits
             └─ output rank copies the drafts asynchronously D2H ─ DraftsReady
                                                   ↓
                                      coordination-side matcher builds sharded masks
                                                   ↓
                              Compute result dependency + masks ready → Decide graph
```

The copy branch of the output rank uses the frame's own pinned staging and an external event observable by the Host;
after the execution thread observes this round's event, it publishes `DecisionInputNeeded` and the target computation continues.
The external event is kept until its Graph and transfer consumers finish, and events are not queried during capture.
This mechanism can be implemented with [CUDA external event nodes](https://docs.nvidia.com/dl-cuda-graph/cuda-graph-basics/constraints.html#external-events-and-streams)
and does not need to call the Host matcher in the graph or wait for a callback of control messages.
Once the mask input and the common command order are fixed, Decide can be enqueued, and the GPU inherits the Compute
completion dependency through stream/event; the Host does not need to wait for the whole Compute to finish first.

The mask is applied on each vocabulary shard. Ordinary sampling uses candidate gather, output rank decision and result
distribution; greedy's global argmax is already visible on all ranks, and the integer prefix comparison can yield the same
acceptance on every machine, published by the output rank. RNG and the ownership of the public output decision stay unchanged.

Finish consumes the a that the Frontend finally accepts, and runs the local GDN fold, QSA/PLE extract, the final hidden
selection, the token counts adoption and the MTP history alignment. Its physical window is fixed, and a is a device input.
This stage needs to convert the per-row copy/metadata preparation in the current Host loop into capturable device preparation actions;
MTP history continues to run only the stem, attention HC, QSA prepare/tail and does not append a complete draft.
The logical adoption of the canonical state is managed by the Session, and Finish hands its device completion dependency to the next round.

### 10.5 Complete Replay and Local Graph Updates

The resident Main of the single-machine `TextGraph` already uses complete outer replay, and the QSA envelope and the PLE upload slot
are updated through children. The distributed implementation follows this assembly so that each device segment in the table above has only one outer replay, and the
capture includes the computation and NCCL directly.
PLE's model computation lives inside the Main graph; the upstream preparation position is not a reason to split the graph layer by layer. The address switch of the two E slots
is handled through input binding or local consumer updates, and the complete model executable is not copied for each slot.

Keep the existing ability to update the QSA launch with the context, using "complete outer graph + updatable local computation
subgraph":

1. At startup, capture the envelope definitions of the QSA local computation region by the actual bound frame address.
   In the base path with the replicated Indexer, this region has no communication.
2. When capturing the outer native device segment, insert the corresponding definition as a child graph and keep the node handle;
   NCCL is captured directly in the outer flow in the common order.
3. When switching context slots, update only the parameters/computation definitions of these children and keep the communication nodes and the backing.
   Do not look up nodes by kernel name, and do not capture the complete layers and communication again.
4. The child's topology, node insertion order and updatable parameters must satisfy the CUDA update requirements; if the computation/communication topology
   changes, use another prepared common variant. The large-M query division places the local computation region on both sides of the
   selection communication, and the communication cannot be hidden inside a child that can be replaced arbitrarily.

The Core Graph wrapper already provides the ability to insert children during capture, keep bound handles and update children.
The implementation combines `cudaStreamGetCaptureInfo` / `cudaStreamUpdateCaptureDependencies` with the explicit
Graph API and updates with `cudaGraphExecChildGraphNodeSetParams`.
See [CUDA capture and explicit node composition](https://developer.nvidia.com/blog/constructing-cuda-graphs-with-dynamic-parameters/)
and [CUDA child graph updates](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__GRAPH.html).
This is an extension of the existing Graph basis and does not require Ops to expose kernel parameters or build a generic model DAG.

The outer executable is built from the ParallelPlan, the device segment/communication protocol, T/B, the state update mode and the frame binding.
The context slots are installed local computation profiles, and the whole Main graph is not copied for every slot.
The workspace is reserved for the peak of all allowed profiles; the child definitions, the outer executable and the NCCL
retention are all counted in the actual Graph preparation cost. Multiple frames do not automatically share writable logits/records; ordered computation
can share scratch, and results awaited across stages are kept until the actual consumer finishes.
The frame views and the parallel scratch partition are fixed before capture; the ordinary enqueue and the child insertion consume
the same bindings and capacities, and the storage reservation must not be missed because inserting a child skipped a Host computation call.

The local kernels, addresses and output-side copy branches of each rank may differ, but the communication protocol must be chosen in common.
NCCL requires the capture/replay participants to be consistent, and each rank uses the graph derived from the corresponding common capture;
if its group call mixes multiple streams, it adds dependencies on those streams, so the communication here stays in
the same ordered stream. See [NCCL Graph contract](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html)
and [NCCL stream semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html).

### 10.6 Cost and Scalable Boundaries

The 48 layers of the base Main each have a mixer output and a MoE output reduction boundary, 96 in total.
The PLE table sharding adds one merge; each draft step has two block reductions and one global argmax.
Under the base layout of this document, the device communication of greedy ordinary is 98 calls, and that of K-step greedy MTP is
`98+3K`; the candidate gather and result distribution of general sampling add one more than the greedy output. This does not include Host
control transfer, Vision feature distribution or later layout optimizations. The complete graph reduces Host launches and handoffs;
the communication in these mathematical dependencies still exists.
The BF16 logical payload of each of the 96 block reductions is `2*2560*M` B:

| M | Single H reduction payload | 96 total |
|---:|---:|---:|
| 1 | 5 KiB | 480 KiB |
| 8 | 40 KiB | 3.75 MiB |
| 48 | 240 KiB | 22.5 MiB |
| 4096 | 20 MiB | 1.875 GiB |

The above excludes PLE, head and MTP and is not the actual wire bytes of the link. The reductions have nonlinear dependencies such as HC, gate and
norm between them, so the 96 calls cannot be combined into one transfer by total bytes. Small M needs control of the per-call launch and
synchronization cost; large M must consider the effective link bandwidth, the computation overlap and the completion time of the slower rank.
The BF16 local contribution and communication are fixed contracts and do not force FP32 for parallelism; a new sharding combination still needs to be checked against its
mathematical and numerical contract and is not required to match another single-card reduction order bit for bit.

The unit for judging the inter-layer time is: local pre-computation → the local result of the slower rank → the exposed communication →
the following HC/projection, rather than halving each Linear's read volume and adding them directly. The shared Router/Indexer/HC
still take time, the EP per-step load may be uneven, and the head halving is only one item.

A rank with no local active expert still outputs the correct zero/shared contribution and participates in the agreed communication. Indexer
full selection and mixed contexts continue to be handled inside the kernel. Different prepared variants can be selected, but a rank cannot
add or remove collectives or change the message size on its own based on local observation. The graph dispatch and update do not introduce per-layer
control messages or a per-round whole-group Host barrier.

Eager and Graph share these boundaries. Before the real compute-communication fusion, keep the clear local Op outputs and the collective
contracts; afterwards the reduction can be connected with the HC combine/norm while keeping the original numerical order of the BF16 residual.
The execution framework does not need to know these Ops' private tiles, tuning tables or expert GEMM algorithms.

Weight placement and execution-time division also need to be separated. Experts/PLE ownership, the head range, MTP single-owner or
the TP/EP body are chosen at startup; the HC token partition and the Indexer query division can reuse the same weights and
canonical state and change only the frame's temporary layout. If both the HC lowrank TP and the MTP FC replicated/TP paths are kept, the matrix views/encodings each can consume must be prepared explicitly at load time and counted in the capacity;
switching variants must not become execution-time weight movement or reordering.

## 11. Implementation Dependencies and Later Optimization

| Item | Feasibility conclusion | Next dependency |
|---|---|---|
| GDN/QSA head TP, expert EP, vocab head TP | Mathematics and main sharding method settled | Local Op shapes, sharded loading and the complete state schema |
| EP replicated input, unified result reduction | The existing subset/indexed bank can carry it; no AllToAll is needed first | Load the local slots directly and complete shared=320 and the fused filtering |
| Embedding, HC, Indexer base replication | Can form the complete base path; the replicated discrete decisions have common execution requirements | Bind the common decision profile and the local cache |
| PLE table row sharding, computation replication | Settled; encoding kept, lookup merge moved earlier | GPU hash/NF4 lookup, exact word handoff and bypass storage |
| Vocabulary candidates and sampling | Can be executed exactly in a distributed way under the current Top-20 contract | Candidate/decision interfaces, mask/count sharding and the adoption protocol |
| MTP / Vision | The body and the independent item division are feasible; history and Frontend semantics kept | Device decision of the sharded head, history path, feature distribution |
| Large-M HC token partition | Later optimization, disabled in the first version | PLE front/back layout, final features, complete RS/AG time |
| Large-M Indexer query division | Later optimization, disabled in the first version | query work domain, load assignment, selection exchange time |
| HC lowrank TP, MTP FC TP, single-owner MTP | Later optimization, disabled in the first version | Complete candidate path comparison, not only local GEMM measurements |
| Complete device-segment Graph, local computation child updates | Single-machine resident Main, draft chain and Core child updates implemented; distributed not yet implemented | Common capture, communication bypass capacity and device Finish |

D0's basic role ownership, encoding slices, local geometry, state and output decision method follow the first-version path in this document.
The later candidates do not enter the first-version dispatch, and no unmeasured M threshold is preset. Shared=320, sharded sampling, GPU
lookup and the Graph combination basis are explicit new capabilities; they cannot be filled in automatically from the existing TP shape list.
The single-machine resident Main already reuses the existing PLE producer for the complete capture and is decoupled from the dual-machine implementation.

## 12. Source Basis

This round compares the specific implementation of the local vLLM working copy rather than treating all its default policies as NInfer's choices:

The single-machine timings reuse the existing migration records of `hc-mix` and
`bf16-n248320-k2560` under `.agent-tmp/spark-op-migration/units/`; the read bandwidth reference comes from
[roofline](../bench/common/roofline.h). These historical data are used only to judge the order of magnitude of the candidate cost.

| vLLM path, relative to `~/vllm/vllm/` | Corresponding fact |
|---|---|
| `models/qwen4_exp/nvidia/model.py`, `hyperconnection.py` | HC replication, sequence-parallel MoE restriction, PLE position and Vision mode |
| `models/qwen4_exp/nvidia/qsa.py`, `indexer_qsa.py` | Main heads TP, Indexer replication, expanded token selection and MTP reuse option |
| `model_executor/models/qwen3_next.py`, `model_executor/layers/fused_moe/config.py` | Applicability conditions of routed EP, shared TP and AllToAll |
| `models/qwen4_exp/nvidia/ngram_embedding.py`, `models/qwen4_exp/common/ple.py` | ETP table row sharding and checkpoint row ranges |
| `models/qwen4_exp/nvidia/mtp.py` | Column TP/gather of the two FCs, conditioning and the draft layer |
| `model_executor/layers/logits_processor.py`, `model_executor/layers/vocab_parallel_embedding.py` | Distributed argmax/TopK, regular logits gather, vocabulary embedding |
| `v1/cudagraph_dispatcher.py`, `compilation/cuda_graph.py` | Division of responsibility between execution mode/profile dispatch and capture replay |

NInfer's landing points include:

- [GDN/QSA/MTP execution](../src/models/qwen4_exp/execution/),
  [StateImage](../src/models/qwen4_exp/state/state_image.cpp),
  [state adoption](../src/models/qwen4_exp/program/commit.cpp).
- [MoE common contract](../include/ninfer/ops/moe.h),
  [bank view](../src/core/weight_bank.h), [bank loading](../src/models/common/moe/weights.cpp),
  [MoE parameter support](../src/ops/moe/operands.cpp),
  [block-scaled shared computation](../src/ops/moe/kernels/cuda/block_scaled/shared.cu).
- [HC contract](../include/ninfer/ops/gated_residual.h), [QSA contract](../include/ninfer/ops/qsa.h),
  [Indexer](../src/ops/softmax_attention/qsa/indexer/).
- [Sampling](../include/ninfer/ops/sampling.h),
  [Speculative acceptance](../include/ninfer/ops/speculative_accept.h),
  [existing candidate/merge implementation](../src/ops/sampling/kernels/distribution.cuh).
- [PLE parameters](../src/models/qwen4_exp/ple/parameters.cpp),
  [hashing](../src/models/qwen4_exp/ple/hash.cpp), [exact NF4 LUT](../src/core/nf4.h),
  [weight materialization](../src/artifact/materializer.cpp),
  [Vision item](../src/models/common/qwen/vision/control.h).
