# GLM5Next dual-Spark parallelism and resource design

Nature: a parallelism and resource design that still has to land. This document maintains the module
split, the weight and state ownership and the deployment budget. The [design index](README.md) holds
the decisions and the implementation status in one place.
[Model semantics](../../maintainer/glm5next-model.md) and the
[numerical policy](../../maintainer/glm5next-quantization.md) are independent of the placement
described here. Execution ownership follows the
[distributed execution design](../../distributed-execution-design.md) and the
[Engine architecture](../../maintainer/engine-architecture.md).

## 1. The first parallel split

One group of two Spark ranks splits by module: MLA/Indexer DP2, KDA TP2, dense/shared MLP TP2, LM head
TP2, routed experts EP2, and replicated mHC weights and four-stream residual. The total GPU count is
2. The parallelism of the modules does not multiply.

| Part | Fixed split |
|---|---|
| Main 11 layers and MTP 1 layer MLA | DP2 by request; the complete logical weights of Q_a/Q_b/KV_a/KV_b/O are replicated on both hosts; the owner computes all 64 heads of that request |
| Main/MTP Indexer | Shares the request owner of the matching MLA; the owner computes the complete projection, the 32 query heads, the pooling, the scoring and the selection |
| MLA/Indexer history | The complete latent, the pooled index cache and the unfinished pool tail of one request are kept only on the owner; Main and MTP share the owner but use different pools and frontiers |
| KDA | TP2 by heads, 32 heads per host; Q/K/V/O, beta, f_b/g_b, the convolution and the large state split by the matching heads; f_a/g_a stay replicated |
| Main first three dense MLP layers | intermediate 12288 splits into 6144 + 6144; down produces a local contribution |
| Main/MTP shared expert | intermediate 2048 splits into 1024 + 1024; the local shared and routed outputs merge and then reduce together |
| Main/MTP routed experts | EP2, 144 complete experts per host, no further TP inside an expert; the global Top8 and the route coefficients keep the complete semantics |
| Router | The complete weights and the control computation are replicated and consume the same block input |
| Main mHC | Both hosts replicate every weight and the complete four-stream residual and process the same group of physical columns; after the complete block output is summed and gathered, each runs its own post and the next pre |
| Main/MTP LM head | vocabulary TP2, `[77440,4096]` per host; Main and MTP share the sharded weight |

The allocation unit of DP is the request. Ordinary execution keeps the MLA/Indexer owner of a request
stable, and that owner consumes all the history of the request. It does not split one history by
sequence position. KDA is still held sharded by both hosts and processes the same group of requests.
ModelSession/ContextDirectory must express both owner sets at the same time. The checkpoint and the
restore cover the shards each one needs. Do not reserve by a rule that both hosts must have an MLA
cache. The cross-owner prefix restore or migration policy converges separately in a later context
interface design.

MLA delivers the complete 4096-wide block output to the later TP/EP computation. The local
contributions of KDA, dense/shared and routed finish their sum before the Main mHC or the MTP residual
consumes them. The output of the Main MLA owner is gathered by the frozen request/token mapping into
a complete BF16 y that agrees on both hosts, and then enters each mHC post. mHC does not split by
stream, hidden or owner and does not carry communication. Its replication adds no MLA/Indexer cache
copy. Both hosts use the same input words, physical column mapping and precision policy.

The concrete collective, the pack/unpack, the model Graph and the compute-communication fusion still
belong to a later execution integration design. Arrange the input exchange, the output reduction and
the layout conversion together so that one handoff is not communicated twice. The interface that
receives the complete y and the residual placement are no longer open items. The local KDA/FFN/head
geometry of the existing base checklist still applies.

The local vLLM setting `TP=1, DP=2, enable_expert_parallel=true` replicates the non-routed-expert
weights as well, including KDA, dense/shared and head. Its MLA and KDA use the same TP group. This
section adopts a mixed split chosen per module. It is not directly equivalent to that vLLM parameter
combination.

## 2. Parameter merging and local geometry

The [base Op checklist](op-checklist.md) maintains the TP2/EP2 geometry and the P/F/M/R/E entries of
the pure projections and the local FFN/MoE. KDA Q/K/V split by heads separately and then merge in
Q/K/V row order. BF16 beta, f_a and g_a do not enter the Q8 parent. KDA O splits along the K axis of
the matching heads. The channel and tap mapping of the Convolution agrees with the projection. The
conversion keeps the source FP32 words and the time order.

The GU of dense/shared splits along intermediate and down splits along the matching K axis. Each
routed bank holds 144 complete experts per host, with no further TP inside an expert. The mapping
from a global ID to the local bank is provided explicitly. The numerical conditions of the
quantization domain and of the GU merge are in the numerical document. They do not change with these
shards.

The [Kpool Sparse MLA](kpool-sparse-mla.md#2-weights-projections-and-the-execution-chain) document
maintains the W_A/W_B merge of MLA/Indexer, the U_K/U_V factor layout and the independent O. The
[mHC design](mhc.md#1-weights-and-tensor-layout) maintains the F row order and the four-stream layout
of mHC. Keep exactly one definition for each physical layout.

## 3. Persistent state capacity for a 1M context

Set `L=1048576` and B=1 and compute by the agreed [family layout](kpool-sparse-mla.md). Main has 11
MLA layers and keeps one 512-wide latent per token per layer:

```text
BF16 latent = layers * L * 1024 B
FP8 latent  = layers * L * (512 + 4*4) B
Indexer    = layers * (L/4) * 128 * 2 B
```

| Scope | BF16 latent | FP8 codes | FP32 scales | BF16 Indexer |
|---|---:|---:|---:|---:|
| Main 11 layers | 11 GiB | 5.5 GiB | 176 MiB | 704 MiB |
| MTP extra layer | 1 GiB | 0.5 GiB | 16 MiB | 64 MiB |
| Main + MTP | 12 GiB | 6 GiB | 192 MiB | 768 MiB |

This is one complete latent cache. Section 1 chooses MLA/Indexer DP2 by request, so the complete
latent of one request resides only on its owner rank. This capacity for a single request does not
divide by two. A comparison scheme that shards MLA TP2 by heads would keep one complete 512-wide
latent on each of the two hosts.

The growing cache totals follow. The FP8 column already contains all MLA scales. Every Indexer entry
is BF16:

| Scope | BF16 MLA + BF16 Indexer | FP8 MLA + BF16 Indexer |
|---|---:|---:|
| Main 11 layers | 11.6875 GiB | 6.359375 GiB |
| MTP extra layer | 1.0625 GiB | 0.578125 GiB |
| Main + MTP | 12.75 GiB | 6.9375 GiB |

The three-column K/gate BF16 canonical tail is 1536 B per layer, which is 16.5 KiB for Main and
18 KiB for Main + MTP. The complete 34-layer FP32 state of KDA is 136 MiB and the three-column BF16
convolution history is 4.78125 MiB. After KDA TP2 by heads, each rank holds 68 MiB and 2.390625 MiB
respectively.

With the selected "MLA/Indexer DP2, KDA TP2", and counting one local KDA state/history and the tail:

| Occupancy of one request on the given rank | Main | Main + native MTP |
|---|---:|---:|
| Owner: FP8 MLA + BF16 Indexer + local KDA state/history + tail | about 6.43 GiB | about 7.01 GiB |
| Owner: BF16 MLA + BF16 Indexer + local KDA state/history + tail | about 11.76 GiB | about 12.82 GiB |
| Non-owner: the other KDA state/history shard of that request | about 0.069 GiB | about 0.069 GiB |

The table charges every layer as full at 1M. The family contract computes the effective row count of
the real Main/MTP frontier for each. The table excludes the execution records and workspace, the
Graph, extra StateImage and checkpoint copies, the page tables, the allocation alignment and the other
small state fields. MTP conditioning and similar items still need their own charge. Multiple requests
without a shared prefix add up in the growing cache by their own owner and length. Both hosts hold
the local KDA shard of each request, and that capacity grows with the number of state instances. It
does not grow with L. Apart from the token/latent frontier, a prefix restore must also restore the
KDA state and convolution history of both hosts, and the Indexer tail and the MTP state on the owner.

### 3.1 Additional accounting for pages, the Indexer and raw records

The growing payload per original token per layer is 1088 B for BF16 MLA + BF16 Indexer, or 592 B for
FP8 MLA + BF16 Indexer. That is 68 KiB or 37 KiB for each 64-token page. Each effective raw record
column per layer is `128*2*2=512 B`. The two planes of the cross-layer frame total `512*W*L*R` B,
without slab alignment. W, L and R are the raw record window capacity, the layer count and the row
capacity. The layout follows the Kpool design. The update exit does not have to keep the whole raw
record range.

At 1M, the Indexer of one layer has 262144 BF16 pooled keys, 64 MiB in total. The latent payload of
the 2048 tokens that MLA selects is BF16 2 MiB or FP8 1.03125 MiB. These are the only data volumes.
They do not represent a measured HBM traffic or a measured latency.

## 4. mHC execution resources

The payload below is for one rank. It excludes arena alignment, the private workspace and the other
model modules:

| Object | Capacity |
|---|---:|
| BF16 F of one site | 786432 B = 0.75 MiB |
| BF16 F of 90 sites | 67.5 MiB |
| 90 groups of BF16 base + scale | 4860 B |
| 90 branch gamma | 720 KiB |
| Main final gamma | 8 KiB |
| Four-stream residual per physical token | 32768 B = 32 KiB |
| Block input or output per physical token | 8192 B = 8 KiB each |
| One group of FP32 post + C per physical token | 80 B |

Each group of FP32 coefficients has only 4 post and 16 mix. The pre is not kept across branches. The
old and new coefficients of a fused call do not alias, so two groups alternate for a total of
160 B/token. A separate residual double buffer is 64 KiB/token. A legal in-place route can reduce the
matching public buffer. The actual workspace must still be charged separately by the capacity query.

These activations grow with the current physical window `M=T*B`. They are reused by the execution
layer order and are not all kept for 45 layers. For example, the four-stream residual of one
`M=4096` is 128 MiB. M here is not the accumulated request context L. Do not allocate mHC activation
as permanently resident for every layer from a 1M history length.

For the relation between the non-historical property of mHC and MTP conditioning, see
[logical state](../../maintainer/glm5next-model.md#7-logical-state-and-historical-dependencies).
These buffers belong to the RankProgram frame. They add no growing KV pool and no request StateImage
field.

## 5. KDA records capacity and cost

By the [KDA record layout](replayssm-kda.md#3-geometry-row-identity-and-record-layout), L is the
local layer count, T is the window width, R is the record row capacity and C is the Q/K/V conv
channel count. Compute with BF16 K/V/conv and FP32 log_decay/beta:

```text
StateBytes/instance = L * (4*Hv*Dv*Dk + 2*C*3)
RecordBytes/column/row = L * (2*C + 2*Hq*Dk + 2*Hv*Dv + 4*Hv*Dk + 4*Hv)
RecordAllocationPayload = T * R * RecordBytes/column/row
```

The payload of one GLM TP2 rank is:

| Content | MiB |
|---|---:|
| one recurrent state | 68.000000 |
| one conv history | 2.390625 |
| one recurrent + conv | 70.390625 |
| conv records per column | 0.796875 |
| key records per column | 0.265625 |
| value records per column | 0.265625 |
| log_decay records per column | 0.531250 |
| beta records per column | 0.004150 |
| all records per column per row | 1.863525 |
| records at T=6, R=1 | 11.181152 |
| records at T=16, R=1 | 29.816406 |
| records at T=6, R=8 | 89.449219 |
| records at T=16, R=8 | 238.531250 |

The table excludes the alignment padding, the inputs and outputs, the other model records, the Graph
and the count of source and destination images. The payload ratio of the complete state snapshot per
column to the raw records is about 37.77. It is not a speedup ratio. The cost change is fewer
per-position large state writes and more compact record writes, plus one recurrent fold after
acceptance. The fold does not rerun the weight projection or the readout.

## 6. Capacity and performance judgement for the parallel scheme

Use MLA heads TP2 with the latent/Indexer history replicated on both hosts as the control. Changing to
the DP2 of this section removes one duplicate history and adds MLA weight replication. Compute with
the agreed BF16 factor projection and without generating an extra absorption matrix: Q_b/KV_b/O total
208 MiB per layer, which is 104 MiB per rank at TP2. Main + MTP adds 1.21875 GiB per rank in total.
If the resident representation is explicitly changed later, charge it again.

For a single 1M request, one growing cache of Main + MTP is 6.9375 GiB with FP8 MLA or 12.75 GiB with
BF16 MLA. Both include the fixed BF16 Indexer, and the FP8 column includes all FP32 scales. After
removing the copy and deducting the 2.4375 GiB of added weight across both hosts, the two-host payload
drops by a net 4.5 / 10.3125 GiB respectively. The owner still has to keep the complete history. DP
does not halve the cache of a single request on the owner. The final startup occupancy also depends on
the fixed pool budget, the temporary data and the communication resources.

Against a scheme that puts every non-routed-expert part on DP, keeping KDA/shared/head at TP2 is the
main expected source of low-concurrency performance. The Q8 weight of KDA Q/K/V/O is 136 MiB per
layer, which is 68 MiB per host at TP2. The 4096-wide BF16 block output is 8 KiB/token. This data
volume relation supports sharing the weight read and the computation at a small batch, but the real
gain depends on the kernel efficiency, the communication latency and the complete layer order. The
direct goal of MLA DP is to reduce the cache duplication. It does not promise a lower single-request
latency than MLA TP2. There is no GLM two-host end-to-end performance result yet.

The mHC replication scheme keeps the existing TP/EP output handoff. If the residual were placed only
on the request owner, every KDA/FFN branch would need input distribution and output collection.
Whether that pays must be measured together with the communication stage. The current choice is
replication, and no measured control proves its performance advantage.

## 7. Takeover and retirement conditions

The later execution integration defines the collective, the pack/unpack, the Graph, the prefix restore
and the cross-rank submission, and verifies the complete path. The concrete open work is collected in
the design index. Do not maintain a duplicate task list in each module document. After the
implementation lands, the actual placement/binding, the model execution and the tests take over. The
necessary current execution note goes into the `glm5next-execution.md` established at that time. The
budget formulas stay maintained with the actual geometry and layout. After they move into the matching
resource note, delete this design. Do not keep a second, expired execution specification.
