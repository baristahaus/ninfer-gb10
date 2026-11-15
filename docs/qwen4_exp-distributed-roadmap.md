# Qwen4Exp Dual-Spark Loading and Inference Implementation Roadmap

Goal: on two DGX Sparks, load the existing Qwen3.8-Flash-Next artifact through the public Engine, CLI and serving, complete TP2/EP2 inference, and connect Text, MTP, Vision, constraints, context reuse and CUDA Graph, while keeping the existing single-machine execution and Op dispatch.

This document maintains the implementation order, dependencies, shared capabilities and stage outputs. The execution and lifecycle contracts are maintained by
[the distributed execution design](distributed-execution-design.md); the specific sharding, mathematical boundaries and Graph composition are maintained by [the Qwen4Exp parallel strategy](qwen4_exp-parallel-design.md). The implemented single-rank ownership is in [engine architecture](maintainer/engine-architecture.md). After implementation, the stable contracts are updated to the corresponding authorities, and this file records the stage completion state.

## 1. Starting Point and Overall Path

The starting point on 2026-10-18: the D0 first-version strategy is settled, and the D1 single-rank execution ownership refactor is complete,
committed as `71cdd615`. The Engine centralizes scheduling, the Session manages logical state, RankProgram manages physical
resources and device execution, and the LocalExecutor drives the local rank directly on the Engine worker thread.

The local Op results are spread across existing work branches; migrating them is a prerequisite for the later integration:

| Source | Existing results | Next action |
|---|---|---|
| `.50`'s `feat/spark-tp2-ep2-ops` | TP2 main Linear, split, GDN, Sparse GQA and EP local expert computation; the corresponding checklist is marked complete | Check the mainline against the Qwen-related commit range and migrate the results not yet merged |
| `feat/spark-parallel-ops` | local QSA projection and prepare, shared=320, GPU sharded PLE lookup, sharded vocabulary decision, scoring statistics | Migrate to the current refactored mainline and align the call interfaces |
| `dev` | Definition/Shard/Parameters, Session/RankProgram, LocalExecutor, logical directory and physical storage | Add the multi-rank implementation on top of the existing ownership |

The existing local mathematics and performance evidence are reused with the migration; the integration verification covers the new calls, layouts and communication combinations.

| Stage | Core work | Actual capability after completion |
|---|---|---|
| D2: distributed execution basis | worker startup, control protocol, DistributedExecutor, communication group, NCCL, failure and retirement | Both machines accept a common plan and complete real GPU communication and ordered execution |
| D3: parallel materialization and resources | ParallelPlan, encoded sharded loading, local parameters, State/KV, frame, common capacity | Both machines load the model at the final layout and complete resource preparation and destruction |
| D4: complete eager text | TP/EP layer combination, PLE, sharded output, commit, cache and restore | The public Engine/serve supports dual-machine ordinary generation, concurrency and context reuse |
| D5: complete model features | MTP, Vision, constraints, scoring and state handoff | Both machines cover the complete usage flow of the target model |
| D6: Graph and performance | coordinated capture, complete device segments, dynamic subgraphs, communication overlap and fusion | A dual-machine implementation usable long-term and ready for continued tuning |

The execution order is D2 → D3 → D4 → D5 → D6. The Op migration can run in parallel with D2; after the common plan and the
owner/payload contracts are fixed, sharded materialization can be developed in parallel with communication execution. The Graph address lifetime, communication
order and failure rules are fixed in D2/D3, and D6 completes the whole-model capture and optimization.

## 2. Shared Capabilities Established from the Start

| Capability | Common mechanism | What the model provides |
|---|---|---|
| Deployment / Group | node, rank, device, ordered participants, communication group | which computation role uses which group |
| ParallelPlan | replication, sharding, owner set, parameter selection, layout identity and execution boundaries | the specific division of heads, experts, PLE, HC and so on |
| Executor | execution quota, command order, result handoff, commit, retirement, failure | the native Work and computation stages |
| Encoded weight materialization | source range reads, encoded planes movement, local backing, deduplication | how each logical parameter is sliced and concatenated |
| ContextDirectory | identity, references, coverage, owner completion conditions, capacity reservation | which states form a restorable frontier |
| Communication and execution boundaries | communication group, buffer lifetime, stream dependencies, complete result or local sum | reduction positions, residual consumption and applicable fusion |

### 2.1 Global Definitions and Local Geometry

Config keeps the global model meaning. The address-free, immutable ParallelPlan drives the parameter sharding,
local geometry, state schema, communication boundaries and capacity planning at the same time; RankExecutionPlan then combines the actual
representation and device resources to produce the local execution resource plan.

The weights, workspace, StateImage, KV and native execution consume the same derived geometry. The specific format and
shape capabilities are decided by Parameters and the Op preparation entries, and errors are returned with the rank and the preparation stage.

### 2.2 Local Tensors and Distributed Boundaries

Tensor continues to express local addresses, positions and layouts. The execution boundary describes the replication or
sharding of tokens/features, whether the result is a complete value or a local sum of a group, and the consumption order of residual/norm.
The preparation stage binds the collective, the layout conversion or the complete fused Op; local operators keep their own shape policy.

For example, the `[2560,M]` mixer output is a local contribution and enters HC combine/norm only after the reduction.
When the reduction and HC are fused later, the same mathematical boundary and lifetime contract apply.

### 2.3 Owners, State and Future Models

The Group and owner sets use explicit ordered members, separating the node ID, the global rank, the rank within the group
and the local CUDA ordinal. The relationship between model roles and physical communicators is resolved by the plan.

State can be sharded, replicated, or exist only on some owners. The directory keeps the logical image/pool and the common
frontier, and each rank describes the local schema; per-field replication is resolved into local byte ranges before dispatch.
The completion condition of a restorable source covers all necessary owners.

With four machines the two QSA KV heads involve replication; other models may also place the attention state only on
one owner. These cases reuse the same directory and execution contracts, and the specific layout comes from the new model plan.
The checkpoint layout identity includes the fixed ParallelPlan; this round's reuse happens within the same plan.

A single rank uses the corresponding degenerate plan, size-one boundaries and LocalExecutor. Spark uses GPU
model computation and Device State/KV; the existing CPU expert and Host cache paths of the RTX 5090 continue
to be provided by the local execution implementation.

## 3. D2: Distributed Execution Basis

### 3.1 Startup, Control Transfer and Build

The coordination node holds the Engine, the Frontend and the service entry, and each GPU rank uses an independent worker
process; the GPU on the coordination node is also held by a local worker. The deployment file describes the node addresses, the artifact
local paths, the rank/device mapping and the model parallel configuration.

Startup advances through common stages:

```text
Join → Resolve → Connect → Materialize → Prepare → Ready
```

The local control connection uses a Unix socket and the remote one uses persistent TCP. GPU tensors go through the communication layer; control messages
carry execution descriptions, resource increments and the necessary Host input/results. Network send/receive is separated from CUDA execution,
and the worker execution thread applies commands and observes device progress.

Add the `NINFER_ENABLE_DISTRIBUTED` build option; when enabled it compiles the communication backend, the collective
implementation and the worker application, and discovers NCCL. `NINFER_TARGET` continues to select the hardware target; single-machine and
multi-machine are deployment modes; the existing single-machine and Ops-only builds stay independent.

### 3.2 Typed Protocol and DistributedExecutor

Reuse the existing `submit/supply/commit/discard/poll` contract and complete the cross-process implementation:

- Messages carry the epoch, command sequence, ticket, stage, plan/variant identity and participants.
- The native Work is encoded field by field. In-process owners such as Vision become explicit data or bound resource identities.
- The first binding sends the required input and resources; later messages send the execution window, page table changes, effective lengths and commit increments.
- The output rank returns candidate outputs; Context materialization and resource retirement are aggregated by the necessary owners.
- The execution reservation also keeps the advance capacity of the following supply, commit/discard, results and completion events.
- CommitReceipt means the execution group has accepted the ordered commit; the formal logical adoption is separate from the device completion.
- DeviceRetired covers the compute/copy/communication and other consumers of the corresponding rank.

The first version keeps one generation transaction per sequence in flight. Later cross-ticket early execution uses the address-free
DeviceResultRef to express the producer, result fields, version and lifetime; the current same-ticket device
stage handoff continues to be held by RankProgram.

### 3.3 Communication and Failure

CommunicationContext owns the communicator, the internal communication resources and the registration lease; the caller
holds the input/output buffers. It implements AllReduce, Broadcast, AllGather, and the ReduceScatter/P2P basis required by the design. GPU communication buffers use the same storage and completion contracts as the formal model.

Distinguish the host enqueue completion from the GPU completion. The progress, asynchronous errors and termination of NCCL nonblocking
are handled by the communication owner; dependent operations or completion credentials can be submitted only after the enqueue completes. See
[NCCL communicator contract](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html).

Each variant specifies the order within and across communicators, and each rank adopts the common count, dtype,
root/peer and capture choices. D2 completes real GPU buffer communication and minimal capture/replay,
confirms the actual transfer path and representative message latency. The whole-model Graph is connected in D6; its participant rules follow
[NCCL Graph contract](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html).

A fixed-member execution group stops new submissions after a communication disconnect or device error, terminates pending requests and closes uniformly.
An ordinary request cancellation completes the necessary actions along the established communication sequence, then discards the results and retires the resources. The preparation stage first
confirms that all participants have their local resources, then enters the operations that include communication together.

**Stage output: two workers can run a common command sequence and complete device communication, result aggregation, cancellation and failure closure.**

## 4. D3: Parallel Materialization and Resources

### 4.1 Plan, Local Parameters and Weight Reads

First turn the settled D0 role table into a ParallelPlan, then generate each rank's parameters and state geometry:

| Role | Basic placement per rank |
|---|---|
| GDN | Q/K 8 heads each, V 24 heads, and the corresponding controls, convolution and recurrent state |
| QSA | Q 12 heads, KV 1 head |
| Indexer | 4 query heads, index cache and raw tail replicated |
| Routed experts | 256 complete experts per layer; the global ID is separate from the physical bank slot |
| Shared | intermediate=320, forming a local contribution |
| HC / residual / embedding | fully replicated |
| LM head | half of the vocabulary, Main/MTP share the local physical head |
| PLE | 64 complete NF4 shards, later computation replicated |
| MTP | QSA TP2, experts EP2, the two stem FCs replicated |

Encoded materialization must support: output row selection, K slicing by quantization group, assembling multiple logical segments into a local parent,
collecting complete planes by expert set, and reading a complete PLE shard.

The GDN QKVZ, AB and QSA joint parents are sliced by role; the convolution keeps the tap order and rebuilds
the local Q/K/V channel arrangement. The Q8 K slice keeps the G32 encoding groups; the expert's codes/scales/divisors
are moved together. After loading, the local encodings and stable views are used directly.

Each node reads its own artifact and materializes only what it needs locally. The source encoding stays unchanged, and sharing relationships such as Main/MTP
continue to be deduplicated; loading uses bounded staging and keeps no extra complete writable weight copy in unified memory.
The generic materializer handles the encoding ranges and backing, and the model parallel module handles the parameter semantics and slice recipes.

### 4.2 State/KV, Frame and Common Capacity

ContextDirectory manages the common logical page groups, and each rank builds the physical backing according to its own schema.
One page group represents all necessary shards of the same token range; the local address and byte span of the same ID can differ.

The resource plan covers:

- GDN source/destination images, checkpoint, verify records and fold scratch;
- Main/MTP main KV shards and the replicated Indexer cache/tail;
- PLE history, sample hidden, conditioning and the device raw-ID tail mirror;
- the execution frame, logits, candidates, decision inputs, mapping staging and communication buffers;
- independent regions where the main computation and the PLE bypass live at the same time, the Graph and the communication internal retention.

The logical KV/State capacity is decided by the tightest constraint among the necessary owners. Each rank reports its local capacity and
resource facts, and the coordination side chooses the common capacity; an explicit user capacity keeps the original requirement.

Establish the communication environment first, then obtain the planning snapshot. Spark accounts physical memory per node, and the resources that the coordination node additionally
bears for the Frontend/media are counted in this node's budget. Preparation retries use a common attempt/phase ID
to keep the order of resource destruction, communication entry and formal rebuild consistent.

**Stage output: both machines load the model at the final sharding, establish consistent parameter and state layouts, report the real resource demand, and complete startup and destruction.**

## 5. D4: Complete Eager Text

### 5.1 Native Layer Combination and EP

Start from the complete single-request chain and extend within the same implementation to long prefill, multiple requests and cache restore. The Main layer
boundaries are as follows; eager and the later Graph reuse the same enqueue flow:

```text
replicated residual → HC → local GDN/QSA → O projection local contribution → AllReduce
  → HC combine/norm/mix → local routed + shared TP local contribution → AllReduce
  → HC combine
```

The BF16 local contribution participates in the reduction under the established contract, and the GDN FP32 controls/state keep their original representation.
The native execution calls the derived local geometry; the Op dispatches by the actual format, shape and token count.

EP keeps the global top-k and the normalized coefficients, and filters the local assignments before reading weights and building grouped jobs. A rank with no local routed work still outputs the correct shared/zero contribution and participates in the reduction.
The local combination of shared and routed keeps the established rounding and addition order.

The replicated Router/Indexer bind a common discrete decision path, input layout and global-ID tie-breaking.
Different contexts, the short-context full selection and the effective column count continue to be handled by the corresponding Op, and the communication shape and stage order are chosen in common.

### 5.2 GPU PLE Producer

```text
device IDs + the committed raw-ID tail
  → local NF4 shard lookup → exact merge of the lookup words
  → stable E[2560,T,B] → PLE projection/gate/convolution
```

The existing lookup keeps the BF16 words, including signed zero. The merge uses the integer word semantics of the exclusive owner;
Main consumes the stable E, the readiness dependency and the consumption lifetime.

The lookup can run early once the IDs are known and overlap with the embedding and layer 0; the projection can also run early when the dependencies allow it. PLE uses an independently living region, and the communication sequence and the following block reduction keep the D0 agreement.
The GPU producer aligns with Main's computation interface, and the storage location does not decide Main's graph granularity.

### 5.3 Sharded Output and Formal Adoption

Greedy merges the candidate keys with a total-order tie-breaking; stochastic sampling gathers the local candidates, and the output rank completes the global filtering and sampling and then distributes.
The physical vocabulary padding is excluded, an empty local shard is allowed, and the dead-end is decided by the global candidates. The persistent counts update only the output finally adopted by the Frontend.

The current sharded candidate Op addresses counts/mask by global token ID. The first version keeps the replicated counts,
about 1 MiB per request; the mask transfer and local placement follow the same contract. The vocabulary slice does not automatically change the indexing of these inputs.

### 5.4 Cache, Scheduling and Request Lifecycle

After ordinary generation is connected end to end, complete multiple requests and different contexts, exact/partial prefix hits, tail-page COW, pause,
Device checkpoint restore, Replay, cancellation, output truncation, forced control and resource reclamation.

Prefix matching runs once, centrally. The selected manifest ties the necessary owner shards of the same logical state;
the directory, Session and Executor are respectively responsible for references/capacity, model frontiers and physical completion aggregation. The replica
publication waits for the necessary shards to complete, and slot reuse inherits the last consumer's dependency.

The policy consumes the actual T/B, effective length and context and keeps the existing resource scheduling algorithm; the multi-rank completion semantics
are carried by the directory and the Executor. The checkpoint's replicated fields and sharded fields are saved, restored and adopted together.

**Stage output: the public Engine/serve supports dual-machine ordinary generation with the complete concurrency, cache and restore lifecycle.**

## 6. D5: Complete Model Features

### 6.1 MTP and Constraint Handoff

```text
common conditioning → distributed draft chain → Main verify
  → sharded candidates and acceptance → Frontend determines the adopted prefix → each rank Finish
```

Each draft step's ID and conditioning are connected on the device side, the whole batch's actual depth is chosen in common, and each row's
effective length uses the device metadata. Once the whole-window IDs of Main verify are fixed, one GPU PLE lookup starts.

Finish consumes the same final accepted length and runs the local GDN fold, QSA/PLE tail extraction, the end
hidden/conditioning, token counts and MTP history alignment. The Session manages the formal frontiers of Main `F`
and MTP `max(F-1,0)`, and the temporary draft tail stays in the candidate state.

The constraint matcher sits on the coordination side. After the drafts are ready, it returns the necessary facts and generates the masks; Main verify
and the local head continue to advance, and Decide consumes their results and the masks. RNG, global token ID, logical
position, final-state truncation and adoption semantics keep the existing contract.

### 6.2 Vision, Protocols and Scoring

Vision divides by complete item: the patches are sent to the encoder owner, and the completely encoded merger
features are distributed to the Text ranks. The item order, MRoPE, scatter column order and the Main/MTP
last-consumer lifetime are kept; the workspace counts the actual parallel encoding peak.

Tool calls, thinking and the OpenAI/Anthropic protocol output continue to be interpreted by the coordination-side Frontend,
and the input resources and the finally adopted token stream connect to the existing paths.

CausalScoring uses the existing local statistics Op: each rank computes `(max, exp-sum, target-logit)`,
and after the gather the full-vocabulary logprob is composed. Scoring goes through the public Engine and keeps the independent purpose and fresh-root
state isolation; communication transfers only the necessary statistics.

**Stage output: both machines connect MTP, Vision, constraints, scoring and their cancellation, cache and commit semantics.**

## 7. D6: Graph and Performance

### 7.1 Complete Device Segments

| Path | Target graph organization |
|---|---|
| ordinary, decision inputs ready | Main, PLE, head and distributed decision form the complete graph |
| masks not yet ready | two segments, Compute and Decide, keeping the logits and input lifetime |
| MTP, no draft-dependent Host input | draft chain, verify, head and acceptance in one graph |
| MTP, needs Host masks | Compute advances first, and Decide consumes the masks that arrive later |
| final adoption | the accepted length is a device input and runs the Finish device segment |
| prefill / Replay | executes at the real T; a complete fixed chunk can add a capture profile |

The QSA dynamic envelope is kept and implemented through a pure local child update in the complete outer graph. The communication nodes,
the common order and the buffer bindings are stable; a case that changes the communication contract selects another prepared variant.

Each rank completes its local resource preparation first, then enters warmup/capture/replay together. The Graph, registration
lease and backing are destroyed according to the consumer lifetime. The bypass scratch and cross-stage results use the planned regions,
and the memory use of the complete Graph is included in the common capacity negotiation.

### 7.2 Judging the Bottleneck from the Complete Chain

In the current 48-layer base plan, Main has 96 block reductions; plus PLE and the greedy head,
ordinary has 98 device communications, and K-step greedy MTP has `98+3K`. The BF16 logical payload of each block is `5120*M` bytes:

| M | Single payload | 96 total |
|---:|---:|---:|
| 1 | 5 KiB | 480 KiB |
| 8 | 40 KiB | 3.75 MiB |
| 4096 | 20 MiB | 1.875 GiB |

These are the model's logical payloads; the actual wire traffic and time are decided by the communication implementation and measurement. For small M, observe the per-call
communication latency, the rank skew and the Host commit gaps; for large M, observe the link throughput, the local computation and the overlap at the same time.

The optimization order: remove redundant copies/synchronization/Host round trips, improve the PLE overlap and the EP load assignment, and then advance
the reduction-boundary fusion. Later candidates such as the HC token partition and the Indexer query division are chosen by the complete stage cost.
The local tuning of a single rank is still maintained by the corresponding Op.

Traces are correlated by rank, ticket, command and stage; the group latency is measured at the actual completion boundary, and each rank's
time is reported separately. Finally, measure the actual output speed of ordinary/MTP, concurrency and long prefill through the public serve, separating the accepted length, the per-step computation time, the exposed communication and the resource cost.

**Stage output: complete the whole-model distributed Graph, identify the main performance overhead, and converge the implementations on the critical path.**

## 8. File Organization and Progress Checklist

```text
apps/worker/                          worker startup and shutdown
src/distributed/                      bootstrap, control transfer, groups, NCCL
src/runtime/executor/                 DistributedExecutor, quota, events and retirement
src/models/common/parallel/           shared sharding and execution boundary descriptions
src/models/common/context/            owner-aware directory, references and completion conditions
src/models/common/moe/                combination of local experts and communication boundaries
src/artifact/                         encoded range materialization
src/models/qwen4_exp/parallel/         model sharding recipes, local geometry and execution plan
src/models/qwen4_exp/work.* / wire.*   native Work value contracts and field-by-field encoding
src/models/qwen4_exp/session/          logical transactions and cache semantics
src/models/qwen4_exp/program/          local resources, device stages and Graph
src/models/qwen4_exp/execution/        native layer computation and reduction positions
src/ops/                              local mathematics, collectives and complete fused implementations
```

The common structures are organized by actual responsibility; each model provides its own role mapping, schema and boundaries.
Other models reuse the deployment, materialization, communication, directory and execution lifecycle and keep their own mathematics and tuning.

- [x] D0: converge the first-version TP2/EP2 model strategy.
- [x] D1: complete the single-rank native execution ownership refactor.
- [ ] Prerequisite: check and migrate the two completed Qwen Op results.
- [x] D2 execution basis: the execution group, control protocol, communication, event aggregation and failure closure pass dual-Spark validation.
- [ ] D2/D3 wiring: the product worker, the native codec adapter, the Engine startup and the per-rank resource presentation; completed together with the actual rank materialization.
- [ ] D3: the final weight sharding, local parameters, State/KV/frame and common capacity are loadable.
- [ ] D4: complete eager text, concurrency, prefix cache, pause/resume and Replay run.
- [ ] D5: MTP, Vision, constraints, scoring and the complete commit semantics are connected.
- [ ] D6: the distributed Graph, dynamic profiles and the critical performance paths converge.

Each stage directly checks its new mathematics, state or execution path and produces a corresponding runnable output. The current D2 validation entries are
`ninfer_distributed_execution_test` and `ninfer_collective_bench`, both using the formal process,
transfer, communication and Executor implementations. A product worker that is independent of the actual model preparation is not yet provided; the product entry is bound to the ModelShard, RankProgram and common capacity in D3, and D4 then connects the complete request chain.
