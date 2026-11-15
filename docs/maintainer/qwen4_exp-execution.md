# Qwen4Exp Execution and State

This document is the entry point for Qwen3.8-Flash-Next execution support in NInfer; it maintains the layer connections, resource ownership,
the PLE pipeline, MTP/Vision, CUDA Graph and the commit rules of the complete continuation.
The model mathematics, quantization and persistent layout are in [model and weights](qwen4_exp-model.md).
Request scheduling and publication are maintained by [Engine architecture](engine-architecture.md); the common
[context cache](resource-scheduling-and-context-cache.md),
[Paged KV](paged-kv-cache.md) and [expert cache](expert-weight-cache.md) each maintain their own common contracts.

## 1. Current Support Scope

| Capability | Current implementation |
|---|---|
| Text | prefill, ordinary decode, forced control, Replay and cancellation of the public Engine |
| Concurrency and context | a startup-fixed 1–8 resident lanes; this instance's maximum context is 262144, and the actual capacity is decided by the startup budget |
| Vision | images, video, multimedia messages; the shared Vision encoder, MRoPE and feature handoff across Text chunks |
| Native MTP | draft count 1–5, supports Text/Vision, 1–8 lanes, sampling penalty and constrained output |
| MTP head | shares the Main BF16 head by default; an optional complete-vocabulary Q4 optimized head, supported on one machine and at TP2 |
| CUDA Graph | the resident Main whole graph, the MTP draft chain, independent sampling/acceptance graphs, a single-column control graph without the head; expert offload keeps segmented handoffs |
| KV storage | BF16, INT8-G64, FP8-row256, NVFP4-G16, K8V4; index keys are fixed at BF16 |
| Context reuse | complete StateImage and Main/MTP page groups, Device prefix hits, Snapshot/Replay; 5090 also provides a Host replica |
| Frontend and serving | thinking, thinking budget, tool call, GBNF/JSON/schema/choice/regex; shares the Qwen Frontend and the existing HTTP protocols |
| CausalScoring | single-lane Text scoring of the public Engine; Vision or the speculative backend is not enabled |
| Expert placement | Main supports GPU resident; 5090 also provides registered Host backing, GPU cache and whole-layer streaming execution; MTP/shared are fixed on the GPU |
| PLE | V3 disk table, Host hash/NF4 decode, bounded row cache, incremental preparation and double-slot H2D |

Current limitations:

- The speculative backend supports only native MTP; DFlash, DFlash2 and the indexed proposal head are not supported.
- Offload mode keeps the complete Host expert source. The GPU cache has no request-adaptive ordering or decode dynamic swap-in;
  small-batch missing experts are computed on the CPU, and there is no third division path of on-demand swap-in for GPU computation.
- PLE currently has only the disk preparation path; there is no whole-table RAM/GPU lookup.
- Prefill, Replay, Vision encode, MTP history alignment and state adoption all use eager.
  The resident Main graph includes all layers and the PLE consumption; the PLE Host preparation delivers the input before the graph is committed, and offload keeps the CPU expert handoff.

User commands are in [CLI](../cli.md) and [Serving](../serving.md); artifact generation is in
[the conversion guide](../weight-conversion.md). These capabilities describe the current implementation; the published throughput and quality results are maintained separately by
[the performance documentation](../performance.md) and the model cards.

## 2. Axes, Frontiers and Ownership

| Symbol | Meaning |
|---|---|
| T / B | number of physical input columns per row / compact batch row count of this round |
| M | `T*B`, the flattened column number `t+T*b`; Linear, HC and MoE use this axis |
| n[b] | effective input prefix length of this window, Device `valid_columns[B]` |
| F / F0 | executed-input frontier / candidate-window base frontier, both counted in original tokens |
| a[b] | input prefix length finally adopted, not equal to the prefetch, sampling or candidate total length |
| State slot | physical slot of the StateImage; independent of the resident lane, batch row and KV table row |

Under the canonical boundary, the Main frontier is F, and with MTP enabled the Backend frontier is `max(F-1,0)`.
Backend position j pairs Main conditioning j with input embedding j+1, and RoPE uses the coordinates of position j.
The Ledger can hold one more anchor than Main that has been published but not executed; I/O prefetch advances no model frontier.

The four streams of HC are the feature axis; the token count of GDN/QSA/MoE is not multiplied by four.
Only MTP's shared fc_hidden treats the four streams as independent Linear columns and uses 4M columns.
Ordinary `[H,T,B]` and `[H,M]` can be viewed directly; a parent subrange must keep the actual strides.

| Owner | Content |
|---|---|
| ModelDefinition | Config, Binding/Use, representation geometry, Frontend resources and PLE hash controls |
| ModelShard / Parameters | local immutable Host/GPU weights, PLE file backing and the actual Op parameter views |
| Session / ContextDirectory | request continuation, checkpoint, Main/MTP frontiers, RoPE delta, PLE raw-ID tail, allocation and reference of state/page/table/Host ranges |
| RankProgram | fixed backing of StateImage and Main/MTP KV, records, workspace, PLE service and upload slots, Graphs; offload additionally holds the expert cache/CPU bridge |
| Ops | borrowed inputs and outputs and scratch; computation or state extraction, does not publish request state |
| Runtime / Frontend | request scheduling, retention policy, output preview/commit and protocol semantics |

HC/Linear can compute padding physical columns; RankProgram provides bounded safe input.
QSA/PLE allow zero effective columns, and the GDN active batch does not contain empty rows. Main determines the
compact members before the complete forward and keeps the row correspondence of the layers, records and commit selectors.

## 3. Main Layer Calls

```text
embedding + optional visual replacement
  → hc_expand_norm: residual, normalized
  → attention hc_mix: block input, FP32 injection
  → GDN or QSA: block output
  → hc_combine_norm: new residual, MoE normalized
  → MoE hc_mix → routed + shared MoE
  → hc_combine_norm: new residual, next-layer attention normalized
    (the last layer uses final norm instead)
  → final hc_mix → sample hidden → head
```

Combine fuses with the next norm, but the norm consumes the actual BF16 residual.
Before entering PLE, materialize the complete residual through `hc_combine`; after PLE outputs the residual and this layer's
attention normalized, the same layer flow continues. The conditioning kept by the last Main layer
is the residual before the final norm/mixer; it is not normalized or block output.

### 3.1 GDN

The AB projection goes through `gdn_control_proj` to produce FP32 g/beta, and QKVZ goes through the input path in the table below to produce Q/K/V/Z.
Recurrence uses `normalize_qk=true`, `scale=1/sqrt(128)`,
reads the FP32 state updated by the current token; then it does the sigmoid-gated RMSNorm and the output Linear.

| Path | Input combination | State handling |
|---|---|---|
| Prefill / Replay / scoring | B=1; `linear_split → causal_conv1d_silu_split → gated_delta_net` | saves the conv/recurrent state at the end of the whole window |
| Ordinary / single-column control | T=1, B=1..8; `gdn_input_proj_conv_snapshot → gated_delta_net_batch_update` | writes the candidate state slot |
| MTP target verify | T=K+1; `gdn_input_proj_conv_record → gated_delta_net_replay_record` | the base is read-only, saves the actual records, and exact fold on adoption |

Z bypasses the convolution and is kept for the output gate. The BF16 output of the independent LinearSplit is a common boundary;
the fused projection/conv does not add a private BF16 projection intermediate boundary.

Verify records contain the latest history column before the convolution, the K/V after convolution/SiLU and before the L2 norm,
and the actually consumed FP32 g/beta. `GdnReplayFoldPlan` replays the accepted prefix from the same base state;
the result aligns bit for bit with the snapshot of the same physical record-producing block.
The projection must not be rerun with a shortened T instead of the fold. The original physical B, row identity and record layout stay unchanged before adoption.
See [GDN input](../../include/ninfer/ops/gdn_input_proj.h) and
[ReplaySSM](replayssm-gdn.md).

### 3.2 QSA

Main's Q8 Q/gate parent and BF16 KV/index parent call LinearSplit separately;
MTP uses one BF16 joint parent and outputs contiguous segments in the order K, V, index-K, Q, gate, index-Q.
The weight row order and the usable prefix views are in [model weight layout](qwen4_exp-model.md#4-merged-matrices-and-persistent-layout).
After the complete QSA projection, the calls run in order:

| Stage | Handoff |
|---|---|
| `qsa_prepare` | projection, norm/RoPE, old raw tail → BF16 Q/index-Q, main KV, complete block index keys, raw-key/coordinate records |
| `qsa_indexer` | index-Q and compressed cache → Device I32 block IDs `[512,T,B]`, counts `[T,B]` |
| `qsa_sparse_gqa` | Q, original gate logits, selection, main KV → gated BF16 output, then to the O Linear |

Selection is the four-token logical block number in the original token domain and is unrelated to the physical page ID.
Sparse GQA consumes the first count IDs directly and adds the visible tail of the current incomplete block; it does not build an expanded token
list or a globally gathered KV. Each query reads only its own causal range; a candidate suffix is invisible even after it is written.
When the number of visible complete blocks C≤512, the Indexer selects all of them in the same kernel and skips scoring.

Compression uses the raw index-K; across windows it first concatenates the old tail, then does the four-column average, keeping the specified BF16 mean rounding,
and then norm/RoPE. The block-head rotation coordinates come from the actual MRoPE and cannot be guessed from the token ordinal.
The main K norm/RoPE result goes directly into the storage codec without an extra BF16 rounding.
Sparse GQA keeps A16 computation and does not quantize Q or the softmax probabilities to low bits;
the quantized KV is decoded on chip, and the gate is applied only after the necessary value-domain inverse transform and the complete reduction.
The exact formula, codec and read/write domains are in [QSA](../../include/ninfer/ops/qsa.h).

### 3.3 MoE

MoE consumes the `[2560,M]` from HC and uses `MoeEpilogue::Write` to produce an independent block output.
The persistent merge of the Router/shared parent does not force a single computation; shared is always counted once on the GPU.
The complete GPU fusion and the routed subset entry are both implemented; the CPU subset produces an FP32 contribution,
which the common mixed execution merges and rounds at the end.

The resident path calls the complete GPU fused MoE directly, and the bank addresses are kept stable by the ModelShard backing.
The offload path selects a strategy by the actual M in the common `models/common/moe`:

- Small M: the GPU computes the ready experts and the CPU computes the complement; weights are not moved when the state is stable.
- Large M: the fixed cache and the missing set filled in the window form the complete bank, all computation runs on the GPU, and the next layer is prefetched.
- Switching: the fixed region's addresses and count stay unchanged; the window is reused between the large-M streaming region and the small-M extra cache.
  When returning to small M, restore layer by layer, and the parts not yet ready are computed by the CPU.

The per-layer quota is allocated evenly by bytes, and the set is a prefix of the startup priority. Flash Next uses the calibrated static
per-layer ordering by default; `MoeOptions::expert_priority` can override it; the ordering does not affect the router or the coefficients.
The threshold is resolved from the computation strategy and the startup CPU team size; the prefill/decode/verify names do not participate in the dispatch.
The common contract for capacity, upload, lease, CPU/GPU handoff and segmented graphs is in [Expert cache](expert-weight-cache.md).

## 4. PLE Preparation, Transfer and Consumption

### 4.1 Disk Table and Read Service

`ple::disk::Table` receives the bound logical parameters and holds the long-lived read-only file backing, the frozen range mapping and
the per-shard NF4 LUT. Startup does not scan the whole table. A row address is first mapped to the actual V3 file range and then expanded to
the direct-I/O alignment range; a range that crosses a volume or an unaligned boundary can produce multiple reads. A short read is acceptable only when it covers all required bytes;
missing data cannot be replaced with zero padding.

RankProgram's `PrepareService` uses one I/O worker and executes
`hash → row deduplication/cache → physical read merge → O_DIRECT/io_uring → gather → LUT decode`.
The current configuration is a 64 MiB four-way raw row cache, 128 in-flight reads and two window slots.
The row cache and in-flight reads can be reused across windows; required windows come first, and prefetch keeps the capacity of the required windows.
The worker does not call CUDA and does not modify the ledger, frontier or StateImage.

The output is a contiguous BF16 `E[2560,T,B]`; invalid tail columns are set to exactly +0.
The lookup hash, NF4 exact rounding and layout are in [the model document](qwen4_exp-model.md#5-ple-table-hash-and-lookup-exit).

### 4.2 Windows, Tickets and Upload Slots

A ticket consists of a physical slot and a generation. Each row carries the sequence, epoch, start frontier
and raw-ID tail; prepare/append copies the caller IDs before returning and does not borrow mutable caller spans afterwards.

| Operation | Effect |
|---|---|
| prepare | fixes T/B, row identity and the known ID prefix; returns not-preparable when no slot is free |
| append | appends candidate IDs per row consecutively; can start a read immediately |
| seal | freezes the final effective input length; tail columns are truncated and zeroed; no append afterwards |
| HostReady | sealed and all effective E are complete; does not yet mean the Device can consume |
| upload / acquire | enqueues H2D and the event on the independent transfer stream, then lends this generation's Device E, ready/consumed events |
| consumed / release | the slot is reused only after the last Device consumption, H2D and Host writers all retire |

`UploadSlots` owns the pinned/Device E independently; PrepareService borrows its Host slot.
The consumer waits for ready before the first E reader, records consumed after the last reader, and then notifies the owner that it has committed.
The Graph captures these two dependencies as external event nodes; this generation's H2D/event must be enqueued on the producer stream before graph replay. An uncommitted next-generation event cannot serve as this round's readiness signal.
RankProgram creates the upload slots first and the preparation service afterwards; on destruction it finishes the Host writers first and then drains the upload and consumption.
Cancellation revokes the publication intent of that window; shared reads still used by other windows complete.
A late completion can fill the correct immutable row cache but cannot write into a reused ticket.
Read failures propagate through the current unit's error path and do not publish a partial E.

`PlePipeline` combines preparation, incremental ID D2H, H2D, consumption and reclamation.
It keeps advancing the upload of ready windows while waiting for the compute to finish.
Canonical state and accepted IDs do not borrow tickets that are about to be released.

### 4.3 Overlap with the GPU

| Scenario | Current preparation timing |
|---|---|
| First input window | committed once the IDs are known; can overlap with Vision encode, embedding and the layers before PLE |
| Prefill / Replay | prepares the next actual Text window after consuming the current PLE, overlapping with the following Main layer computation |
| Ordinary decode | prepared once the anchor is known; the resident Main waits before commit until HostReady/H2D are enqueued, and waits inside the graph at the PLE entry for the upload to finish |
| MTP verify | prepares the anchor first; each step's draft IDs are asynchronously D2H'd and appended in order; seal when all effective IDs are present |
| Control / scoring | uses the same real input frontier and preparation pipeline |

The GPU handoff is:

```text
raw IDs + raw-ID tail → PrepareService → HostReady → H2D/event → E → KV Linear → K/V
Main embedding / layers before PLE → complete residual
K/V + residual → ple_gate: BF16 U, BF16 N records
residual + U + N + old history → ple_conv_add_norm: BF16 residual, next set of normalized
  → continue Main layers
```

The PLE convolution consumes the N records of norm(actual BF16 U), and the four taps correspond to -9, -6, -3 and 0.
Conv/SiLU adds with the residual/U and writes BF16, then the next HC norm runs.
The last reader of E is the KV Linear; RankProgram records consumed after the whole PLE GPU segment.
The external record in the resident graph lets the Host reclaim that slot while the following layers execute.
The N records and the IDs needed for adoption are kept until commit and are not released with the E slot.
The GPU mathematics is in [PLE contract](../../include/ninfer/ops/ple.h).

## 5. MTP Generation and Adoption

The MTP draft count K is fixed at startup to 1–5; the physical target window is T=K+1.
Each row shortens its effective length by the remaining output/context budget; a mixed batch keeps this physical window,
and when the whole batch can no longer produce a draft, it uses ordinary.
The draft does argmax on the complete vocabulary. `--proposal-head full` shares the Main BF16 head;
`--proposal-head optimized` loads and binds the artifact's independent Q4 head and runs an A16 Linear with BF16 activations.
The column count of each draft head call is B, and a normal K-step draft calls it K times. TP2 computes it by vocabulary shard and
uses the device-side global candidate selection; Eager and CUDA Graph use the same binding.
The workspace plan includes the actual demand of the selected draft head, and the weights are materialized at startup.
The weight layout and capacity are in [model and weights](qwen4_exp-model.md#33-optimized-mtp-head).

The target still uses the Main head; acceptance supports the request's greedy or sampling configuration
and handles penalty and the constraint mask. The acceptance mathematics is maintained by
[SpeculativeAccept](../../include/ninfer/ops/speculative_accept.h).

1. **History setup.** Prefill, Replay and control pair the input embedding with the previous column's Main conditioning.
   The root skips the first column, and a consecutive chunk carries the previous column's old conditioning. Only the MTP stem,
   attention HC, QSA projection/prepare and tail extraction run; attention readout, MoE and the head do not.
2. **Draft.** The first step pairs Main conditioning with the anchor at Backend F-1 and keeps its
   bridge tail separately; later steps use the previous step's MTP conditioning. The working tail/hidden does not become canonical state.
3. **Target.** Main executes `[anchor,drafts...]` and records the GDN transitions, QSA raw keys/coordinates,
   PLE N and the sample/conditioning window. All records keep the original T/B and member order.
4. **Adoption.** The backend accepts c drafts and permits c+1 outputs; the Frontend can further truncate to a.
   Main adopts the first a inputs, and both end hidden take column a-1. The backend starts from the real bridge tail,
   rebuilds its following a-1 columns of history with Main conditioning and the accepted shifted IDs, and commits to F+a-1.
5. **Settlement.** The formal penalty counts increase only by the finally published tokens; the rejected suffix does not enter the effective
   State/KV coverage, and the remaining correction/bonus becomes the next round's anchor.

Even when zero drafts are accepted, a normal round still adopts the anchor, that is a=1. Cancelled or failed rows with zero acceptance do not adopt candidates
and are released by the Engine's final-state flow. The checkpoint saves the Main end pair and the committed Backend state;
restore does not depend on the internal features of pending drafts and does not rerun the head to rebuild the saved hidden.

## 6. Vision and Frontend

The shared Qwen Frontend provides the original IDs, media patches, vision spans and three-axis MRoPE.
RankProgram replaces the visual columns after the embedding and before the HC expansion; MTP history setup also uses the corresponding visual embedding.
The QSA causal order still uses the token ordinal, RoPE consumes the coordinates independently, and generated text uses the saved rope_delta.

One Text window consumes at most one visual item, and the item can span multiple chunks.
RankProgram holds one workspace and a vision handoff independent of the general computation area; each residency's vision execution
borrows that handoff by owner/generation. The Session keeps only the owning Host input and the consumption plan. The encoding scratch can reuse the general area;
Text/decode does not overwrite an effective handoff, another Vision encoding can overwrite it, and the retained patches re-encode it later if needed.
Main and MTP retire the item binding after their corresponding consumption; pause/cancel also retires it.

Visual items already covered by a reused prefix are not encoded again. The checkpoint does not truncate the consumption range of the same visual item;
an item paused midway is restored through Replay. Vision adds no StateImage field or KV pool;
its historical effect has already entered the Main/MTP state.

The converter packs
[the Flash Next template](../../tools/chat_templates/qwen3_8_flash_next.jinja) by default,
and `--chat-template` can override it at runtime. Thinking, reasoning effort, tool calls and structured output
follow the common Frontend implementation; the model code does not handle HTTP protocols or execute tools.
The template default behavior and the override method are in [the conversion guide](../weight-conversion.md),
and the output constraints are in [constrained decoding](constrained-decoding.md).
The sampling defaults come from the artifact's generation_config; thinking and non-thinking share these values,
and the user can override them explicitly.

## 7. Complete State, Checkpoint and Restore

### 7.1 StateImage and Metadata

The state of one canonical Main frontier F contains:

| Data | Storage and meaning |
|---|---|
| GDN recurrent | 36 layers of FP32 `[128,128,48]` |
| GDN conv history | BF16 `[10240,3]` per layer, history before the convolution |
| Main QSA tail | 12 layers, each at most three columns of raw BF16 index-K and the three-axis coordinates of the incomplete block head |
| PLE conv history | BF16 `[10240,9]`, the most recent nine columns of N |
| Main sample hidden | BF16 `[2560]`, from the final mixer output at F-1 |
| Main MTP conditioning | BF16 `[10240]` when MTP is enabled, from the pre-final residual at the same F-1 |
| Backend QSA tail | one layer of raw tail/coordinates when MTP is enabled |
| Host metadata | Main/MTP frontiers, rope_delta, the two most recent executed raw IDs |

The model declares the fields with `StateImageBuilder`; Core stores each complete image contiguously, and Host/Device
use the same internal layout. The raw-ID tail is kept in the continuation metadata and is not put into the Device image again.
At F=0 the Device state is zeroed, the raw-ID tail is filled with EOS, and the end hidden has no valid token yet.

The tail of each QSA pool is decided by its own frontier f; it saves f%4 raw keys and sets the rest to +0;
at a complete block boundary the tail/coordinates are all zero. It is independent of the main KV compressed encoding and
cannot be replaced by already norm/RoPE'd keys.

### 7.2 Common Logical Page Group of Main/MTP

Each logical page covers 64 original tokens: the main K/V data/scale planes of all layers in that pool each have 64 rows,
and the BF16 index plane of each layer has 16 rows. They share the page ID, block table, coverage, COW and migration transactions.
Original token p looks up `table[p/64]`; index block g looks up `table[g/16]`.
The 12 Main layers belong to one pool, and the one MTP layer belongs to an independent Backend pool.

When a page has c effective original tokens, the index has only floor(c/4) effective rows, and the index cursor cannot advance independently.
Prepare can write the candidate suffix; coverage changes only on adoption. COW protects all planes that may be overwritten,
and Device/Host migration keeps their original words; restore must also have the StateImage and metadata of the same frontier.
The specific row density, Host packing and transfer rules belong to [Paged KV](paged-kv-cache.md).

### 7.3 Commit and Restore Boundaries

```text
Session obtains the execution permit, reserves state / pages, and forms the COW and input description
  → RankProgram runs the input preparation, candidate forward and backend acceptance
  → ResultReady; the Frontend preview obtains the accepted prefix a
  → Session commits Commit and publishes the output after obtaining the CommitReceipt
  → RankProgram runs the GDN exact fold, QSA/PLE extract, hidden selection and MTP history alignment
  → DeviceRetired; Session settles the StateImage, each pool's coverage, input identity and resource references
```

`qsa_tail_extract` takes the tail from the actual raw-key/coordinate records;
`ple_state_extract` takes the nine-column history from the N records; the raw-ID tail is updated by the same a.
At a=0, QSA/PLE can copy the old state to an independent destination; GDN fold at a=0 is a strict no-op.
Whether the destination is adopted is decided by the Session's Commit, and RankProgram performs the corresponding physical writes.

Sample and conditioning must come from the same accepted column of the same execution and be copied by the original words;
the final mixer cannot be recomputed for a shorter window. The accepted frontier may not align to a four-token block or a 64-token page.
For example, when F0=6 and a=1, the new QSA tail is raw 4/5/6, and blocks 4..7 written during the candidate stage are still invisible.

A long window produces only the end GDN recurrent state, so the complete-model checkpoint aligns with the actual execution boundary.
The Session splits the window at the selected capture point; QSA/PLE can extract an intermediate prefix but do not by themselves give that point complete
model restore capability. Saving a checkpoint keeps the exact HistoryPrefix and does not immediately copy the tail page because of non-page alignment;
COW runs only on a real write conflict.

Snapshot copies/restores the complete image, each pool's prefix and the metadata; Replay re-executes the accepted input,
keeping the original request identity, output ledger and budget, and does not publish again. Pause can release the lane, and restore uses a new
state slot/table row. Prefix hits use the common resource policy and do not build a QSA/PLE-specific matching or retention algorithm.
The immutable PLE row cache, expert cache, tickets, scratch and draft working features are not part of the checkpoint.

## 8. CUDA Graph and Startup Resources

### 8.1 Graph Boundaries

`program/graphs.*` prepares the ordinary, control, target verify, draft chain and decision graphs by exact-B.
One resident Main replay includes the embedding, all layers, PLE, final mixer and the head needed for generation.
Control uses the head-less graph; the first sampling after a prefix hit can still project the head separately from the saved hidden.
Eager/capture call the same set of GPU bodies; CPU experts, cache leases and disk I/O run outside the graph.

`text_graph.*` separates the organization of resident and expert offload. The resident QSA mixer holds each context envelope as a child
graph; the remaining layer computation is captured directly in the outer graph. PLE consumption is also a child,
including the ready wait, the computation and the consumed record. Switching the upload slot updates only that child and does not copy the complete Main.
The current Host producer delivers the E already enqueued for H2D before the Main commit; H2D can overlap with the preceding in-graph computation,
and the Host read wait no longer overlaps the resident preceding layers. The table residence location does not enter Main's mathematical interface.
Offload is still segmented by the real CPU/cache handoff, and the layers before PLE can execute during Host preparation.

Within the same exact-B, the position, valid columns, state selectors, table rows, page IDs and route/mask
content are Device inputs. RankProgram selects the context envelope and updates the local child; the Op keeps its own
dispatch logic. Startup validates all envelope updates and PLE slot switches; the workspace takes the peak of the allowed profiles.

MTP selects one draft chain by this round's actual depth S=1..K, keeping each row's effective length and the step-0 bridge.
Each step asynchronously copies the ID on the transfer branch and records an external event; the Host appends it to PLE in order;
the chain end reconnects to the compute stream. Then Main verify (including the head) is submitted, followed by the acceptance graph that consumes the logits.
Ordinary likewise separates Main/head from the sampling graph; the constraint mask and sampling configuration are uploaded before the decision.
The final accepted length is still decided by the Host transaction, and the GDN fold, QSA/PLE tail extraction and Backend alignment run outside the graph.

### 8.2 Capacity Planning

Before loading, build the execution profile from the config, Binding/Use and each Op's capacity query.
Startup provides the resident and the target-available offload placement candidates; the Loader builds the original encoding backing at the selected location.
After the weights are materialized, the Graph probe uses a few private KV pages and keeps the actual batch, draft count,
KV format, context envelope and graph sharing relationships. Resident borrows the loaded GPU banks directly; offload
uses the common MoE calibration bank. The Graph preparation description distinguishes the two execution modes.
After destroying the probe, plan the formal resources from the actual available VRAM and the Graph rebuild estimate, following the common startup 1 GiB
headroom and the bounded capacity correction rule. The Graph is destroyed before the workspace, bridge and events it references.

Flash Next's prefill chunk is 4096 by default and can be set explicitly to a positive multiple of 128;
the actual window is limited by the remaining context, the capture point and the media boundary. There is no automatic chunk tuning at startup.
Resident's automatic KV uses the remaining budget after the required resources are deducted; offload first guarantees one complete context and
uses the remaining budget for the expert cache. An explicit KV capacity must be honored.
The Device pool is fixed at startup and does not expand during requests or dynamically exchange quota with the expert cache.

The workspace takes the peak of sequentially exclusive scratch; records, conditioning and the vision
handoff that live across stages are counted separately. The PLE capacity is
`Mmax=max(prefill_chunk,max_concurrency*(draft_tokens+1))`,
two pinned E and two Device E each occupy `2*Mmax*2560*2` B in total,
plus the 64 MiB row cache, I/O buffers and bounded Host metadata. The PLE capacity query uses an independent
TableGeometry; startup retries do not borrow the table of a destroyed ModelShard. Spark counts these Host allocations and the GPU
allocations in the same physical budget, and the Host State/KV cache is zero; the Vision live media quota is reserved separately in the startup budget.

The following is this instance's logical state payload, not yet counting field/slot alignment:

| Content | Per state copy |
|---|---:|
| Main GDN recurrent | 108 MiB |
| Main GDN conv history | 2.109375 MiB |
| PLE N history | 180 KiB |
| QSA raw tail + block-head coordinates | 780 B/layer; Main 12 layers, MTP another 1 layer |
| Main sample / MTP conditioning | 5 KiB / another 20 KiB when enabled |

Each logical page group contains all layers of the pool; the table below lists the main KV plus the 4096 B index plane per single layer:

| Main KV profile | Single-layer 64-token payload | Main 12 layers bytes/original token |
|---|---:|---:|
| BFloat16 (BF16 K / FP16 V) | 135168 B | 25344 |
| Int8Group64 | 71680 B | 13440 |
| Fp8E4M3Row256 | 70144 B | 13152 |
| Nvfp4Group16 | 40960 B | 7680 |
| Fp8KeyNvfp4Value | 55552 B | 10416 |

The MTP pool is counted separately per layer, and its capacity includes the provisional lead.
The actual allocation comes from the expanded plane schema, the StateImage layout and the workspace query;
this payload table does not replace the startup budget. Complete Host experts, the registered subset, pinned context,
the PLE cache and process memory are each accounted separately; the registered range is not another copy of the expert weights.

## 9. Implementation and Maintenance Entry Points

| Responsibility | Implementation |
|---|---|
| Config, read-only definitions and local weight loading | [config](../../src/models/qwen4_exp/config.cpp), [definition](../../src/models/qwen4_exp/definition.cpp), [load](../../src/models/qwen4_exp/load.cpp) |
| Address-free profile, Parameters and layer computation | [execution](../../src/models/qwen4_exp/execution/) |
| Startup budget, resources and graphs | [planning](../../src/models/qwen4_exp/program/planning.cpp), [resources](../../src/models/qwen4_exp/program/resources.cpp), [graphs](../../src/models/qwen4_exp/program/graphs.cpp) |
| PLE hash, read service and upload slots | [ple](../../src/models/qwen4_exp/ple/), [PlePipeline](../../src/models/qwen4_exp/program/ple_pipeline.cpp) |
| Main/MTP physical execution and adoption | [execution](../../src/models/qwen4_exp/program/execution.cpp), [mtp](../../src/models/qwen4_exp/program/mtp.cpp), [commit](../../src/models/qwen4_exp/program/commit.cpp) |
| Request transactions, page groups and checkpoint | [session](../../src/models/qwen4_exp/session/), [directory](../../src/models/common/context/directory.h) |
| StateImage and local context execution | [state](../../src/models/qwen4_exp/state/), [context](../../src/models/qwen4_exp/program/context.cpp) |
| Static expert ordering | [expert_priority](../../src/models/qwen4_exp/parallel/expert_priority.cpp) |

[Tests](../../tests/README.md) maintain the entry points of the Op mathematics, state semantics, PLE and model loading tests;
[Benchmarks](../../bench/README.md) maintain the local and complete Engine measurements.
