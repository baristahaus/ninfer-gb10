# Expert weight sources and the GPU replica cache

This document specifies the implemented bank views, placement loading, GPU replica container and MoE compute handoff.
The [artifact format](artifact-container.md) and the Op contracts own the mathematics and the quantization layouts;
the [Engine architecture](engine-architecture.md) owns execution and lifetime.

The Engine supports a resident GPU and MoE mixed execution bounded by the device memory budget. The CPU `routed_swiglu` Op
supports Q4 GU + Q5/Q6 down and NVFP4 GU/down. It consumes the Host bank directly and produces FP32 contributions.
The GPU uses the same set of logical expert IDs. It can compute a fixed subset, or it can consume the completed whole-layer bank.
Qwen3.5 MoE and Qwen4Exp Main both use this common schedule, with eager and segmented CUDA Graph support.

## Source and identity

In `core/weight_bank.h`, `WeightBankLayout` describes one affine matrix set: the local N/K,
the matrix count, and for each codes/high/scales plane the parent offset, the single-matrix byte count and the matrix stride.
`WeightBankRegion` explicitly specifies the start row on the flattened parent, the matrix row count, the count and the row stride.
The geometry comes from `WeightGeometry`; the bank defines no codec of its own.

`WeightBankView` borrows the ModelShard's immutable `WeightParent`. `matrix_weight` returns one
native matrix view, including the activation divisor passed separately per Use. The matrix byte count and the stride may differ:
for example, the view of all gates in a GU bank has matrix row count I and stride 2I.

- Groupwise INT uses the existing rank-2 parent `[E*N,K]`; the model Binding states E/N explicitly.
- NVFP4 uses rank-3 `[E,N,K]`, with codes, swizzled G16 scales and FP32 divisors in separate planes.
  Slices keep the 128-row boundary, and each local matrix sits in the same divisor domain.
- Block-FP8 uses rank-2/3 codes and BF16 `[E,N/128,K/128]` scales; it has no matrix divisor.
- Paged NF4 is governed by its row-read contract and is not represented as an affine expert bank.

`ExpertBankSource` combines the GU and down banks of the same logical expert, and for a Host source it provides
the `CudaHostView` of the corresponding parent. The CPU uses the Host address of the mathematical view; upload uses the registered transfer view.
`ExpertKey={ExpertBankId,logical_expert}` does not change with cache allocation.
Qwen3.5 builds bank IDs in Main layer order and appends the selected MTP bank at the end.
The model loader checks the parent, start and shape of each gate/up/down Binding before it builds the bank;
it does not guess the encoding from physical object IDs or parameter names.

## Logical loading and placement

The load order is `read_definition → resolve_load → materialize_model`:

1. `Binder` checks the logical references, the Uses and the selected resources; an auxiliary value takes the owning Host value of the corresponding element range.
   `ModelDefinition` holds these read Host resources and their `PreparedResources`, and the Frontend borrows their stable addresses.
2. `ModelDefinition` keeps the configuration, the logical references, the Uses and the expert source descriptions, and provides the weight geometry without backing.
   The Engine explicitly builds the immutable `ModelProfile`, queries the
   actual Op workspace before it reads large weights, and resolves the capacity jointly. A purely logical binder carries no execution layout constraint.
   `resolve_load` produces the `ResolvedLoadPlan` of one local materialization, which names the target GPU
   and whether Main experts use a GPU or a CPU source.
   MTP, router, shared expert and the remaining fixed weights stay on the GPU; the placement unfolds along explicit WeightIds.
3. `plan_materialization` deduplicates by `(physical object, Device)` and uses `LayoutBuilder`
   to plan the Host and Device backing separately. The Host `CpuOnly` and `CudaTransfer` needs merge into one source;
   registered objects occupy the aligned prefix of the Host backing, and ordinary CPU objects use the separate region after it.
   When the same parent needs both CPU and GPU backing, both capacities count.
4. The materializer allocates a cacheable `HostBuffer` for a CPU tensor and a
   `DeviceBuffer` for a GPU tensor. Direct I/O writes a CPU-only aligned payload straight into the final Host backing;
   boundary pages and dual CPU/GPU needs use bounded staging, read once and then dispatched.
   Host placement keeps the in-page offset of the file; large Host backing regions are touched in parallel before the read, and the `CudaTransfer`
   region is then registered for its lifetime by `CudaHostRegistration`, still as cacheable CPU memory.
   ModelShard holds the allocation and the registration; after the RankProgram finishes using them, it deregisters first and then releases the Host backing.
   A load with `CpuOnly` needs alone creates no CUDA stream, event or pinned buffer.
5. `bind_view(reference, backing, device)` selects the actual location explicitly.
   ModelShard holds the final backing; all native weight and bank views borrow it and do not depend on the Reader lifetime.

This path keeps the complete parent in its original encoding: it does no runtime repack, and it does not store a Host source as many single-expert objects.
`Parameters` holds the bank identity, the matrix profile and the Use metadata, and it can describe a CPU source.
The workspace query consumes the mathematical profile; a full GPU FFN explicitly accepts a resident or cached CUDA bank,
and mixed single-layer execution additionally borrows the native Host bank and the handoff resources.
Qwen3.5 MoE and Qwen4Exp both support full GPU placement; when the 5090 budget is short, Host Main experts
and the RankProgram GPU cache are available. Spark uses full GPU placement. The Qwen4Exp block-FP8 MTP experts are resident on the GPU;
see [Qwen4Exp execution](qwen4_exp-execution.md) for the model orchestration.

## Replica capacity and GPU addressing

`describe_expert_cache` first returns the layout and the capacities without pointers: the data pool C, each bank entry layout,
the Host directory entry count, the GPU offset/divisor tables and the pinned index snapshot capacity.
`ExpertCache` is built with the RankProgram; the caller supplies the upload execution view explicitly and the consumer execution view at acquire/release.
The ModelShard source, the DeviceContext and the borrowed streams must outlive the cache.
A fully resident source requires C=0 and creates no dynamic offset/divisor/upload-slot storage.
ModelShard charges the Host capacity of a registered source; the registered byte count is a subset of it, so no second weight capacity is added.

Each entry of a CPU source is the GU/down data block of a complete expert and holds only codes/high/scales.
Each plane is aligned to at least 16 bytes and each entry to 256 bytes; different banks use their own entry size.
All banks share one byte pool; the RankProgram explicitly plans the fixed region and the reusable region, and it does not pad a small expert up to the largest expert.
An entry is not a serialized parent, and it carries no artificial divisor trailer.

The NVFP4 matrix divisor is static metadata: a CPU source mirrors it to the GPU at startup, and a resident GPU source
references the original divisor plane directly. The divisor in the Host metadata and in the payload must represent the same set of values;
the source builder guarantees that, and the artifact materializer extracts it from the original payload and verifies it.
Runtime upload moves only codes/high/scales, and the divisor is always addressed by logical expert ID.

`device_weight_bank` / `device_expert_bank` build borrowing views from the native CUDA bank; they allocate no weights and reorder none.
`ExpertBankDeviceView` records both the logical expert count and the CUDA device.
The addressing contract of `ExpertMatrixDeviceView`:

```text
resident: offsets == nullptr; plane_address = plane_base + expert * plane_stride
cached:   offsets != nullptr; plane_address = plane_base + offsets[expert]
divisor_address = divisor_base + expert * divisor_stride
```

For the cached form, plane_base is pool_base plus the bank's plane offset; `indexed_storage` describes the whole borrowed
pool range, and it is used to check that outputs and scratch do not overlap the weights. The allocation addresses of the offset table, the pool and the divisor
are stable for their lifetime, so Graph capture can use them. `UINT64_MAX` is the unpublished sentinel; a consumer may read only
the keys protected by a successful acquire, and it must not treat the contents of other table entries as a guarantee of availability.

The encoded data sizes of the current real weights are below; none includes the allocator or the execution workspace:

| Source | Single-expert dynamic data | Whole-bank dynamic data |
|---|---:|---:|
| 35B Q4 GU + Q5 down, 256 experts | 1,802,240 B | 461,373,440 B |
| 35B Q4 GU + Q6 down, 256 experts | 1,933,312 B | 494,927,872 B |
| Flash Next NVFP4 GU + down, 512 experts | 2,764,800 B | 1,415,577,600 B |

The two divisor sets of the Flash Next Main 48 banks total 196,608 B; the offset table is also 196,608 B.
Its original MTP block-FP8 expert payload is 2,516,889,600 B and counts toward capacity as a fixed GPU source.

## Preparation, publication and consumption

`ExpertPlacement` names the logical key and its 256-byte aligned offset inside the region. Construction installs
the fixed layout `[0,fixed_bytes)`; a fixed entry cannot be replaced during the cache lifetime. `replace(region,entries)`
rebuilds one reusable region completely: the new layout must not overlap, cut an existing entry, or borrow the same key outside the region.
The caller releases the old consumers first; the upload stream waits for their completion events and then overwrites the payload and the index. Uncommitted old uploads are cancelled,
and committed uploads are ordered through the same stream. Replacement moves no replica outside the region and performs no weight D2D.
The reader wait of a replacement enters the upload stream immediately. The RankProgram must first commit the complete upload and publication event of the current layer,
and only then replace the later prefetch windows, so a later window's wait does not block current-layer data that is not yet committed.

`prepare(ExpertCacheRequest)` selects the required and prefetch keys of existing placements; a repeated key counts once.
It changes only the upload demand and keeps the addresses and the cross-batch progress of the selected entries. The RankProgram decides the layout, the window rotation and the retention set.

`progress(ExpertUploadBudget)` submits work within this call's payload-byte and batch-count budget and does not wait on the GPU explicitly.
The startup configuration also bounds the single-batch payload, the copy descriptor count and the in-flight batch count; a zero budget only checks completion state.
The return value says whether required is published, whether prefetch is ready, whether an upload slot is awaited, and the batch count and byte count submitted by this call.

Weights copy directly from the registered source with no CPU packing. A larger contiguous bank range uses pitched H2D,
and a small set or a scattered range uses batched H2D; the choice depends only on the actual addresses and sizes. Transfer descriptors are preallocated and reused.
The plain Host address and the CUDA mapped address are stored separately; batched copies consume the latter, and CPU mathematics keeps consuming the former.
One expert may span batches; its offset is published only after the complete payload is committed. Padding inside the encoding keeps its original bytes,
and entry alignment gaps do not count toward the transfer payload.

An upload slot holds only descriptors, a dirty-index snapshot and a completion event. The index batch is submitted after the data batches;
the slot's snapshot and descriptors are not rewritten before completion. A Host source with no valid registered view is refused before the cache is built.

Publication order:

```text
encoded data DMA → changed offset entries DMA → slot completion event
    → acquire: consumer stream wait → kernels / Graph replay
    → release: consumer completion event → region replacement
```

The index updates only entries that are already safe to modify, may merge adjacent dirty entries, and never overwrites a table entry that another use is reading.
Each published entry records its upload slot and publication generation; acquire waits only for the generations it needs, and a reclaimed old slot generation means
that upload has completed. A later unrelated prefetch does not become a new dependency of an old data consumer.

`acquire(keys,execution)` succeeds as soon as the publication is enqueued on the stream; it does not require the Host to have observed DMA completion.
It returns empty when the key is unpublished. It returns empty when released uses fill the bounded ring, and the caller then waits for reclamation. All slots still holding an unreleased use is a caller error.
Every successfully acquired use must call `release` once on the same consumer stream, including for a resident bank.
release must follow the last real kernel or Graph replay; both handoff operations execute outside capture.
A region replacement waits on the upload stream for the consumer events of released uses, then updates the payload and the offsets;
an unreleased use blocks the replacement. A use records the replica generation, and an old use completing does not reduce the pin of a new replica.
A Host release does not mean the GPU has finished reading. After a CUDA submission failure the cache stops accepting work and drains
committed transfers and consumers before destruction.

The cache holds no CPU thread pool, performs no MoE mathematics and owns no model execution policy.
The layer above expresses actual demand through required/prefetch and the progress budget; a committed copy cannot be withdrawn.
The preparation cost of registration and the first upload is handled at startup. CPU computation and DMA can read the immutable source at the same time,
but they share system memory bandwidth; the RankProgram allocates separate streams for weight upload and the small bridge.

## MoE compute handoff

The mathematics, the inputs/outputs and the alias contract belong to the [MoE Op](../../include/ninfer/ops/moe.h) and
[SumContributions](../../include/ninfer/ops/sum_contributions.h).

- `topk_softmax_route` writes the logical expert IDs, the raw top-k softmax coefficients and the separate shared sigmoid
  coefficients. The routing tensors are I32/FP32 `[K,T]`, and the shared coefficients are FP32 `[T]`.
- `routed_swiglu` consumes the same routing and can compute All, None, or the 1 side or the 0 side of a mask; it does not renormalize.
  The mask expresses the compute split and the GPU offset table expresses the physical location. The two are independent, and the offsets and weights of an unselected expert are not read.
  The optional shared term is computed once on the GPU and also applies to None.
- The CPU branch writes FP32 `[H,T]`; a GPU contribution can write FP32, and the final computation can write BF16 or add the original BF16 residual before a single final rounding.
  `sum_contributions` accepts one or two FP32 contributions and an optional BF16 residual, and produces a BF16 output.
- `topk_softmax_swiglu_moe` defines the fused computation of the complete router, routed/shared SwiGLU and the optional residual.
  It shares the kernel mechanism with the separate Ops and keeps its own format/shape dispatch and tuning policy; the fully GPU Main/MTP
  executes through this entry point.

CUDA mathematics supports the 35B integer expert profiles and the Flash Next NVFP4 and block-FP8 profiles,
and consumes the actual representation and the activation Use through the same set of [MoE semantic entry points](../../include/ninfer/ops/moe.h).
The CPU supports Q4+Q5, Q4+Q6 and Main NVFP4 with BF16 input;
the SiLU and down inputs stay FP32 and the activation is not quantized. CPU block-FP8 is not implemented yet.
`CudaSwiGLUExperts` borrows an affine or indexed GPU bank; `CpuSwiGLUExperts` borrows the complete native Host bank
and consumes it directly through each plane's offset and stride, with NVFP4 using each expert's own matrix divisor.
The CPU weight interface carries no activation divisor. The GPU activation divisor belongs to the Use and the matrix divisor belongs to the weight representation; neither replaces the other.
A fused GU/down takes the intersection of the activation policies of the relevant Uses; NVFP4 A4 needs a common positive activation divisor.

The Op allocates no long-lived resources and uploads no weights; the CUDA branch consumes Device routing, and the CPU branch consumes Host routing synchronously.
The RankProgram provides scratch and the execution view, and owns publication, acquire/release and the bridge completion dependency.
A CUDA Graph with a fixed profile can replay new routing, mask and
index contents; the update must happen after the old consumers finish. shared always stays resident on the GPU and never enters the dynamic expert set.

The single-layer execution in `models/common/moe` borrows `moe::ExecutionView`. `moe::Partition` builds the complementary mask and the cache keys from the
ready GPU ID snapshot of this layer, and the CPU and the GPU consume the same snapshot in this call. The mask is updated on the compute stream,
and pinned staging is not reused before DMA completes. `moe::Bridge` holds the CPU team, the plain Host scratch,
the pinned/Device frame, and the separate bridge stream and events. A resource query returns the frame and
scratch upper bounds for each side, and it can take the maximum over several profiles; the GPU scratch is supplied by the caller and is reused in route/routed order.
A pure GPU split builds no CPU team and no bridge, and a full fused call explicitly accepts a resident or indexed bank.
The common execution supports `Write` and `AddResidual`; the first does not read the old output, and the second adds the existing
BF16 output before the final rounding. Qwen3.5 uses the second, and the HC architecture can consume the independent block output of the first.

The caller first publishes required through cache `prepare/progress`. A mixed call submits the route on the compute stream first,
returns hidden and the contiguous IDs/coefficients, and then submits the GPU routed/shared work. A small T copy is ordered directly on the compute stream,
and a larger T uses a separate bridge stream and events so the transfer overlaps the GPU computation. The caller holds the lease of the GPU split;
the RankWorker runs the synchronous CPU collective once the input is ready, overlapping the GPU branch.
The CPU FP32 contributions upload through the same transfer stream, and the compute stream merges the two contributions and the residual by dependency. The normal path waits only for the CPU input,
and the final output is ordered on the compute stream. One frame is reused continuously by the same RankWorker on fixed streams; the next
route/D2H passes the previous layer's combine, so it does not overwrite a Host contribution that is still uploading. Resource destruction drains both streams.
If this routing has no CPU assignment, the CPU collective and the contribution return are skipped. The single-layer cache handoff executes outside capture.

The RankProgram records the number of CPU layers that actually take part in each complete traversal; under a repeated identical T, it keeps the team polling between layers
when the CPU participates often, and it lets the team sleep when participation is sparse. Start and stop use different thresholds, and the observation restarts
when T changes or the traversal is incomplete. The policy depends on no specific expert ID, no GPU hit-rate assumption and no request phase; its scope is released
when the traversal ends, and a fully GPU streaming computation keeps no CPU team active.

## Engine budget and layer-by-layer execution

`NINFER_VRAM_BUDGET_MIB` is the upper bound on the Device capacity the engine may use, and the free device memory at startup bounds it further.
The budget covers the fixed weights, the RankProgram State/KV, the workspace, the cache, the offset/divisor tables, the bridge/masks
and the runtime margin. Automatic KV first guarantees one complete context; Host mode spends the remaining capacity on experts first.
An explicit KV configuration is still delivered in full. The Host source, the registration range and the pinned frame are reported separately, and the registration range is
a subset of the Host backing. MTP, router, shared, Vision and draft always stay on the GPU.

When full GPU placement is feasible for both architectures, the fused path is used and no Host expert replica is created; the Host path keeps the complete
original encoding source.
`EngineOptions::moe.expert_priority` can specify a priority order over all logical IDs of each Main layer.
The 35B-A3B and Flash Next configurations default to the model-private static calibration order, and other configurations use ID order;
an explicit order overrides the default. The order decides only the cache placement: it changes no router, no coefficients and no expert mathematics,
and it is not updated during a request. The calibration source description and the arrays live in each model's `moe_priority.cpp` and
`expert_priority.cpp`; other training weights with the same configuration can provide their own order.
Each layer gets a byte-balanced allowance and selects the priority prefix in whole experts; the order itself
does not force offload. The priority shapes the target set, and the ready replicas decide the actual GPU split.
`moe.cpu_workers=0` selects the physical cores automatically and leaves cores for the executing caller thread; a nonzero value sets the team size.

The RankProgram holds `moe::Schedule`, which wraps the offload traversal, the CPU activity and the graph handoff; its concrete
`offload::Resources`, the CPU/team plan and the segmented graph compile only for a target that provides offload.
A resident call consumes the ModelShard bank directly and reuses the full GPU fused Op. The common plan follows the actual MoE call order,
and bank IDs are independent of model layer numbers. The model provides the bank placement and the priority of Main/MTP; the mixed-compute threshold and the reference CPU team size are
policy inputs of the compute profile, and the cache plan fixes no model geometry. After the fixed resources, KV, workspace, bridge/masks and indexes are subtracted,
the remaining capacity C splits into the fixed region R and the streaming windows. Startup solves jointly for the balanced fixed allowance and the window capacity: each layer's complete
bank is W_l and its fixed experts are R_l, and every window must hold `max_l(W_l-R_l)`. It does not subtract two complete banks first.
The plan enumerates the allowance boundaries in whole experts and uses the rounding remainder; the fixed set takes each layer's priority prefix, and its count and addresses are stable for their lifetime.
Two windows are the default; when the budget cannot hold two windows it uses one, and when even one window is infeasible it refuses that capacity plan.

For a small T, all window space merges into one contiguous extra cache, which keeps extending each layer's priority prefix by the balanced byte allowance.
The fixed layout, the extra layout and each layer's missing set are all determined at startup; execution starts only after the initial fixed and extra uploads complete.
Each Main traversal forms its demand from the actual T alone:

- Small T: the GPU and the CPU consume the same ready set and its complement. When stable, the snapshot and the lease are reused and no weight uploads occur.
  When returning from a large T, the extra layout is restored between layers in layer order and in batches; there is no need to wait for the whole restore, and an expert that is not ready is
  computed by the CPU. Each complete expert enters the snapshot only after its upload completes, and the state becomes stable again after the restore finishes.
- Large T: the window packs the current layer's missing set and combines it with the fixed replicas into the complete indexed bank; the other window prefetches the next layer.
  The current data publishes its own completion event first, and then the next layer's upload advances. After the window's last read, the next overwrite can be enqueued,
  and single-window mode loads in the same dependency order. The tail remainder of a window caused by different layers' missing byte counts is not used yet.

prefill, Replay, ordinary/control and speculative verification share this entry point, and the MoE policy does not change with the request phase.
Weight upload overlaps the current layer's computation and the next layer's mixer; a run of large T does not restore the small-T layout at a chunk boundary.
Switching between small and large T rebuilds only the windows and leaves the fixed region unchanged; the H2D of one complete extra-cache restore does not exceed the total window capacity.
When a large T starts again during a restore, the uncommitted restore is cancelled and the streaming layout is rebuilt after the uploads already made.
The cache never enters a StateImage, a checkpoint or a speculative rollback.

MemorySummary reports the pool capacity, the fixed region, the total window capacity and the window count, the actual occupancy, the complete small-T target and the actual ready
bytes and per-layer counts. The fixed region plus the windows equal cache_bytes; ready and target are a state or a target within it, so they cannot be added twice.
It also reports the total H2D bytes and the prefetch bytes inside them, and it no longer provides a defragmentation count.

`prefill_chunk=0` means unspecified: the resident model defaults to 1024 and CPU expert offload to 4096.
`--prefill-chunk N` accepts a positive multiple of 128, and startup plans the capacity from the given value or the fixed default; the actual call
is bounded by the remaining context length. The measurement signature includes the priority, each layer's cache allowance, the team, the pool capacity and the chunk,
and it does not apply the resident model's recompute-cost calibration to offload.

Qwen3.5 resident execution uses one whole CUDA Graph; Main offload uses a segmented Graph and also keeps eager submission.
A stable small T fuses the mixer with the leading MoE GPU segment; during window streaming or extra-cache restore the two are submitted separately,
and the snapshot and the weight-ready dependency are taken after the mixer. The CPU waits on the input-ready external event of the graph
and computes in parallel with the GPU routed/shared work; when this round has no CPU contribution, the collective and the return are skipped.
The capture path places the input D2H before the GPU experts so the separate copy branch is not delayed; the result H2D of a larger T
still uses the bridge stream. Eager keeps the input copy branch of a larger T.
The cache handoff and the prefetch stay outside the graph, and the lease release is ordered after the last GPU weight consumer.

MTP's Main verification and DFlash's target use the same segmented schedule; MTP itself and draft
continue the resident Graph. A target executable with the same exact-B is shared across the draft's different topologies,
and selecting a profile installs the matching target envelope at the same time; the MoE segment is not replicated with the context tier.
The constraint mask is prepared after the first target GPU segment is submitted. The Graph instance and the bridge/masks
are both held by the RankProgram. Startup calibration borrows the loaded Host bank and carries the weights in a single-layer window, but captures All/None/MaskOnes and the steady segments
according to the candidate's formal plan; the CPU and the GPU still use the same set of execution primitives.
A change of expert ID or offset does not require recalibration; when the set of compute variants changes, the corresponding set is calibrated.
After the temporary resources are destroyed, the rebuild estimate is subtracted and the expert cache allowance is then fixed; destruction ends the device
work and releases the Graphs first, and then releases the events and storage they reference. Prefill and Replay continue eager submission, and the MoE
compute policy still depends only on the actual T.

The test and measurement entry points are in [Tests](../../tests/README.md) and [Benchmarks](../../bench/README.md).
