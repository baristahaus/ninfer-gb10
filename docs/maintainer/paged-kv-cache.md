# NInfer Paged KV Context Store

This document defines the physical storage and consumption contract of the NInfer growing KV. It is
the maintainer authority for typed KV pools, logical pages, Device/Host replicas, address spaces,
reservations, block tables, and GPU consumer views.

See [Engine architecture](engine-architecture.md) for request order and lifetime, and
[resource scheduling and context cache](resource-scheduling-and-context-cache.md) for checkpoints,
cache retention, preemption, and resource reclamation policy. The KV Store fulfils the physical
requirements of the selected operation and provides the model execution unit with a stable
direct-access view.

The centralized `ContextDirectory` is held by the ModelSession and owns State, pages, history,
execution rows, Host ranges, references, and reservations. A RankProgram holds the fixed backing and
the `ContextStorage`, performs copies and mapping updates by value identity, and keeps no second
free-page allocator. A single rank uses the same contract, driven in-process by the LocalExecutor.

---

## 1. Physical model

The growing KV uses a set of homogeneous pools that are fixed at startup. Each pool:

- stores every plane that shares one frontier and one lifetime;
- uses a fixed original-token page size, plane order, per-plane row span, and page-group count;
- owns its own physical page-ID namespace and free capacity;
- provides one logical address space per active sequence;
- is addressed directly by the consumer through a block table.

A request's KV is not required to be physically contiguous and is not fixed to a control lane. All
active and inactive address spaces share the pool capacity. An ordinary reservation secures the next
unit's increment; a pause-and-resume reservation covers the rebuild up to the old frontier plus the
first real new unit, and is held across chunks.

Paged storage covers the KV that grows with the context. DFlash local cyclic state, Vision/query
temporary K/V, and other fixed state have different lifetimes and are managed by their own StateImage
or workspace contract.

---

## 2. Three independent granularities

The KV architecture distinguishes:

| Granularity | Meaning | Current contract |
|---|---|---|
| allocation granularity | how much token payload a pool takes or releases at one time | `P=64` for each growing pool |
| valid-frontier granularity | up to which logical position the consumer can read | 1 token |
| reusable-state granularity | which frontier has a complete model continuation | a target-defined checkpoint |

A page boundary is not an Attention mask boundary, and it is not a prefix hit boundary. A valid
frontier can lie at any offset inside a page.

The physical row density of each plane is described separately by
`KVPlaneGeometry::tokens_per_row`: one row can cover 1 or 4 original tokens. It changes neither the
allocation, the frontier, nor the original-token domain of a checkpoint; the effective row count is
derived from the coverage as described in section 4.

The KV Store can represent, truncate, or protect a prefix at any token frontier; that does not prove
the model can resume from that position. A reusable frontier requires both a complete StateImage and
the target-defined backend state; see
[Continuation and checkpoint](resource-scheduling-and-context-cache.md#4-continuation-records-and-complete-checkpoints) for the exact
rules.

---

## 3. Typed pool set and capacity

### 3.1 Pool set

The model config and the selected speculative backend determine the pool set at startup:

```text
ordinary:
    Main Text

MTP:
    Main Text
    MTP

DFlash / DFlash2:
    Main Text
    Draft Full, when the selected draft config contains full-attention layers
```

The Qwen3_5 speculative backends are mutually exclusive within one Engine, so that architecture
currently has at most two growing pools. The common resource contract uses pool IDs fixed at startup
and does not limit the number of pools a model has. All five layers of the official DFlash2 config are
local attention, so only the Main growing pool is needed and its draft context uses cyclic storage.

| Pool | Content | Logical frontier |
|---|---|---|
| Main Text | target attention K/V, code/scale, and the compressed index planes the model declares | the target materialized KV frontier |
| MTP | MTP persistent K/V, code/scale, and the compressed index planes the model declares | the MTP KV frontier |
| Draft Full | the selected draft's full-context K/V | the draft context frontier |

Main Text and MTP use the BF16, INT8-G64, FP8-E4M3FN-row256, NVFP4-G16, or K8V4 KV profile the Engine
selects; Draft Full uses its own BF16 profile. The physical layout under the `BFloat16` name is BF16 K
and FP16 V, and the writer converts BF16 V to FP16 once. K8V4 is a closed asymmetric profile, not a
runtime bit-width combination: K is always FP8-E4M3FN-row256 and V is always NVFP4-G16.

`PagedKVStorageLayout` resolves the selected closed profile into the K/V data/scale plane schema; the
target planner expands that schema per layer and fixes each plane ordinal. The common pool
implementation still receives only the expanded `KVPageGeometry`, the plane inventory, and the
capacity, and does not interpret the storage mode. The main K/V data/scale planes of Qwen3.5 and
Qwen4Exp both use `tokens_per_row=1`. Qwen4Exp appends a BF16 index plane at each QSA layer with
`tokens_per_row=4`, and that plane shares the page group and the original-token coverage of that
layer's main KV; the raw tail of an incomplete block is kept in the StateImage. See
[Qwen4Exp execution](qwen4_exp-execution.md) for its mathematics and model handoff, and section 4 for
the common row geometry.

### 3.2 Main capacity

Let:

- \(S\): the `max_context` of one sequence;
- \(C\): `max_concurrency`;
- \(P=64\): the Main page size;
- \(L=\lceil S/P\rceil\): the logical page capacity of one address space;
- \(M\): the physical page-group count of the Main pool.

The usable range is:

\[
M_{min}=\max(L,C)
\]

\[
M_{max}=C\,L
\]

\(M_{min}\) satisfies both the case where a single request alone reaches \(S\) and the geometric
lower bound where each of C requests needs one minimum page. \(M_{max}\) is the physical upper bound
where all active requests reach the per-sequence ceiling at the same time.

The explicit policy of `kv_capacity` resolves to:

\[
M=\left\lceil K_{main}/P\right\rceil
\]

and it requires \(K_{main}\ge S\) and \(M\in[M_{min},M_{max}]\).

### 3.3 Automatic capacity

After the weights load and the Graph calibration resources are destroyed, the automatic policy
resolves \(M\) from the remaining device memory \(F\) within the budget, the planning reserve \(R\),
and the Graph rebuild estimate \(G\). The driver reservation cost is already reflected in \(F\) and is
not deducted again. The model planner provides an affine `SequenceCapacityCurve` from the bound
parameters and the selected execution domain:

\[
B(M)=B_{min}+(M-M_{min})B_{step}
\]

where \(B(M)\) is the explicit runtime Device backing for that Main capacity, comprising:

- the Main and selected-backend typed pools;
- the active/checkpoint State storage;
- the block tables and the fixed persistent state;
- the unified workspace;
- the MoE backing when offload is enabled.

The automatic policy chooses:

\[
M=
\min\left(
M_{max},
M_{min}+
\left\lfloor
\frac{F-R-G-B_{min}}{B_{step}}
\right\rfloor
\right)
\]

and it requires \(F\ge R+G+B_{min}\). When \(M_{min}=M_{max}\), that single point is used directly.
The target generates \(B_{min}\), \(B_{step}\), and the final layout with the same production layout
builder and verifies the capacity curve; the resolver does not copy the model dimension formulas and
does not guess the explicit layout size by allocation probing. The automatic policy for MoE offload
first keeps one complete context and then uses the remaining budget for the expert cache; it
recalibrates when the Graph variant changes, see
[engine architecture](engine-architecture.md#8-top-level-constraints-on-physical-execution).

The finally published Main KV capacity is \(M\,P\) token-equivalents. Page rounding adds only
physical padding and does not enlarge a single sequence's logical ceiling \(S\).

### 3.4 Backend capacity

The selected backend's logical page capacity is still \(L\). Its physical capacity is derived from the
Main \(M\):

```text
backend off:
    Main physical pages = M

MTP with D draft tokens (verification width D+1):
    Main physical pages = M
    MTP physical pages  = M

DFlash / DFlash2 with full-attention layers:
    Main physical pages        = M
    Draft Full physical pages = M
```

MTP generates drafts in the current round. The Main peak for this round is `F+D+1`, and the MTP peak
is `F+D`; each reservation covers its own actual write range, and no extra backend pages are allocated
for drafts pre-generated across rounds. Draft Full is retained according to its own context frontier;
a draft config with no full layer does not allocate this pool.

The pools are physically separate. A free page of one pool cannot become another pool's payload. The
startup plan builds the complete typed capacity vector at once, the directory adopts these capacities,
and the rank binds the corresponding storage; pool geometry is never expanded or re-partitioned at
run time.

### 3.5 Host capacity

The StateImage, the Main KV, and the selected backend KV share the startup-fixed pinned backing of the
rank Resources. The `HostAllocator` in the directory allocates their offsets and lengths centrally,
and the `HostDirectory` manages the extent members of each KV layout; the StateDirectory directly
holds the State allocation of the same allocator. The physical views are resolved by the
ContextStorage. A Host pool ID names the backing it belongs to, and the free bytes of different pools
cannot be offset against each other. The Spark Host context capacity is zero. Host capacity is charged
by the actual packed bytes and the extent geometry, and it does not enlarge the Device capacity or a
single sequence's context ceiling. Inputs, the request ledger, and other CPU data have their own
lifetimes and do not count toward this physical context backing.

---

## 4. Page groups and physical layout

### 4.1 Grouping invariant

Planes enter the same pool only when they satisfy all of the following at once:

1. they use the same logical cache ordinal;
2. they share one committed frontier;
3. they are reserved, materialized, truncated, retained, and released together;
4. they use the same page size;
5. they have no semantics of independent release or independent capacity reuse.

Therefore the K, V, code, scale, and compressed index of the Main layers can share one page-group ID,
while Main, MTP, and DFlash Full must belong to different pools.

### 4.2 Page group

One pool-local page-group ID \(g\) selects the payload of that logical block in every grouped plane at
once:

```text
page group g
├── layer/plane 0 slice g
├── layer/plane 1 slice g
├── ...
└── layer/plane n slice g
```

The planes have backing that is stable for the Engine lifetime, but they are not required to form one
contiguous blob. A page group is the smallest Device unit of allocation, reservation, reference, and
transfer.

Physical IDs may be arranged arbitrarily. The allocator may prefer contiguous IDs to improve locality,
but correctness, admission, and kernel launch topology do not depend on contiguity. Fixed-size page
groups produce no variable-size external fragmentation and need no Device compaction.

### 4.3 Plane row span and Device order

The Device pool uses the original-token page span \(P=64\). Each `KVPlaneGeometry` gives its own
dtype, leading extent \(X\), head extent \(H\), alignment, and `tokens_per_row` \(s\), and the physical
row count is \(R=P/s\). The following combinations are currently accepted:

| \(s\) | rows per page \(R\) | Device order |
|---:|---:|---|
| 1 | 64 | PageMajor or HeadMajor |
| 4 | 16 | PageMajor |

`kv_plane_rows_per_page()` provides the same check for Device and Host planning: \(s\) can be only 1
or 4, \(P\) must be positive and divisible by \(s\), and \(R\) must be representable as a positive
Tensor extent; a four-token row requires PageMajor. The dtype, \(X\), \(H\), and alignment remain the
storage facts of each plane, and \(X\) and \(H\) must be positive. The row span does not denote the
quantization group size of the feature axis; the K/V code and the scale of one token each take one
row. The row span takes part in geometry equality and in Host layout selection; the actual shape of
the Device backing must agree with that geometry.

The two closed orders are:

\[
PageMajor:[X,R,H,N_{physical}],\qquad HeadMajor:[X,R,N_{physical},H]
\]

The K/V planes of Main Text and MTP use PageMajor; DFlash Full uses HeadMajor. For these existing K/V
profiles \(s=1\), where:

- \(X=D\) denotes a K/V or quantized code plane;
- the INT8-G64 scale plane uses \(X=D/64\);
- the FP8-E4M3FN-row256 scale plane uses \(X=1\);
- the NVFP4-G16 code plane uses packed U8 \(X=D/2\), and the scale plane uses U8 \(X=D/16\);
- \(H\) is the KV heads;
- \(N_{physical}\) is the physical page count of that pool.

For logical position \(p\):

\[
b=\lfloor p/P\rfloor,\qquad o=p\bmod P,\qquad r=\lfloor o/s\rfloor,\qquad g=block\_table[b]
\]

\(r\) locates the physical row that contains that original position; whether that row is complete and
readable is still decided by the coverage of section 4.4.

The page-major address is:

\[
address=base+d\,nb_0+r\,nb_1+h\,nb_2+g\,nb_3
\]

The head-major address is:

\[
address=base+d\,nb_0+r\,nb_1+g\,nb_2+h\,nb_3
\]

All grouped planes use the same \(g\), each with its own row span, leading coordinate, and Tensor
strides. The feature axis is contiguous, and address and capacity arithmetic use an integer type that
can cover the complete Tensor span. The exact persistent codec is defined by
[`kv_cache_append.h`](../../include/ninfer/ops/kv_cache_append.h), and the consumer arithmetic by
[`softmax_attention.h`](../../include/ninfer/ops/softmax_attention.h) and
[`qsa.h`](../../include/ninfer/ops/qsa.h); the allocator interprets only the plane bytes, the order,
and the page-group identity.

The physical payload of one token and one head in the D256 Main/MTP profile is:

| profile | K code + scale | V code + scale | K+V |
|---|---:|---:|---:|
| BF16 | 512 B | 512 B | 1024 B |
| INT8-G64 | 256 B + 8 B | 256 B + 8 B | 528 B |
| FP8-E4M3FN-row256 | 256 B + 2 B | 256 B + 2 B | 516 B |
| NVFP4-G16 | 128 B + 16 B | 128 B + 16 B | 288 B |
| K8V4 | 256 B + 2 B | 128 B + 16 B | 402 B |

The K/V code and scale planes have their own dtype, leading extent, and group size; they still share
the page-group identity, the frontier, and the lifetime. The capacity curve, the Device/Host replicas,
the continuation transfer, and the memory summary are all computed from this typed plane inventory,
and `2 * vector_bytes` must not replace the asymmetric byte count of K8V4.

### 4.4 Logical position domain

The block table uses the autoregressive cache ordinal:

```text
logical block b covers positions [b*P, (b+1)*P)
```

RoPE, MRoPE, the Vision axes, and `rope_delta` are Attention input metadata and do not change KV slot
ownership.

For a committed frontier \(F\), the number of valid original tokens in logical page \(b\) is:

\[
c=\operatorname{clamp}(F-bP,0,P)
\]

For a plane with span \(s\), only rows \([0,\lfloor c/s\rfloor)\) are completely valid; row \(r\)
corresponds to original positions \([bP+rs,bP+(r+1)s)\). For example, with \(c=5,s=4\) only the first
row is valid, and the fifth token does not publish the second row early. All planes share the same
original-token coverage and keep no separate row frontier. The consumer must also respect its own
causal read domain and must not treat the existence of a physical row or the coverage at the end of a
chunk as the visible range of each query.

Model state that has not yet formed a complete row is kept by its StateImage contract; the page group
itself cannot replace that state. A provisional write, or a copy of invalid suffix bytes, does not
enlarge the committed coverage.

### 4.5 Page payload

The complete page payload of each plane and the page-group payload are:

\[
PlanePageBytes=sizeof(dtype)\,X\,H\,(P/s)
\]

\[
PageBytes=\sum_{plane} PlanePageBytes
\]

The startup physical bytes come from the complete span and alignment of each plane slab. The
`PageBytes` of different pools may differ, but all page groups within one pool are equivalent. The
Device slab, the Host packed offsets, the transfer bytes, and the capacity query all use the same
plane geometry; a four-token plane must not be charged as \(P\) rows.

The Host layout aligns the start of each plane to 256 B and aligns the end to 256 B to form
`page_stride`. That stride is used for the Host allocation; the payload bytes exclude the padding
between planes and at the end of a page.

`P=64` satisfies the current 32/64-key Attention tiles, the 128-token aligned prefill chunks, the
bounded block-table metadata, and the bounded tail slack at the same time. The prefix hit granularity
takes no part in choosing the page size. Changing the page size, the grouping, or the closed plane
order is an architecture change.

---

## 5. Logical pages and replicas

### 5.1 Logical page identity

A Device page ID is only the physical position in the current Device replica, not a prefix identity.
The centralized PageDirectory maintains generation-checked logical pages for each pool. A `PageId`
contains the pool, the descriptor index, and the generation, and the instance identity is decided by
the execution session; no directory owner pointer crosses an execution boundary:

```text
LogicalKVPage
├── object generation
├── content epoch
├── committed columns [0, P]
├── required/protected columns
├── history references
├── active references and writer state
├── optional Device page-group lease + valid columns
├── optional Host extent membership
└── transaction pins
```

The canonical content of one logical page is identified by `content epoch + committed columns`.
Committed and protected columns always count original tokens, and planes with different spans derive
their effective row count from them. Speculative bytes, or bytes not yet committed, do not extend the
committed coverage.

One logical page keeps the same page ordinal in every history that references it. A fork and a view
share the prefix by original position, COW and growth create a new logical page, and truncation
removes only the suffix; a single history does not reference the same page twice either. Therefore,
when checking the complete holders of a selected page, the actual handle can be queried by that page
ordinal within each checkpoint's coverage.

A replica is valid for the first \(n\) columns a checkpoint needs if and only if:

\[
replica.epoch=page.epoch
\quad\land\quad
replica.coverage\ge n
\]

### 5.2 Device replica

A Device replica uses the consumer-native plane layout of `4`. The directory `PageLease` exclusively
holds one pool-local physical page group; the value-type `PageSlot` contains the pool, the index, and
the generation. The PageAllocator checks the allocation identity, and the rank uses the corresponding
physical position according to the accepted ordered command.

For the same capacity unit, a pool distinguishes:

```text
allocated page lease
reserved but not materialized page
globally available page
```

\[
allocated+reserved+available=capacity
\]

The PageAllocator prefers to continue the complete contiguous range after the preceding page, then
chooses a complete free run, and finally uses scattered runs. Materialize turns a reservation into a
lease without a second capacity admission; dematerialize returns the lease to the same reservation. An
ordinary unit settlement releases the remaining balance; a protected recovery keeps the balance that
future coverage needs and returns it only after real new progress. A checkpoint alias and a shared
history reference do not occupy the physical page twice.

### 5.3 Host replica

A Host replica uses the logical-order packed `HostKVPageLayout`:

- it stores no Device page ID and no block-table holes;
- each page contains the complete grouped plane payload of that typed pool;
- a variable-size extent pays only for the actual page count;
- the Main and backend layouts can allocate extents of different stride in the same arena.

The HostAllocator manages bounded variable-size ranges on the startup-fixed backing. State and the
different KV layouts compete for the same capacity and have no fixed quota against each other. A
request must satisfy the alignment and the contiguous extent geometry; `free_bytes` is only an
occupancy summary, not sufficient proof of allocatability. A shared Host extent is charged once per
actual allocation, and a pending destination already occupies capacity before publication.

Each copy reader holds the epoch and coverage of the concrete replica. A Host reader and a Device
reader protect their actual source separately; removing the last history/checkpoint reference does not
reclaim bytes that a copy still reads. A Host extent can split when other members retire, the read page
range keeps its original address, and a new member capability does not affect DMA already submitted. A
Host replica replacement holds its own destination claim, and no new Host reader pointing at the old
replica can be obtained before publication.

### 5.4 Replica transfer

A Device COW and a D2H/H2D transfer copy the complete plane payload of the page group, moving the
original words by each plane's actual row count, bytes, and offset. Planes of different density are
not allocated, retained, or reclaimed separately. A transfer can merge adjacent physical IDs into
fewer transfer runs.
The DevicePageStorage chooses a 2D copy according to the bound plane geometry: a short PageMajor run
can merge planes of equal width and equal pitch, and HeadMajor can choose page or head as the outer
submission axis. The merged path is used only when it reduces the call count and the pitch is legal;
the Host layout and the payload do not change. After preparation the rank exports an address-free
`KVTransferProfile`, which contains the payload, the Device copy count, and the Host copy count chosen
by the run length; the directory quotes from that actual plan. Migration forms real Host/Device runs
once the destination position is known, and the physical execution checks the plan against the actual
work; the observation records the actual bytes and copy operations. At source-selection time there is
no destination physical ID yet, so only a nominal contiguous-transfer estimate is given from the known
Host runs, and it must not be treated as the actual submission count.
The publication order of a replacement replica is:

```text
reserve destination
  -> copy page payload
  -> verify epoch and committed coverage
  -> publish replica
  -> release source when no longer required
```

The source and the destination are both pinned while the copy runs. The copy, the mapping upload, and
the consumer use ordered dependencies; the consumer cannot read the destination page before the
required copy and table upload complete. The directory publishes the replica after the migration
completes, and the only valid source stays available until the last consumer retires.

### 5.5 Descriptor lifetime

A logical descriptor does not constitute a third payload. It can continue to exist under Device-only,
Host-only, or Both placements. A descriptor can be reclaimed and advance the generation only when the
references, the replicas, and the transaction pins are all zero.

---

## 6. KV history and prefix references

`HistoryOwner` is the single writable owner of a pool, and `HistoryPrefix` holds `[0, F)` of the same
actual history. A reference owns only the logical content and does not pin Device residency. Main and
the backend use their own owners; the model decides their frontier relation. A complete snapshot saves
each pool's prefix together with the StateImage, the metadata, and the input identity of the same
execution.

```text
HistoryDirectory
├── generation-checked history descriptors
├── shared prefix directory / private append path
├── committed frontier and provisional memberships
├── counted retained frontiers
├── unit / recovery page reservation
└── optional execution-row lease
```

Each history contributes one required coverage for its pages and contributes protection according to
the deepest surviving prefix. Adding or removing a reference updates only the changed range and does
not scan the Session's checkpoint table. After the writable owner is released, the directory shrinks
to the deepest surviving prefix; when the last reference disappears, its page references are released.
Structural sharing of directory nodes decides neither content protection nor resource ownership.

Saving a non-aligned prefix does not copy the tail page. The same writer can keep appending, and a
reader reads only the frontier it retained. When a rollback would overwrite retained content, or an
independent writer competes for the same partial tail, the directory branches first and private COW
pages are created. Public publication and private saving use the same prefix mechanism and do not
change the physical layout for a retention purpose.

Each pool establishes a Device block-table matrix at a fixed address at startup:

\[
block\_tables[N_{logical},C]
\]

where `N_logical=L` and `C=max_concurrency`. An active history leases one row, and an entry is a
pool-local I32 physical page ID. A binding publishes the complete mapping, and an append or a COW
publishes only the changed slice. An execution row owns no logical pages and no frontier; a resume
after a pause can lease another row.

## 7. Binding, writing, and settlement

### 7.1 Binding

`binding_demand` computes the missing replicas, the COW pages, and the growth pages from the source
prefix, the target position, and the peak position. It accepts the list of prefixes this authorization
retires and computes the actual protection after retirement; the query itself consumes no source.
`HistoryBinding` then holds the source reference, the destination history, the Device reservation, the
replicas to be restored, and the execution row; the ContextTransaction holds the binding, the State
destination, the Host reservation, and the pure-value actions:

```text
reserve complete binding
  -> enqueue restore, conflicting-tail COW and reserved-row mapping
  -> wait for this batch's completion event
  -> publish replicas and consume transferred source ownership
  -> adopt writer
```

An existing read-only prefix can stay on the history that was taken over. Even when the first unit
appends no token, the binding still takes the writer identity of the partial tail page, which prevents
several pending requests from quoting it as zero COW at the same time. Where MTP and similar paths
must start from an earlier position, the native model provides the target position; the original
frontier and content stay valid until the source is consumed.

The Session checks State and all enabled pools together through the directory. A pause-and-resume
takes the complete coverage of "rebuild to the old frontier plus the first real new unit". If any pool
is short, the authorized retirement does not happen. A retirement already accepted takes effect
according to the request contract; a cancellation of an unadopted destination must finish the actual
DMA before it releases the reservation and the source references.
During the copy, conflicting execution is blocked by the actual writer of the logical page; the
history that holds the source prefix cannot replace the writer identity.

### 7.2 Explicit write ranges

`write_pages(history, begin, end)` computes the growth and COW requirements.
`prepare_write(owner, begin, end, transaction)` handles protection, branching, and mapping from the
actual write start and materializes pages within the reservation already obtained; it does not advance
the committed frontier. A longer speculative mapping that was already prepared can be kept until
settlement, and a shorter write request does not trim it implicitly.

The Session first obtains the complete execution and typed resource license of the call set, and then
enters the irrevocable directory preparation. prepare_write/trim produce the COW and mapping actions;
a later cancellation changes only the intent to adopt the output, and the actions already accepted
still run to retirement, while a preparation failure fails the execution instance and drains it. The
independent destination reservation of a binding can be withdrawn while it is not yet committed. An
ordinary unit reserves only the current peak; a recovery reservation is held across chunks up to the
complete recovery coverage, and other requests cannot use its balance. The Ops and the Graph continue
to consume only typed views, selectors, and block tables at fixed addresses.

### 7.3 Commit and rollback

`prepare_commit` checks the actually accepted prefix and tidies the unadopted provisional memberships,
keeping the retired leases; the committed frontier and coverage do not change at this point. Only
after the Executor accepts the physical commit does the transaction `adopt` advance each pool's KV
frontier, and the model adopts State and metadata at the same time. The bytes of a rejected suffix may
remain but do not enlarge the valid range. Planes of different density still use
`floor(coverage/tokens_per_row)`.

`trim` removes the provisional memberships beyond the adopted frontier; if the old prefix still needs
longer content, the writer branches and keeps the old directory. The necessary COW is performed by
`prepare_write` before the partial tail is actually overwritten. An ordinary settlement returns the
unused pages, and a recovery settlement keeps the balance the target coverage needs; reaching the old
frontier alone does not end the recovery protection.

Logical trim and physical reclamation happen separately. The pages and row leases referenced by an old
execution stay in the ContextTransaction until the device consumers finish. Commit preparation records
the recovery coverage requirement and tops up the reservation after retirement returns the old pages;
an independent settlement of an already stable state uses `settle_growth`.
This avoids temporarily occupying two copies of capacity for the old pages and the next round's
reservation. The current execution slot cannot be reused before the settlement completes.

### 7.4 Release and shared accounting

A pause or a finish releases the execution row, the active references, and the unused reservation,
while a complete snapshot keeps holding its exact prefix. Deleting a checkpoint revokes only the
corresponding references. `released_pages` computes the last physical holder from the prefix set
actually deleted in this run, and a public alias and a shared complete page are not charged twice; the
StateImage is also settled by the actual number of content references.

From the mapping publication to the completion of the GPU unit, the selected rows, the page
membership, and the payload actually consumed stay stable. Prefix references and the chosen mapping
are updated centrally by the coordination side; the COW source leases, the page-table upload source,
and the actual reuse of a row obey that execution's completion dependencies. Cancelling an in-flight
transfer does not return the space early; when a partial commit without a recorded completion fence
fails, the rank drains the actual stream before the directory is allowed to retire.

## 8. Partial pages and replica coverage

The Device and Host replicas of one page group can cover different lengths. For example, the Device
already has the first 48 columns and the Host keeps only the first 40: the Host is still valid for
`[0,40)`. If a reference still needs 48 columns, the Device must not be dropped and 48 columns must
not be restored from that Host. After the last reference that needs the longer range is released, the
shorter Host replica can be used directly to retain the remaining prefix.

Write preparation allows modification only in ranges that no other content owner retains. An in-place
overwrite narrows the Host's valid coverage, and the Device commit then extends its own valid range.
When a longer prefix must be offloaded, a new Host extent is requested and the copy completes before
the old membership is replaced; the quote includes the Host space in which the old and new replicas
coexist temporarily. The other pages of a shared extent stay retained.

The COW of an independent writer copies all the original plane words, and the frontier stays exact to
the original token. For example, with (P=64,F=1000), two writers share the first 15 complete pages and
each writes into the partial page that holds the last 40 columns; a read-only save itself needs no
copy.

---

## 9. Speculative and non-growing KV

### 9.1 Independent pool frontiers

MTP and a DFlash backend with full layers use the backend pool of the ContextDirectory and establish
no private allocator.

In one speculative unit, Main and the backend:

- materialize from their own unit reservations;
- may have different mapped/provisional frontiers;
- commit their accepted frontiers separately;
- trim their rejected trailing mappings separately.

During an MTP draft the backend mapped extent can temporarily lead Main; DFlash Full usually trails
Main. A provisional lead does not constitute committed checkpoint coverage. Rejected bytes may stay in
the partial page, but a new canonical write must overwrite them before any later read.

### 9.2 Fixed and transient K/V

The following storage does not enter the growing pools:

| Resource | Owner |
|---|---|
| DFlash/DFlash2 local sliding-window K/V | fixed per-sequence StateImage |
| DFlash/DFlash2 boundary-local snapshot | fixed checkpoint StateImage |
| Vision/query temporary K/V | RankProgram workspace |

The DFlash/DFlash2 cyclic K/V uses its own `CyclicKVCacheLayerView` and modulo/window semantics; it
holds no page ID, no block table, and no growing reservation. The model config decides its layer
count, heads, and window.

---

## 10. Consumer contract

### 10.1 Single-sequence view

A single-sequence growing-cache Op uses the non-owning `PagedKVLayerView`:

```text
PagedKVLayerView
├── k_pages / v_pages
├── optional k_scale_pages / v_scale_pages
├── block_table          I32 [Nlogical]
├── head_dim
├── num_kv_heads
└── storage
```

A view contains only one layer's plane tensors and one block-table row. It contains no allocator
handle, no request identity, no reservation, no ownership, and no frontier.

### 10.2 Batched view

A batched Op uses the `PagedKVBatchLayerView`:

```text
PagedKVBatchLayerView
├── shared layer planes
├── block_tables         I32 [Nlogical, C]
├── head_dim / num_kv_heads
├── storage
└── table_rows[B]        separate Op input
```

`table_rows[b]` selects the corresponding active address-space row for compact row \(b\). The compact
batch order, the Engine lane, and the table row can differ from one another. The per-row context
length, the valid columns, and the positions are provided by the Op's other typed inputs.

### 10.3 Address translation

Currently \(P=64\):

```text
logical_block = position >> 6
page_offset   = position & 63
physical_page = block_table[logical_block]
plane_row     = page_offset / tokens_per_row
```

The element address is then computed with the closed plane order of `4.3`, and that plane row is
checked to lie within the valid complete rows. A batched consumer first selects the table row with
`table_rows[b]` and then performs the same translation.
The existing codec planes of the K/V views use `tokens_per_row=1`. Planes of other densities are
expressed by the typed view of the corresponding Op; the storage geometry itself does not extend the
mathematical contract of `PagedKVLayerView`.

The wrapper validates the Tensor dtype, the geometry, the closed strides, the table shape, and the
execution envelope. The caller guarantees:

- the logical blocks this run may access are materialized;
- the read domain does not exceed that sequence's frozen valid frontier;
- the writable K/V code and scale all complete before the frontier publication;
- the selected table rows are stable within the GPU unit.

### 10.4 Direct paged execution

Growing-cache Ops consume paged views directly. Page translation is computed at page/tile granularity
and reused, and the inner loop does not interpret the request, the pool kind, or the allocator state.
Kernel correctness cannot depend on adjacent logical pages mapping to adjacent physical IDs.

Paging introduces no gather-to-contiguous cache and no staging copy proportional to the context
length. A route-specific tile, split, warp, and shared-memory scheme can be optimized independently as
long as it keeps the same logical Attention, the persistent codec, and the address contract above. See
[Op development](op-development.md) for the numeric and performance admission rules of an Op.

---

## 11. CUDA Graph and table publication

The plane bases and the block-table matrix base are stable for the Engine lifetime. What changes
across a replay is:

- the table content;
- the `table_rows` selectors;
- the positions, the context lengths, and the valid counts;
- the model state selectors.

Page IDs, request identity, and physical contiguity do not enter the graph key.

Before an execution unit that needs new mappings:

```text
Session materializes pages and prepares mapping actions
  -> Executor transfers owning value descriptors
  -> rank copies mapping IDs into its pinned execution-slot staging
  -> uploads the changed table slices
  -> launches/replays consumers on the ordered stream
```

Batched publication is used when one boundary adds several pages or reactivates a complete membership.
A mapping update must precede the consumer, and the same row must not be rewritten while a replay is
in flight. The Host source of each H2D must also stay valid until the transfer completes; stream order
must not be relied on to let the CPU overwrite pinned mapping data early. Compute preparation and the
independent context migration each have their own execution slot, mapping staging, and completion
event, whose addresses are stable after startup.

---

## 12. Core invariants

1. Each Device page-group lease carries at most one logical page replica in its pool.
2. One pool combines only planes that share the frontier, the lifetime, the page size, and the
   allocation semantics.
3. All planes of one pool use the same page-group ID for the same logical block, and the row density
   produces no second page table.
4. Device occupancy counts allocated leases and reservations; logical aliases are not charged twice.
5. Logical page identity, content epoch, and physical page ID are independent of one another.
6. The valid frontier is exact to the original token; a plane's effective complete row count is
   floor(coverage/tokens_per_row). A page boundary changes neither the Attention nor the checkpoint
   semantics.
7. The coverage a published checkpoint needs cannot be overwritten by a writer; any logical page has
   at most one writer.
8. A read-only prefix does not copy the tail page; on a writer conflict, or when protected content
   would be overwritten, the directory must branch and COW first.
9. A Host/Device replacement is published only after the copy and the epoch/coverage verification
   complete.
10. An ordinary unit settlement returns the unused capacity; a recovery reservation is held across
    chunks until real new progress or a terminal state.
11. Within one GPU execution unit the membership, the block tables, the replicas, and the read
    frontier are stable.
12. A history with no active writer occupies no execution row; an execution row owns no logical
    pages.
13. The Main and backend pools reserve, materialize, commit, and truncate separately.
14. A growing-cache consumer accesses the KV only through the paged view and the block table, and
    takes no allocator or ownership authority.
15. Kernel correctness does not depend on physical page ID contiguity and does not build
    request-contiguous KV through a gather.
16. Checkpoint reusability is proved by a complete target continuation; the existence of a KV page
    does not itself constitute a hit.

---

## 13. Implementation locations

| Responsibility | Primary location |
|---|---|
| pure page geometry, Host packed layout | `src/core/paged_kv_geometry.*` |
| Device layout planning and public paged views | `src/core/paged_kv_cache.*` |
| closed K/V data/scale plane schema | `src/core/paged_kv_storage.h` |
| centralized directory, transactions, and value actions | `src/models/common/context/directory.*`, `contract.h` |
| contiguous pages, Host range allocation | `src/models/common/context/directory/allocation.*` |
| logical pages, replicas, coverage, and references | `src/models/common/context/directory/pages.*` |
| history, prefix references, writing, and binding | `src/models/common/context/directory/history.*` |
| Host extent membership and partial reclamation | `src/models/common/context/directory/host.*` |
| rank-local pages, State, page-table views, and DMA | `src/models/common/context/storage.*` |
| asynchronous action slots, staging, completion dependencies, and observation | `src/models/common/context/transfer.*` |
| StateImage geometry and field views | `src/core/state_image.*`; the architecture declares the fields and typed views |
| State logical identity, slots, and replica reservations | `src/models/common/context/directory/state.*` |
| complete snapshots and record leases | `src/models/common/context/snapshot.h`, `checkpoint_store.h` |
| native context, checkpoint, and recovery mathematics | `src/models/qwen3_5/session/`, `src/models/qwen4_exp/session/` |
| model pool layout and capacity curve | each architecture's `program/planning/` or `program/planning.*` |
| growing-cache Ops | `include/ninfer/ops/`, `src/ops/` |

See [Qwen3.5 model](qwen3_5-model.md) and [DFlash](dflash.md) for the exact model state and the
backend mathematics; the growing-cache Ops in the table above define the persistent KV codec and the
causal consumer numerical contract. The paths locate the current implementation and do not promote a
file or a class name itself into an external interface.
