# Kpool Sparse MLA implementation design

Nature: a complete-chain interface and state handoff design that still has to land. The
[design index](README.md) holds the decisions, the implementation status and the task scope. The
[model reference](../../maintainer/glm5next-model.md#4-sparse-mla-and-the-kpool4-indexer) maintains
the model mathematics. The
[numerical policy](../../maintainer/glm5next-quantization.md#3-mlaindexer-cache-and-compute-policy)
maintains the precision, the codec and the representation boundaries. This document defines the
concrete weight merge, the page, the views, the write domain, the selection and the accepted-prefix
interface.

Ownership follows the [Engine architecture](../../maintainer/engine-architecture.md) and the
[Paged KV context store](../../maintainer/paged-kv-cache.md). Qualification follows
[Op development](../../maintainer/op-development.md). This design does not extend the implementation
scope of the [base checklist](op-checklist.md).

## 1. Scope of use and geometry

| Item | Contract |
|---|---|
| first deployment | dual Spark, MLA/Indexer DP2 by request; the owner computes the complete heads |
| use sites | the 11 MLA layers of Main and the 1 MLA layer of MTP; each layer has its own cache and selection |
| Input / Q bottleneck | hidden=4096, Q bottleneck=1536 |
| MLA | 64 heads, Q/K/V head dim=256, latent dim=512, NoPE |
| Indexer | 32 query heads, dim=128, kpool=4, Top512 complete pools |
| valid context | at most 1048576 original tokens |
| local batch | B=1..8; T is any positive physical width, limited by the available views and storage |

This family uses the BF16 and E4M3FN-G128-FP32-scale latent profiles, the fixed BF16 Indexer and the
same compute precision that the numerical document already fixed. Decode, verify and prefill use the
same semantics. The kernel count, the tiles, the pipeline and the legal route dispatch are
implementation choices and add no precision mode.

## 2. Weights, projections and the execution chain

Matrices use the logical `W[N,K]` notation. Native Tensor dimension zero is contiguous, `M=T*B` and
`column=t+T*b`. x is the BF16 `[4096,T,B]` after the external attention input norm. This family does
not run the block input norm again and does not receive the four-stream residual of mHC.

### 2.1 The fixed weight layout

Prepare receives one BF16 contiguous `W_A[2336,4096]` with this row order:

| Row range | Source role | Output use |
|---|---|---|
| [0,1536) | q_a_proj | the Q bottleneck, 1536-wide RMSNorm |
| [1536,2048) | kv_a_proj_with_mqa | the latent, 512-wide RMSNorm |
| [2048,2176) | indexer.wk | 128-wide affine LayerNorm |
| [2176,2304) | index_kpool_compress_gate | the raw pool gate |
| [2304,2336) | indexer.weights_proj | FP32 raw head coefficients |

The last 32 rows keep the FP32 exit directly and must not be converted to BF16 together with the other
projections. They contain no sigmoid, no softmax, no query scale and no `1/sqrt(32)`.

The Q projection uses a BF16 contiguous `W_B[20480,1536]`: the first 16384 rows are the MLA Q_b and the
last 4096 rows are the Indexer Q. An ordinary call writes two independent contiguous outputs directly
through Linear/split. When MTP reuses a selection it uses only a read-only view of the first 16384
rows, with no runtime packing and no duplicated resident weight.

KV_b keeps the factor representation and stores U_K/U_V separately per head. Each head is logically
`[256,512]` and natively BF16 `[512,256,64]`, with the latent axis contiguous. The row order of the
source KV_b within each head is `[K256,V256]`, and the converter keeps the words by this mapping. U_K
and U_V are not multiplied with Q_b/O to make a larger weight. The final `W_O[4096,16384]` is an
independent BF16 Linear.

The norm gain and bias and the pool APE keep the source BF16 words and use a direct gamma or affine.
They do not use the `1+weight` of Qwen4Exp. The APE native layout is `[128,4]` and is raised to FP32
during computation.

### 2.2 The chain

```text
x + W_A
    → prepare_record / prepare_update
        → q_r BF16 [1536,T,B]
        → head_coeff FP32 [32,T,B]
        → latent cache, complete pooled keys, records or the destination tail
    → W_B Linear/split(q_r)
        → query BF16 [256,64,T,B]
        → index_query BF16 [128,32,T,B]
    → kpool_indexer(index_query, head_coeff, index cache)
        → pool selection + selection_end
    → kpool_sparse_mla(query, U_K/U_V, latent cache, selection)
        → heads BF16 [256,64,T,B]
    → W_O Linear
        → block output BF16 [4096,T,B]
```

Prepare is a complete fused Op that contains the projection, the norm and the cache/state effect. It
can reuse the private compute mechanism of Linear, but it does not require calling the public Linear
first and materializing a BF16 W_A output. The pure projections of W_B and W_O still belong to the
Linear family.

## 3. The 64-token page and four-token contiguous loading

P is fixed at 64 original tokens, with 16 pools per page. The per-layer page-major planes are:

| Profile / plane | Native Tensor | tokens_per_row |
|---|---|---:|
| BF16 latent | BF16 [512,64,1,Npage] | 1 |
| FP8 latent codes | E4M3FN [512,64,1,Npage] | 1 |
| FP8 latent scales | FP32 [4,64,1,Npage] | 1 |
| Indexer pooled keys, used by both profiles | BF16 [128,16,1,Npage] | 4 |

Instantiate only the planes of the selected latent profile. Every plane start is aligned to 256 B and
the rows inside a page are packed tightly. The latent row stride is 1024/512 B, the FP8 scale row
stride is 16 B and the index row stride is 256 B. The page stride covers the complete page payload and
keeps the 256 B alignment, and can keep an explicit outer page gap. Do not insert a scale or metadata
between the latent data of adjacent tokens.

All layers and planes of one pool share the page-group ID, the original token frontier and the block
table. `block_tables` is Device I32 `[logical_pages,table_rows]`, with contiguous page entries and an
explicit row span. Physical IDs can be arranged arbitrarily, and several logical pages are not
required to be physically contiguous.

For a selected logical pool g:

```text
logical_page  = g / 16
pool_in_page  = g % 16
physical_page = block_tables[logical_page, table_rows[b]]

index row     = pool_in_page
latent rows   = 4*pool_in_page + {0,1,2,3}
```

One page-table lookup locates four consecutive latents and does not cross a KV page. The four-token
payload is BF16 4096 B, or FP8 codes 2048 B plus 64 B of scales in another plane. The data start keeps
the 256 B alignment naturally. A complete pool can move as one group. An unfinished tail must be
masked by the valid token count and must not read an invalid fourth entry. Selection is not expanded
into 2048 scattered token IDs and does not require a global gather buffer to be built first.

The 11 layers of Main and the 1 layer of MTP belong to different growing pools and each has its own
page namespace, block table and frontier. The latent, the scales and the index inside one pool are
reserved, protected, shortened and released together. The existing `tokens_per_row=1/4` geometry can
carry these planes, but the consumer uses dedicated MLA/Indexer typed views and does not disguise them
as an ordinary K/V pair.

## 4. Public views, metadata and identity

### 4.1 Batch and cache views

```cpp
enum class KpoolMlaStorage {
    BFloat16,
    Fp8E4m3fnGroup128F32,
};

struct KpoolMlaBatchView {
    Tensor positions;      // Device I32 [T,B], original token ordinal.
    Tensor valid_columns;  // Device I32 [B], 0 <= n[b] <= T.
    Tensor table_rows;     // Device I32 [B].
};

struct KpoolMlaLatentCacheView {
    Tensor data;           // the latent data plane of section 3.
    Tensor scales;         // empty for the BF16 profile; FP32 [4,64,1,Npage] for the FP8 profile.
    Tensor block_tables;
    KpoolMlaStorage storage;
};

struct KpoolIndexCacheView {
    Tensor keys;           // BF16 [128,16,1,Npage].
    Tensor block_tables;
};

struct KpoolMlaCacheView {
    KpoolMlaLatentCacheView latent;
    KpoolIndexCacheView index;
};

struct KpoolExecutionEnvelope {
    uint32_t max_visible_tokens; // 0..1048576, the Host resource upper bound.
};
```

The two tables of a combined cache must have the same address, shape and strides, and the same Npage.
The separate read-only entries of MLA and of the Indexer receive only the view they actually need, and
an unused plane is not read.

The positions of the active rows are contiguous and non-negative. Padding consumes no ordinal. Every
valid `p+1<=1048576` and lies inside the matching tables and the resource envelope. Envelope=0 allows
only every row to be invalid. It is not a mathematical mask and not a second sequence frontier. When
B=0 the caller does not submit this Op.

A state slot, a table row, a batch row, a record row and a selection row are different identities and
are mapped explicitly. The Op receives no request ID, no rank, no world size and no communicator. The
caller guarantees that these views belong to the same request, the same layer and the correct history.

### 4.2 Tail and records

```cpp
struct KpoolTailStateView {
    Tensor raw_keys;        // BF16 [128,3,Slots], this layer.
    Tensor raw_gates;       // BF16 [128,3,Slots], without APE.
    Tensor slots;          // Device I32 [B].
};

struct KpoolRawRecordWriteView {
    Tensor raw_keys;        // BF16 [128,W,B], this layer.
    Tensor raw_gates;       // BF16 [128,W,B].
    Tensor offsets;        // Device I32 [B].
};

struct KpoolTailAllLayersView {
    Tensor raw_keys;        // BF16 [128,3,L,Slots].
    Tensor raw_gates;       // BF16 [128,3,L,Slots].
    Tensor slots;          // Device I32 [B], all layers select the same state image.
};

struct KpoolRawRecordReadView {
    Tensor raw_keys;        // BF16 [128,W,L*R].
    Tensor raw_gates;       // BF16 [128,W,L*R].
    Tensor base_positions;  // Device I32 [R], F0, the position of record column 0.
    Tensor valid_columns;   // Device I32 [R], u, the valid record prefix.
    int32_t layers;         // L.
    int32_t record_capacity;// R, 1..8; this call has B<=R.
};
```

L is the layer count that the call contains and supports a single-layer view and all Main MLA layers.
The different frontiers of Main and MTP do not mix inside one tail extraction. The outer index of a
cross-layer record is fixed at `layer*R+b`, and the active B must not be mistaken for the layer stride.
Prepare receives the view of the first B rows of that layer. A physical record row keeps its identity
from production until the acceptance result retires.

W is a positive record width. A write requires `0<=offset[b]` and `offset[b]+n[b]<=W`. When records
accumulate consecutively, the current start is `base_positions[b]+offset[b]`. The caller manages the
valid length and the source identity, and the Op does not publish this metadata itself. A tail and a
record require only the feature axis to be contiguous. The other strides are explicit, positive and
non-overlapping. Do not assume that a layer, a field or a slot is adjacent.

### 4.3 Selection

```cpp
struct KpoolBlockSelection {
    Tensor pool_ids;        // Device I32 [512,Nselection].
    Tensor pool_counts;     // Device I32 [Nselection].
    Tensor selection_end;   // Device I32 [Nselection].
};
```

The Indexer outputs `Nselection=T*B`, with row number `t+T*b`. Sparse MLA also receives a Device I32
`selection_rows[T,B]`, which can reference a still-valid selection table of a different width,
including the first-step result of MTP. A selection is execution frame data and cannot replace the
read lease or the checkpoint identity of its cache.

## 5. The complete Prepare computation

### 5.1 Parameters and representation boundaries

```cpp
struct KpoolMlaPrepareParameters {
    Weight a_projection;   // BF16 contiguous W_A[2336,4096], fixed A16.
    Tensor query_norm;     // BF16 [1536].
    Tensor latent_norm;    // BF16 [512].
    Tensor index_norm;     // BF16 [128].
    Tensor index_bias;     // BF16 [128].
    Tensor pool_ape;        // BF16 [128,4].
};
```

The five projection results are produced by the
[preparation numerical boundaries](../../maintainer/glm5next-quantization.md#42-mlaindexer-input-preparation):
q_r, the latent encoder input, the raw K/gate and the FP32 head_coeff. The norm, the pooling and the
records consume the representation already specified together, and this interface adds no extra
rounding.

The public q_r is BF16 `[1536,T,B]` and head_coeff is FP32 `[32,T,B]`, contiguous and aligned to
16 B. The W_A output has no forced BF16 landing point before the norm. k_raw/g_raw is the common
representation of the pooling, the records and the tail: even when all four tokens of one pool are
inside the current call, these BF16 words must be used. Do not bypass the boundary and use the
unrounded FP32 values instead.

### 5.2 Pooling and the write domain

Let the start of an active row be `F=positions[0,b]` and its length be n. before is the canonical tail
at F. For each newly completed `g in [floor(F/4),floor((F+n)/4))`, and for each channel d:

Compute the pooled key by the per-channel four-token weighted pooling of the model, and follow the
numerical document for the numerical order. The raw words earlier than F come from before, and the
rest use the words that this call actually produced. APE is added only during pooling. The LayerNorm
is before the pooling.

Prepare writes the latent of `[F,F+n)`, the complete pool range above and the selected state exit. An
unfinished pool gets no index cache write. Preparing the whole window in advance is allowed, but
section 6 decides the visible set of each query independently. Do not use the coverage at the end of
the window in place of the per-query causal boundary.

### 5.3 The two state exits

```cpp
void kpool_mla_prepare_record(
    const Tensor& x, const KpoolMlaPrepareParameters& parameters,
    const KpoolMlaBatchView& batch, const KpoolTailStateView& before,
    KpoolMlaCacheView cache, KpoolRawRecordWriteView records,
    Tensor& q_r, Tensor& head_coeff, MemoryArena& workspace,
    CudaExecutionView execution);

void kpool_mla_prepare_update(
    const Tensor& x, const KpoolMlaPrepareParameters& parameters,
    const KpoolMlaBatchView& batch, const KpoolTailStateView& before,
    KpoolMlaCacheView cache, KpoolTailStateView after,
    Tensor& q_r, Tensor& head_coeff, MemoryArena& workspace,
    CudaExecutionView execution);
```

The record exit saves the actual k_raw/g_raw of this call word by word at `offset[b]+t` and keeps
before unchanged. The valid raw words are the only record source. Do not shorten T or rerun another
producer to restore an accepted prefix. Verify keeps these records until the decision retires. They do
not enter a long-term checkpoint.

The update exit generates after at F+n directly by section 8 and does not need to keep the whole raw
record range persistently. An ordinary prefill or decode uses update. An MTP provisional draft state
can also update its own independent target. The two exits use the same computation formula and
representation boundary and do not change the numerical policy because of the exit.

When n=0, q_r/head_coeff write exact positive zero and no cache, record or after is written. No payload
of the state, the positions or the table selectors is read. The matrix projection is allowed to read
the physical padding x, so every physical x column must be valid finite BF16 data, and the padding
must not affect the valid output. An invalid record suffix and an unaddressed cache stay as they are.

## 6. The Indexer and grouped selection

```cpp
void kpool_indexer(
    const Tensor& index_query, const Tensor& head_coeff,
    const KpoolIndexCacheView& cache, const KpoolMlaBatchView& batch,
    KpoolExecutionEnvelope envelope, MemoryArena& workspace,
    KpoolBlockSelection& out, CudaExecutionView execution);
```

The input index_query is contiguous BF16 `[128,32,T,B]` and head_coeff is contiguous FP32
`[32,T,B]`. For a valid query p, `E=p+1` and `C=floor(E/4)`, and only `g in [0,C)` is evaluated:

```text
dot[h,g] = sum_d(FP32(index_query[d,h]) * FP32(index.K[d,g]))
S[g]     = sum_h(head_coeff[h] * ReLU(dot[h,g]))
```

The operands of the Q/K dot product are BF16, and the accumulator, the head multiplication and the
reduction are FP32. S differs from the mathematical score of section 4 of the model only by the common
positive factor 1/64, and omitting it is part of this contract. The coefficients are signed and must
not move inside the ReLU. The final score can be negative. An illegal candidate is excluded directly.
Do not zero-fill it and let it join the TopK. The Indexer scores take part in the selection only and do
not enter the MLA logits.

Select a fixed `min(512,C)` distinct pools. When two scores are equal, the smaller logical g wins. The
output order is not specified and is not required to sort by score or ID. When C<=512, select
everything directly, read no payload of the query, the coefficients, the keys or the tables, and still
output the correct count and E. This optimisation covers E<=2051.

For a valid output row `m=t+T*b`, the count is the selection count above and end is E. Only the first
count IDs are valid, and the content after them is unspecified. An invalid row writes I32 zero to
count/end and reads no positions, query, coefficients, cache or table selector. The ID content is
unspecified. Every input and cache stays unchanged.

Do not approximate the TopK. The implementation can partition, merge or privately sort, but it must
satisfy the same Top512 rule and the same tie rule. Only the tail of an unfinished pool is forced into
the later attention. The newest complete pool is not forced in as well.

## 7. Sparse MLA and the frozen tail boundary

```cpp
struct KpoolMlaFactors {
    Tensor key_up;         // BF16 [512,256,64], logically U_K[h][256,512].
    Tensor value_up;       // BF16 [512,256,64], logically U_V[h][256,512].
};

void kpool_sparse_mla(
    const Tensor& query, const KpoolMlaFactors& factors,
    const KpoolMlaLatentCacheView& cache, const KpoolMlaBatchView& batch,
    const KpoolBlockSelection& selected, const Tensor& selection_rows,
    KpoolExecutionEnvelope envelope, MemoryArena& workspace,
    Tensor& out, CudaExecutionView execution);
```

Query/out are contiguous BF16 `[256,64,T,B]` and selection_rows is contiguous Device I32 `[T,B]`. A
valid query takes E, count and the IDs from the given selection row and requires:

- `0<=E<=p+1`, and the selection belongs to the same request, the same layer and the correct cache history;
- `0<=count<=min(512,floor(E/4))`;
- the first count IDs are distinct, non-negative and smaller than floor(E/4).

The consumer can receive any legal set independently and does not require it to be exactly the actual
TopK of the current query. Let J be the four tokens of each selected g, and merge in
`[4*floor(E/4),E)`. These two parts do not overlap and hold at most 2051 tokens. E is the frozen
boundary of the selection and must not be replaced by p+1 of the current query. Only a valid selected
row and a valid tail are read.

For each head:

The query produces BF16 heads after the U_K mapping, the latent attention and the U_V mapping. The
complete mathematics and rounding order are in the
[latent attention numerical policy](../../maintainer/glm5next-quantization.md#43-attention-in-the-latent-domain).

c comes from the selected cache, and the FP8 profile follows the
[cache consumption rule of the numerical document](../../maintainer/glm5next-quantization.md#33-consumption).
The scale is fixed at 1/16 and matches the 256-wide QK of training. It does not change because
latent=512. There is no RoPE, no attention bias, no output gate, no extra sink and no extra forced
self token.

QK, the weighted accumulation, the factor projection and the private q_latent/u all follow the
numerical policy above. The public output is the BF16 heads after the complete U_V mapping. The final O
projection is not in this Op.

An empty J and an invalid query output exact BF16 positive zero and read no query, factor or cache
payload. An invalid query also reads no positions, selection or table selector. The inputs, the
selection, the metadata and the cache stay unchanged. Private tiling does not change this read domain.
Do not load an invalid fourth tail entry.

## 8. The canonical tail and exact prefix extraction

### 8.1 State

At the source slot of frontier F, the first `F%4` raw K/gate columns of each layer are the unfinished
pool and are stored in time order. The other columns are BF16 positive zero. The phase is derived from
F and no second counter is stored per layer. When F=0 both planes are all zero. APE is not stored in
the raw gate.

The StateImage holds the tail and the complete pools stay in the growing pool. The existing directory
owns the selection, the copy and the protection of the active image, the immutable checkpoint and the
target image. The tail cannot be recovered backwards from the MLA latent.

### 8.2 Extraction and zero acceptance

```cpp
void kpool_tail_extract(
    const KpoolTailAllLayersView& before, const KpoolRawRecordReadView& records,
    const Tensor& accepted_columns, KpoolTailAllLayersView after,
    CudaExecutionView execution);
```

accepted_columns is Device I32 `[B]`. Each row requires `0<=a<=u<=W`, before corresponds to F0 and
`F0+u<=1048576`. L, record_capacity and the views match. The slot counts of before and after can
differ. A view with L=1 extracts one layer, or one call covers all Main MLA layers. The model does not
replay token by token.

For a>0:

```text
F1 = F0 + a
r  = F1 % 4
s  = F1 - r

for j in [0,3):
    if j >= r:
        after.K[:,j] = BF16 +0
        after.G[:,j] = BF16 +0
    elif s+j < F0:
        old_column = s+j - (F0 - F0%4)
        after.K/G[:,j] = before.K/G[:,old_column]
    else:
        after.K/G[:,j] = records.K/G[:,s+j-F0]
```

This is an exact word copy and zero fill with no floating-point operation, and it keeps the bit
pattern of a copied word. Read only the accepted prefix that is needed. Do not read a rejected suffix
and do not scan pools that completed earlier.

a=0 is a strict no-op. It reads and writes no state or record payload and copies nothing even when
source!=destination. That row keeps using the original image and must not adopt an unwritten target.
This differs explicitly from the cross-address zero-acceptance copy behaviour of the existing QSA
tail_extract and agrees with the zero-acceptance rule of the agreed KDA commit.

An in-place extraction of the same row is allowed and uses the original source words. The target
regions do not overlap and must not overwrite the source of another active row or any record or
metadata. A read-only source can be shared. The alias is defined by the address actually selected and
is not inferred from a view object or a slot number. Only the tail fields of this family are modified,
and KDA, hidden and the other StateImage fields are not touched. There is no global scratch and no
hidden allocation.

Prepare/update uses the same tail formula for a valid n and treats the actual raw words of this call
as a temporary record. n=0 likewise writes no target. It does not need to expose or keep the whole
record range to the caller.

### 8.3 Adoption of the growing cache

The valid range at F1=F0+a is:

```text
latent       : [0,F1)
pooled keys  : [0,floor(F1/4))
tail         : [4*floor(F1/4),F1)
```

For example, F0=63, this round prepares up to 69 and only 65 is accepted: pool [60,64) is valid.
[64,68) is not visible even when it was physically written, and the new tail keeps only the raw words
at position 64. A rejected suffix does not widen the coverage. The caller owns the protection and the
copy-on-write of a shared page, the reservations, the table update and the frontier publication. A
complete checkpoint holds the correct growing prefix, the tail and the other model state together.

## 9. MTP and the DP owner handoff

The Main and MTP caches and tails are separate, the owner is the same, and no selection is shared
across requests or layers. The first step of an MTP draft round generates the selection normally, and
the later steps reuse the pool IDs, the count and E. Do not freeze only the pools and then recompute
the tail at a new position.

The later steps omit the Index Q projection, the scoring and the TopK, and the MLA Q uses the first
16384 rows of W_B. Prepare still maintains an independent provisional latent, pooled key and tail and
produces head_coeff by the interface. A reuse step does not consume that coefficient. In this way the
provisional content of the grouped planes stays complete. The selection and its mapping live in the
frame, and the cache read leases stay until the last consumer retires. A new selection write must not
overwrite a frozen row that is still referenced.

The canonical MTP history updates from the real Main conditioning. A recursive draft hidden does not
equal a real Main hidden because its token was accepted. The accepted part that needs rebuilding uses
the normal prepare/update chain and must not adopt these states by shortening the draft cache. The
family uses the ordinal, state and cache views that the caller provides and does not derive the
Main/MTP frontier offset, the input token or the conditioning itself.

RankProgram holds the fixed backing, the frames, the workspace and the Graph. ModelSession selects the
source/destination, the accepted length, the frontier, the checkpoint and the distributed adoption
relation. The Op allocates no page, chooses no owner and runs no communication. The model execution
hands the complete block output of the owner to the later TP/EP. KDA still uses the independent TP
state shard of the two ranks.

## 10. Memory, workspace and Graph

The activation and the output are contiguous and aligned to at least 16 B, and the BF16/FP32 parameter
vectors are naturally aligned. The Device I32 metadata is contiguous and naturally aligned. The
state/record feature axis is contiguous and positive explicit outer strides are allowed. The access
span must be representable and must not overlap. The layout inside a cache page strictly follows
section 3.

Apart from the in-place tail update of the same row that section 8 explicitly allows, every write
region and read region do not overlap. Read-only views can share backing. The regions actually
accessed by an output, metadata, cache, record and active scratch must not have an illegal write or
read overlap. The Host checks the visible dtype, shape, stride, alignment, capacity, device and fixed
address relation. The caller guarantees the device counts, the selectors, the readable pages and the
post-selection alias relation, and the implementation does not read device metadata back.

Prepare, Indexer and sparse MLA each provide a caller-owned workspace capacity query:

```cpp
size_t kpool_mla_prepare_record_workspace_capacity_bytes(
    KpoolMlaStorage storage, int32_t min_tokens, int32_t max_tokens,
    int32_t batch_size, int32_t multiprocessor_count);
size_t kpool_mla_prepare_update_workspace_capacity_bytes(
    KpoolMlaStorage storage, int32_t min_tokens, int32_t max_tokens,
    int32_t batch_size, int32_t multiprocessor_count);
size_t kpool_indexer_workspace_capacity_bytes(
    int32_t min_tokens, int32_t max_tokens, int32_t batch_size,
    KpoolExecutionEnvelope envelope, int32_t multiprocessor_count);
size_t kpool_sparse_mla_workspace_capacity_bytes(
    KpoolMlaStorage storage, int32_t min_tokens, int32_t max_tokens,
    int32_t batch_size, KpoolExecutionEnvelope envelope,
    int32_t multiprocessor_count);
```

The query covers the whole positive T range under the fixed profile, B and envelope, including the
dispatch boundaries. It counts the internal peak and the alignment gaps with an Op-local cursor
aligned to 256 B and excludes the output, cache, state and record backing and the trailing padding.
The SM count is the same as for execution. Execution and sizing share the dispatch and the allocation
rules.

The scratch uses bounded query windows. Do not materialize the complete Indexer score of
[query,head,context] and do not build a globally expanded BF16 latent cache. The frame keeps the
independent records and selection until their last reader retires, and they cannot serve as reusable
temporary arena space.

A Graph replay with the same fixed T/B, profile, views, envelope and device facts reads the new
counts, positions, slots, page mappings and selection mappings from the device. There is no Host
readback, no hidden allocation and no Host launch topology that depends on a changed device value. When
the Host envelope changes, prepare a capture or an update that meets the resource upper bound.
Rewriting a Host variable does not modify an already captured Graph. The operation submits to
execution.stream(), and the same-stream order or an explicit dependency makes production and
consumption visible. The caller keeps the actual device lifetime of the workspace, the records, the
state and the page leases.

## 11. Capacity and performance criteria

The unified budget of the 1M growing cache, the request tail, the TP2 KDA state and the execution
resources is in the
[parallel resource design](parallel.md#3-persistent-state-capacity-for-a-1m-context). The raw records,
the selection and the workspace are charged separately by the actual T/B/capacity of this design and
by the sizing of section 10. Do not estimate the startup memory from the cache payload alone.

The performance comparison covers the complete prepare/indexer/sparse-MLA/tail chain, separates BF16
and FP8 latent storage, and records the actual context, T/B, valid length, selection reuse and Graph
condition. An FP8 gain must count the scale read, the on-chip decode and the branch projection, and the
gain of four-token contiguous loading must be measured on the complete consumer. Local kernel data
cannot replace a two-host model result.

Measure through the complete public Op with fixed hardware/toolchain, input, T/B, context, storage,
selection and run conditions, and report the two prepare exits, the Indexer, sparse MLA, the tail
extraction and the necessary combinations separately. Cover the four phases, the short-context
select-everything case, the long-context scan and the MTP frozen selection. Report the temporary space,
the distribution, the worst change and the exceptions. The control and the workloads are fixed before
the evaluation, and a microbenchmark does not replace two-host model performance.

## 12. Acceptance contract and implementation boundary

| Object | Acceptance basis |
|---|---|
| projection, norm, pooling, MLA | an independent FP64 mathematical oracle from the represented inputs; the fusion is checked against the complete formula |
| FP8 encoder | an independent exact codec oracle for a given FP32 input, covering rounding, zero, the smallest values and saturation |
| tail, raw record save, invalid regions | an exact word/byte check, including the untouched range |
| Indexer | an independent scoring oracle; an explicit tie case must match exactly, and a near tie is checked against the FP32 dot-product/head-reduction error bound |
| FP8 MLA consumption | an independent interpretation of the actual code/scale; the storage error and the attention arithmetic error are separate, and the FP8 cache does not lower the compute policy |

The actually produced BF16 raw K/gate is an observable state boundary. After the mathematical
correctness of that production computation is verified, the tail extraction must copy the actual words
of the same physical window exactly. Rerunning with a shorter T, the approximate output of another
kernel or a readable generated result cannot replace these two independent requirements.

The finite conformance cases cover:

- both MLA storage profiles, both prepare exits, the five semantic entries and the actual production dispatch boundaries;
- the four pool phases, an arbitrary chunk start, a page or chunk crossing, mixed lengths and a zero-length row;
- the select-everything boundary at E=2051/2052, negative coefficients, a tie, a different selection-row mapping, a newest complete pool that can be excluded and an unfinished tail that must be included;
- the 1M index domain, an arbitrary physical page arrangement, and a pool already written being invisible to an earlier query;
- zero acceptance, short acceptance, full acceptance, in-place and out-of-place, a shared read-only source and an illegal cross-row overwrite;
- the actual records of the same physical window, a rejected suffix whose rewrite does not affect the commit, unaddressed regions and the guards;
- the first MTP step and the frozen-E reuse, the real-conditioning rebuild, and the behaviour after a Graph metadata change.

A floating-point reduction is not required to agree bit for bit across different legal routes. The
mathematical acceptance of a fixed numerical policy must not be relaxed selectively by T/B, by storage
or by a private route. An exact codec, a copy and a no-op are still verified bit for bit. This contract
claims no measured accuracy loss and no measured performance gain.

Implementation ownership is fixed at `include/ninfer/ops/kpool_sparse_mla.h` and the matching
`src/ops/softmax_attention/kpool_sparse_mla/` family. The pure Q/O projections stay in Linear. Core
manages the non-owning views, the checked layout and the fixed storage, and the model execution calls
only through the public contract. After the header, the Core views/layout, the codec and the
qualification tests land, each takes over its matching contract. After the model integration
completes, the actual execution code and the current execution note take over the combination
relation. The remaining parts stay in this document. After everything is taken over, delete this
document. The implementation status and the task scope are maintained only in the design index and the
base checklist, and a document pass does not authorize an execution agent to re-select the codec, the
layout or the state semantics.
