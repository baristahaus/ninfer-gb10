#pragma once

// The Qwen3.5 family's sharded forward: the arithmetic half of tensor parallelism.
//
// WHAT SHARDING MEANS FOR THE FORWARD. At width 2 each rank holds its own slices of the weights
// (src/models/qwen3_5/shard_rules.h) and the same replicated activations. The layer arithmetic
// then has exactly two kinds of projection:
//
//   Row-parallel   (`attention/{query,key,value,gate}`, `mlp/{gate,up}`, `text/output_head`)
//                  Each rank's weight is a block of the OUTPUT rows, so rank r computes its own
//                  output rows from the FULL input. Nothing is communicated, and the ranks'
//                  outputs concatenate along the output-feature axis. This is an ordinary
//                  `ops::linear` against the rank's own weight; there is no sharded code path.
//
//   Column-parallel (`attention/output`, `mlp/down`, `gdn/output`)
//                  Each rank's weight is a block of the INPUT columns, so rank r computes a
//                  PARTIAL output over its own columns from the full-width output. The partials
//                  must be summed across ranks before the result means anything. That sum is
//                  `ops::allreduce_sum` over the peer mailbox, and afterwards BOTH ranks hold the
//                  identical full-width result.
//
// Norms, the embedding, the residual stream and the attention/GDN state are replicated, so they
// need no communication at all.
//
// THE TRAP THIS HEADER EXISTS TO AVOID. `linear_add` fuses the residual into the projection, and a
// column-parallel projection must NOT do that: the residual is replicated, so adding it on both
// ranks and then summing adds it twice. The order is forced —
//
//     partial = linear(input, column_shard)      // no residual
//     sum     = allreduce(partial)               // both ranks now hold the full result
//     residual = residual + sum                  // once, after the sum
//
// which is why the column-parallel helpers below call `ops::linear` + `ops::residual_add` and
// never `ops::linear_add`.
//
// CALLER OBLIGATIONS, all of them ordering or ownership rather than arithmetic:
//
//   * EACH RANK'S OWN PROGRAM calls a column-parallel helper once, for its own rank. The collective
//     carries that rank, and the helper reduces only that rank's exchange
//     (`ops::allreduce_sum_rank`), so the ordering the arithmetic needs is stream order: the
//     projection and the exchange are enqueued on the same stream by the same owner, in that order.
//     The single-owner `ops::allreduce_sum` — one caller enqueues both ranks' exchanges — is
//     deliberately NOT reachable from here: with two Programs it would take two slots and enqueue
//     four exchanges per reduce, and its ordering would become a host-enqueue-order race between
//     two threads that no stream event fixes.
//   * `partials` are the same-shaped full-width partials of both ranks, in rank order, resident on
//     their own devices. BOTH PROGRAMS MUST PASS THE SAME PAIR — one allocation per rank, shared by
//     the two Programs — because the combine reads and writes both buffers. Two Programs each
//     allocating their own pair would diverge silently, which is why this is a checked precondition
//     and an ownership rule rather than something a helper can infer.
//   * The peer's buffer is written by the reduce, so neither rank may read it before the reduce
//     completes on both streams.
//   * The reduce's slot protocol needs the two ranks to call the same sequence of reduces in the
//     same order; that is the executor's obligation, not a per-layer one.
//
// WIDTH 1. Every helper here is width-2 only and throws otherwise. The single-device lane keeps
// using `project` / `project_add` / `dense_mlp` unchanged, which is what makes width-1 behaviour
// byte-identical by construction rather than by argument.

#include "core/arena.h"
#include "core/device.h"
#include "models/common/sharded_group.h"
#include "models/qwen3_5/execution/linear.h"
#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/peer_mailbox.h"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::execution {

// The group-scoped resources of the two-Program lane. Created ONCE by the owner that builds both
// ranks (the executor), and borrowed by every Program of the group:
//
//   * ONE mailbox for the whole group. Two mailboxes would never rendezvous — each owns its own
//     pinned slab and its own per-device arrival words, so a rank's exchange kernel would wait for
//     a peer that never writes to the same slab.
//   * ONE partial buffer per rank, shared by both Programs, because the combine reads and writes
//     both. A pair per Program would diverge silently.
//
// A Program does not create these. It is handed this struct, and it refuses to run at width > 1
// without it, which is what makes "one mailbox, one pair" an ownership rule rather than a comment.
struct ShardedGroup {
    const ExecutionContext& execution;
    ops::PeerMailbox& mailbox;
    std::array<Tensor, 2> partials;
};

// The cross-rank reduction a column-parallel projection owes. Owned by the caller (a Program):
// the mailbox is created once before any graph capture and outlives every layer.
struct ShardedCollective {
    const ExecutionContext& execution;
    ops::PeerMailbox& mailbox;
    // The rank this Program owns. It is what makes the reduce a per-rank operation: the helpers
    // reduce only this rank's exchange, so the caller's own stream order covers projection →
    // exchange. A Program cannot pass a different rank to a helper; it can only get one wrong here,
    // once, at construction.
    int rank = 0;

    // The number of reduces this Program has issued. Both ranks issue the same reduces in the same
    // order, so the index is the same on both and `PeerMailbox::slot_for` derives the same slot on
    // both — which is what the per-rank form needs and a shared counter cannot give.
    mutable std::size_t reduce_index = 0;

    // Sums this rank's exchange only. The sharded helpers are the only callers, which is how the
    // single-owner form stays out of the sharded path.
    void reduce(std::span<void* const> data, std::size_t bytes) const {
        ops::allreduce_sum_rank(execution, mailbox, rank, mailbox.slot_for(reduce_index, bytes),
                                data, bytes);
        ++reduce_index;
    }
};

// Row-parallel projection. Rank `rank`'s weight already holds its block of the output rows, so this
// is the unsharded projection against that weight — stated as its own name so a forward pass reads
// as the parallelism it uses rather than as a bare `project`.
inline void sharded_project_rows(const Tensor& input, const LinearParameters& p, Tensor& output,
                                 MemoryArena& workspace, CudaExecutionView execution) {
    project(input, p, output, workspace, execution);
}

// Column-parallel projection: rank `rank`'s partial into `partials[rank]`, then the cross-rank sum,
// which leaves the full-width result in both `partials`. No residual.
void sharded_project_columns(const Tensor& input, const LinearParameters& p,
                             std::array<Tensor, 2>& partials,
                             const ShardedCollective& collective, MemoryArena& workspace,
                             CudaExecutionView execution);

// The same, then the replicated residual added once after the sum:
//   residual = residual + allreduce(linear(input, column_shard))
void sharded_project_columns_add(const Tensor& input, const LinearParameters& p,
                                 std::array<Tensor, 2>& partials, Tensor& residual,
                                 const ShardedCollective& collective, MemoryArena& workspace,
                                 CudaExecutionView execution);

// The dense MLP of one layer at width 2, which is the whole pattern in one function:
//   gate/up   row-parallel, no communication
//   silu_mul  elementwise over the rank's own intermediate rows, no communication
//   down      column-parallel, summed across ranks
//   residual  added once after the sum
//
// `partials` is the down projection's full-width scratch on both ranks, and `gate_out`/`up_out`
// are the rank's own [intermediate/2, T] intermediates.
void sharded_dense_mlp(const Tensor& hidden, const DenseParameters& parameters, Tensor& residual,
                       std::array<Tensor, 2>& partials, const ShardedCollective& collective,
                       MemoryArena& workspace, CudaExecutionView execution,
                       DenseMlpUse use = DenseMlpUse::Text);

// THE ROW-PARALLEL HEAD'S GATHER. `text/output_head` is Rows-sharded, so rank r's projection
// produces only its block of the vocabulary. Sampling must see the whole vocabulary — and it must
// see the SAME values on both ranks, or the two ranks could pick different tokens and the lane would
// silently diverge. So the two rank-sized blocks are gathered into one full-vocabulary buffer that
// both ranks hold identically, and then sampled exactly as width 1 samples.
//
// HOW, and why it is exact. Rank r zero-fills its full buffer, copies its own block into the row
// range [rank * rank_rows, (rank + 1) * rank_rows), and calls the per-rank allreduce. Every row
// outside that range is BF16 0.0 on that rank and its real value on the peer, and adding 0.0 to a
// BF16 value is exact in the exchange's own arithmetic (float(x) + 0.0f == float(x), one
// round-to-nearest-even store returns x). So the sum IS the concatenation, bit for bit, and no new
// transport or kernel is involved.
//
// THE SLOT, which is why the gather is chunked by columns. The mailbox slot is sized for one
// column-parallel partial — hidden_size x the plan's token capacity — and the logits payload is
// vocabulary x columns, which is far wider: for the 27B one full verify round is ~64 MiB against a
// ~42 MiB slot at a 4096-token capacity. Enlarging the slot to the logits payload would cost four
// times that in pinned host memory (2 slots x 2 ranks), and 2 GiB for a causal-scoring lane. So the
// gather exchanges COLUMN SLICES that fit the slot it has; a slice of the full buffer is contiguous
// because dim 0 is fastest, so each slice is one payload. The number of exchanges is a property of
// the plan below, and `slot` advances by exactly that many, which keeps the two ranks' slot
// sequences in lockstep with the layer reduces that share the same mailbox.
struct LogitsGatherPlan {
    std::int32_t vocabulary    = 0; // rows of the full buffer
    std::int32_t rank_rows     = 0; // rows this rank writes
    std::int32_t columns       = 0; // columns to gather
    std::int32_t chunk_columns = 0; // columns one exchange carries
    std::int32_t exchanges     = 0; // ceil(columns / chunk_columns)
    std::size_t bytes_per_column = 0;
    std::size_t chunk_bytes      = 0; // the widest exchange's payload
};

// The gather's geometry, pure so the refusal is testable without a device. Throws
// std::invalid_argument, naming the quantities, when the extents are not two rank halves, when the
// rank is outside [0, 2), when a column's payload is not a whole number of 16-byte exchange units,
// or when one column does not fit the mailbox slot — the case that cannot be chunked away.
[[nodiscard]] LogitsGatherPlan plan_logits_gather(std::int32_t vocabulary, std::int32_t rank_rows,
                                                  int rank, std::int32_t columns,
                                                  std::size_t slot_bytes);

// Gathers rank `rank`'s half of the vocabulary into the group's full-vocabulary logits pair. `local`
// is this rank's [rank_rows, columns] projection; `full` is this rank's member of the group's pair,
// [vocabulary, logits_columns] with logits_columns >= columns. `slot` is this rank's next exchange
// index in the Program's deterministic sequence (`ShardedCollective::reduce_index`); it is advanced
// by the number of exchanges the plan needs, so the caller passes it straight through.
void all_gather_logits(const models::ShardedGroup& group, int rank, std::size_t& slot,
                       const Tensor& local, const Tensor& full, CudaExecutionView execution);

// This rank's member of the group's full-vocabulary logits pair, viewed at the call's column count.
// The pair is [vocabulary, logits_columns]; a view at [vocabulary, columns] is its first `columns`
// columns, which is exactly the region a gather of that many columns wrote, because dim 0 is fastest.
// Sampling and the speculative accept ops read this instead of the frame's buffer at width 2.
[[nodiscard]] inline Tensor sharded_logits_view(const models::ShardedGroup& group, int rank,
                                                std::int32_t rows, std::int32_t columns) {
    const Tensor& pair = group.logits[static_cast<std::size_t>(rank)];
    if (pair.data == nullptr) {
        throw std::logic_error("sharded forward: the group has no logits pair");
    }
    if (columns > pair.ne[1]) {
        throw std::invalid_argument(
            "sharded forward: the logits view is wider than the pair; columns=" +
            std::to_string(columns) + " pair columns=" + std::to_string(pair.ne[1]));
    }
    return Tensor({pair.data, pair.bytes(), pair.device}, pair.dtype, {rows, columns});
}

} // namespace ninfer::models::qwen3_5::execution
