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
#include "models/qwen3_5/execution/linear.h"
#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/peer_mailbox.h"

#include <array>
#include <cstddef>

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

} // namespace ninfer::models::qwen3_5::execution
