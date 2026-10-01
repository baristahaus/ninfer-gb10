#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr std::int32_t kFlashNextPleStageHeads     = 16;
inline constexpr std::int32_t kFlashNextPleStageMaxTokens = 64;

// Host/device handshake for one in-graph PLE gather, in pinned (device-mapped) host memory.
// The device owns request_sequence and late; the host owns answer_sequence. Each field sits on
// its own cache line so neither side's writes share a line with the other's.
struct FlashNextPleStageMailbox {
    alignas(64) std::uint32_t request_sequence = 0;
    std::int32_t tokens                        = 0;
    alignas(64) std::uint32_t answer_sequence  = 0;
    alignas(64) std::uint32_t late             = 0;
    // Token-major row IDs: ids[t * 16 + head].
    alignas(64) std::uint32_t ids[kFlashNextPleStageMaxTokens * kFlashNextPleStageHeads] = {};
};

/**
 * Op: flash_next_ple_publish_ids
 *
 * Math / indexing:
 *   For column j of row b, let t_j = tokens[j,b], and extend the row to the left with
 *   t_-1 = history[0,b] and t_-2 = history[1,b], where a negative history value means the
 *   sequence has no token there. With E the PLE end-of-sequence token 248044:
 *     s1 = t_{j-1} if it exists, else E;
 *     s2 = t_{j-2} if it exists and t_{j-1} != E, else E;
 *     bigram  = (i64(t_j) * M0) ^ (i64(s1) * M1),  trigram = bigram ^ (i64(s2) * M2)  (mod 2^64);
 *     ids[(b*W + j)*16 + h] = offset[h] + (bigram  as i64) mod+ size[h]   for h < 8,
 *                             offset[h] + (trigram as i64) mod+ size[h]   for h >= 8,
 *   with the Flash-Next PLE multipliers, table sizes and offsets, and mod+ the non-negative
 *   remainder. This equals the file-backed reference (compute_ple_ids) over the whole sequence.
 *
 * Shapes / effects:
 *   tokens is contiguous I32 [W,B] with W*B <= kFlashNextPleStageMaxTokens; history is
 *   contiguous I32 [2,B]. mailbox is pinned host memory. The Op writes mailbox->ids and
 *   mailbox->tokens = W*B, then publishes request_sequence + 1 with a system-scope release.
 */
void flash_next_ple_publish_ids(const Tensor& tokens, const Tensor& history,
                                FlashNextPleStageMailbox* mailbox, cudaStream_t stream);

/**
 * Op: flash_next_ple_wait_staged
 *
 * Effects:
 *   Waits, with system-scope acquire, until mailbox->answer_sequence equals
 *   mailbox->request_sequence; stream work after it observes the host's staged rows. After
 *   timeout_ns it sets mailbox->late = 1 and returns, so the caller can report a dead or stalled
 *   host stage instead of hanging the device.
 */
void flash_next_ple_wait_staged(FlashNextPleStageMailbox* mailbox, std::uint64_t timeout_ns,
                                cudaStream_t stream);

} // namespace ninfer::ops
