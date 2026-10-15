#pragma once

#include "core/weight_view.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::artifact {

struct TensorObject;

// Interpret and validate a requested tensor's complete encoded parent.
[[nodiscard]] WeightGeometry describe_tensor(const TensorObject& object);

// Resolve one exl3_expert_bank's complete geometry from its payload (SP-7 contract section 4).
//
// The bank's byte size is data-dependent (mixed per-expert bitrates give unequal trellis
// extents), so the geometry resolves from the payload's per-expert directory rather than the
// shape alone. `directory` holds the directory plane's bytes: entry 0's u64 trellis offset
// first, so the caller reads the first 16 bytes, takes entry 0's offset as the directory's
// byte bound, and re-reads that prefix. `codebook_word` is the payload's final four bytes.
//
// Validation (the contract's load validation): the shape is rank 2 [E*N, K] with K and N
// multiples of 128; the directory's offsets increase in expert id order, are 256-byte aligned,
// and match the extents implied by each entry's 2K; an odd 2K requires the mul1 codebook;
// every extent is 256-byte aligned; and the directory-implied planes end exactly at the
// object's byte size. Exactly one expert count is consistent; ambiguity or none is an error.
[[nodiscard]] WeightGeometry describe_exl3_bank(const TensorObject& object,
                                                std::span<const std::byte> directory,
                                                std::array<std::byte, 4> codebook_word);

} // namespace ninfer::artifact
