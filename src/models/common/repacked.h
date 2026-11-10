#pragma once

#include "core/buffer.h"
#include "core/weight_view.h"
#include "models/common/weights.h"

#include <cstddef>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::models {

// Gathers a weight whose parts are not contiguous within one parent into one contiguous buffer, so
// a consumer that requires a single parent region (`ops::WeightInput`) sees one.
//
// The condition is structural, not a width: a column shard of a row-major matrix gives one part per
// row, a segmented row shard gives one part per segment, and an unsharded weight gives one part
// covering its whole object. So this cannot fire at width 1, and no caller needs a width guard.
//
// The gather is generic over the encoded layout because the bytes of a row are a linear prefix in
// the column axis for every layout in the tree: the target is simply the geometry of the rank's own
// shape, and each source row contributes a byte range at a constant offset. No per-format code.
// One fused consumer: several parameters whose bytes form one matrix for the op that reads them.
// The GDN q|k|v role is the only one in this family today -- its [5120, 5120] rank-2 matrix is
// query then key then value -- but the list is the same structure the profile enumeration walks
// (`execution/profile.cpp`'s role profiles), so the two cannot disagree.
struct FusedGroup {
    std::vector<WeightId> members; // in destination row order
};

class RepackedWeights {
public:
    RepackedWeights();
    ~RepackedWeights();
    RepackedWeights(RepackedWeights&&) noexcept;
    RepackedWeights& operator=(RepackedWeights&&) noexcept;
    RepackedWeights(const RepackedWeights&)            = delete;
    RepackedWeights& operator=(const RepackedWeights&) = delete;

    // Rewrites every non-contiguous view in `weights` to a single region over a parent owned here.
    // A view that already has one contiguous region is left untouched, which is why the
    // single-device lane never enters this path.
    void gather(std::vector<BoundWeight>& weights,
                std::span<const FusedGroup> fused = {});

    [[nodiscard]] std::size_t count() const noexcept { return owned_.size(); }

private:
    struct Owned {
        // A device parent uses the CUDA buffer; a host-resident parent (the tiered expert lane, and
        // the host-only test) uses the byte vector. Exactly one is live.
        DeviceBuffer device_buffer;
        std::vector<std::byte> host_buffer;
        std::vector<float> divisors;
        WeightParent parent;
    };
    std::vector<std::unique_ptr<Owned>> owned_;
};

} // namespace ninfer::models
