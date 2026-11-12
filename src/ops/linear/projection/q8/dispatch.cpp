#include "ops/linear/projection/q8/q8_dispatch.h"
#include "ops/linear/projection/q8/shapes.h"
#include "core/platform.h"

#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
using ShapeSelector = Q8Launch (*)(std::int32_t);

struct ShapeEntry {
    std::int32_t n;
    std::int32_t k;
    ShapeSelector select;
};

template <class Geometry>
constexpr ShapeEntry shape(ShapeSelector select) {
    return {Geometry::kOutputRows, Geometry::kInputRows, select};
}

constexpr std::array kShapes{
#if NINFER_TARGET_SPARK
    shape<Q8N4096K4096>(select_q8_n4096_k4096),   shape<Q8N12288K4096>(select_q8_n12288_k4096),
    shape<Q8N2560K3072>(select_q8_n2560_k3072),   shape<Q8N8192K2560>(select_q8_n8192_k2560),
#endif
    shape<Q8N1024K2048>(select_q8_n1024_k2048),   shape<Q8N1024K5120>(select_q8_n1024_k5120),
    shape<Q8N2048K4096>(select_q8_n2048_k4096),   shape<Q8N2048K4608>(select_q8_n2048_k4608),
    shape<Q8N2048K16384>(select_q8_n2048_k16384), shape<Q8N2560K6144>(select_q8_n2560_k6144),
    shape<Q8N4608K4608>(select_q8_n4608_k4608),   shape<Q8N2560K4608>(select_q8_n2560_k4608), shape<Q8N2560K4096>(select_q8_n2560_k4096), shape<Q8N2560K7680>(select_q8_n2560_k7680),   shape<Q8N5120K4608>(select_q8_n5120_k4608),
    shape<Q8N5120K6144>(select_q8_n5120_k6144),   shape<Q8N5120K10240>(select_q8_n5120_k10240),
    shape<Q8N5120K17408>(select_q8_n5120_k17408), shape<Q8N5120K25600>(select_q8_n5120_k25600),
    shape<Q8N6144K2560>(select_q8_n6144_k2560),   shape<Q8N6144K5120>(select_q8_n6144_k5120),
    shape<Q8N9216K2048>(select_q8_n9216_k2048),   shape<Q8N10240K2560>(select_q8_n10240_k2560),
    shape<Q8N12288K2048>(select_q8_n12288_k2048), shape<Q8N12288K2560>(select_q8_n12288_k2560),
    shape<Q8N14336K5120>(select_q8_n14336_k5120), shape<Q8N16384K2560>(select_q8_n16384_k2560),
    shape<Q8N34816K5120>(select_q8_n34816_k5120), shape<Q8N248320K5120>(select_q8_n248320_k5120),
    // The width-2 MTP geometries. Appended, never reordered: the lookup is linear first-match, so
    // every entry above resolves exactly as it did.
    shape<Q8N7168K5120>(select_q8_n7168_k5120),   shape<Q8N8704K5120>(select_q8_n8704_k5120),
    shape<Q8N124160K5120>(select_q8_n124160_k5120),
    shape<Q8N5120K3072>(select_q8_n5120_k3072),   shape<Q8N5120K8704>(select_q8_n5120_k8704),
    // The fused mlp gate|up parent's Rows-shard half: 34816 halves to 17408 at width 2.
    shape<Q8N17408K5120>(select_q8_n17408_k5120),
};
} // namespace

Q8Launch select_q8_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) throw std::invalid_argument("q8 linear: T must be positive");
    for (const auto& entry : kShapes) {
        if (entry.n == n && entry.k == k) return entry.select(t);
    }
    // Name the shape and the token extent: on the two-GPU lane the caller is looking at a
    // rank-sized geometry it did not write by hand, and "unsupported shape" alone does not say
    // which one. The FP8 dispatch already names its refusals for the same reason.
    throw std::invalid_argument("q8 linear: unsupported shape N" + std::to_string(n) + "K" +
                                std::to_string(k) + " T" + std::to_string(t));
}

} // namespace ninfer::ops::detail
