#include "ops/linear/pair/dispatch.h"
#include "ops/common/host/operands.h"
#include <cstddef>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
void require_adjacent_parent_views(const Weight& first_weight, const Weight& second_weight) {
    constexpr std::int32_t kParentRows     = 6144;
    constexpr std::int32_t kHidden         = 2048;
    constexpr std::int32_t kFirstRow       = 4096;
    constexpr std::int32_t kSecondRow      = 5120;
    constexpr std::uint64_t kCodeBytes     = static_cast<std::uint64_t>(kParentRows) * kHidden;
    constexpr std::uint64_t kScaleRowBytes = (kHidden / 32) * 2;
    constexpr std::uint64_t kPayloadBytes =
        kCodeBytes + static_cast<std::uint64_t>(kParentRows) * kScaleRowBytes;

    const auto* payload = static_cast<const std::byte*>(first_weight.payload);
    if (payload == nullptr || second_weight.payload != first_weight.payload ||
        first_weight.payload_bytes < kPayloadBytes || second_weight.payload_bytes < kPayloadBytes ||
        first_weight.qdata != payload + static_cast<std::uint64_t>(kFirstRow) * kHidden ||
        second_weight.qdata != payload + static_cast<std::uint64_t>(kSecondRow) * kHidden ||
        first_weight.scales !=
            payload + kCodeBytes + static_cast<std::uint64_t>(kFirstRow) * kScaleRowBytes ||
        second_weight.scales !=
            payload + kCodeBytes + static_cast<std::uint64_t>(kSecondRow) * kScaleRowBytes) {
        throw std::invalid_argument(
            "q8 pair: [1024,2048] weights must be exact adjacent parent row views");
    }
}

// N is the weights' own row count, not a fixed 1024: the paired kernel masks its rows and columns
// (`q8_pair_gemm_mma_kernel`'s `grow < m`, `r0 < m && c0 < n`) and the SIMT arm derives its grid
// from `operands.rows`, so any positive N the two weights share is a supported geometry. The 27B's
// MTP K/V projection is 1024 rows at width 1 and 512 at width 2, because the four KV heads shard by
// rows; both are the same kernel. The only shape that still needs an exact row count is the K=2048
// adjacent-parent form, which `require_adjacent_parent_views` checks separately.
void require_weight(const Weight& w, int k) {
    const std::uint64_t payload_bytes =
        std::uint64_t(w.n) * k + std::uint64_t(w.n) * (k / 32) * 2;
    if (w.qtype != QType::Q8_G32_FP16 || w.layout != QuantLayout::RowSplit ||
        w.scale_dtype != DType::FP16 || w.group != 32 || w.group_size != 32 || w.ndim != 2 ||
        w.n <= 0 || w.k != k || w.shape[0] != w.n || w.shape[1] != k ||
        w.padded_shape[0] != w.n || w.padded_shape[1] != k || w.payload_bytes < payload_bytes ||
        w.qhigh != nullptr || w.high_plane_bytes != 0 || !common::operand::aligned(w.qdata, 16) ||
        !common::operand::aligned(w.scales, 16))
        throw std::invalid_argument(
            "linear_pair: invalid Q8 row-split weight representation or alignment");
}
} // namespace

LinearPairLaunch select_linear_pair(int k, int tokens) {
    if (tokens > 0) {
        if (k == 2048) return select_q8_pair_n1024_k2048(tokens);
        if (k == 5120) return select_q8_pair_n1024_k5120(tokens);
    }
    throw std::invalid_argument("linear_pair: unsupported shape or token extent");
}

void validate_pair_weights(const Weight& first, const Weight& second, int max_tokens) {
    (void)select_linear_pair(first.k, max_tokens);
    require_weight(first, first.k);
    require_weight(second, first.k);
    if (first.k == 2048) require_adjacent_parent_views(first, second);
}
} // namespace ninfer::ops::detail
