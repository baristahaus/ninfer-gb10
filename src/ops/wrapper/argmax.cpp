// ninfer::ops - argmax wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/argmax.h"

#include "ops/launcher/argmax.h" // detail::argmax_launch

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

std::int64_t numel_allow_zero(const Tensor& t, const char* label) {
    bool has_zero = false;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument(std::string("argmax: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { has_zero = true; }
    }
    if (has_zero) { return 0; }

    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("argmax: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

} // namespace

void argmax(const Tensor& logits, Tensor& out, std::int32_t valid_rows, cudaStream_t stream) {
    if (logits.dtype != DType::BF16) { throw std::invalid_argument("argmax: logits must be BF16"); }
    if (out.dtype != DType::I32) { throw std::invalid_argument("argmax: out must be I32"); }

    const std::int64_t logits_n = numel_allow_zero(logits, "logits");
    (void)numel_allow_zero(out, "out");

    if (logits.ne[2] != 1 || logits.ne[3] != 1) {
        throw std::invalid_argument("argmax: logits must be rank-2 [vocab,T]");
    }
    if (out.ne[1] != 1 || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("argmax: out must be rank-1 [T]");
    }
    if (logits.ne[0] <= 0) {
        throw std::invalid_argument("argmax: physical rows must be positive");
    }
    if (valid_rows <= 0 || valid_rows > logits.ne[0]) {
        throw std::invalid_argument("argmax: valid_rows must be in [1, logits.ne[0]]");
    }
    if (out.ne[0] != logits.ne[1]) {
        throw std::invalid_argument("argmax: out shape must be [logits.ne[1]]");
    }
    if (logits_n == 0) { return; }

    if (!logits.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("argmax: logits/out must be contiguous");
    }
    if (logits.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("argmax: logits/out data must be non-null");
    }

    detail::argmax_launch(logits, out, valid_rows, stream);
}

void shortlist_exact_argmax(const Tensor& hidden, const Tensor& approximate_logits,
                            std::int32_t valid_rows,
                            const Weight& exact_head, const std::int32_t* id_map,
                            Tensor& candidate_ids, Tensor& candidate_scores, Tensor& out,
                            cudaStream_t stream) {
    constexpr int kTile = 512;
    constexpr int kPerTile = 2;
    if (hidden.dtype != DType::BF16 || approximate_logits.dtype != DType::BF16 ||
        candidate_ids.dtype != DType::I32 || candidate_scores.dtype != DType::FP32 ||
        out.dtype != DType::I32) {
        throw std::invalid_argument("shortlist_exact_argmax: invalid tensor dtype");
    }
    if (!hidden.is_contiguous() || !approximate_logits.is_contiguous() ||
        !candidate_ids.is_contiguous() || !candidate_scores.is_contiguous() ||
        !out.is_contiguous()) {
        throw std::invalid_argument("shortlist_exact_argmax: tensors must be contiguous");
    }
    const int tokens = hidden.ne[1];
    const int shortlist_rows = valid_rows;
    const int candidate_rows = ((shortlist_rows + kTile - 1) / kTile) * kPerTile;
    if (tokens <= 0 || hidden.ne[0] <= 0 || hidden.ne[2] != 1 || hidden.ne[3] != 1 ||
        shortlist_rows < 2 || shortlist_rows > approximate_logits.ne[0] ||
        approximate_logits.ne[1] != tokens ||
        approximate_logits.ne[2] != 1 || approximate_logits.ne[3] != 1 ||
        candidate_ids.ne[0] != candidate_rows || candidate_ids.ne[1] != tokens ||
        candidate_ids.ne[2] != 1 || candidate_ids.ne[3] != 1 ||
        candidate_scores.ne[0] != candidate_rows || candidate_scores.ne[1] != tokens ||
        candidate_scores.ne[2] != 1 || candidate_scores.ne[3] != 1 || out.ne[0] != tokens ||
        out.ne[1] != 1 || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("shortlist_exact_argmax: invalid tensor shape");
    }
    const bool bf16_head = exact_head.qtype == QType::BF16 &&
                           exact_head.layout == QuantLayout::Contiguous;
    const bool fp8_head = exact_head.qtype == QType::FP8_E4M3FN_ROW_BF16 &&
                          exact_head.layout == QuantLayout::RowScale &&
                          exact_head.scales != nullptr;
    if ((!bf16_head && !fp8_head) ||
        exact_head.k != hidden.ne[0] || exact_head.n <= 0 || exact_head.qdata == nullptr ||
        id_map == nullptr || hidden.data == nullptr || approximate_logits.data == nullptr ||
        candidate_ids.data == nullptr || candidate_scores.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("shortlist_exact_argmax: invalid exact head or storage");
    }
    detail::shortlist_exact_argmax_launch(hidden, approximate_logits, valid_rows, exact_head, id_map,
                                          candidate_ids, candidate_scores, out, stream);
}

} // namespace ninfer::ops
