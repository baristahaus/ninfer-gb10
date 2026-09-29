// ninfer::ops - target_logprobs wrapper: public contract validation and launcher dispatch.
#include "ninfer/ops/target_logprobs.h"

#include "ops/launcher/target_logprobs.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_rank_two(const Tensor& tensor, const char* label) {
    if (tensor.ne[0] <= 0 || tensor.ne[1] <= 0 || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("target_logprobs: ") + label +
                                    " must be rank-2 with positive dimensions");
    }
}

void require_vector(const Tensor& tensor, std::int32_t columns, const char* label) {
    if (tensor.ne[0] != columns || tensor.ne[1] != 1 || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("target_logprobs: ") + label +
                                    " must have shape [columns]");
    }
}

void require_accessible(const Tensor& tensor, std::size_t alignment, const char* label) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("target_logprobs: ") + label +
                                    " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string("target_logprobs: ") + label +
                                    " data must be non-null");
    }
    if ((reinterpret_cast<std::uintptr_t>(tensor.data) & (alignment - 1)) != 0) {
        throw std::invalid_argument(std::string("target_logprobs: ") + label +
                                    " data is not naturally aligned");
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    if (lhs_begin <= rhs_begin) { return rhs_begin - lhs_begin < lhs.bytes(); }
    return lhs_begin - rhs_begin < rhs.bytes();
}

// Validates an optional ranking request: either both outputs are supplied with a common rank count
// or neither is, so a caller cannot receive ids without probabilities or the reverse.
std::int32_t validate_ranking(const Tensor& logits, Tensor* top_ids, Tensor* top_logprobs) {
    if ((top_ids == nullptr) != (top_logprobs == nullptr)) {
        throw std::invalid_argument("target_logprobs: top_ids and top_logprobs are requested "
                                    "together or not at all");
    }
    if (top_ids == nullptr) { return 0; }
    if (top_ids->dtype != DType::I32) {
        throw std::invalid_argument("target_logprobs: top_ids must be I32");
    }
    if (top_logprobs->dtype != DType::FP32) {
        throw std::invalid_argument("target_logprobs: top_logprobs must be FP32");
    }
    require_rank_two(*top_ids, "top_ids");
    if (top_ids->ne[0] > kMaxReportedLogprobRanks) {
        throw std::invalid_argument("target_logprobs: top_ids requests more than " +
                                    std::to_string(kMaxReportedLogprobRanks) + " ranks");
    }
    if (top_ids->ne[1] != logits.ne[1] || top_ids->ne[2] != 1 || top_ids->ne[3] != 1) {
        throw std::invalid_argument("target_logprobs: top_ids must have shape [ranks,columns]");
    }
    if (top_logprobs->ne[0] != top_ids->ne[0] || top_logprobs->ne[1] != top_ids->ne[1] ||
        top_logprobs->ne[2] != 1 || top_logprobs->ne[3] != 1) {
        throw std::invalid_argument("target_logprobs: top_logprobs must match the top_ids shape");
    }
    require_accessible(*top_ids, alignof(std::int32_t), "top_ids");
    require_accessible(*top_logprobs, alignof(float), "top_logprobs");
    return top_ids->ne[0];
}

} // namespace

void target_logprobs(const Tensor& logits, const Tensor& target_ids, std::int32_t valid_rows,
                     Tensor& output, Tensor* top_ids, Tensor* top_logprobs, cudaStream_t stream) {
    if (logits.dtype != DType::BF16) {
        throw std::invalid_argument("target_logprobs: logits must be BF16");
    }
    if (target_ids.dtype != DType::I32) {
        throw std::invalid_argument("target_logprobs: target_ids must be I32");
    }
    if (output.dtype != DType::FP32) {
        throw std::invalid_argument("target_logprobs: output must be FP32");
    }

    require_rank_two(logits, "logits");
    const std::int32_t columns = logits.ne[1];
    require_vector(target_ids, columns, "target_ids");
    require_vector(output, columns, "output");
    if (valid_rows <= 0 || valid_rows > logits.ne[0]) {
        throw std::invalid_argument("target_logprobs: valid_rows must be in [1, physical_rows]");
    }

    require_accessible(logits, alignof(std::uint16_t), "logits");
    require_accessible(target_ids, alignof(std::int32_t), "target_ids");
    require_accessible(output, alignof(float), "output");
    if (overlaps(output, logits) || overlaps(output, target_ids)) {
        throw std::invalid_argument("target_logprobs: output must not overlap either input");
    }

    const std::int32_t ranks = validate_ranking(logits, top_ids, top_logprobs);
    if (ranks > 0) {
        if (overlaps(*top_ids, logits) || overlaps(*top_ids, target_ids) ||
            overlaps(*top_logprobs, logits) || overlaps(*top_logprobs, target_ids) ||
            overlaps(*top_ids, output) || overlaps(*top_logprobs, output) ||
            overlaps(*top_ids, *top_logprobs)) {
            throw std::invalid_argument(
                "target_logprobs: ranking outputs must not overlap any other tensor");
        }
    }

    detail::target_logprobs_launch(logits, target_ids, valid_rows, output, top_ids, top_logprobs,
                                   stream);
}

} // namespace ninfer::ops
