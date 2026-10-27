#include "ninfer/ops/speculative_round.h"
#include "ops/launcher/speculative_round.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kSparseMaxDrafts    = 15;
constexpr std::int32_t kSparseCandidates   = 16;
constexpr std::int32_t kSparsePhysicalRows = 248320;
constexpr std::int32_t kSparseTokenDomain  = 248077;
constexpr std::int32_t kSparseMaxBatch     = 8;

void require_contiguous_nonnull(const Tensor& t, const char* op, const char* name) {
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

void require_dtype(const Tensor& t, DType dtype, const char* op, const char* name) {
    if (t.dtype != dtype) {
        throw std::invalid_argument(std::string(op) + ": invalid dtype for " + name);
    }
    require_contiguous_nonnull(t, op, name);
}

void require_scalar(const Tensor& t, DType dtype, const char* op, const char* name) {
    require_dtype(t, dtype, op, name);
    if (t.ne[0] != 1 || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid scalar shape for " + name);
    }
}

void require_vector(const Tensor& t, DType dtype, std::int32_t n, const char* op,
                    const char* name) {
    require_dtype(t, dtype, op, name);
    if (n <= 0 || t.ne[0] != n || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid vector shape for " + name);
    }
}

void require_matrix(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t cols,
                    const char* op, const char* name) {
    require_dtype(t, dtype, op, name);
    if (rows <= 0 || cols <= 0 || t.ne[0] != rows || t.ne[1] != cols || t.ne[2] != 1 ||
        t.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid matrix shape for " + name);
    }
}

void require_tensor3(const Tensor& t, DType dtype, std::int32_t n0, std::int32_t n1,
                     std::int32_t n2, const char* op, const char* name) {
    require_dtype(t, dtype, op, name);
    if (n0 <= 0 || n1 <= 0 || n2 <= 0 || t.ne[0] != n0 || t.ne[1] != n1 || t.ne[2] != n2 ||
        t.ne[3] != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

} // namespace

void speculative_prepare_verify_inputs(const Tensor& anchors, const Tensor& drafts,
                                       const Tensor& base_positions, const Tensor& current_extents,
                                       Tensor& verify_ids, Tensor& positions, cudaStream_t stream) {
    constexpr const char* op = "speculative_prepare_verify_inputs";
    const std::int32_t k     = drafts.ne[0];
    const std::int32_t batch = drafts.ne[1];
    if (k < 1) { throw std::invalid_argument("speculative_prepare_verify_inputs: K must be >=1"); }
    if (batch < 1) {
        throw std::invalid_argument("speculative_prepare_verify_inputs: B must be >=1");
    }
    require_vector(anchors, DType::I32, batch, op, "anchors");
    require_matrix(drafts, DType::I32, k, batch, op, "drafts");
    require_vector(base_positions, DType::I32, batch, op, "base_positions");
    require_vector(current_extents, DType::I32, batch, op, "current_extents");
    require_matrix(verify_ids, DType::I32, k + 1, batch, op, "verify_ids");
    require_matrix(positions, DType::I32, k + 1, batch, op, "positions");
    detail::speculative_prepare_verify_inputs_launch(
        anchors, drafts, base_positions, current_extents, verify_ids, positions, stream);
}

void speculative_prepare_verify_ids(const Tensor& anchors, const Tensor& drafts,
                                    const Tensor& current_extents, Tensor& verify_ids,
                                    cudaStream_t stream) {
    constexpr const char* op = "speculative_prepare_verify_ids";
    const std::int32_t k     = drafts.ne[0];
    const std::int32_t batch = drafts.ne[1];
    if (k < 1 || batch < 1) {
        throw std::invalid_argument("speculative_prepare_verify_ids: K and B must be positive");
    }
    require_vector(anchors, DType::I32, batch, op, "anchors");
    require_matrix(drafts, DType::I32, k, batch, op, "drafts");
    require_vector(current_extents, DType::I32, batch, op, "current_extents");
    require_matrix(verify_ids, DType::I32, k + 1, batch, op, "verify_ids");
    detail::speculative_prepare_verify_ids_launch(anchors, drafts, current_extents, verify_ids,
                                                  stream);
}

void speculative_select_accepted_hidden(const Tensor& hidden, const Tensor& selectors, Tensor& out,
                                        cudaStream_t stream) {
    constexpr const char* op = "speculative_select_accepted_hidden";
    require_dtype(hidden, DType::BF16, op, "hidden");
    if (hidden.ne[0] <= 0 || hidden.ne[1] <= 0 || hidden.ne[2] <= 0 || hidden.ne[3] != 1) {
        throw std::invalid_argument("speculative_select_accepted_hidden: invalid hidden shape");
    }
    require_vector(selectors, DType::I32, hidden.ne[2], op, "selectors");
    require_dtype(out, DType::BF16, op, "out");
    if (out.ne[0] != hidden.ne[0] || out.ne[1] != hidden.ne[2] || out.ne[2] != 1 ||
        out.ne[3] != 1) {
        throw std::invalid_argument("speculative_select_accepted_hidden: invalid output shape");
    }
    detail::speculative_select_accepted_hidden_launch(hidden, selectors, out, stream);
}

void proposal_remap_token_ids(Tensor& proposal_tokens, const std::int32_t* id_map, std::int32_t n,
                              cudaStream_t stream) {
    constexpr const char* op = "proposal_remap_token_ids";
    require_dtype(proposal_tokens, DType::I32, op, "proposal_tokens");
    if (proposal_tokens.ne[0] <= 0 || proposal_tokens.ne[1] != 1 || proposal_tokens.ne[2] != 1 ||
        proposal_tokens.ne[3] != 1) {
        throw std::invalid_argument(
            "proposal_remap_token_ids: proposal_tokens must be a non-empty vector");
    }
    if (id_map == nullptr || n <= 0) {
        throw std::invalid_argument("proposal_remap_token_ids: id_map must be non-null and n>0");
    }
    detail::proposal_remap_token_ids_launch(proposal_tokens, id_map, n, stream);
}

} // namespace ninfer::ops
