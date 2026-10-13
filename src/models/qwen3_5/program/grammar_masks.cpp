#include "models/qwen3_5/program/program_impl.h"
#include "core/device.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

// Decode row r owns positions [r*(K+1), (r+1)*(K+1)) of the stable mask buffers and the prefill
// lane owns the region after the last decode row. The returned operand keeps a fixed device
// address, so a captured graph reads whatever the latest fill uploaded.
ops::SamplingMask ProgramImpl::bind_grammar_mask(runtime::TokenMaskProvider* provider,
                                                 std::size_t row) {
    grammar_dead_positions[row] = 0;
    if (provider == nullptr || !provider->constrained(row)) { return {}; }
    const auto words = static_cast<std::size_t>(grammar_masks_device.ne[0]);
    return {static_cast<const std::uint32_t*>(grammar_masks_device.data) +
                row * (draft_window + 1U) * words,
            static_cast<std::int32_t>(words)};
}

ops::SamplingMask ProgramImpl::fill_grammar_mask(runtime::TokenMaskProvider* provider,
                                                 std::size_t row, std::span<const TokenId> drafts,
                                                 bool prefill) {
    grammar_dead_positions[row] = 0;
    if (provider == nullptr || !provider->constrained(row)) { return {}; }
    if (!grammar_masks_host || drafts.size() > draft_window) {
        throw std::logic_error("grammar mask request exceeds the planned proposal window");
    }
    const auto words  = static_cast<std::size_t>(grammar_masks_device.ne[0]);
    const auto region = prefill ? static_cast<std::size_t>(max_concurrency) : row;
    const auto offset = region * (draft_window + 1U) * words;
    const std::span<std::uint32_t> host(static_cast<std::uint32_t*>(grammar_masks_host->data()) +
                                            offset,
                                        (drafts.size() + 1U) * words);
    grammar_dead_positions[row] = provider->fill(row, drafts, host);
    auto* device_words          = static_cast<std::uint32_t*>(grammar_masks_device.data) + offset;
    // The pinned source stays untouched until the round synchronizes after its consumer.
    CUDA_CHECK(cudaMemcpyAsync(device_words, host.data(), host.size_bytes(), cudaMemcpyHostToDevice,
                               device.stream));
    provider->uploaded(row, host.size_bytes());
    return {device_words, static_cast<std::int32_t>(words)};
}

} // namespace ninfer::models::qwen3_5::detail
