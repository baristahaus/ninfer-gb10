#pragma once

#include "models/common/qwen/frontend/processor.h"
#include "models/qwen3_8_flash_next/impl/frontend/media_types.h"

#include <ninfer/models/qwen3_8_flash_next/prepared_prompt.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next::frontend_internal {

struct ProcessedInput {
    std::vector<int> input_ids;
    std::vector<std::uint8_t> token_types;
    // Axis-major [3, input_ids.size()] in temporal, height, width order.
    std::vector<std::int32_t> positions;
    std::int32_t rope_delta = 0;
    std::vector<VisionItem> vision_items;
    // One immutable row-major [raw_patches, 1536] payload per Vision item.
    std::vector<std::shared_ptr<const qwen3_8_flash_next::PreparedMediaPayload>> media_payloads;
    std::optional<RewriteCheckpointSpec> rewrite_checkpoint;
    std::vector<std::uint32_t> rewrite_execution_frontiers;
    std::vector<std::optional<std::uint32_t>> message_boundaries;
    std::vector<std::optional<std::uint32_t>> cache_boundaries;
    qwen::frontend::PreprocessStats stats;
};

struct EncodedChat {
    std::vector<int> input_ids;

    struct MediaTokenRun {
        TokenSpan tokens;
        Modality modality       = Modality::Image;
        std::size_t item_index  = 0;
        std::size_t frame_index = 0;
    };

    std::vector<MediaTokenRun> media_token_runs;
    std::optional<RewriteCheckpointSpec> rewrite_checkpoint;
    std::vector<std::uint32_t> rewrite_execution_frontiers;
    std::vector<std::optional<std::uint32_t>> message_boundaries;
    std::vector<std::optional<std::uint32_t>> cache_boundaries;
};

} // namespace ninfer::models::qwen3_8_flash_next::frontend_internal
