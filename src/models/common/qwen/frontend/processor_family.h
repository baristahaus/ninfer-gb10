#pragma once

#include "models/common/qwen/frontend/chat_template.h"
#include "models/common/qwen/frontend/media_cache.h"
#include "models/common/qwen/frontend/processor.h"
#include "models/common/qwen/frontend/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::models::qwen::frontend {

// The common Qwen frontend supplies the family traits for Processor. The trait surface is the
// only place where the family's types and the few policy choices meet the common mechanics.
struct CommonProcessorFamily {
    using Modality               = ninfer::models::qwen::Modality;
    using VisionItem             = ninfer::models::qwen::VisionItem;
    using TokenSpan              = ninfer::models::qwen::TokenSpan;
    using PreparedMediaPayload   = ninfer::models::qwen::PreparedMediaPayload;
    using MediaPreprocessCache   = ninfer::models::qwen::frontend::MediaPreprocessCache;
    using MediaCacheKey          = ninfer::models::qwen::frontend::MediaCacheKey;
    using PreparedMedia          = ninfer::models::qwen::frontend::PreparedMedia;
    using PendingMedia           = ninfer::models::qwen::frontend::PendingMedia;
    using MediaCacheRequestStats = ninfer::models::qwen::frontend::MediaCacheRequestStats;
    using RenderedChat           = ninfer::models::qwen::frontend::RenderedChat;
    using MediaTokenRunByteSpec  = ninfer::models::qwen::frontend::MediaTokenRunByteSpec;
    using RewriteCheckpointKind  = ninfer::models::qwen::RewriteCheckpointKind;
    using EncodedChat            = ninfer::models::qwen::frontend::EncodedChat;
    using ProcessedInput         = ninfer::models::qwen::frontend::ProcessedInput;
    using ChatMessage            = ninfer::models::qwen::frontend::ChatMessage;
    using ChatPart               = ninfer::models::qwen::frontend::ChatPart;
    using ChatPartKind           = ninfer::models::qwen::frontend::ChatPartKind;
    using ChatRole               = ninfer::ChatRole;
    using ChatRenderOptions      = ninfer::models::qwen::frontend::ChatRenderOptions;
    using Tokenizer              = ninfer::models::qwen::frontend::Tokenizer;
    using CompiledChatTemplate   = ninfer::models::qwen::frontend::CompiledChatTemplate;

    static constexpr std::size_t kRewriteCheckpointBoundaries = 2;

    static RenderedChat render(const CompiledChatTemplate& chat_template,
                               const std::vector<ChatMessage>& messages, ChatRenderOptions options,
                               const PreparationControl& control) {
        return chat_template.render(messages, std::move(options), control);
    }

    static int vision_token_id(const Tokenizer& tokenizer, Modality modality) {
        const std::string_view text = modality == Modality::Image ? kImagePad : kVideoPad;
        const std::vector<int> ids  = tokenizer.encode(text);
        if (ids.size() != 1) {
            throw std::invalid_argument("Vision placeholder must encode to one tokenizer token: " +
                                        std::string(text));
        }
        return ids.front();
    }

    static void set_processed_metadata(ProcessedInput& output, const RenderedChat& rendered) {
        output.starts_in_reasoning  = rendered.starts_in_reasoning;
        output.continuation_content = rendered.continuation_content;
    }

    static void set_rewrite_checkpoint(EncodedChat& encoded, RewriteCheckpointKind kind,
                                       std::uint32_t frontier, std::uint32_t recovery) {
        encoded.rewrite_checkpoint = RewriteCheckpointSpec{
            .kind = kind, .frontier = frontier, .recovery_frontier = recovery};
    }

    static std::optional<std::uint32_t> cache_frontier(const TokenBoundaryResult& boundary) {
        if (!boundary.exact_frontier) { return std::nullopt; }
        return to_token_frontier(*boundary.exact_frontier, "cache boundary");
    }

    static void append_rewrite_boundaries(const RenderedChat& rendered,
                                          std::vector<std::size_t>& boundaries) {
        if (rendered.rewrite_checkpoint) {
            boundaries.push_back(rendered.rewrite_checkpoint->offset);
            boundaries.push_back(rendered.rewrite_checkpoint->recovery_offset);
        }
    }

    template <class Map>
    static void map_rewrite_offsets(RenderedChat& rendered, Map&& map) {
        rendered.rewrite_checkpoint->offset =
            map(rendered.rewrite_checkpoint->offset, "rewrite checkpoint");
        rendered.rewrite_checkpoint->recovery_offset =
            map(rendered.rewrite_checkpoint->recovery_offset, "input recovery");
    }
};

} // namespace ninfer::models::qwen::frontend
