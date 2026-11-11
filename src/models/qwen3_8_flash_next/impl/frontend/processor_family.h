#pragma once

#include "models/common/qwen/frontend/processor.h"
#include "models/common/qwen/frontend/tokenizer.h"

#include "models/qwen3_8_flash_next/impl/frontend/chat_template.h"
#include "models/qwen3_8_flash_next/impl/frontend/media_cache_family.h"
#include "models/qwen3_8_flash_next/impl/frontend/processor_types.h"

#include <ninfer/models/qwen3_8_flash_next/prepared_prompt.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next::frontend_internal {

// Flash-Next keeps its own message, template, prepared-prompt and media-cache types. The common
// Qwen Processor consumes those types through this trait surface; the render/encode pipeline,
// media preprocessing, cache boundaries and position construction are not duplicated here.
struct FlashNextProcessorFamily {
    using Modality             = ninfer::models::qwen3_8_flash_next::frontend_internal::Modality;
    using VisionItem           = ninfer::models::qwen3_8_flash_next::frontend_internal::VisionItem;
    using TokenSpan            = ninfer::models::qwen3_8_flash_next::frontend_internal::TokenSpan;
    using PreparedMediaPayload = qwen3_8_flash_next::PreparedMediaPayload;
    using MediaPreprocessCache =
        ninfer::models::qwen::frontend::MediaCache<FlashNextMediaCacheFamily>;
    using MediaCacheKey          = typename FlashNextMediaCacheFamily::Key;
    using PreparedMedia          = typename FlashNextMediaCacheFamily::Media;
    using PendingMedia           = typename FlashNextMediaCacheFamily::Pending;
    using MediaCacheRequestStats = ninfer::models::qwen::frontend::MediaCacheRequestStats;
    using RenderedChat = ninfer::models::qwen3_8_flash_next::frontend_internal::RenderedChat;
    using MediaTokenRunByteSpec =
        ninfer::models::qwen3_8_flash_next::frontend_internal::MediaTokenRunByteSpec;
    using RewriteCheckpointKind = qwen3_8_flash_next::RewriteCheckpointKind;
    using EncodedChat    = ninfer::models::qwen3_8_flash_next::frontend_internal::EncodedChat;
    using ProcessedInput = ninfer::models::qwen3_8_flash_next::frontend_internal::ProcessedInput;
    using ChatMessage    = ninfer::models::qwen3_8_flash_next::frontend_internal::ChatMessage;
    using ChatPart       = ninfer::models::qwen3_8_flash_next::frontend_internal::ChatPart;
    using ChatPartKind   = ninfer::models::qwen3_8_flash_next::frontend_internal::ChatPartKind;
    using ChatRole       = ninfer::ChatRole;
    using ChatRenderOptions =
        ninfer::models::qwen3_8_flash_next::frontend_internal::ChatRenderOptions;
    using Tokenizer = qwen::frontend::Tokenizer;
    using CompiledChatTemplate =
        ninfer::models::qwen3_8_flash_next::frontend_internal::CompiledChatTemplate;

    static constexpr std::size_t kRewriteCheckpointBoundaries = 1;

    static RenderedChat render(const CompiledChatTemplate& chat_template,
                               const std::vector<ChatMessage>& messages, ChatRenderOptions options,
                               const PreparationControl&) {
        return chat_template.render(messages, std::move(options));
    }

    static int vision_token_id(const Tokenizer& tokenizer, Modality modality) {
        const bool video            = modality == Modality::Video;
        const int expected          = video ? kVideoToken : kImageToken;
        const std::string_view text = video ? qwen::frontend::kVideoPad : qwen::frontend::kImagePad;
        const std::vector<int> ids  = tokenizer.encode(text);
        if (ids.size() != 1 || ids.front() != expected) {
            throw std::invalid_argument(
                "Qwen3.8 tokenizer vision token IDs do not match model contract");
        }
        return expected;
    }

    static void set_processed_metadata(ProcessedInput&, const RenderedChat&) {}

    static void set_rewrite_checkpoint(EncodedChat& encoded, RewriteCheckpointKind kind,
                                       std::uint32_t frontier, std::uint32_t) {
        encoded.rewrite_checkpoint = RewriteCheckpointSpec{.kind = kind, .frontier = frontier};
    }

    static std::optional<std::uint32_t>
    cache_frontier(const qwen::frontend::TokenBoundaryResult& boundary) {
        return qwen::frontend::to_token_frontier(boundary.stable_frontier, "cache boundary");
    }

    static void append_rewrite_boundaries(const RenderedChat& rendered,
                                          std::vector<std::size_t>& boundaries) {
        if (rendered.rewrite_checkpoint) {
            boundaries.push_back(rendered.rewrite_checkpoint->offset);
        }
    }

    template <class Map>
    static void map_rewrite_offsets(RenderedChat& rendered, Map&& map) {
        rendered.rewrite_checkpoint->offset =
            map(rendered.rewrite_checkpoint->offset, "rewrite checkpoint");
    }

private:
    static constexpr int kImageToken = 248056;
    static constexpr int kVideoToken = 248057;
};

} // namespace ninfer::models::qwen3_8_flash_next::frontend_internal
