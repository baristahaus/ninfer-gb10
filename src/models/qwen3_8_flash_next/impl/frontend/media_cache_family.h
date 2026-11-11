#pragma once

#include "models/common/qwen/frontend/media_cache.h"

#include "models/qwen3_8_flash_next/impl/frontend/media_types.h"

#include <ninfer/models/qwen3_8_flash_next/prepared_prompt.h>

#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next::frontend_internal {

// Flash-Next keeps its own payload, modality and Vision item types, and its existing host worker
// policy. The cache admission/lookup/eviction machinery, keying and accounting live in the common
// MediaCache; this trait surface is the only family-specific part.
struct FlashNextMediaCacheFamily {
    using Modality   = ninfer::models::qwen3_8_flash_next::frontend_internal::Modality;
    using VisionItem = ninfer::models::qwen3_8_flash_next::frontend_internal::VisionItem;
    using Payload    = qwen3_8_flash_next::PreparedMediaPayload;
    using Key        = qwen::frontend::media_cache_detail::Key<Modality, Modality::Image>;
    using Media      = qwen::frontend::media_cache_detail::Prepared<VisionItem, Payload>;
    using Pending    = qwen::frontend::media_cache_detail::Pending<Media>;
    using Workers    = ninfer::HostWorkerPool;

    static std::uint32_t resolve_threads(std::uint32_t requested);
};

} // namespace ninfer::models::qwen3_8_flash_next::frontend_internal
