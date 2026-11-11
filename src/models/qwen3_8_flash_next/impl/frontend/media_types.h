#pragma once

#include "models/qwen3_8_flash_next/impl/frontend/chat_template.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next::frontend_internal {

struct VisionGrid {
    std::int32_t temporal = 0;
    std::int32_t height   = 0;
    std::int32_t width    = 0;
};

struct TokenSpan {
    std::size_t begin = 0;
    std::size_t count = 0;
};

struct VisionItem {
    Modality modality = Modality::Image;
    VisionGrid grid;
    std::size_t patch_begin = 0;
    std::size_t patch_count = 0;
    std::array<std::uint8_t, 32> content_digest{};
    std::vector<double> timestamps;
    std::vector<TokenSpan> token_spans;
};

} // namespace ninfer::models::qwen3_8_flash_next::frontend_internal
