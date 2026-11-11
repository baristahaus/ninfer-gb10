#include "models/common/qwen/frontend/media_cache_impl.h"

#include "models/qwen3_8_flash_next/impl/frontend/media_cache_family.h"

#include <algorithm>
#include <thread>

namespace ninfer::models::qwen3_8_flash_next::frontend_internal {

std::uint32_t FlashNextMediaCacheFamily::resolve_threads(std::uint32_t requested) {
    if (requested != 0) { return requested; }
    const unsigned detected = std::thread::hardware_concurrency();
    return std::min<std::uint32_t>(detected == 0 ? 1U : detected, 16U);
}

} // namespace ninfer::models::qwen3_8_flash_next::frontend_internal

template class ninfer::models::qwen::frontend::MediaCache<
    ninfer::models::qwen3_8_flash_next::frontend_internal::FlashNextMediaCacheFamily>;
