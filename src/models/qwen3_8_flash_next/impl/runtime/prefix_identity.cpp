#include "models/common/qwen/prefix_identity_impl.h"

#include <ninfer/models/qwen3_8_flash_next/prepared_prompt.h>

namespace ninfer::models::qwen {

template class ResidentPrefixIdentityT<ninfer::models::qwen3_8_flash_next::PreparedPromptData>;
template class PrefixShortlistDigestsT<ninfer::models::qwen3_8_flash_next::PreparedPromptData>;
template bool prefix_matches<ninfer::models::qwen3_8_flash_next::PreparedPromptData>(
    const ninfer::models::qwen3_8_flash_next::PreparedPromptData&, std::span<const TokenId>,
    const ResidentPrefixIdentityT<ninfer::models::qwen3_8_flash_next::PreparedPromptData>&,
    std::size_t);

} // namespace ninfer::models::qwen
