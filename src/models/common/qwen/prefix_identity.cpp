#include "models/common/qwen/prefix_identity_impl.h"

namespace ninfer::models::qwen {

template class ResidentPrefixIdentityT<PreparedPromptData>;
template class PrefixShortlistDigestsT<PreparedPromptData>;
template bool
recovery_prefix_matches<PreparedPromptData>(const RecoveryIdentityViewT<PreparedPromptData>&,
                                            const RecoveryIdentityViewT<PreparedPromptData>&);
template bool prefix_matches<PreparedPromptData>(const PreparedPromptData&,
                                                 std::span<const TokenId>,
                                                 const ResidentPrefixIdentityT<PreparedPromptData>&,
                                                 std::size_t);

} // namespace ninfer::models::qwen
