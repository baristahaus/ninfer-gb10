#pragma once

// The SHA-256 digest helpers are family-neutral: the common Qwen implementation is the single
// source, re-exported under this family's private namespace.
#include "models/common/qwen/frontend/digest.h"

namespace ninfer::models::qwen3_8_flash_next::frontend_internal {

using qwen::frontend::Sha256Digest;
using qwen::frontend::sha256;
using qwen::frontend::sha256_hex;

} // namespace ninfer::models::qwen3_8_flash_next::frontend_internal
