#pragma once
#include "artifact/reader.h"

namespace ninfer::artifact {
// Encoded-object identity. Source safetensors identities cannot be recovered after conversion.
Json encoded_identity(const std::filesystem::path& path,
                      const std::filesystem::path& template_override = {});
} // namespace ninfer::artifact
