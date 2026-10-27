#pragma once

// The Flash-Next startup adapter: loads the registered 125B-A6B package through the artifact
// binder, sizes the KV cache against the family capacity curve, and builds the Program.

#include "runtime/engine/flash_next/instance.h"
#include "runtime/engine/startup.h"

#include "artifact/reader.h"

#include <chrono>

namespace ninfer::flashnext_runtime {

using ConstructedFlashNextModel = runtime::ConstructedModel;

// Resolves the master-generation context-cache defaults for the Flash-Next route on `options`.
void normalize_flash_next_cache_options(EngineOptions& options);

// Reads the artifact, materializes the weights, and constructs the Flash-Next instance. The
// returned model carries its own DeviceContext (ConstructedModel::device); the Engine keeps it
// alive beside the engine core. Resolves Flash-Next option defaults on `options` in place.
[[nodiscard]] ConstructedFlashNextModel
construct_flash_next(EngineOptions& options, artifact::Reader& reader,
                     std::chrono::steady_clock::time_point start);

} // namespace ninfer::flashnext_runtime
