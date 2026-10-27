#include "runtime/engine/flash_next/instance.h"

#include "ninfer/types.h"

namespace ninfer::flashnext_runtime {

FlashNextInstance::FlashNextInstance(std::unique_ptr<ModelContract::LoadedModel> source,
                                     const EngineOptions& options)
    : model(std::move(source)), frontend(ModelContract::make_frontend(*model, options)),
      capacity(options.max_context) {}

} // namespace ninfer::flashnext_runtime
