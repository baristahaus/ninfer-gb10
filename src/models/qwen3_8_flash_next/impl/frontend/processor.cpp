#include "models/common/qwen/frontend/processor_impl.h"

#include "models/qwen3_8_flash_next/impl/frontend/processor_family.h"

template class ninfer::models::qwen::frontend::Processor<
    ninfer::models::qwen3_8_flash_next::frontend_internal::FlashNextProcessorFamily>;
