#pragma once

// Include this once from an exact target translation unit after defining
// NINFER_QWEN38_FLASH_NEXT_VARIANT and NINFER_QWEN38_FLASH_NEXT_RUNTIME_NS. The body is shared
// source; the selected Variant is compile-time data and the only target-dependent calls are its
// three closed leaves.

#include "models/qwen3_8_flash_next/impl/runtime/layouts.h"
#include "models/qwen3_8_flash_next/impl/runtime/dflash_context.h"
#include "models/qwen3_8_flash_next/impl/runtime/text_context.h"
#include "models/qwen3_8_flash_next/impl/runtime/vision_context.h"
#include "models/qwen3_8_flash_next/impl/runtime/schedule.h"
#include "models/qwen3_8_flash_next/impl/runtime/program.h"
#include "models/qwen3_8_flash_next/impl/runtime/pressure_planner.h"
#include "models/qwen3_8_flash_next/impl/runtime/api_impl.h"

#include "models/qwen3_8_flash_next/impl/runtime/layouts_impl.h"
#include "models/qwen3_8_flash_next/impl/runtime/dflash_context_impl.h"
#include "models/qwen3_8_flash_next/impl/runtime/text_context_impl.h"
#include "models/qwen3_8_flash_next/impl/runtime/vision_context_impl.h"
#include "models/qwen3_8_flash_next/impl/runtime/text_prefill_impl.h"
#include "models/qwen3_8_flash_next/impl/runtime/graph_impl.h"
#include "models/qwen3_8_flash_next/impl/runtime/speculative_target_impl.h"
#include "models/qwen3_8_flash_next/impl/runtime/dflash_impl.h"
#include "models/qwen3_8_flash_next/impl/runtime/dflash2_impl.h"
#include "models/qwen3_8_flash_next/impl/runtime/decode_impl.h"
#include "models/qwen3_8_flash_next/impl/runtime/mtp_impl.h"
#include "models/qwen3_8_flash_next/impl/runtime/request_plan_impl.h"
#include "models/qwen3_8_flash_next/impl/runtime/program_impl.h"
