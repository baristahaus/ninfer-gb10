#include "artifact/binder.h"
#include "artifact/reader.h"
#include "models/qwen3_8_flash_next_125b_a6b/impl/load_plan.h"
#include <ninfer/models/qwen3_8_flash_next/startup_features.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>

int main() {
    const char* path = std::getenv("NINFER_QWEN38_FLASH_NEXT_WEIGHTS");
    if (path == nullptr || *path == '\0') { return 77; }
    try {
        ninfer::artifact::Reader reader{std::filesystem::path(path)};
        ninfer::artifact::Binder binder(reader);
        const auto plan = ninfer::models::qwen3_8_flash_next_125b_a6b::plan_artifact(binder);
        const auto& table_binding = reader.directory().bindings.at(
            ninfer::models::qwen3_8_flash_next_125b_a6b::kPleTableName);
        const auto& table = reader.directory().tensor(table_binding.parts.front().object);
        const std::uint64_t ple_bytes = 320001536ULL * 160 *
            (table.format == "bf16" ? 2ULL : 1ULL);
        if (plan.bindings.ple_mapping.size() != ple_bytes) {
            throw std::runtime_error("PLE table mapping has the wrong extent");
        }
        std::uint64_t logical_parameters = 0;
        for (const auto handle : plan.materialization.device_objects) {
            const auto& tensor = reader.directory().tensor(handle.object);
            std::uint64_t count = 1;
            for (auto dimension : tensor.shape) count *= dimension;
            logical_parameters += count;
        }
        if (logical_parameters < 124000000000ULL || logical_parameters > 126000000000ULL)
            throw std::runtime_error("text logical parameter range check failed (expected 124B..126B, PLE excluded)");
        std::cout << "text_logical_parameters=" << logical_parameters << '\n';
        const std::size_t expected_text_objects = table.format == "bf16" ? 1259 : 1260;
        if (plan.materialization.device_objects.size() != expected_text_objects) {
            throw std::runtime_error("MTP0/Vision-off load plan uploaded optional tensors");
        }
        ninfer::artifact::Binder mtp_binder(reader);
        const auto mtp_plan = ninfer::models::qwen3_8_flash_next_125b_a6b::plan_artifact(
            mtp_binder, {.speculative = ninfer::SpeculativeBackend::Mtp});
        // NVFP4 drafter experts add an activation-divisor object to each of the two banks.
        const bool nvfp4_mtp =
            reader.directory().bindings.contains("mtp.layers.0.mlp.experts.gate_up");
        const std::size_t expected_mtp_objects = expected_text_objects + (nvfp4_mtp ? 33 : 31);
        if (mtp_plan.materialization.device_objects.size() != expected_mtp_objects ||
            mtp_plan.materialization.device_capacity_bytes <=
                plan.materialization.device_capacity_bytes) {
            throw std::runtime_error("MTP3 load plan did not upload exactly the draft tensors");
        }
        ninfer::artifact::Binder full_binder(reader);
        const auto full_plan = ninfer::models::qwen3_8_flash_next_125b_a6b::plan_artifact(
            full_binder, {.vision = true, .speculative = ninfer::SpeculativeBackend::Mtp});
        if (full_plan.materialization.device_objects.size() != expected_mtp_objects + 333 ||
            full_plan.materialization.device_capacity_bytes <=
                mtp_plan.materialization.device_capacity_bytes) {
            throw std::runtime_error("Vision load plan did not upload exactly the Vision tensors");
        }
        std::cout << "device=" << plan.materialization.device_capacity_bytes
                  << " mtp_device=" << mtp_plan.materialization.device_capacity_bytes
                  << " full_device=" << full_plan.materialization.device_capacity_bytes
                  << " file_backed="
                  << plan.bindings.ple_mapping.size()
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
