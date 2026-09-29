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
        const auto& provenance = reader.directory().provenance;
        const std::string recipe = provenance.contains("recipe") && provenance.at("recipe").is_string()
                                       ? provenance.at("recipe").get<std::string>()
                                       : std::string{};
        const bool fp8_mtp = recipe == "qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp-v3";
        const bool fp8_projections =
            fp8_mtp || recipe == "qwen3_8_flash_next_125b_a6b_nvfp4_fp8_projections-v3";
        // The FP8 profiles pack each text shared expert's gate/up into one [1280,2560] FP8
        // parent: 48 fewer device objects in every selection (96 gate/up objects become 48
        // parents).
        const std::size_t text_objects = fp8_projections ? 1212 : 1260;
        // The MTP layer binds 31 objects. The fp8_mtp profile packs its shared gate/up into one
        // parent (one fewer) and adds the two NVFP4 banks' activation divisors (two more).
        const std::size_t mtp_objects = fp8_mtp ? 32 : 31;
        ninfer::artifact::Binder binder(reader);
        const auto plan = ninfer::models::qwen3_8_flash_next_125b_a6b::plan_artifact(binder);
        if (plan.bindings.ple_mapping.size() != 320001536ULL * 160) {
            throw std::runtime_error("PLE table mapping has the wrong extent");
        }
        if (plan.materialization.device_objects.size() != text_objects) {
            throw std::runtime_error("MTP0/Vision-off load plan uploaded optional tensors");
        }
        ninfer::artifact::Binder mtp_binder(reader);
        const auto mtp_plan = ninfer::models::qwen3_8_flash_next_125b_a6b::plan_artifact(
            mtp_binder, {.speculative = ninfer::SpeculativeBackend::Mtp});
        if (mtp_plan.materialization.device_objects.size() != text_objects + mtp_objects ||
            mtp_plan.materialization.device_capacity_bytes <=
                plan.materialization.device_capacity_bytes) {
            throw std::runtime_error("MTP3 load plan did not upload exactly the draft tensors");
        }
        ninfer::artifact::Binder full_binder(reader);
        const auto full_plan = ninfer::models::qwen3_8_flash_next_125b_a6b::plan_artifact(
            full_binder, {.vision = true, .speculative = ninfer::SpeculativeBackend::Mtp});
        if (full_plan.materialization.device_objects.size() != text_objects + mtp_objects + 333 ||
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
