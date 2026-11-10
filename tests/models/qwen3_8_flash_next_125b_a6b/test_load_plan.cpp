#include "artifact/binder.h"
#include "artifact/reader.h"
#include "models/qwen3_8_flash_next_125b_a6b/impl/load_plan.h"
#include <ninfer/models/qwen3_8_flash_next/startup_features.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>

int main() {
    const char* path = std::getenv("NINFER_QWEN38_FLASH_NEXT_WEIGHTS");
    if (path == nullptr || *path == '\0') {
        std::cerr << "SKIP: NINFER_QWEN38_FLASH_NEXT_WEIGHTS is not set; the load-plan test did not run\n";
        return 77;
    }
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
        if (plan.device_objects.size() != text_objects) {
            throw std::runtime_error("MTP0/Vision-off load plan uploaded optional tensors");
        }
        ninfer::artifact::Binder mtp_binder(reader);
        const auto mtp_plan = ninfer::models::qwen3_8_flash_next_125b_a6b::plan_artifact(
            mtp_binder, {.speculative = ninfer::SpeculativeBackend::Mtp});
        if (mtp_plan.device_objects.size() != text_objects + mtp_objects) {
            throw std::runtime_error("MTP3 load plan did not upload exactly the draft tensors");
        }
        ninfer::artifact::Binder full_binder(reader);
        const auto full_plan = ninfer::models::qwen3_8_flash_next_125b_a6b::plan_artifact(
            full_binder, {.vision = true, .speculative = ninfer::SpeculativeBackend::Mtp});
        if (full_plan.device_objects.size() != text_objects + mtp_objects + 333) {
            throw std::runtime_error("Vision load plan did not upload exactly the Vision tensors");
        }
        // The DFlash2 option on an artifact without the component must be refused up
        // front (the draft execution is a separate work item; the load plan binds only
        // a matching component).
        ninfer::artifact::Binder dflash2_binder(reader);
        try {
            (void)ninfer::models::qwen3_8_flash_next_125b_a6b::plan_artifact(
                dflash2_binder, {.speculative = ninfer::SpeculativeBackend::DFlash2});
        } catch (const ninfer::artifact::ArtifactError& error) {
            if (std::string(error.what()).find("DFlash2 component") == std::string::npos) {
                throw;
            }
            std::cout << "dflash2 refused as expected: " << error.what() << '\n';
        }
        std::cout << "device_objects=" << plan.device_objects.size()
                  << " mtp_objects=" << mtp_plan.device_objects.size()
                  << " full_objects=" << full_plan.device_objects.size()
                  << " file_backed="
                  << plan.bindings.ple_mapping.size()
                  << '\n';
        // The DFlash2-enabled load plan: runs only against a dflash2-carrying artifact
        // (the 5090-side conversion's output, NINFER_QWEN38_FLASH_NEXT_DFLASH2_WEIGHTS).
        // Asserts the draft's 85 device bindings plus the refusal-free component
        // validation; skips when the artifact is absent.
        const char* dflash2_path = std::getenv("NINFER_QWEN38_FLASH_NEXT_DFLASH2_WEIGHTS");
        if (dflash2_path != nullptr && *dflash2_path != '\0') {
            ninfer::artifact::Reader dflash2_reader{std::filesystem::path(dflash2_path)};
            ninfer::artifact::Binder dflash2_binder(dflash2_reader);
            const auto dflash2_plan = ninfer::models::qwen3_8_flash_next_125b_a6b::plan_artifact(
                dflash2_binder, {.speculative = ninfer::SpeculativeBackend::DFlash2});
            // The DFlash2 backend replaces MTP as the speculative lane: the plan binds
            // the text body plus the draft's 76 device objects, and no MTP tensors.
            const std::size_t expected_dflash2_objects = text_objects + 76;
            if (dflash2_plan.device_objects.size() != expected_dflash2_objects) {
                throw std::runtime_error(
                    "DFlash2 load plan uploaded " +
                    std::to_string(dflash2_plan.device_objects.size()) +
                    " device objects, expected " +
                    std::to_string(expected_dflash2_objects));
            }
            if (dflash2_plan.bindings.mtp.embedding_norm.name ==
                "mtp.pre_fc_norm_embedding.weight") {
                throw std::runtime_error("the DFlash2 plan must not bind MTP tensors");
            }
            if (dflash2_plan.bindings.dflash2.feature_projection.name !=
                "dflash2/feature_projection") {
                throw std::runtime_error("the DFlash2 feature projection is unbound");
            }
            std::cout << "dflash2_objects=" << dflash2_plan.device_objects.size()
                      << " (no MTP tensors bound)\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
