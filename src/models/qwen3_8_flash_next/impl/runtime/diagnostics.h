#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "core/cyclic_kv_cache.h"
#include "core/linear_attention_state.h"
#include <ninfer/models/qwen3_8_flash_next/state_image.h>

#include <span>

#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next::detail {

// Maintainer-only capture. Synchronous copies must never enter a CUDA graph.
inline void capture_target_logits(const char* route, const Tensor& ids, const Tensor& positions,
                                  const Tensor* valid_columns, const Tensor& table_rows,
                                  const Tensor& logits, cudaStream_t stream) {
    static const char* directory = std::getenv("NINFER_FLASH_NEXT_LOGITS_DIR");
    if (directory == nullptr || *directory == '\0') { return; }
    cudaStreamCaptureStatus status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    if (status != cudaStreamCaptureStatusNone) {
        throw std::logic_error("Flash-Next logit capture requires --no-cuda-graph");
    }
    if (!ids.is_contiguous() || !positions.is_contiguous() || !logits.is_contiguous()) {
        throw std::logic_error("Flash-Next logit capture requires contiguous tensors");
    }
    std::vector<std::int32_t> tokens(ids.numel()), offsets(positions.numel());
    std::vector<std::uint16_t> values(logits.numel());
    std::vector<std::int32_t> valid(table_rows.numel(), 1), rows(table_rows.numel());
    CUDA_CHECK(
        cudaMemcpyAsync(tokens.data(), ids.data, ids.bytes(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(offsets.data(), positions.data, positions.bytes(),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(values.data(), logits.data, logits.bytes(), cudaMemcpyDeviceToHost,
                               stream));
    if (valid_columns != nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(valid.data(), valid_columns->data, valid_columns->bytes(),
                                   cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(rows.data(), table_rows.data, table_rows.bytes(),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    static std::atomic<std::uint64_t> serial{0};
    const auto stem = std::filesystem::path(directory) / std::to_string(serial.fetch_add(1));
    std::filesystem::create_directories(directory);
    std::ofstream binary(stem.string() + ".bf16", std::ios::binary);
    binary.write(reinterpret_cast<const char*>(values.data()), logits.bytes());
    std::ofstream metadata(stem.string() + ".json");
    metadata << nlohmann::json{
        {"route", route},
        {"input_tokens", tokens},
        {"positions", offsets},
        {"valid_columns", valid},
        {"kv_table_rows", rows},
        {"vocab", logits.ne[0]},
        {"width", std::string_view(route) == "ordinary" ? 1 : ids.ne[0]},
        {"batch", std::string_view(route) == "ordinary" ? ids.ne[0] : ids.ne[1]}};
    if (!binary || !metadata) { throw std::runtime_error("cannot write Flash-Next logit capture"); }
}

// Select one execution frontier explicitly: full recurrent images are large. This is
// called after commit/rollback, outside graph capture, and never samples proposal state.
//
// Every per-request state component the pool owns is dumped, because a restore that misses one
// component is exactly the class of defect this seam exists to catch: the GDN state (conv and
// recurrent, every layer), the continuation hidden, the MTP hidden, the PLE state and, for a
// masked-draft Program, the local cyclic K/V. The files are raw little-endian bytes with no
// header; the suffix names the component, the layer and the element format.
// `phase` names the point in the sequence's life the dump was taken at: "commit" after a resolved
// round, "materialization" right after a Fork bound the restored source. A copy-identity test
// compares a "materialization" dump against the "commit" dump of the slot it was copied from.
inline void capture_committed_state(const char* route, const StateImageDevicePool& images, int slot,
                                    std::uint32_t lane, std::uint32_t frontier,
                                    std::span<const std::int32_t> ledger, cudaStream_t stream,
                                    const char* phase = "commit") {
    static const char* directory = std::getenv("NINFER_FLASH_NEXT_STATE_DIR");
    static const char* selected  = std::getenv("NINFER_FLASH_NEXT_STATE_FRONTIER");
    if (directory == nullptr || selected == nullptr || *directory == '\0') { return; }
    if (frontier != std::stoul(selected)) { return; }
    static std::atomic<std::uint64_t> serial{0};
    const auto stem = std::filesystem::path(directory) / std::to_string(serial.fetch_add(1));
    std::filesystem::create_directories(directory);
    nlohmann::json metadata{{"route", route},
                            {"phase", phase},
                            {"lane", lane},
                            {"physical_slot", slot},
                            {"frontier", frontier},
                            {"ledger", std::vector<std::int32_t>(ledger.begin(), ledger.end())}};
    std::vector<std::string> components;
    const auto dump = [&](const Tensor& tensor, const std::string& suffix,
                          const char* component) {
        if (tensor.data == nullptr || tensor.bytes() == 0) {
            throw std::logic_error("Flash-Next state capture got an empty component");
        }
        std::vector<std::byte> bytes(tensor.bytes());
        CUDA_CHECK(cudaMemcpyAsync(bytes.data(), tensor.data, bytes.size(),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::ofstream file(stem.string() + suffix, std::ios::binary);
        file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        if (!file) { throw std::runtime_error("cannot write Flash-Next state capture"); }
        if (std::find(components.begin(), components.end(), component) == components.end()) {
            components.emplace_back(component);
        }
    };
    const LinearAttentionStatePool& pool = images.linear();
    for (std::uint32_t layer = 0; layer < pool.layer_count(); ++layer) {
        dump(pool.conv_slot(layer, slot), ".conv." + std::to_string(layer) + ".bf16", "conv");
        dump(pool.recurrent_slot(layer, slot), ".recurrent." + std::to_string(layer) + ".fp32",
             "recurrent");
    }
    dump(images.continuation_hidden_slot(slot), ".continuation_hidden.bf16", "continuation_hidden");
    if (images.mtp_hidden_store() != nullptr) {
        dump(images.mtp_hidden_slot(slot), ".mtp_hidden.bf16", "mtp_hidden");
    }
    if (images.ple_store() != nullptr) { dump(images.ple_slot(slot), ".ple.bf16", "ple"); }
    std::uint32_t dflash_layers = 0;
    if (const CyclicKVCache* local = images.dflash_local(); local != nullptr) {
        dflash_layers = local->layer_count();
        for (std::uint32_t layer = 0; layer < dflash_layers; ++layer) {
            const CyclicKVCacheLayerView view = local->layer_view(layer);
            // The slot is the cyclic cache's own axis, exactly as copy_to_host addresses it.
            dump(view.k.slice(3, slot, 1),
                 ".dflash_local_k." + std::to_string(layer) + ".bf16", "dflash_local_k");
            dump(view.v.slice(3, slot, 1),
                 ".dflash_local_v." + std::to_string(layer) + ".bf16", "dflash_local_v");
        }
    }
    metadata["layers"]              = pool.layer_count();
    metadata["conv_channels"]       = pool.spec().conv_channels;
    metadata["conv_width"]          = pool.spec().conv_width;
    metadata["dflash_local_layers"] = dflash_layers;
    metadata["components"]          = components;
    std::ofstream file(stem.string() + ".json");
    file << metadata;
    if (!file) { throw std::runtime_error("cannot write Flash-Next state metadata"); }
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
