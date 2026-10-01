#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "core/linear_attention_state.h"

#include <span>

#include <nlohmann/json.hpp>
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
        throw std::logic_error("Flash-Next logit capture must run outside CUDA graph capture");
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
inline void capture_committed_state(const char* route, const LinearAttentionStatePool& pool,
                                    int slot, std::uint32_t lane, std::uint32_t frontier,
                                    std::span<const std::int32_t> ledger, cudaStream_t stream) {
    static const char* directory = std::getenv("NINFER_FLASH_NEXT_STATE_DIR");
    static const char* selected  = std::getenv("NINFER_FLASH_NEXT_STATE_FRONTIER");
    if (directory == nullptr || selected == nullptr || *directory == '\0') { return; }
    if (frontier != std::stoul(selected)) { return; }
    static std::atomic<std::uint64_t> serial{0};
    const auto stem = std::filesystem::path(directory) / std::to_string(serial.fetch_add(1));
    std::filesystem::create_directories(directory);
    nlohmann::json metadata{{"route", route},
                            {"lane", lane},
                            {"physical_slot", slot},
                            {"frontier", frontier},
                            {"ledger", std::vector<std::int32_t>(ledger.begin(), ledger.end())}};
    for (std::uint32_t layer = 0; layer < pool.layer_count(); ++layer) {
        for (bool recurrent : {false, true}) {
            const Tensor tensor =
                recurrent ? pool.recurrent_slot(layer, slot) : pool.conv_slot(layer, slot);
            std::vector<std::byte> bytes(tensor.bytes());
            CUDA_CHECK(cudaMemcpyAsync(bytes.data(), tensor.data, bytes.size(),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const auto suffix = std::string(recurrent ? ".recurrent." : ".conv.") +
                                std::to_string(layer) + (recurrent ? ".fp32" : ".bf16");
            std::ofstream file(stem.string() + suffix, std::ios::binary);
            file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            if (!file) { throw std::runtime_error("cannot write Flash-Next state capture"); }
        }
    }
    metadata["layers"] = pool.layer_count();
    std::ofstream file(stem.string() + ".json");
    file << metadata;
    if (!file) { throw std::runtime_error("cannot write Flash-Next state metadata"); }
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
