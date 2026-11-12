#pragma once

#include "core/device.h"
#include "core/device_kv_page_pool.h"
#include "core/host_kv_arena.h"
#include "core/tensor.h"
#include "core/cyclic_kv_cache.h"
#include "core/linear_attention_state.h"
#include <ninfer/models/qwen3_8_flash_next/state_image.h>

#include <span>

#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
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
    // A comma-separated list selects several frontiers in one process: a comparison needs dumps of
    // the same state at more than one point, and one process cannot be re-run at a second frontier.
    const bool frontier_selected = std::strcmp(phase, "commit") == 0;
    bool wanted                  = !frontier_selected;
    for (std::string_view rest(selected); !rest.empty() && !wanted;) {
        const auto comma              = rest.find(',');
        const std::string_view token  = rest.substr(0, comma);
        try {
            if (!token.empty() && frontier == std::stoul(std::string(token))) { wanted = true; }
        } catch (const std::exception&) {
            throw std::invalid_argument("Flash-Next state capture frontier is not a number list");
        }
        if (comma == std::string_view::npos) { break; }
        rest.remove_prefix(comma + 1);
    }
    if (!wanted) { return; }
    static std::atomic<std::uint64_t> serial{0};
    const auto stem = std::filesystem::path(directory) / std::to_string(serial.fetch_add(1));
    std::filesystem::create_directories(directory);
    nlohmann::json metadata{{"route", route},
                            {"phase", phase},
                            {"lane", lane},
                            {"physical_slot", slot},
                            {"frontier", frontier},
                            {"execution_frontier", frontier},
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

// Maintainer-only prefix KV page capture. At the head of a decode round the paged text KV a prefix
// reuse is about to read must hold the bytes a cold prefill of the same tokens wrote. This copies
// ONE page at a time from the physical pool to a HostKVAllocationView and writes the page record to
// `<serial>.<frontier>.<page>.bin`. The serial is a per-process monotonic run counter; both ranks
// keep their own copy, so a process that runs both ranks numbers a run the same way on each. Pair
// runs by (frontier, run order), never by lane or slot: several runs can present the same lane and
// frontier, and the physical page a logical page names differs between a reused and a cold run by
// design.
//
// `page` in the file name is the LOGICAL page index, not the physical one. Pairing a reused run
// against a cold run requires the same key in both, and the two runs' physical pages differ.
//
// The file is the HostKVPageLayout page: every plane at its `offset`, `page_payload_bytes` long, in
// `planes` order. K and V (and any scale or auxiliary plane) are separate regions of ONE page
// allocation, not two allocations. `page_stride` bytes are written, with the padding between planes
// and at the end zeroed before the copy so two runs never differ on uninitialized bytes. The
// `.json` sidecar records the layout, the physical page, the lane, the committed columns and the
// first-differing-column result.
//
// A first-differing-column result is required: a page is addressable as a tensor
// (DeviceKVPagePool::plane(i), sliced along the page axis) but the dump is already on host, so the
// comparison walks the host layout by (plane, head, row). A difference at global token
// `page * page_tokens + column` equal to the frontier is the boundary; one below it is inside the
// prefix; one above it is beyond the prefix (a stale tail, not the boundary).
struct KVPageDumpComparison {
    bool compared                    = false;
    bool identical                   = false;
    int previous_serial              = 0;
    std::uint32_t first_column       = 0;
    std::uint32_t first_global_token = 0;
    std::size_t plane                = 0;
    std::size_t head                 = 0;
    std::size_t row                  = 0;
    std::size_t byte_in_row          = 0;
    const char* boundary             = "not compared";
};

// Per-plane classification of the first differing column range. A first differing column only
// locates a fault; it cannot distinguish an unwritten (zero) region from a region written by a
// different numerical route. The classification accumulates over the DIFFERING values inside the
// range: whether either side is entirely zero, whether every differing value is zero on one side,
// the largest absolute difference, and the mean absolute magnitude of each side. It also keeps the
// all-zero test over the whole range, because an all-zero pattern is a property of the region, not
// only of the elements that happened to differ.
struct KVPlaneClassification {
    std::size_t plane                    = 0;
    DType dtype                          = DType::BF16;
    const char* domain                   = "value";
    std::size_t values                   = 0;
    std::size_t differing_values         = 0;
    std::size_t finite_diffs             = 0;
    std::size_t nonfinite_current        = 0;
    std::size_t nonfinite_previous       = 0;
    std::size_t nonfinite_diffs          = 0;
    bool current_all_zero                = true;
    bool previous_all_zero               = true;
    bool current_diff_zero               = true;
    bool previous_diff_zero              = true;
    std::uint32_t first_differing_column = 0;
    std::uint32_t last_differing_column  = 0;
    double max_abs_diff                  = 0.0;
    double max_abs_current               = 0.0;
    double max_abs_previous              = 0.0;
    double sum_abs_current_diff          = 0.0;
    double sum_abs_previous_diff         = 0.0;
    double sum_abs_current_range         = 0.0;
    double sum_abs_previous_range        = 0.0;
    // The tail, not the maximum. A single max_abs_diff cannot distinguish diffuse rounding (most
    // elements differ slightly) from a localised defect (a few elements differ a lot), and those
    // two have opposite implications. The ULP distance is measured against the larger side's own
    // bf16 spacing at that element, so it is comparable across magnitudes.
    std::size_t over_one_ulp              = 0;
    std::size_t over_four_ulp             = 0;
    std::size_t over_sixteen_ulp          = 0;
    std::uint32_t max_ulp                 = 0;
    std::size_t worst_column             = 0;
    double relative_l2                   = 0.0;

    [[nodiscard]] double mean_abs_current_diff() const {
        return finite_diffs == 0 ? std::numeric_limits<double>::quiet_NaN()
                                 : sum_abs_current_diff / static_cast<double>(finite_diffs);
    }

    [[nodiscard]] double mean_abs_previous_diff() const {
        return finite_diffs == 0 ? std::numeric_limits<double>::quiet_NaN()
                                 : sum_abs_previous_diff / static_cast<double>(finite_diffs);
    }

    [[nodiscard]] double mean_abs_current_range() const {
        const std::size_t finite = values - nonfinite_current;
        return finite == 0 ? std::numeric_limits<double>::quiet_NaN()
                           : sum_abs_current_range / static_cast<double>(finite);
    }

    [[nodiscard]] double mean_abs_previous_range() const {
        const std::size_t finite = values - nonfinite_previous;
        return finite == 0 ? std::numeric_limits<double>::quiet_NaN()
                           : sum_abs_previous_range / static_cast<double>(finite);
    }
};

inline const char* kv_plane_dtype_name(DType dtype) {
    switch (dtype) {
    case DType::BF16:
        return "bf16";
    case DType::FP16:
        return "fp16";
    case DType::FP32:
        return "fp32";
    case DType::FP8_E4M3FN:
        return "fp8_e4m3fn";
    case DType::I8:
        return "i8";
    case DType::U8:
        return "u8";
    case DType::I32:
        return "i32";
    case DType::U32:
        return "u32";
    case DType::I64:
        return "i64";
    case DType::U64:
        return "u64";
    }
    return "unknown";
}

// "value" means the absolute numbers are decoded floating-point values. "code" means the plane is
// an integer code (or a packed code) and the reported magnitudes are code differences, not model
// values; the domain field on every line makes that distinction explicit.
inline const char* kv_plane_domain(DType dtype) {
    switch (dtype) {
    case DType::BF16:
    case DType::FP16:
    case DType::FP32:
    case DType::FP8_E4M3FN:
        return "value";
    default:
        return "code";
    }
}

inline double kv_plane_decode(DType dtype, const std::byte* bytes) {
    switch (dtype) {
    case DType::BF16: {
        std::uint16_t bits = 0;
        std::memcpy(&bits, bytes, sizeof(bits));
        const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
        float value              = 0.0f;
        std::memcpy(&value, &wide, sizeof(value));
        return static_cast<double>(value);
    }
    case DType::FP16: {
        std::uint16_t bits = 0;
        std::memcpy(&bits, bytes, sizeof(bits));
        const std::uint32_t sign     = (bits >> 15) & 1U;
        const std::uint32_t exponent = (bits >> 10) & 0x1fU;
        const std::uint32_t mantissa = bits & 0x3ffU;
        double value                 = 0.0;
        if (exponent == 0) {
            value = std::ldexp(static_cast<double>(mantissa), -24);
        } else if (exponent == 0x1fU) {
            value = mantissa == 0 ? std::numeric_limits<double>::infinity()
                                  : std::numeric_limits<double>::quiet_NaN();
        } else {
            value =
                std::ldexp(static_cast<double>(mantissa | 0x400U), static_cast<int>(exponent) - 25);
        }
        return sign != 0 ? -value : value;
    }
    case DType::FP32: {
        float value = 0.0f;
        std::memcpy(&value, bytes, sizeof(value));
        return static_cast<double>(value);
    }
    case DType::FP8_E4M3FN: {
        const std::uint8_t bits      = *reinterpret_cast<const std::uint8_t*>(bytes);
        const std::uint32_t sign     = (bits >> 7) & 1U;
        const std::uint32_t exponent = (bits >> 3) & 0xfU;
        const std::uint32_t mantissa = bits & 0x7U;
        double value                 = 0.0;
        if (exponent == 0) {
            value = std::ldexp(static_cast<double>(mantissa), -9);
        } else if (exponent == 0xfU && mantissa == 0x7U) {
            value = std::numeric_limits<double>::quiet_NaN();
        } else {
            value =
                std::ldexp(static_cast<double>(mantissa | 0x8U), static_cast<int>(exponent) - 10);
        }
        return sign != 0 ? -value : value;
    }
    case DType::I8: {
        std::int8_t value = 0;
        std::memcpy(&value, bytes, sizeof(value));
        return static_cast<double>(value);
    }
    case DType::U8:
        return static_cast<double>(*reinterpret_cast<const std::uint8_t*>(bytes));
    case DType::I32: {
        std::int32_t value = 0;
        std::memcpy(&value, bytes, sizeof(value));
        return static_cast<double>(value);
    }
    case DType::U32: {
        std::uint32_t value = 0;
        std::memcpy(&value, bytes, sizeof(value));
        return static_cast<double>(value);
    }
    case DType::I64: {
        std::int64_t value = 0;
        std::memcpy(&value, bytes, sizeof(value));
        return static_cast<double>(value);
    }
    case DType::U64: {
        std::uint64_t value = 0;
        std::memcpy(&value, bytes, sizeof(value));
        return static_cast<double>(value);
    }
    }
    throw std::logic_error("Flash-Next KV page dump cannot decode an unknown plane dtype");
}

inline const char* kv_zero_pattern(const KVPlaneClassification& classification) {
    if (classification.current_all_zero && classification.previous_all_zero) { return "both"; }
    if (classification.current_all_zero) { return "current"; }
    if (classification.previous_all_zero) { return "previous"; }
    return "none";
}

inline const char* kv_diff_zero_side(const KVPlaneClassification& classification) {
    if (classification.differing_values == 0) { return "none"; }
    if (classification.current_diff_zero && classification.previous_diff_zero) { return "both"; }
    if (classification.current_diff_zero) { return "current"; }
    if (classification.previous_diff_zero) { return "previous"; }
    return "none";
}

// Walk one plane over rows whose token span intersects [first_column, end_column). Each row is
// compared element by element; the differing elements feed the difference statistics while every
// element feeds the all-zero pattern. Rows are the comparison unit because a four-token row packs
// four original tokens into one leading span, and the dump cannot split that span without the
// plane's private row layout.
inline KVPlaneClassification
classify_kv_plane_range(const HostKVPageLayout& layout, std::size_t plane_index,
                        const std::byte* current, const std::byte* previous,
                        std::uint32_t first_column, std::uint32_t end_column) {
    const HostKVPlaneLayout& host_plane = layout.planes[plane_index];
    const KVPlaneGeometry& plane        = layout.geometry.planes[plane_index];
    const std::size_t element_size      = dtype_size(plane.dtype);
    const std::size_t element_bytes = static_cast<std::size_t>(plane.leading_extent) * element_size;
    const std::uint32_t rows        = layout.geometry.page_tokens / plane.tokens_per_row;
    KVPlaneClassification out;
    out.plane                  = plane_index;
    out.dtype                  = plane.dtype;
    out.domain                 = kv_plane_domain(plane.dtype);
    bool first_difference_seen = false;
    for (std::uint32_t row = 0; row < rows; ++row) {
        const std::uint32_t column = row * plane.tokens_per_row;
        if (column + plane.tokens_per_row <= first_column) { continue; }
        if (column >= end_column) { break; }
        for (std::int32_t head = 0; head < plane.head_extent; ++head) {
            const std::size_t base =
                host_plane.offset + static_cast<std::size_t>(head) * host_plane.head_payload_bytes +
                static_cast<std::size_t>(row) * element_bytes;
            for (std::int32_t element = 0; element < plane.leading_extent; ++element) {
                const std::size_t offset = base + static_cast<std::size_t>(element) * element_size;
                const double current_value  = kv_plane_decode(plane.dtype, current + offset);
                const double previous_value = kv_plane_decode(plane.dtype, previous + offset);
                out.values += 1;
                out.current_all_zero  = out.current_all_zero && current_value == 0.0;
                out.previous_all_zero = out.previous_all_zero && previous_value == 0.0;
                if (std::isfinite(current_value)) {
                    out.sum_abs_current_range += std::fabs(current_value);
                } else {
                    out.nonfinite_current += 1;
                }
                if (std::isfinite(previous_value)) {
                    out.sum_abs_previous_range += std::fabs(previous_value);
                } else {
                    out.nonfinite_previous += 1;
                }
                if (std::memcmp(current + offset, previous + offset, element_size) == 0) {
                    continue;
                }
                out.differing_values += 1;
                out.current_diff_zero  = out.current_diff_zero && current_value == 0.0;
                out.previous_diff_zero = out.previous_diff_zero && previous_value == 0.0;
                if (!first_difference_seen) {
                    out.first_differing_column = column;
                    first_difference_seen      = true;
                }
                out.last_differing_column = column + plane.tokens_per_row - 1U;
                if (std::isfinite(current_value) && std::isfinite(previous_value)) {
                    out.finite_diffs += 1;
                    // ULP distance against the larger side's own spacing at this element, so the
                    // tail is comparable across magnitudes. bf16 spacing is 2^-8 relative, so a
                    // one-ULP reorder is 1 here and a 5% difference is about 13.
                    {
                        const double scale = std::max(std::fabs(current_value),
                                                      std::fabs(previous_value));
                        if (scale > 0.0) {
                            const double ulp =
                                std::fabs(current_value - previous_value) / (scale / 256.0);
                            const auto ulp_count = static_cast<std::uint32_t>(
                                std::min(ulp, 65535.0));
                            out.over_one_ulp += ulp_count >= 1 ? 1U : 0U;
                            out.over_four_ulp += ulp_count >= 4 ? 1U : 0U;
                            out.over_sixteen_ulp += ulp_count >= 16 ? 1U : 0U;
                            if (ulp_count > out.max_ulp) {
                                out.max_ulp     = ulp_count;
                                out.worst_column = column;
                            }
                        }
                    }
                    out.sum_abs_current_diff += std::fabs(current_value);
                    out.sum_abs_previous_diff += std::fabs(previous_value);
                    out.max_abs_current = std::max(out.max_abs_current, std::fabs(current_value));
                    out.max_abs_previous =
                        std::max(out.max_abs_previous, std::fabs(previous_value));
                    out.max_abs_diff =
                        std::max(out.max_abs_diff, std::fabs(current_value - previous_value));
                } else {
                    out.nonfinite_diffs += 1;
                    out.max_abs_diff = std::numeric_limits<double>::infinity();
                }
            }
        }
    }
    return out;
}

// Maintainer-only KV write-target probe. The page a KV append writes is a DEVICE fact: the kernel
// resolves it from the execution-table entry for the token's position, not from the host
// membership the logical store keeps. A copy-on-write branch that publishes the host membership
// but leaves the execution row naming the parent is therefore invisible to the host probe and to
// the page dump (which reads the membership's physical page), yet the suffix writes land on the
// parent. This reads the row's device content for one logical page and prints it beside the
// membership's physical page: DIFFER names the resolution point that did not move.
//
// `phase` names the caller: "suffix prefill" at the chunk seam, "decode" at the round head. The
// matrix is `KVExecutionTablePool::matrix()`, I32 [logical pages, rows]; row r's logical page p is
// element p + logical_pages * r. Synchronous, so it must never run under CUDA graph capture.
// Inert when NINFER_FLASH_NEXT_KV_WRITE_PROBE is unset.
inline void probe_kv_table_binding(const char* phase, const Tensor& table_matrix, std::int32_t row,
                                   std::uint32_t logical_page, std::int32_t membership_physical,
                                   std::uint32_t committed, std::uint32_t refs,
                                   std::uint32_t writers, std::uint32_t lane, int rank,
                                   std::uint32_t frontier, cudaStream_t stream) {
    static const char* enabled = std::getenv("NINFER_FLASH_NEXT_KV_WRITE_PROBE");
    if (enabled == nullptr || *enabled == '\0') { return; }
    if (stream == nullptr) {
        throw std::logic_error("Flash-Next KV write probe requires the compute stream");
    }
    if (table_matrix.data == nullptr || table_matrix.dtype != DType::I32 ||
        table_matrix.ne[0] <= 0 || table_matrix.ne[1] <= 0 || row < 0 ||
        row >= table_matrix.ne[1] || logical_page >= static_cast<std::uint32_t>(table_matrix.ne[0])) {
        throw std::logic_error("Flash-Next KV write probe got an invalid execution-table row");
    }
    std::int32_t device_physical = -1;
    const std::size_t index =
        static_cast<std::size_t>(logical_page) +
        static_cast<std::size_t>(table_matrix.ne[0]) * static_cast<std::size_t>(row);
    CUDA_CHECK(cudaMemcpyAsync(&device_physical,
                               static_cast<const std::int32_t*>(table_matrix.data) + index,
                               sizeof(device_physical), cudaMemcpyDeviceToHost, stream));
    // The engine's streams are cudaStreamNonBlocking: the copy is ordered only against `stream`,
    // so the host read must wait on that same stream.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::fprintf(stderr,
                 "prefix kv write: %s lane %u rank %d frontier %u row %d page %u device %d "
                 "membership %d %s committed %u refs %u writers %u\n",
                 phase, lane, rank, frontier, row, logical_page, device_physical,
                 membership_physical, device_physical == membership_physical ? "AGREE" : "DIFFER",
                 committed, refs, writers);
}

inline void capture_prefix_kv_page(const char* directory, const DeviceKVPagePool& pool,
                                   DeviceKVPageHandle physical, std::uint32_t logical_page,
                                   std::uint32_t frontier, std::uint32_t committed_columns,
                                   std::uint32_t lane, int rank, cudaStream_t stream,
                                   const char* phase = nullptr) {
    if (directory == nullptr || *directory == '\0') { return; }
    if (rank < 0 || rank >= 2) { return; }
    if (stream == nullptr) {
        throw std::logic_error("Flash-Next KV page dump requires the compute stream");
    }
    const HostKVPageLayout layout = plan_host_kv_page_layout(pool.geometry());
    if (layout.page_stride == 0 || layout.planes.empty() ||
        layout.planes.size() != layout.geometry.planes.size()) {
        throw std::logic_error("Flash-Next KV page dump got an empty page layout");
    }

    // Per-rank state: this process can run both ranks, and one static int would let a rank consume
    // the other's serial. The host arena is per-rank too; its page allocation is pinned.
    static std::array<int, 2> serial_by_rank{};
    static std::array<std::optional<HostKVArena>, 2> host_arenas;
    static std::array<std::map<std::pair<std::uint32_t, std::uint32_t>, std::vector<int>>, 2>
        run_history;
    const auto rank_index = static_cast<std::size_t>(rank);
    auto& arena           = host_arenas[rank_index];
    if (!arena || arena->layout_for(layout.geometry) == nullptr) {
        arena.reset();
        arena.emplace(layout.page_stride, std::span<const HostKVPageLayout>(&layout, 1));
    }
    std::optional<HostKVAllocation> allocation = arena->allocate(layout, 1);
    if (!allocation) {
        throw std::runtime_error("Flash-Next KV page dump cannot allocate one host page");
    }
    HostKVAllocationView view = arena->writable_view(*allocation);
    std::memset(view.data(), 0, layout.page_stride);
    const DeviceKVPageHandle source[1] = {physical};
    pool.copy_to_host(std::span<const DeviceKVPageHandle>(source, 1), view, stream);
    // The engine's streams are cudaStreamNonBlocking, so this copy is ordered only against
    // `stream`. A host read after this point must wait on that same stream; the default stream
    // would not order against it and would report a false result.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
        throw std::runtime_error("cannot create Flash-Next KV dump directory: " + error.message());
    }
    const int serial = serial_by_rank[rank_index]++;
    // The phase is part of the file name, so one run's prefill snapshot and its decode dump never
    // collide even at the same frontier and logical page.
    const std::string phase_prefix =
        phase == nullptr ? std::string() : std::string(phase) + ".";
    const std::string stem =
        (std::filesystem::path(directory) /
         (std::to_string(serial) + "." + phase_prefix + std::to_string(frontier) + "." +
          std::to_string(logical_page)))
            .string();
    {
        std::ofstream binary(stem + ".bin", std::ios::binary);
        binary.write(reinterpret_cast<const char*>(view.data()),
                     static_cast<std::streamsize>(layout.page_stride));
        if (!binary) { throw std::runtime_error("cannot write Flash-Next KV page dump"); }
    }

    KVPageDumpComparison comparison;
    std::vector<KVPlaneClassification> classifications;
    std::uint32_t classify_begin        = 0;
    std::uint32_t classify_end          = 0;
    const char* classify_basis          = "none";
    std::size_t differing_planes        = 0;
    std::size_t differing_values        = 0;
    std::uint32_t last_differing_column = 0;
    bool classified                     = false;
    auto& history = run_history[rank_index][std::make_pair(frontier, logical_page)];
    // A phase snapshot is never paired: its purpose is to compare ONE run's prefill seam against
    // the same run's decode seam, and the paired-run history is keyed by (frontier, page) alone.
    if (phase == nullptr) { history.push_back(serial); }
    // Run order pairing: the first and second dump at one (frontier, page) are a pair, the third
    // and fourth another. A reused run and its cold control are consecutive dumps of the same
    // request shape; lane and physical page are deliberately not keys.
    if (phase == nullptr && history.size() >= 2 && history.size() % 2 == 0) {
        comparison.compared        = true;
        comparison.previous_serial = history[history.size() - 2];
        const std::string previous_stem =
            (std::filesystem::path(directory) /
             (std::to_string(comparison.previous_serial) + "." + std::to_string(frontier) + "." +
              std::to_string(logical_page)))
                .string();
        std::ifstream previous_file(previous_stem + ".bin", std::ios::binary);
        std::vector<std::byte> previous(layout.page_stride);
        if (previous_file.read(reinterpret_cast<char*>(previous.data()),
                               static_cast<std::streamsize>(layout.page_stride))) {
            const auto* current = view.data();
            if (std::memcmp(current, previous.data(), layout.page_stride) == 0) {
                comparison.identical = true;
                comparison.boundary  = "identical";
            } else {
                comparison.first_column = layout.geometry.page_tokens;
                for (std::size_t plane_index = 0; plane_index < layout.planes.size();
                     ++plane_index) {
                    const HostKVPlaneLayout& host_plane = layout.planes[plane_index];
                    const KVPlaneGeometry& plane         = layout.geometry.planes[plane_index];
                    const std::size_t element_bytes =
                        static_cast<std::size_t>(plane.leading_extent) * dtype_size(plane.dtype);
                    const std::uint32_t rows =
                        layout.geometry.page_tokens / plane.tokens_per_row;
                    for (std::uint32_t row = 0; row < rows; ++row) {
                        const std::uint32_t column = row * plane.tokens_per_row;
                        if (column >= comparison.first_column) { break; }
                        for (std::int32_t head = 0; head < plane.head_extent; ++head) {
                            const std::size_t offset =
                                host_plane.offset +
                                static_cast<std::size_t>(head) * host_plane.head_payload_bytes +
                                static_cast<std::size_t>(row) * element_bytes;
                            if (std::memcmp(current + offset, previous.data() + offset,
                                            element_bytes) == 0) {
                                continue;
                            }
                            comparison.first_column = column;
                            comparison.plane        = plane_index;
                            comparison.head         = static_cast<std::size_t>(head);
                            comparison.row          = row;
                            comparison.byte_in_row  = 0;
                            for (std::size_t byte = 0; byte < element_bytes; ++byte) {
                                if (current[offset + byte] != previous[offset + byte]) {
                                    comparison.byte_in_row = byte;
                                    break;
                                }
                            }
                            break;
                        }
                    }
                }
                if (comparison.first_column == layout.geometry.page_tokens) {
                    // The whole-page compare said the pages differ but no plane/head/row did:
                    // only padding can account for it, and padding is zeroed before the copy.
                    throw std::logic_error("Flash-Next KV page dump found a difference outside "
                                           "every plane payload");
                }
                comparison.first_global_token =
                    logical_page * layout.geometry.page_tokens + comparison.first_column;
                comparison.boundary = comparison.first_global_token < frontier
                                          ? "inside the prefix (N < frontier)"
                                      : comparison.first_global_token == frontier
                                          ? "at the boundary (N == frontier)"
                                          : "beyond the prefix (N > frontier)";
                // Classify the region the page claims is committed, starting at the first
                // differing column. A first difference at or beyond the committed count is
                // classified over the rest of the page instead, and the basis field says so.
                const std::uint32_t page_tokens = layout.geometry.page_tokens;
                const std::uint32_t committed =
                    committed_columns > 0 ? std::min(committed_columns, page_tokens) : page_tokens;
                classify_begin = comparison.first_column;
                if (classify_begin < committed) {
                    classify_end   = committed;
                    classify_basis = "committed";
                } else {
                    classify_end   = page_tokens;
                    classify_basis = "page";
                }
                if (classify_end > classify_begin) {
                    classifications.reserve(layout.planes.size());
                    for (std::size_t plane_index = 0; plane_index < layout.planes.size();
                         ++plane_index) {
                        KVPlaneClassification plane_classification =
                            classify_kv_plane_range(layout, plane_index, view.data(),
                                                    previous.data(), classify_begin, classify_end);
                        if (plane_classification.differing_values != 0) {
                            differing_planes += 1;
                            differing_values += plane_classification.differing_values;
                            last_differing_column = std::max(
                                last_differing_column, plane_classification.last_differing_column);
                        }
                        classifications.push_back(plane_classification);
                    }
                    classified = true;
                }
            }
        } else {
            comparison.compared = false;
        }
    }

    if (phase != nullptr) {
        std::fprintf(stderr,
                     "prefix kv dump: phase %s serial %d frontier %u page %u physical %d lane %u "
                     "rank %d committed %u page_bytes %zu -> %s.bin\n",
                     phase, serial, frontier, logical_page, physical.index(), lane, rank,
                     committed_columns, layout.page_stride, stem.c_str());
    } else {
        std::fprintf(stderr,
                     "prefix kv dump: serial %d frontier %u page %u physical %d lane %u rank %d "
                     "committed %u page_bytes %zu -> %s.bin\n",
                     serial, frontier, logical_page, physical.index(), lane, rank,
                     committed_columns, layout.page_stride, stem.c_str());
    }
    if (comparison.compared) {
        if (comparison.identical) {
            std::fprintf(stderr,
                         "prefix kv dump: compare serial %d vs %d frontier %u page %u: bitwise "
                         "identical over columns 0..%u\n",
                         serial, comparison.previous_serial, frontier, logical_page,
                         layout.geometry.page_tokens - 1U);
        } else if (comparison.first_column == 0) {
            std::fprintf(stderr,
                         "prefix kv dump: compare serial %d vs %d frontier %u page %u: differs "
                         "from column 0 (global token %u, plane %zu head %zu row %zu byte %zu; "
                         "%s)\n",
                         serial, comparison.previous_serial, frontier, logical_page,
                         comparison.first_global_token, comparison.plane, comparison.head,
                         comparison.row, comparison.byte_in_row, comparison.boundary);
        } else {
            std::fprintf(stderr,
                         "prefix kv dump: compare serial %d vs %d frontier %u page %u: agrees "
                         "for columns 0..%u, differs from column %u (global token %u, plane %zu "
                         "head %zu row %zu byte %zu; %s)\n",
                         serial, comparison.previous_serial, frontier, logical_page,
                         comparison.first_column - 1U, comparison.first_column,
                         comparison.first_global_token, comparison.plane, comparison.head,
                         comparison.row, comparison.byte_in_row, comparison.boundary);
        }
    } else {
        if (phase == nullptr) {
            std::fprintf(stderr,
                         "prefix kv dump: serial %d frontier %u page %u: no previous run at this "
                         "frontier/page to compare\n",
                         serial, frontier, logical_page);
        }
    }
    if (classified) {
        // The header names the range and the plane census; each following line is one plane's
        // verdict. Only planes with a differing value print in full, so a page whose K plane
        // differs and whose V plane agrees is one line plus a census, not two dozen empty rows.
        std::fprintf(stderr,
                     "prefix kv dump: classify serial %d vs %d frontier %u page %u columns %u..%u "
                     "basis=%s committed=%u differing_planes=%zu/%zu differing_values=%zu "
                     "last_differing_column=%u\n",
                     serial, comparison.previous_serial, frontier, logical_page, classify_begin,
                     classify_end - 1U, classify_basis, committed_columns, differing_planes,
                     layout.planes.size(), differing_values, last_differing_column);
        for (const KVPlaneClassification& plane_classification : classifications) {
            if (plane_classification.differing_values == 0) { continue; }
            std::fprintf(
                stderr,
                "prefix kv dump: classify serial %d vs %d frontier %u page %u plane %zu/%zu %s "
                "%s n=%zu diffs=%zu max_abs_diff=%.9g max_abs_current=%.9g "
                "max_abs_previous=%.9g mean_abs_current_diff=%.9g "
                "mean_abs_previous_diff=%.9g zero_pattern=%s diff_zero_side=%s "
                "nonfinite=%zu/%zu/%zu\n",
                serial, comparison.previous_serial, frontier, logical_page,
                plane_classification.plane, layout.planes.size(),
                kv_plane_dtype_name(plane_classification.dtype), plane_classification.domain,
                plane_classification.values, plane_classification.differing_values,
                plane_classification.max_abs_diff, plane_classification.max_abs_current,
                plane_classification.max_abs_previous, plane_classification.mean_abs_current_diff(),
                plane_classification.mean_abs_previous_diff(),
                kv_zero_pattern(plane_classification), kv_diff_zero_side(plane_classification),
                plane_classification.nonfinite_current, plane_classification.nonfinite_previous,
                plane_classification.nonfinite_diffs, plane_classification.over_one_ulp,
                plane_classification.over_four_ulp, plane_classification.over_sixteen_ulp,
                plane_classification.max_ulp, plane_classification.worst_column);
        }
    }

    nlohmann::json planes = nlohmann::json::array();
    for (std::size_t plane_index = 0; plane_index < layout.planes.size(); ++plane_index) {
        const HostKVPlaneLayout& host_plane = layout.planes[plane_index];
        const KVPlaneGeometry& plane        = layout.geometry.planes[plane_index];
        planes.push_back(nlohmann::json{
            {"index", plane_index},
            {"dtype", static_cast<int>(plane.dtype)},
            {"leading_extent", plane.leading_extent},
            {"head_extent", plane.head_extent},
            {"tokens_per_row", plane.tokens_per_row},
            {"offset", host_plane.offset},
            {"page_payload_bytes", host_plane.page_payload_bytes},
            {"head_payload_bytes", host_plane.head_payload_bytes},
        });
    }
    const nlohmann::json comparison_json{
        {"compared", comparison.compared},
        {"previous_serial", comparison.previous_serial},
        {"identical", comparison.identical},
        {"first_column", comparison.first_column},
        {"first_global_token", comparison.first_global_token},
        {"plane", comparison.plane},
        {"head", comparison.head},
        {"row", comparison.row},
        {"byte_in_row", comparison.byte_in_row},
        {"boundary", comparison.boundary},
    };
    nlohmann::json classification_planes = nlohmann::json::array();
    for (const KVPlaneClassification& plane_classification : classifications) {
        classification_planes.push_back(nlohmann::json{
            {"plane", plane_classification.plane},
            {"dtype", kv_plane_dtype_name(plane_classification.dtype)},
            {"domain", plane_classification.domain},
            {"values", plane_classification.values},
            {"differing_values", plane_classification.differing_values},
            {"finite_diffs", plane_classification.finite_diffs},
            {"nonfinite_current", plane_classification.nonfinite_current},
            {"nonfinite_previous", plane_classification.nonfinite_previous},
            {"nonfinite_diffs", plane_classification.nonfinite_diffs},
            {"current_all_zero", plane_classification.current_all_zero},
            {"previous_all_zero", plane_classification.previous_all_zero},
            {"current_diff_zero", plane_classification.current_diff_zero},
            {"previous_diff_zero", plane_classification.previous_diff_zero},
            {"zero_pattern", kv_zero_pattern(plane_classification)},
            {"diff_zero_side", kv_diff_zero_side(plane_classification)},
            {"first_differing_column", plane_classification.first_differing_column},
            {"last_differing_column", plane_classification.last_differing_column},
            {"max_abs_diff", plane_classification.max_abs_diff},
            {"max_abs_current", plane_classification.max_abs_current},
            {"max_abs_previous", plane_classification.max_abs_previous},
            {"mean_abs_current_diff", plane_classification.mean_abs_current_diff()},
            {"mean_abs_previous_diff", plane_classification.mean_abs_previous_diff()},
            {"mean_abs_current_range", plane_classification.mean_abs_current_range()},
            {"mean_abs_previous_range", plane_classification.mean_abs_previous_range()},
        });
    }
    const nlohmann::json classification_json{
        {"classified", classified},
        {"begin_column", classify_begin},
        {"end_column", classify_end},
        {"basis", classify_basis},
        {"committed_columns", committed_columns},
        {"differing_planes", differing_planes},
        {"plane_count", layout.planes.size()},
        {"differing_values", differing_values},
        {"last_differing_column", last_differing_column},
        {"planes", classification_planes},
    };
    const nlohmann::json metadata{
        {"serial", serial},
        {"phase", phase == nullptr ? "decode" : phase},
        {"frontier", frontier},
        {"page", logical_page},
        {"physical", physical.index()},
        {"lane", lane},
        {"rank", rank},
        {"committed_columns", committed_columns},
        {"page_tokens", layout.geometry.page_tokens},
        {"page_stride", layout.page_stride},
        {"planes", planes},
        {"comparison", comparison_json},
        {"classification", classification_json},
    };
    std::ofstream metadata_file(stem + ".json");
    metadata_file << metadata;
    if (!metadata_file) { throw std::runtime_error("cannot write Flash-Next KV page metadata"); }
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
