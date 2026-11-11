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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
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

inline void capture_prefix_kv_page(const char* directory, const DeviceKVPagePool& pool,
                                   DeviceKVPageHandle physical, std::uint32_t logical_page,
                                   std::uint32_t frontier, std::uint32_t committed_columns,
                                   std::uint32_t lane, int rank, cudaStream_t stream) {
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
    const std::string stem =
        (std::filesystem::path(directory) /
         (std::to_string(serial) + "." + std::to_string(frontier) + "." +
          std::to_string(logical_page)))
            .string();
    {
        std::ofstream binary(stem + ".bin", std::ios::binary);
        binary.write(reinterpret_cast<const char*>(view.data()),
                     static_cast<std::streamsize>(layout.page_stride));
        if (!binary) { throw std::runtime_error("cannot write Flash-Next KV page dump"); }
    }

    KVPageDumpComparison comparison;
    auto& history = run_history[rank_index][std::make_pair(frontier, logical_page)];
    history.push_back(serial);
    // Run order pairing: the first and second dump at one (frontier, page) are a pair, the third
    // and fourth another. A reused run and its cold control are consecutive dumps of the same
    // request shape; lane and physical page are deliberately not keys.
    if (history.size() >= 2 && history.size() % 2 == 0) {
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
                comparison.boundary =
                    comparison.first_global_token < frontier
                        ? "inside the prefix (N < frontier)"
                        : comparison.first_global_token == frontier
                              ? "at the boundary (N == frontier)"
                              : "beyond the prefix (N > frontier)";
            }
        } else {
            comparison.compared = false;
        }
    }

    std::fprintf(stderr,
                 "prefix kv dump: serial %d frontier %u page %u physical %d lane %u rank %d "
                 "committed %u page_bytes %zu -> %s.bin\n",
                 serial, frontier, logical_page, physical.index(), lane, rank, committed_columns,
                 layout.page_stride, stem.c_str());
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
        std::fprintf(stderr,
                     "prefix kv dump: serial %d frontier %u page %u: no previous run at this "
                     "frontier/page to compare\n",
                     serial, frontier, logical_page);
    }

    nlohmann::json planes = nlohmann::json::array();
    for (std::size_t plane_index = 0; plane_index < layout.planes.size(); ++plane_index) {
        const HostKVPlaneLayout& host_plane = layout.planes[plane_index];
        const KVPlaneGeometry& plane         = layout.geometry.planes[plane_index];
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
    const nlohmann::json metadata{
        {"serial", serial},
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
    };
    std::ofstream metadata_file(stem + ".json");
    metadata_file << metadata;
    if (!metadata_file) { throw std::runtime_error("cannot write Flash-Next KV page metadata"); }
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
