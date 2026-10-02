#include "core/gdn_replay_records.h"

#include "core/device.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer {
namespace {

struct RecordPlaneCopy {
    const uint4* source;
    uint4* destination;
    std::int64_t layer_pitch; // uint4 units
    std::int64_t row_vectors; // uint4 units per record row
};

struct RecordPlaneCopies {
    RecordPlaneCopy planes[4];
};

// Rows [0, rows) of every layer of every plane; grid (chunks, layers, planes). A kernel rather
// than a 2D memcpy so a captured node updates across batch profiles (cudaGraphExecUpdate rejects
// a 2D memcpy whose extent changes).
__global__ void copy_record_rows_kernel(const __grid_constant__ RecordPlaneCopies copies,
                                        std::int32_t rows) {
    const RecordPlaneCopy plane = copies.planes[blockIdx.z];
    const std::int64_t base     = static_cast<std::int64_t>(blockIdx.y) * plane.layer_pitch;
    const std::int64_t count    = plane.row_vectors * rows;
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        plane.destination[base + index] = plane.source[base + index];
    }
}

} // namespace

void copy_gdn_replay_record_rows(const GdnReplayRecords& source,
                                 const GdnReplayRecords& destination, std::int32_t rows,
                                 cudaStream_t stream) {
    const GdnReplayRecordSpec& a = source.spec;
    const GdnReplayRecordSpec& b = destination.spec;
    if (a.layers <= 0 || a.record_capacity <= 0 || a.layers != b.layers ||
        a.record_capacity != b.record_capacity || a.width != b.width ||
        a.conv_channels != b.conv_channels || a.qk_heads != b.qk_heads ||
        a.value_heads != b.value_heads || a.key_dim != b.key_dim || a.value_dim != b.value_dim) {
        throw std::invalid_argument("GDN replay record copy requires one valid spec");
    }
    if (rows <= 0 || rows > a.record_capacity) {
        throw std::out_of_range("GDN replay record copy row count out of range");
    }
    const std::array<std::pair<const Tensor*, const Tensor*>, 4> planes{{
        {&source.conv, &destination.conv},
        {&source.key, &destination.key},
        {&source.value, &destination.value},
        {&source.gate, &destination.gate},
    }};
    RecordPlaneCopies copies{};
    std::int64_t widest_row = 0;
    for (std::size_t index = 0; index < planes.size(); ++index) {
        const auto& [from, to]  = planes[index];
        const std::size_t bytes = from->bytes();
        const auto* begin       = static_cast<const std::byte*>(from->data);
        const auto* other       = static_cast<const std::byte*>(to->data);
        const std::size_t outer = static_cast<std::size_t>(a.layers) * a.record_capacity;
        if (bytes != to->bytes() || !from->is_contiguous() || !to->is_contiguous() ||
            bytes % (outer * sizeof(uint4)) != 0 ||
            reinterpret_cast<std::uintptr_t>(begin) % alignof(uint4) != 0 ||
            reinterpret_cast<std::uintptr_t>(other) % alignof(uint4) != 0) {
            throw std::invalid_argument("GDN replay record copy planes do not match");
        }
        if (begin < other + bytes && other < begin + bytes) {
            throw std::invalid_argument("GDN replay record copy planes overlap");
        }
        // Outer index layer * capacity + row: one pitch per layer, the first rows of each.
        const auto row_vectors = static_cast<std::int64_t>(bytes / outer / sizeof(uint4));
        copies.planes[index]   = {static_cast<const uint4*>(from->data),
                                  static_cast<uint4*>(to->data), row_vectors * a.record_capacity,
                                  row_vectors};
        widest_row             = std::max(widest_row, row_vectors);
    }
    constexpr int kThreads = 256;
    const auto chunks      = static_cast<unsigned>(
        std::min<std::int64_t>((widest_row * rows + kThreads - 1) / kThreads, 64));
    copy_record_rows_kernel<<<dim3(chunks, static_cast<unsigned>(a.layers), 4), kThreads, 0,
                              stream>>>(copies, rows);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer
