#include "models/common/repacked.h"

#include "core/device.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models {
namespace {

// True when `view`'s parts are one contiguous element run inside one parent. That is the
// single-device shape and the shape a whole-object parameter keeps after any placement that does
// not split it.
bool contiguous(const WeightView& view) {
    if (view.parts.empty()) { return false; }
    const auto* parent = view.parts.front().parent;
    std::uint64_t end  = view.parts.front().begin;
    for (const auto& part : view.parts) {
        if (part.parent != parent || part.begin != end) { return false; }
        end = part.end;
    }
    return true;
}

// One run of target rows taken from consecutive source rows at one column offset. A column shard
// gives one run covering every row; a segmented row shard gives one run per segment.
struct Run {
    std::uint64_t target_row = 0;
    std::uint64_t source_row = 0;
    std::uint64_t rows       = 0;
    std::uint64_t column     = 0;
};

std::uint64_t scale_bytes(const WeightGeometry& geometry, std::uint64_t column) {
    const std::uint64_t group = geometry.group_size ? geometry.group_size : 1;
    const std::uint64_t groups_per_row = geometry.padded_columns / group;
    if (groups_per_row == 0 || column % group != 0) {
        throw std::invalid_argument("repack: a column shard must align to the scale group");
    }
    const std::uint64_t scale_column = column / group;
    const std::uint64_t bytes        = scale_column * geometry.scale_row_bytes / groups_per_row;
    if (scale_column * geometry.scale_row_bytes % groups_per_row != 0) {
        throw std::invalid_argument("repack: a scale slice does not map to whole bytes");
    }
    return bytes;
}

std::uint64_t code_bytes(const WeightGeometry& geometry, std::uint64_t column) {
    if (geometry.padded_columns == 0) { return 0; }
    const std::uint64_t bytes = column * geometry.code_bytes_per_row / geometry.padded_columns;
    if (column * geometry.code_bytes_per_row % geometry.padded_columns != 0) {
        throw std::invalid_argument("repack: a column slice does not map to whole code bytes");
    }
    return bytes;
}

void copy_2d(const std::byte* source, std::size_t source_pitch, std::byte* destination,
             std::size_t destination_pitch, std::size_t width, std::size_t rows, Device device) {
    if (rows == 0 || width == 0) { return; }
    if (device.type() == DeviceType::CPU) {
        for (std::size_t row = 0; row < rows; ++row) {
            std::memcpy(destination + row * destination_pitch, source + row * source_pitch, width);
        }
        return;
    }
    CUDA_CHECK(cudaMemcpy2DAsync(destination, destination_pitch, source, source_pitch, width, rows,
                                 cudaMemcpyDeviceToDevice, nullptr));
}

} // namespace

RepackedWeights::RepackedWeights() = default;
RepackedWeights::~RepackedWeights() = default;
RepackedWeights::RepackedWeights(RepackedWeights&&) noexcept            = default;
RepackedWeights& RepackedWeights::operator=(RepackedWeights&&) noexcept = default;

void RepackedWeights::gather(std::vector<BoundWeight>& weights) {
    for (auto& weight : weights) {
        WeightView& view = weight.view;
        if (view.shape.size() != 2) { continue; }
        if (contiguous(view)) { continue; }
        const WeightParent& source = *view.parts.front().parent;
        for (const auto& part : view.parts) {
            if (part.parent != &source) {
                throw std::invalid_argument(weight.name + ": a repack cannot span two parents");
            }
        }
        const WeightGeometry target = weight_geometry(source.geometry.format,
                                                      source.geometry.layout, view.shape);
        // Gathered on the parent's own device, so a host-resident weight (the tiered expert lane)
        // works as well as a device one.
        const Device device = source.device;
        auto owned          = std::make_unique<Owned>();
        if (device.type() == DeviceType::CPU) {
            owned->host_buffer.assign(target.bytes, std::byte{0});
        } else {
            owned->device_buffer = DeviceBuffer(device, target.bytes);
        }
        owned->parent.geometry = target;
        owned->parent.device   = device;
        owned->parent.data     = device.type() == DeviceType::CPU
                                     ? owned->host_buffer.data()
                                     : static_cast<const std::byte*>(owned->device_buffer.data());
        if (!source.weight_divisors.empty()) {
            owned->divisors.assign(source.weight_divisors.begin(), source.weight_divisors.end());
            owned->parent.weight_divisors = owned->divisors;
        }
        auto* destination = const_cast<std::byte*>(owned->parent.data);

        // Group the parts into runs of consecutive rows at one column offset. A column shard gives
        // one run over every row; a segmented row shard gives one run per segment.
        const std::uint64_t columns = source.geometry.padded_columns;
        if (columns == 0) { throw std::invalid_argument("repack: parent has no columns"); }
        std::vector<Run> runs;
        for (const auto& part : view.parts) {
            const std::uint64_t extent = part.end - part.begin;
            if (extent == 0) { continue; }
            const std::uint64_t first_row = part.begin / columns;
            const std::uint64_t column    = part.begin % columns;
            const std::uint64_t rows      = extent / columns;
            if (rows == 0) {
                // One row's column prefix. Consecutive rows at the same column coalesce.
                if (!runs.empty() && runs.back().column == column &&
                    runs.back().source_row + runs.back().rows == first_row) {
                    ++runs.back().rows;
                } else {
                    runs.push_back({0, first_row, 1, column});
                }
                continue;
            }
            if (extent % columns != 0) {
                throw std::invalid_argument(weight.name +
                                            ": a repack needs row-aligned or column-prefix parts");
            }
            runs.push_back({0, first_row, rows, column});
        }
        // Assign target rows in part order; a column shard's runs all start at the same column and
        // each covers one target row.
        std::uint64_t target_row = 0;
        for (auto& run : runs) {
            run.target_row = target_row;
            target_row += run.rows;
        }
        if (target_row != view.shape[0]) {
            throw std::invalid_argument(weight.name + ": a repack does not cover its own rows");
        }

        const std::size_t source_code_pitch = source.geometry.code_bytes_per_row;
        const std::size_t target_code_pitch = target.code_bytes_per_row;
        const std::size_t source_scale_pitch = source.geometry.scale_row_bytes;
        const std::size_t target_scale_pitch = target.scale_row_bytes;
        for (const auto& run : runs) {
            const std::size_t code_offset = code_bytes(source.geometry, run.column);
            copy_2d(source.data + run.source_row * source_code_pitch + code_offset,
                    source_code_pitch, destination + run.target_row * target_code_pitch,
                    target_code_pitch, target_code_pitch, run.rows, device);
            if (target.scale_row_bytes != 0 && source.geometry.scale_row_bytes != 0) {
                const std::size_t scale_offset = scale_bytes(source.geometry, run.column);
                copy_2d(source.data + source.geometry.scale_offset +
                            (run.source_row / source.geometry.scale_block_rows) *
                                source_scale_pitch +
                            scale_offset,
                        source_scale_pitch,
                        destination + target.scale_offset +
                            (run.target_row / target.scale_block_rows) * target_scale_pitch,
                        target_scale_pitch, target_scale_pitch,
                        run.rows / source.geometry.scale_block_rows, device);
            }
        }
        if (target.divisor_count != 0 && !source.weight_divisors.empty()) {
            if (source.weight_divisors.size() < target.divisor_count) {
                throw std::invalid_argument(weight.name + ": a repack needs every divisor");
            }
            if (device.type() == DeviceType::CPU) {
                std::memcpy(destination + target.divisor_offset, owned->divisors.data(),
                            target.divisor_count * sizeof(float));
            } else {
                CUDA_CHECK(cudaMemcpyAsync(destination + target.divisor_offset,
                                           owned->divisors.data(),
                                           target.divisor_count * sizeof(float),
                                           cudaMemcpyHostToDevice, nullptr));
            }
        }
        view.parts = {WeightRegion{&owned->parent, 0, target.elements}};
        owned_.push_back(std::move(owned));
    }
}

} // namespace ninfer::models
