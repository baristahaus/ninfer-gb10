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

namespace {

// Copies `weight`'s parts into `destination`, which is a parent of `target` geometry, starting at
// destination row `first_row`. Returns the rows written. The caller owns the buffer and the parent.
std::uint64_t gather_view_into(const BoundWeight& weight, const WeightGeometry& target,
                               std::byte* destination, std::uint64_t first_row) {
    const WeightView& view = weight.view;
    if (view.parts.empty()) { throw std::invalid_argument(weight.name + ": empty weight view"); }
    const WeightParent& source = *view.parts.front().parent;
    for (const auto& part : view.parts) {
        if (part.parent != &source) {
            throw std::invalid_argument(weight.name + ": a repack cannot span two parents");
        }
    }
    // Group the parts into runs of consecutive rows at one column offset. A column shard gives one
    // run over every row; a segmented row shard gives one run per segment.
    const std::uint64_t columns = source.geometry.padded_columns;
    if (columns == 0) { throw std::invalid_argument("repack: parent has no columns"); }
    std::vector<Run> runs;
    for (const auto& part : view.parts) {
        const std::uint64_t extent = part.end - part.begin;
        if (extent == 0) { continue; }
        const std::uint64_t row    = part.begin / columns;
        const std::uint64_t column = part.begin % columns;
        const std::uint64_t rows   = extent / columns;
        if (rows == 0) {
            // One row's column prefix. Consecutive rows at the same column coalesce.
            if (!runs.empty() && runs.back().column == column &&
                runs.back().source_row + runs.back().rows == row) {
                ++runs.back().rows;
            } else {
                runs.push_back({0, row, 1, column});
            }
            continue;
        }
        if (extent % columns != 0) {
            throw std::invalid_argument(weight.name +
                                        ": a repack needs row-aligned or column-prefix parts");
        }
        runs.push_back({0, row, rows, column});
    }
    std::uint64_t target_row = first_row;
    for (auto& run : runs) {
        run.target_row = target_row;
        target_row += run.rows;
    }
    if (target_row - first_row != view.shape[0]) {
        throw std::invalid_argument(weight.name + ": a repack does not cover its own rows");
    }

    const Device device                 = source.device;
    const std::size_t source_code_pitch = source.geometry.code_bytes_per_row;
    const std::size_t target_code_pitch = target.code_bytes_per_row;
    const std::size_t source_scale_pitch = source.geometry.scale_row_bytes;
    const std::size_t target_scale_pitch = target.scale_row_bytes;
    for (const auto& run : runs) {
        const std::size_t code_offset = code_bytes(source.geometry, run.column);
        copy_2d(source.data + run.source_row * source_code_pitch + code_offset, source_code_pitch,
                destination + run.target_row * target_code_pitch, target_code_pitch,
                target_code_pitch, run.rows, device);
        if (target.scale_row_bytes != 0 && source.geometry.scale_row_bytes != 0) {
            const std::size_t scale_offset = scale_bytes(source.geometry, run.column);
            copy_2d(source.data + source.geometry.scale_offset +
                        (run.source_row / source.geometry.scale_block_rows) * source_scale_pitch +
                        scale_offset,
                    source_scale_pitch,
                    destination + target.scale_offset +
                        (run.target_row / target.scale_block_rows) * target_scale_pitch,
                    target_scale_pitch, target_scale_pitch,
                    run.rows / source.geometry.scale_block_rows, device);
        }
    }
    return view.shape[0];
}

} // namespace

void RepackedWeights::gather(std::vector<BoundWeight>& weights,
                             std::span<const FusedGroup> groups) {
    // A fused consumer needs one parent across several parameters: the GDN q|k|v role is one
    // [5120, 5120] matrix at width 2 whose rows are query then key then value. So the gather's unit
    // is the group, not the parameter. The grouping is the same one the profile enumeration walks
    // (`execution/profile.cpp`'s role profiles), which is why the two cannot disagree.
    std::vector<bool> fused(weights.size(), false);
    for (const auto& group : groups) {
        if (group.members.size() < 2) { continue; }
        std::uint64_t rows            = 0;
        const WeightParent* source    = nullptr;
        const WeightParent* first     = nullptr;
        bool needs                    = false;
        for (const auto id : group.members) {
            const auto& view = weights.at(id.index).view;
            if (view.shape.size() != 2 || view.parts.empty()) {
                throw std::invalid_argument(weights.at(id.index).name +
                                            ": a fused member must be a matrix");
            }
            if (first == nullptr) { first = view.parts.front().parent; }
            // The consumer needs one parent: it fires when the members sit in more than one parent
            // (three separate objects) or when a member is not one contiguous region of its own.
            if (view.parts.front().parent != first || !contiguous(view)) { needs = true; }
            if (source == nullptr) { source = view.parts.front().parent; }
            rows += view.shape[0];
        }
        if (!needs || source == nullptr) { continue; } // width 1: one parent, every member whole
        for (const auto id : group.members) {
            fused.at(id.index) = true;
        }
        const std::uint64_t columns = weights.at(group.members.front().index).view.shape[1];
        const std::vector<std::uint64_t> fused_shape{rows, columns};
        const WeightGeometry target =
            weight_geometry(source->geometry.format, source->geometry.layout, fused_shape);
        auto owned = std::make_unique<Owned>();
        const Device device = source->device;
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
        if (!source->weight_divisors.empty()) {
            owned->divisors.assign(source->weight_divisors.begin(), source->weight_divisors.end());
            owned->parent.weight_divisors = owned->divisors;
        }
        auto* destination = const_cast<std::byte*>(owned->parent.data);
        std::uint64_t first_row = 0;
        for (const auto id : group.members) {
            auto& weight = weights.at(id.index);
            const std::uint64_t written =
                gather_view_into(weight, target, destination, first_row);
            const std::uint64_t begin = first_row * target.code_bytes_per_row;
            weight.view.parts = {WeightRegion{&owned->parent, begin,
                                              begin + weight_element_count(weight.view.shape)}};
            first_row += written;
        }
        owned_.push_back(std::move(owned));
    }

    // Every weight a fused group did not claim is gathered on its own, exactly as before.
    for (std::size_t index = 0; index < weights.size(); ++index) {
        if (fused.at(index)) { continue; }
        auto& weight = weights.at(index);
        WeightView& view = weight.view;
        if (view.shape.size() != 2) { continue; }
        if (contiguous(view)) { continue; }
        const WeightParent& source = *view.parts.front().parent;
        for (const auto& part : view.parts) {
            if (part.parent != &source) {
                throw std::invalid_argument(weight.name + ": a repack cannot span two parents");
            }
        }
        const WeightGeometry target =
            weight_geometry(source.geometry.format, source.geometry.layout, view.shape);
        auto owned          = std::make_unique<Owned>();
        const Device device = source.device;
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
        (void)gather_view_into(weight, target, destination, 0);
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
