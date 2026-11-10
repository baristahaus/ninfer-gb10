// The plan asks for the geometry a consumer will see; the load-time repack builds a buffer of that
// geometry. This test pins that the two answers are the same value, because if they can disagree a
// future change will make them.
//
// Host-only: `consumer_geometry` reads the view's parts and the parent's geometry and touches no
// device.

#include "core/weight_view.h"

#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

struct HostParent {
    std::vector<std::uint16_t> words;
    ninfer::WeightParent parent;
};

HostParent make_parent(std::uint64_t rows, std::uint64_t columns) {
    HostParent out;
    out.words.resize(rows * columns);
    const std::vector<std::uint64_t> shape{rows, columns};
    out.parent.geometry = ninfer::weight_geometry(ninfer::QType::BF16,
                                                  ninfer::QuantLayout::Contiguous, shape);
    out.parent.device = ninfer::Device::cpu();
    out.parent.data   = reinterpret_cast<const std::byte*>(out.words.data());
    return out;
}

// The rank's column shard as `shard_parameter` produces it: one part per row.
ninfer::WeightView column_shard(const HostParent& source, std::uint64_t rows,
                                std::uint64_t columns, std::uint64_t begin,
                                std::uint64_t extent) {
    ninfer::WeightView view;
    view.shape = {rows, extent};
    for (std::uint64_t row = 0; row < rows; ++row) {
        view.parts.push_back(
            {&source.parent, row * columns + begin, row * columns + begin + extent});
    }
    return view;
}

ninfer::WeightView whole(const HostParent& source, std::uint64_t rows, std::uint64_t columns) {
    ninfer::WeightView view;
    view.shape = {rows, columns};
    view.parts.push_back({&source.parent, 0, rows * columns});
    return view;
}

int check_agrees(const ninfer::WeightView& view, const ninfer::WeightGeometry& expected,
                 const char* label) {
    const auto described = ninfer::consumer_geometry(view);
    if (described.format != expected.format || described.layout != expected.layout ||
        described.shape != expected.shape || described.bytes != expected.bytes ||
        described.code_bytes_per_row != expected.code_bytes_per_row ||
        described.scale_row_bytes != expected.scale_row_bytes ||
        described.divisor_count != expected.divisor_count) {
        return fail(std::string(label) + ": the plan's geometry differs from the buffer's");
    }
    return 0;
}

} // namespace

int main() {
    try {
        int failures = 0;
        constexpr std::uint64_t rows = 64;
        constexpr std::uint64_t columns = 512;
        HostParent source = make_parent(rows, columns);

        // A column shard: the plan must describe the rank's own [rows, columns/2] matrix, which is
        // exactly what the repack builds.
        for (std::uint64_t rank = 0; rank < 2; ++rank) {
            const auto view = column_shard(source, rows, columns, rank * (columns / 2),
                                           columns / 2);
            const std::vector<std::uint64_t> shape{rows, columns / 2};
            const auto expected = ninfer::weight_geometry(ninfer::QType::BF16,
                                                          ninfer::QuantLayout::Contiguous, shape);
            failures += check_agrees(view, expected,
                                     rank == 0 ? "rank 0 column shard" : "rank 1 column shard");
        }

        // A whole weight is its parent's geometry, unchanged: the single-device lane's answer.
        failures +=
            check_agrees(whole(source, rows, columns), source.parent.geometry, "whole weight");

        // A view spanning two parents has no single answer and must refuse rather than pick one.
        HostParent other = make_parent(rows, columns);
        auto split = whole(source, rows, columns);
        split.parts.push_back({&other.parent, 0, 8});
        bool threw = false;
        try {
            (void)ninfer::consumer_geometry(split);
        } catch (const std::invalid_argument&) { threw = true; }
        if (!threw) { return fail("a view spanning two parents did not refuse"); }

        if (failures == 0) { std::cout << "ok\n"; }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
