// The column-shard repack, checked by value rather than by shape. Every defect in this area so far
// has been one that "looked right": a coverage assertion passes a wrong offset, so this decodes the
// gathered parent back to floats and compares it element by element against the source matrix's
// columns for the rank.
//
// Host-only: the gather runs on a host-backed parent, which is the same arithmetic the device path
// uses (only the copy primitive differs).

#include "models/common/repacked.h"

#include <cmath>
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

// A BF16 matrix over pageable host memory, plus the WeightParent that describes it.
struct HostMatrix {
    std::vector<std::uint16_t> words;
    ninfer::WeightParent parent;
};

float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16;
    const std::uint32_t exponent = (bits >> 10) & 0x1FU;
    const std::uint32_t mantissa = bits & 0x3FFU;
    std::uint32_t out = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            out = sign;
        } else {
            // Subnormal: normalize it.
            std::uint32_t m = mantissa;
            std::int32_t e  = -14;
            while ((m & 0x400U) == 0) {
                m <<= 1;
                --e;
            }
            out = sign | static_cast<std::uint32_t>(e + 127) << 23 | ((m & 0x3FFU) << 13);
        }
    } else if (exponent == 0x1FU) {
        out = sign | 0x7F800000U | (mantissa << 13);
    } else {
        out = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    float value = 0.0F;
    std::memcpy(&value, &out, sizeof(value));
    return value;
}

HostMatrix make_matrix(std::uint64_t rows, std::uint64_t columns) {
    HostMatrix out;
    out.words.resize(rows * columns);
    // A value that depends on both coordinates, so a transposed or shifted gather cannot match.
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < columns; ++column) {
            const auto value = static_cast<float>(row) * 0.5F + static_cast<float>(column) * 0.25F;
            // Store as BF16 by rounding the FP32.
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            const std::uint16_t half = static_cast<std::uint16_t>(
                ((bits >> 16) & 0x8000U) | (((bits >> 23) & 0xFFU) - 112U) << 7 |
                ((bits >> 16) & 0x7FU));
            out.words[row * columns + column] = half;
        }
    }
    const std::vector<std::uint64_t> shape{rows, columns};
    out.parent.geometry = ninfer::weight_geometry(ninfer::QType::BF16,
                                                  ninfer::QuantLayout::Contiguous, shape);
    out.parent.device = ninfer::Device::cpu();
    out.parent.data   = reinterpret_cast<const std::byte*>(out.words.data());
    return out;
}

// The rank's column shard as `shard_parameter` produces it: one part per row.
ninfer::WeightView column_shard(const HostMatrix& matrix, std::uint64_t rows,
                                std::uint64_t columns, std::uint64_t begin,
                                std::uint64_t extent) {
    ninfer::WeightView view;
    view.shape = {rows, extent};
    for (std::uint64_t row = 0; row < rows; ++row) {
        view.parts.push_back(
            {&matrix.parent, row * columns + begin, row * columns + begin + extent});
    }
    return view;
}

int check_shard(std::uint64_t rows, std::uint64_t columns, std::uint64_t begin,
                std::uint64_t extent, const char* label) {
    HostMatrix source = make_matrix(rows, columns);
    ninfer::models::BoundWeight weight;
    weight.name = label;
    weight.view = column_shard(source, rows, columns, begin, extent);

    std::vector<ninfer::models::BoundWeight> weights;
    weights.push_back(std::move(weight));
    ninfer::models::RepackedWeights repacked;
    repacked.gather(weights);

    if (weights.front().view.parts.size() != 1) {
        return fail(std::string(label) + ": the gathered view is not one region");
    }
    const auto& part = weights.front().view.parts.front();
    if (part.parent == &source.parent) {
        return fail(std::string(label) + ": the gathered view still points at the source");
    }
    if (part.end - part.begin != rows * extent) {
        return fail(std::string(label) + ": the gathered region does not cover the shard");
    }
    const auto* gathered = reinterpret_cast<const std::uint16_t*>(part.parent->data);
    // The value oracle: every element must equal the source's element at the same logical
    // position, which is row `row`, column `begin + column`.
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t column = 0; column < extent; ++column) {
            const std::uint16_t want = source.words[row * columns + begin + column];
            const std::uint16_t got  = gathered[row * extent + column];
            if (want != got) {
                return fail(std::string(label) + ": value differs at row " + std::to_string(row) +
                            " column " + std::to_string(column) + " (" +
                            std::to_string(half_to_float(got)) + " vs " +
                            std::to_string(half_to_float(want)) + ")");
            }
        }
    }
    return 0;
}

// A view that is already one contiguous range must not be repacked.
int check_untouched(std::uint64_t rows, std::uint64_t columns, std::uint64_t begin,
                    std::uint64_t extent, const char* label) {
    HostMatrix source = make_matrix(rows, columns);
    std::vector<ninfer::models::BoundWeight> weights;
    weights.push_back({label, {}, column_shard(source, rows, columns, begin, extent), {}});
    ninfer::models::RepackedWeights repacked;
    repacked.gather(weights);
    if (repacked.count() != 0) {
        return fail(std::string(label) + ": the gather fired on a contiguous view");
    }
    if (weights.front().view.parts.size() != 1 ||
        weights.front().view.parts.front().parent != &source.parent) {
        return fail(std::string(label) + ": a contiguous view was rewritten");
    }
    return 0;
}

} // namespace

int main() {
    try {
        int failures = 0;
        // Rank 0 and rank 1 of a 64 x 512 matrix: the first and second column halves.
        failures += check_shard(64, 512, 0, 256, "rank 0 column half");
        failures += check_shard(64, 512, 256, 256, "rank 1 column half");
        // A shape whose rows and columns are not equal, so a transposed gather cannot pass.
        failures += check_shard(16, 1024, 512, 512, "wide matrix, second half");
        // A one-row shard is already one contiguous range, so the gather must leave it alone: the
        // detection condition is "not contiguous", and this pins that it does not over-fire.
        failures += check_untouched(1, 64, 32, 32, "single row shard");
        if (failures == 0) { std::cout << "ok\n"; }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
