// Qualifies the test-owned EXL3 codec decoder (exl3_ref.h) against the external fixture produced
// by the pinned exllamav3 (docs/maintainer/exl3-trellis-contract.md 3.3). Every fixture tile and
// coverage linear must decode bit-exactly; the FP16 composition exllamav3 reports for the linear is
// compared with the binary64 composition of contract 2.1 within an FP16 tolerance. Without the
// fixture the test skips.

#include "ops/exl3_ref.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace ref = ninfer::test::exl3_ref;
using Json    = nlohmann::json;

constexpr int kSkip = 77;

std::filesystem::path fixture_path() {
    if (const char* path = std::getenv("NINFER_EXL3_FIXTURE"); path != nullptr && *path != '\0') {
        return path;
    }
    return std::filesystem::path(NINFER_SOURCE_DIR) / "tests/fixtures/exl3/exllamav3-1.6.1.json";
}

std::vector<std::uint16_t> hex_u16(const Json& value) {
    const std::string text = value.get<std::string>();
    if (text.size() % 4 != 0) throw std::runtime_error("fixture array is not whole 16-bit words");
    std::vector<std::uint16_t> out(text.size() / 4);
    const auto nibble = [](char c) -> unsigned {
        if (c >= '0' && c <= '9') return static_cast<unsigned>(c - '0');
        if (c >= 'a' && c <= 'f') return static_cast<unsigned>(c - 'a' + 10);
        throw std::runtime_error("fixture hex is not lowercase hexadecimal");
    };
    for (std::size_t i = 0; i < out.size(); ++i) {
        const auto byte = [&](std::size_t at) {
            return nibble(text[at]) << 4 | nibble(text[at + 1]);
        };
        out[i] = static_cast<std::uint16_t>(byte(4 * i) | byte(4 * i + 2) << 8); // little-endian
    }
    return out;
}

int compare_bits(const std::string& label, const std::vector<std::uint16_t>& actual,
                 const std::vector<std::uint16_t>& expected) {
    if (actual.size() != expected.size()) {
        std::cerr << label << ": decoded " << actual.size() << " values, fixture holds "
                  << expected.size() << '\n';
        return 1;
    }
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] == expected[i]) continue;
        if (mismatches++ == 0) {
            std::cerr << label << ": first mismatch at " << i << ": decoded 0x" << std::hex
                      << actual[i] << ", exllamav3 0x" << expected[i] << std::dec << '\n';
        }
    }
    if (mismatches != 0) {
        std::cerr << label << ": " << mismatches << " of " << actual.size() << " values differ\n";
        return 1;
    }
    return 0;
}

int check_tiles(const Json& tiles) {
    int failures = 0;
    for (const Json& tile : tiles) {
        const int k2           = tile.at("k2").get<int>();
        const std::string name = tile.at("codebook").get<std::string>();
        const auto elements    = hex_u16(tile.at("trellis"));
        if (elements.size() != static_cast<std::size_t>(ref::words_per_tile(k2))) {
            std::cerr << "fixture tile has " << elements.size() << " elements for k2=" << k2
                      << '\n';
            ++failures;
            continue;
        }
        std::vector<std::uint16_t> decoded(ref::kTileWeights);
        ref::decode_tile(elements.data(), k2, ref::parse_codebook(name), decoded.data());
        failures += compare_bits("tile K=" + std::to_string(k2 / 2.0) + " " + name, decoded,
                                 hex_u16(tile.at("inner")));
    }
    return failures;
}

int check_linear_inner(const std::string& label, const Json& linear) {
    const int k2  = linear.at("k2").get<int>();
    const int k   = linear.at("k").get<int>();
    const int n   = linear.at("n").get<int>();
    const auto tr = hex_u16(linear.at("trellis"));
    if (tr.size() != static_cast<std::size_t>(k / 16) * (n / 16) * ref::words_per_tile(k2)) {
        std::cerr << label << ": trellis size does not match its geometry\n";
        return 1;
    }
    const auto codebook = ref::parse_codebook(linear.at("codebook").get<std::string>());
    return compare_bits(label, ref::decode_inner(tr.data(), k, n, k2, codebook),
                        hex_u16(linear.at("inner")));
}

// exllamav3 composes in FP16 (an FP16 Hadamard matrix, FP16 products and scalings), so its result
// differs from the binary64 composition by a few FP16 roundings per element. A wrong transform
// order or side produces O(1) relative error, far outside these bounds.
int check_composition(const Json& linear) {
    const int k      = linear.at("k").get<int>();
    const int n      = linear.at("n").get<int>();
    const auto inner = hex_u16(linear.at("inner"));
    const auto weight =
        ref::linear_weight(inner, hex_u16(linear.at("suh")), hex_u16(linear.at("svh")), k, n);
    const auto fixture = hex_u16(linear.at("weight"));
    double error_sq = 0.0, reference_sq = 0.0, max_error = 0.0, max_reference = 0.0;
    for (std::size_t i = 0; i < weight.size(); ++i) {
        const double error = ref::binary16_to_double(fixture[i]) - weight[i];
        error_sq += error * error;
        reference_sq += weight[i] * weight[i];
        max_error     = std::max(max_error, std::fabs(error));
        max_reference = std::max(max_reference, std::fabs(weight[i]));
    }
    const double relative_l2 = std::sqrt(error_sq / reference_sq);
    const double gross       = max_error / max_reference;
    std::cout << "composition: relative L2 " << relative_l2 << ", max |err| / max |ref| " << gross
              << '\n';
    if (relative_l2 > 2e-3 || gross > 4e-3) {
        std::cerr << "composition: exllamav3 get_weight_tensor() disagrees with contract 2.1\n";
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    const auto path = fixture_path();
    if (!std::filesystem::exists(path)) {
        std::cout << "skip: EXL3 fixture " << path << " has not been generated\n";
        return kSkip;
    }
    try {
        std::ifstream input(path);
        const Json fixture = Json::parse(input);
        if (fixture.at("format") != "ninfer_exl3_fixture" || fixture.at("format_version") != 1) {
            std::cerr << "unknown EXL3 fixture format\n";
            return 1;
        }
        int failures = check_tiles(fixture.at("tiles"));
        for (const Json& coverage : fixture.at("coverage")) {
            failures += check_linear_inner("coverage " + coverage.at("codebook").get<std::string>(),
                                           coverage);
        }
        failures += check_linear_inner("linear", fixture.at("linear"));
        failures += check_composition(fixture.at("linear"));
        if (failures == 0) {
            std::cout << "EXL3 codec decoder reproduces the exllamav3 "
                      << fixture.at("exllamav3").at("version").get<std::string>() << " fixture\n";
        }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "EXL3 codec fixture: " << error.what() << '\n';
        return 1;
    }
}
