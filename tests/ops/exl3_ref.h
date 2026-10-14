#pragma once

// Test-owned exact EXL3 codec decoder (docs/maintainer/exl3-trellis-contract.md 2.1, 2.3-2.5).
// It is the contract's C++ exact codec oracle for kernel qualification (3.1, 3.3): stored trellis
// int16 elements, the bitrate and the codebook map to FP16 inner-weight bit patterns, compared bit
// for bit. It is written from the contract alone and calls no production code. Binary16 values are
// handled as bit patterns, with this file's own round-to-nearest-even; no compiler half type is
// used. Bitrates travel as k2 = 2K (2..16, or 3/5/7 for the MUL1-only half-integer rates).

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::test::exl3_ref {

enum class Codebook { Inst3, Mcg, Mul1 };

inline Codebook parse_codebook(std::string_view name) {
    if (name == "3inst") return Codebook::Inst3;
    if (name == "mcg") return Codebook::Mcg;
    if (name == "mul1") return Codebook::Mul1;
    throw std::invalid_argument("unknown EXL3 codebook: " + std::string(name));
}

inline void require_rate(int k2, Codebook codebook) {
    if (k2 % 2 == 0) {
        if (k2 < 2 || k2 > 16) throw std::invalid_argument("integer EXL3 bitrate must be 1..8");
        return;
    }
    if (k2 != 3 && k2 != 5 && k2 != 7)
        throw std::invalid_argument("half-integer EXL3 bitrate must be 1.5, 2.5 or 3.5");
    if (codebook != Codebook::Mul1)
        throw std::invalid_argument("half-integer EXL3 bitrates require the mul1 codebook");
}

inline constexpr int kTile        = 16;
inline constexpr int kTileWeights = 256;

inline int words_per_tile(int k2) { return 8 * k2; }

// binary16 ------------------------------------------------------------------------------------

inline double binary16_to_double(std::uint16_t bits) {
    const double sign  = (bits & 0x8000U) != 0 ? -1.0 : 1.0;
    const int exponent = (bits >> 10) & 0x1F;
    const int fraction = bits & 0x3FF;
    if (exponent == 0x1F) throw std::domain_error("binary16 pattern is not finite");
    if (exponent == 0) return sign * std::ldexp(static_cast<double>(fraction), -24);
    return sign * std::ldexp(static_cast<double>(1024 + fraction), exponent - 25);
}

// Round-to-nearest-even of a binary64 value to binary16 (subnormals, overflow to infinity, and
// the sign of zero preserved). Callers pass values that are exact in binary64.
inline std::uint16_t round_binary16(double value) {
    const std::uint16_t sign = std::signbit(value) ? 0x8000U : 0U;
    const double magnitude   = std::fabs(value);
    if (magnitude == 0.0) return sign;
    if (!std::isfinite(magnitude))
        throw std::domain_error("binary16 rounding of a non-finite value");
    int binade = 0;
    (void)std::frexp(magnitude, &binade); // magnitude in [2^(binade-1), 2^binade)
    const int unbiased  = binade - 1;
    const int quantum   = (unbiased < -14 ? -14 : unbiased) - 10;
    const double scaled = std::ldexp(magnitude, -quantum); // exact: a power-of-two scaling
    double integer      = std::floor(scaled);
    const double rest   = scaled - integer;
    if (rest > 0.5 || (rest == 0.5 && std::fmod(integer, 2.0) == 1.0)) integer += 1.0;
    auto units = static_cast<std::uint32_t>(integer);
    if (quantum == -24) return static_cast<std::uint16_t>(sign | units); // 2^-24 spacing
    int exponent = quantum + 25;
    if (units == 2048U) {
        units = 1024U;
        exponent += 1;
    }
    if (exponent >= 0x1F) return static_cast<std::uint16_t>(sign | 0x7C00U);
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10) |
                                      (units - 1024U));
}

// Section 2.5 ---------------------------------------------------------------------------------

inline std::uint16_t codebook_value(std::uint32_t state, Codebook codebook) {
    if (codebook == Codebook::Mul1) {
        const std::uint32_t x = state * 0x83DCD12DU;
        const std::uint32_t b = (x & 0xFFU) + ((x >> 8) & 0xFFU) + ((x >> 16) & 0xFFU) + (x >> 24);
        const double h        = binary16_to_double(static_cast<std::uint16_t>(0x6400U + b));
        const double product  = h * binary16_to_double(0x1EEE);
        return round_binary16(product + binary16_to_double(0xC931));
    }
    std::uint32_t x =
        codebook == Codebook::Inst3 ? state * 89226354U + 64248484U : state * 0xCBAC1FEDU;
    x                 = (x & 0x8FFF8FFFU) ^ 0x3B603B60U;
    const double low  = binary16_to_double(static_cast<std::uint16_t>(x & 0xFFFFU));
    const double high = binary16_to_double(static_cast<std::uint16_t>(x >> 16));
    return round_binary16(low + high);
}

// Sections 2.3 and 2.4 ------------------------------------------------------------------------

// Decodes one tile of words_per_tile(k2) stored int16 elements into 256 binary16 patterns,
// row-major over the tile's (r, c).
inline void decode_tile(const std::uint16_t* elements, int k2, Codebook codebook,
                        std::uint16_t* out) {
    require_rate(k2, codebook);
    const int words = 4 * k2; // little-endian uint32 words of the ring
    std::vector<std::uint32_t> ring(static_cast<std::size_t>(words));
    for (int m = 0; m < words; ++m)
        ring[static_cast<std::size_t>(m)] = static_cast<std::uint32_t>(elements[2 * m]) |
                                            (static_cast<std::uint32_t>(elements[2 * m + 1]) << 16);
    const int ring_bits = 32 * words;
    const auto bit      = [&](int p) {
        p %= ring_bits;
        return (ring[static_cast<std::size_t>(p / 32)] >> (31 - p % 32)) & 1U;
    };
    int end = 0;
    for (int i = 0; i < kTileWeights; ++i) {
        end += k2 / 2 + ((k2 % 2 != 0) ? ((0xAAAA >> (i % 16)) & 1) : 0);
        std::uint32_t state = 0;
        for (int p = end - 16; p < end; ++p) state = (state << 1) | bit(p + ring_bits);
        const int t = i >> 3, j = i & 7;
        const int r        = 2 * (t & 3) + (j & 1) + 8 * ((j >> 1) & 1);
        const int c        = (t >> 2) + 8 * ((j >> 2) & 1);
        out[r * kTile + c] = codebook_value(state, codebook);
    }
    if (end != ring_bits) throw std::logic_error("EXL3 step widths do not fill the ring");
}

// W_inner [k, n] row-major from trellis [k/16, n/16, 16 K].
inline std::vector<std::uint16_t> decode_inner(const std::uint16_t* trellis, int k, int n, int k2,
                                               Codebook codebook) {
    if (k % kTile != 0 || n % kTile != 0) throw std::invalid_argument("EXL3 dims must tile by 16");
    std::vector<std::uint16_t> inner(static_cast<std::size_t>(k) * static_cast<std::size_t>(n));
    std::uint16_t tile[kTileWeights];
    const int tiles_n = n / kTile;
    for (int a = 0; a < k / kTile; ++a) {
        for (int b = 0; b < tiles_n; ++b) {
            const std::size_t index = static_cast<std::size_t>(a) * tiles_n + b;
            decode_tile(trellis + index * static_cast<std::size_t>(words_per_tile(k2)), k2,
                        codebook, tile);
            for (int r = 0; r < kTile; ++r)
                for (int c = 0; c < kTile; ++c)
                    inner[static_cast<std::size_t>(kTile * a + r) * n + kTile * b + c] =
                        tile[r * kTile + c];
        }
    }
    return inner;
}

// Section 2.1 ---------------------------------------------------------------------------------

// W = diag(suh) Hn_k W_inner Hn_n diag(svh) in binary64, [k, n] row-major, where Hn_m holds m/128
// copies of H128 / sqrt(128) and H128[i][j] = (-1)^popcount(i AND j).
inline std::vector<double> linear_weight(const std::vector<std::uint16_t>& inner,
                                         const std::vector<std::uint16_t>& suh,
                                         const std::vector<std::uint16_t>& svh, int k, int n) {
    constexpr int block = 128;
    if (k % block != 0 || n % block != 0) throw std::invalid_argument("EXL3 dims must be 128k");
    const double norm = 1.0 / std::sqrt(static_cast<double>(block));
    const auto h      = [](int i, int j) {
        return (__builtin_popcount(static_cast<unsigned>(i & j)) & 1) != 0 ? -1.0 : 1.0;
    };
    const auto at = [n](int row, int col) {
        return static_cast<std::size_t>(row) * static_cast<std::size_t>(n) +
               static_cast<std::size_t>(col);
    };
    std::vector<double> left(inner.size(), 0.0);
    for (int row = 0; row < k; ++row) {
        const int base = row - row % block;
        for (int col = 0; col < n; ++col) {
            double sum = 0.0;
            for (int i = 0; i < block; ++i)
                sum += h(row - base, i) * binary16_to_double(inner[at(base + i, col)]);
            left[at(row, col)] = norm * sum;
        }
    }
    std::vector<double> weight(inner.size(), 0.0);
    for (int row = 0; row < k; ++row) {
        const double u = binary16_to_double(suh[static_cast<std::size_t>(row)]);
        for (int col = 0; col < n; ++col) {
            const int base = col - col % block;
            double sum     = 0.0;
            for (int i = 0; i < block; ++i) sum += left[at(row, base + i)] * h(i, col - base);
            weight[at(row, col)] =
                u * norm * sum * binary16_to_double(svh[static_cast<std::size_t>(col)]);
        }
    }
    return weight;
}

} // namespace ninfer::test::exl3_ref
