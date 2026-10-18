#include "models/qwen3_8_flash_next/impl/ple_table.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace q38 = ninfer::models::qwen3_8_flash_next;

namespace {

// Independent E2M1/E4M3 word decoders in double precision and an independent
// nearest-even BF16 encoder; they share no code with the production decoder.
double e2m1_value(unsigned code) {
    static constexpr double kMagnitudes[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const double magnitude = kMagnitudes[code & 7];
    return (code & 8) ? -magnitude : magnitude;
}

double e4m3_value(unsigned word) {
    const double sign = (word & 0x80) ? -1.0 : 1.0;
    const unsigned exponent = (word >> 3) & 0xF;
    const unsigned fraction = word & 7;
    if (exponent == 0) { return sign * static_cast<double>(fraction) * std::exp2(-9.0); }
    if (exponent == 15 && fraction == 7) { return std::numeric_limits<double>::quiet_NaN(); }
    return sign * (1.0 + static_cast<double>(fraction) / 8.0) * std::exp2(exponent - 7.0);
}

std::uint16_t bf16_bits_ne(double value) {
    // value is exactly representable in BF16 here (<= 6 mantissa bits); the
    // round-to-nearest-even reference therefore never loses a bit. Assert the
    // float representation is exact before folding to BF16.
    const float single = static_cast<float>(value);
    const std::uint32_t word = std::bit_cast<std::uint32_t>(single);
    if ((word & 0x0000FFFFU) != 0) {
        throw std::runtime_error("test value is not exactly representable in float");
    }
    const std::uint32_t lsb     = (word >> 16) & 1;
    const std::uint32_t rounded = (word + 0x7FFFU + lsb) & 0xFFFF0000U;
    return static_cast<std::uint16_t>(rounded >> 16);
}

void check_row(unsigned scale_word, unsigned nibble, unsigned position,
               std::span<std::byte> codes, std::span<std::byte> scales,
               std::span<std::byte> staged) {
    const int half  = static_cast<int>(position / 2);
    const bool high = (position & 1) != 0;
    codes[half]     = static_cast<std::byte>(high ? nibble << 4 : nibble);
    for (auto& scale_byte : scales) { scale_byte = static_cast<std::byte>(scale_word); }
    try {
        q38::decode_ple_nvfp4_row_bf16(codes, scales, staged);
    } catch (const std::exception& error) {
        std::cerr << "unexpected decode failure at scale=0x" << std::hex << scale_word << std::dec
                  << ": " << error.what() << '\n';
        throw;
    }
    const double value = e2m1_value(nibble) * e4m3_value(scale_word);
    const std::uint16_t expected = bf16_bits_ne(value);
    const std::uint16_t actual =
        static_cast<std::uint16_t>(static_cast<unsigned>(staged[2 * position]) |
                                   (static_cast<unsigned>(staged[2 * position + 1]) << 8));
    if (actual != expected) {
        std::cerr << "PLE NVFP4 decode mismatch: scale=0x" << std::hex << scale_word << std::dec
                  << " nibble=" << nibble << " position=" << position << " value=" << value
                  << " staged=0x" << std::hex << actual << " expected=0x" << expected << std::dec
                  << '\n';
        throw std::runtime_error("PLE NVFP4 decode differs from oracle");
    }
}

} // namespace

int main() {
    try {
        std::vector<std::byte> codes(80);
        std::vector<std::byte> scales(10);
        std::vector<std::byte> staged(2 * 160);
        // Exhaustive: every group scale word and every E2M1 nibble in both nibble
        // positions, against the independent double-precision oracle. The E2M1 x
        // E4M3-scale product has at most six mantissa bits, so BF16 holds it
        // exactly; no value can overflow.
        for (unsigned scale_word = 0; scale_word < 256; ++scale_word) {
            if (std::isnan(e4m3_value(scale_word))) { continue; }
            for (unsigned nibble = 0; nibble < 16; ++nibble) {
                for (unsigned position : {0U, 128U}) {
                    check_row(scale_word, nibble, position, codes, scales, staged);
                }
            }
        }

        // The NaN scale word is not a legal NVFP4 scale and must fail loudly.
        std::vector<std::byte> nan_scales(10, static_cast<std::byte>(0x7F));
        std::vector<std::byte> any_codes(80, static_cast<std::byte>(0x12));
        try {
            q38::decode_ple_nvfp4_row_bf16(any_codes, nan_scales, staged);
            throw std::runtime_error("a NaN PLE NVFP4 group scale was not rejected");
        } catch (const std::range_error&) {
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
