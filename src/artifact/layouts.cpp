#include "artifact/layouts.h"

#include "artifact/formats.h"
#include "artifact/schema.h"

#include <limits>
#include <stdexcept>

namespace ninfer::artifact {
namespace {

[[nodiscard]] std::uint64_t add64(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::overflow_error("EXL3 geometry addition overflows u64");
    }
    return a + b;
}

[[nodiscard]] std::uint64_t mul64(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::overflow_error("EXL3 geometry multiplication overflows u64");
    }
    return a * b;
}

[[nodiscard]] std::uint64_t align256(std::uint64_t value) {
    return add64(value, 255) / 256 * 256;
}

[[nodiscard]] std::uint64_t read_u64_le(const std::byte* bytes) {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(bytes[i])) << (8 * i);
    }
    return value;
}

[[nodiscard]] std::uint32_t read_u32_le(const std::byte* bytes) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i) {
        value |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(bytes[i])) << (8 * i);
    }
    return value;
}

struct DirectoryEntry {
    std::uint64_t offset = 0;
    std::uint32_t extent = 0;
    std::uint32_t k2     = 0;
};

// Parse E entries of the directory prefix; returns false when the prefix is too short or an
// entry carries nonzero reserved bytes.
[[nodiscard]] bool parse_entries(std::span<const std::byte> directory, std::uint32_t experts,
                                 std::vector<DirectoryEntry>& entries) {
    if (mul64(experts, 16) > directory.size()) { return false; }
    entries.clear();
    for (std::uint32_t expert = 0; expert < experts; ++expert) {
        const auto* entry = directory.data() + static_cast<std::size_t>(expert) * 16;
        if (static_cast<std::uint8_t>(entry[13]) != 0 || static_cast<std::uint8_t>(entry[14]) != 0 ||
            static_cast<std::uint8_t>(entry[15]) != 0) {
            return false;
        }
        entries.push_back({read_u64_le(entry), read_u32_le(entry + 8),
                           static_cast<std::uint8_t>(entry[12])});
    }
    return true;
}

// The complete plane structure implied by E experts, validated against the object's bytes.
[[nodiscard]] bool planes_consistent(const TensorObject& object, std::uint32_t experts,
                                     const std::vector<DirectoryEntry>& entries,
                                     std::uint64_t rows, std::uint64_t k,
                                     Exl3BankGeometry& bank) {
    if (rows % experts != 0) { return false; }
    const auto rows_per_expert = rows / experts;
    if (rows_per_expert == 0 || rows_per_expert % 128 != 0 || k == 0 || k % 128 != 0) {
        return false;
    }
    const auto trellis_offset = align256(mul64(experts, 16));
    if (trellis_offset != entries.front().offset) { return false; }
    std::uint64_t trellis_bytes = 0;
    std::uint64_t cursor         = trellis_offset;
    for (const auto& entry : entries) {
        if (entry.offset != cursor || entry.offset % 256 != 0 || entry.extent % 256 != 0) {
            return false;
        }
        const auto extent = align256(mul64(mul64(k, rows_per_expert), entry.k2) / 16);
        if (extent != entry.extent) { return false; }
        if (entry.k2 % 2 == 0) {
            if (entry.k2 < 2 || entry.k2 > 16) { return false; }
        } else {
            if (entry.k2 != 3 && entry.k2 != 5 && entry.k2 != 7) { return false; }
            if (bank.codebook != 2) { return false; }  // Half-integer rates require mul1.
        }
        cursor        = add64(cursor, entry.extent);
        trellis_bytes = add64(trellis_bytes, entry.extent);
    }
    const auto suh_offset = cursor;
    const auto suh_bytes   = mul64(mul64(experts, k), 2);
    const auto svh_offset = align256(add64(suh_offset, suh_bytes));
    const auto svh_bytes   = mul64(rows, 2);
    if (add64(svh_offset, add64(svh_bytes, 4)) != object.bytes) { return false; }
    bank.trellis_offset = trellis_offset;
    bank.trellis_bytes  = trellis_bytes;
    bank.suh_offset     = suh_offset;
    bank.suh_bytes      = suh_bytes;
    bank.svh_offset     = svh_offset;
    bank.svh_bytes      = svh_bytes;
    bank.experts        = experts;
    bank.rows_per_expert = static_cast<std::uint32_t>(rows_per_expert);
    bank.trellis_extents.assign(entries.size(), 0);
    bank.k2.assign(entries.size(), 0);
    for (std::size_t expert = 0; expert < entries.size(); ++expert) {
        bank.trellis_extents[expert] = entries[expert].extent;
        bank.k2[expert]              = entries[expert].k2;
    }
    return true;
}

} // namespace

WeightGeometry describe_exl3_dense(const TensorObject& object,
                                   std::array<std::byte, 4> codebook_word) {
    try {
        if (parse_format(object.format) != QType::EXL3 ||
            parse_layout(object.layout) != QuantLayout::Exl3Dense) {
            throw std::invalid_argument("object is not an exl3_dense_v1");
        }
        if (object.shape.size() != 2) {
            throw std::invalid_argument("EXL3 dense linear must be rank 2 [N, K]");
        }
        const auto rows = object.shape[0];
        const auto k    = object.shape[1];
        if (rows % 128 != 0 || k % 128 != 0) {
            throw std::invalid_argument("EXL3 dense linear requires N and K divisible by 128");
        }
        const auto codebook = read_u32_le(codebook_word.data());
        if (codebook > 2) {
            throw std::invalid_argument("unknown EXL3 codebook id " + std::to_string(codebook));
        }
        // The object's byte size is align256(k·n·K/8) + 2·(k + n) + 4. Admissible bitrates
        // differ by n·k/16 bytes (at least 1024), far above the 256-byte trellis alignment,
        // so at most one 2K in 2…16 reproduces the size.
        const auto tail_bytes = add64(mul64(add64(k, rows), 2), 4);
        if (object.bytes < tail_bytes) {
            throw std::invalid_argument("EXL3 dense linear byte size is below its fixed planes");
        }
        std::uint32_t found = 0;
        std::uint64_t trellis_bytes = 0;
        std::uint32_t k2 = 0;
        for (std::uint32_t candidate = 2; candidate <= 16; ++candidate) {
            // Half-integer rates are 1.5, 2.5 and 3.5 only (contract section 2.4).
            if (candidate % 2 != 0 && candidate != 3 && candidate != 5 && candidate != 7) {
                continue;
            }
            const auto extent = align256(mul64(mul64(k, rows), candidate) / 16);
            if (add64(extent, tail_bytes) != object.bytes) { continue; }
            if (found != 0) {
                throw std::invalid_argument("ambiguous EXL3 dense bitrate");
            }
            found        = candidate;
            trellis_bytes = extent;
            k2            = candidate;
        }
        if (found == 0) {
            throw std::invalid_argument(
                "EXL3 dense linear byte size matches no admissible bitrate");
        }
        if (k2 % 2 != 0 && codebook != 2) {
            throw std::invalid_argument("half-integer EXL3 bitrate requires the mul1 codebook");
        }
        Exl3BankGeometry bank;
        bank.codebook        = codebook;
        bank.trellis_offset  = 0;
        bank.trellis_bytes   = trellis_bytes;
        bank.suh_offset      = trellis_bytes;
        bank.suh_bytes       = mul64(k, 2);
        bank.svh_offset      = add64(bank.suh_offset, bank.suh_bytes);
        bank.svh_bytes       = mul64(rows, 2);
        bank.experts         = 1;
        bank.rows_per_expert = static_cast<std::uint32_t>(rows);
        bank.trellis_extents = {trellis_bytes};
        bank.k2              = {k2};
        WeightGeometry geometry;
        geometry.format    = QType::EXL3;
        geometry.layout    = QuantLayout::Exl3Dense;
        geometry.shape     = object.shape;
        geometry.elements  = mul64(rows, k);
        geometry.bytes     = object.bytes;
        geometry.alignment = 256;
        geometry.exl3      = std::move(bank);
        return geometry;
    } catch (const std::exception& error) {
        throw ArtifactError(object.id + ": " + error.what());
    }
}

WeightGeometry describe_tensor(const TensorObject& object) {
    try {
        auto geometry =
            weight_geometry(parse_format(object.format), parse_layout(object.layout), object.shape);
        if (geometry.bytes != object.bytes || object.offset % geometry.alignment) {
            throw ArtifactError("encoded size or object alignment differs from layout");
        }
        return geometry;
    } catch (const std::exception& error) { throw ArtifactError(object.id + ": " + error.what()); }
}

WeightGeometry describe_exl3_bank(const TensorObject& object,
                                   std::span<const std::byte> directory,
                                   std::array<std::byte, 4> codebook_word) {
    try {
        if (parse_format(object.format) != QType::EXL3 ||
            parse_layout(object.layout) != QuantLayout::Exl3ExpertBank) {
            throw std::invalid_argument("object is not an exl3_expert_bank");
        }
        if (object.shape.size() != 2) {
            throw std::invalid_argument("EXL3 expert bank must be rank 2 [E*N, K]");
        }
        const auto rows = object.shape[0];
        const auto k    = object.shape[1];
        const auto codebook = read_u32_le(codebook_word.data());
        if (codebook > 2) {
            throw std::invalid_argument("unknown EXL3 codebook id " + std::to_string(codebook));
        }
        if (directory.size() < 16) {
            throw std::invalid_argument("EXL3 directory is shorter than one entry");
        }
        const auto trellis_offset = read_u64_le(directory.data());
        if (trellis_offset == 0 || trellis_offset % 256 != 0 || trellis_offset > directory.size()) {
            throw std::invalid_argument("EXL3 directory bound is invalid");
        }
        // The directory is E*16 bytes with align_up(E*16, 256) == entry 0's offset, so the
        // expert count satisfies (offset - 256)/16 < E <= offset/16: at most sixteen
        // candidates, of which exactly one must reproduce the object's byte size.
        std::vector<DirectoryEntry> entries;
        Exl3BankGeometry bank;
        bank.codebook = codebook;
        std::uint32_t found = 0;
        const auto first_candidate =
            trellis_offset > 256 ? static_cast<std::uint32_t>((trellis_offset - 256) / 16 + 1) : 1;
        for (std::uint32_t experts = first_candidate; experts * 16 <= trellis_offset; ++experts) {
            if (!parse_entries(directory, experts, entries)) { continue; }
            Exl3BankGeometry candidate = bank;
            if (planes_consistent(object, experts, entries, rows, k, candidate)) {
                if (found != 0) { throw std::invalid_argument("ambiguous EXL3 expert count"); }
                found  = experts;
                bank   = candidate;
            }
        }
        if (found == 0) {
            throw std::invalid_argument("EXL3 directory does not describe the object's bytes");
        }
        WeightGeometry geometry;
        geometry.format    = QType::EXL3;
        geometry.layout    = QuantLayout::Exl3ExpertBank;
        geometry.shape     = object.shape;
        geometry.elements  = mul64(rows, k);
        geometry.bytes     = object.bytes;
        geometry.alignment = 256;
        geometry.exl3      = std::move(bank);
        return geometry;
    } catch (const std::exception& error) {
        throw ArtifactError(object.id + ": " + error.what());
    }
}

} // namespace ninfer::artifact
