#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
template <std::int32_t OutputRows, std::int32_t InputRows>
struct Fp8Geometry {
    static_assert(OutputRows > 0 && (OutputRows % 16) == 0);
    static_assert(InputRows > 0 && (InputRows % 32) == 0);

    static constexpr std::int32_t kOutputRows = OutputRows;
    static constexpr std::int32_t kInputRows  = InputRows;
};

template <std::int32_t InputRows>
struct Fp8ActivationGeometry {
    static_assert(InputRows > 0 && (InputRows % 32) == 0);

    static constexpr std::int32_t kInputRows = InputRows;
};

using Fp8N14336K5120             = Fp8Geometry<14336, 5120>;
using Fp8N16384K5120             = Fp8Geometry<16384, 5120>;
using Fp8N34816K5120             = Fp8Geometry<34816, 5120>;
using Fp8N248320K5120            = Fp8Geometry<248320, 5120>;
using Fp8N5120K6144              = Fp8Geometry<5120, 6144>;
using Fp8N5120K17408             = Fp8Geometry<5120, 17408>;
// One rank's half of the fused FP8 objects the 27B binds at tensor-parallel width 2.
using Fp8N7168K5120              = Fp8Geometry<7168, 5120>;
using Fp8N8192K5120              = Fp8Geometry<8192, 5120>;
using Fp8N124160K5120            = Fp8Geometry<124160, 5120>;
// The role-split shapes of the sharded 27B lane: one role of a fused qkv/qkvz object.
using Fp8N5120K5120              = Fp8Geometry<5120, 5120>;
using Fp8N3072K5120              = Fp8Geometry<3072, 5120>;
using Fp8N512K5120               = Fp8Geometry<512, 5120>;
using Fp8N8704K5120              = Fp8Geometry<8704, 5120>;
using Fp8Activation5120Geometry  = Fp8ActivationGeometry<5120>;
using Fp8Activation6144Geometry  = Fp8ActivationGeometry<6144>;
using Fp8Activation17408Geometry = Fp8ActivationGeometry<17408>;

enum class Fp8GeometryId : std::uint8_t {
    N14336K5120,
    N16384K5120,
    N34816K5120,
    N248320K5120,
    N5120K6144,
    N5120K17408,
    N7168K5120,
    N8192K5120,
    N124160K5120,
    N5120K5120,
    N3072K5120,
    N512K5120,
    N8704K5120,
};

inline Fp8GeometryId resolve_fp8_geometry(std::int32_t output_rows, std::int32_t input_rows) {
    if (output_rows == Fp8N14336K5120::kOutputRows && input_rows == Fp8N14336K5120::kInputRows) {
        return Fp8GeometryId::N14336K5120;
    }
    if (output_rows == Fp8N16384K5120::kOutputRows && input_rows == Fp8N16384K5120::kInputRows) {
        return Fp8GeometryId::N16384K5120;
    }
    if (output_rows == Fp8N34816K5120::kOutputRows && input_rows == Fp8N34816K5120::kInputRows) {
        return Fp8GeometryId::N34816K5120;
    }
    if (output_rows == Fp8N248320K5120::kOutputRows && input_rows == Fp8N248320K5120::kInputRows) {
        return Fp8GeometryId::N248320K5120;
    }
    if (output_rows == Fp8N5120K6144::kOutputRows && input_rows == Fp8N5120K6144::kInputRows) {
        return Fp8GeometryId::N5120K6144;
    }
    if (output_rows == Fp8N5120K17408::kOutputRows && input_rows == Fp8N5120K17408::kInputRows) {
        return Fp8GeometryId::N5120K17408;
    }
    if (output_rows == Fp8N7168K5120::kOutputRows && input_rows == Fp8N7168K5120::kInputRows) {
        return Fp8GeometryId::N7168K5120;
    }
    if (output_rows == Fp8N8192K5120::kOutputRows && input_rows == Fp8N8192K5120::kInputRows) {
        return Fp8GeometryId::N8192K5120;
    }
    if (output_rows == Fp8N124160K5120::kOutputRows && input_rows == Fp8N124160K5120::kInputRows) {
        return Fp8GeometryId::N124160K5120;
    }
    if (output_rows == Fp8N5120K5120::kOutputRows && input_rows == Fp8N5120K5120::kInputRows) {
        return Fp8GeometryId::N5120K5120;
    }
    if (output_rows == Fp8N3072K5120::kOutputRows && input_rows == Fp8N3072K5120::kInputRows) {
        return Fp8GeometryId::N3072K5120;
    }
    if (output_rows == Fp8N512K5120::kOutputRows && input_rows == Fp8N512K5120::kInputRows) {
        return Fp8GeometryId::N512K5120;
    }
    if (output_rows == Fp8N8704K5120::kOutputRows && input_rows == Fp8N8704K5120::kInputRows) {
        return Fp8GeometryId::N8704K5120;
    }
    throw std::invalid_argument("unsupported FP8 problem N" + std::to_string(output_rows) + "K" +
                                std::to_string(input_rows));
}

} // namespace ninfer::ops::detail
