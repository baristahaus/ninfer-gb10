"""Reference EXL3 decoder: `docs/maintainer/exl3-trellis-contract.md` section 2 executed literally.

The deliberately slow, scalar statement of the format, written from the contract alone with the
standard library only. It shares nothing with `exl3.py` (the vectorized exact oracle) or with
production kernels:

- ring bits are read through the contract's int16 phrasing of the storage rule (section 2.4),
  where the oracle uses the uint32 phrasing;
- binary16 values are exact rationals, and every rounding to binary16 is the IEEE 754
  round-to-nearest-even rule implemented here on integers, with subnormals and signed zero;
- section 2.1's linear is evaluated in binary64 from the decoded `W_inner`, `suh`, `svh` and an
  explicitly constructed Sylvester Hadamard matrix.
"""

from __future__ import annotations

from fractions import Fraction

TILE = 16
TILE_WEIGHTS = 256
HADAMARD = 128

_CODEBOOKS = {"3inst", "mcg", "mul1"}


# binary16 ------------------------------------------------------------------------------------


def binary16_value(bits: int) -> Fraction:
    """The exact value of a finite binary16 bit pattern."""

    sign = -1 if bits & 0x8000 else 1
    exponent = (bits >> 10) & 0x1F
    fraction = bits & 0x3FF
    if exponent == 0x1F:
        raise ValueError(f"binary16 pattern {bits:#06x} is not finite")
    if exponent == 0:
        return sign * Fraction(fraction, 1 << 24)  # subnormal: fraction * 2^-24
    return sign * Fraction(1024 + fraction, 1 << 10) * Fraction(2) ** (exponent - 15)


def round_to_binary16(value: Fraction, negative_zero: bool = False) -> int:
    """IEEE 754 binary16 round-to-nearest-even of an exact rational. An exact zero is +0 unless
    `negative_zero` (the sign IEEE assigns to that zero result) is requested."""

    if value == 0:
        return 0x8000 if negative_zero else 0
    sign = 0x8000 if value < 0 else 0
    magnitude = -value if value < 0 else value
    # Smallest exponent e with magnitude < 2^(e+1), i.e. 2^e <= magnitude.
    e = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** e > magnitude:
        e -= 1
    quantum_exponent = max(e, -14) - 10  # spacing of representable values at this magnitude
    scaled = magnitude / Fraction(2) ** quantum_exponent
    integer = scaled.numerator // scaled.denominator
    remainder = scaled - integer
    if remainder > Fraction(1, 2) or (remainder == Fraction(1, 2) and integer % 2 == 1):
        integer += 1
    if quantum_exponent == -24:
        # Subnormals and the first normal binade share the spacing 2^-24, where the encoding
        # is the integer itself (a carry to 2048 is 2^-13, exponent field 2).
        return sign | integer
    if integer == 2048:  # rounding carried into the next binade
        integer = 1024
        quantum_exponent += 1
    exponent = quantum_exponent + 10 + 15
    if exponent >= 0x1F:
        return sign | 0x7C00  # overflow to infinity under round-to-nearest
    return sign | (exponent << 10) | (integer - 1024)


def binary16_add(a: int, b: int) -> int:
    total = binary16_value(a) + binary16_value(b)
    both_negative_zero = a == 0x8000 and b == 0x8000
    return round_to_binary16(total, negative_zero=both_negative_zero)


def binary16_fma(a: int, b: int, c: int) -> int:
    product = binary16_value(a) * binary16_value(b)
    total = product + binary16_value(c)
    product_negative = (a ^ b) & 0x8000 and product == 0
    negative_zero = total == 0 and bool(product_negative) and c == 0x8000
    return round_to_binary16(total, negative_zero=negative_zero)


# Section 2.4: the ring ------------------------------------------------------------------------


def step_width(position: int, k2: int) -> int:
    if k2 % 2 == 0:
        return k2 // 2
    return k2 // 2 + ((0xAAAA >> (position % 16)) & 1)


def ring_bit(elements: list[int], p: int) -> int:
    """Ring bit p: element 2m+1 holds ring bits 32m..32m+15 and element 2m holds 32m+16..32m+31,
    each most significant bit first."""

    m, offset = divmod(p, 32)
    element = elements[2 * m + 1] if offset < 16 else elements[2 * m]
    return (element >> (15 - offset % 16)) & 1


def tile_states(elements: list[int], k2: int) -> list[int]:
    if len(elements) != 8 * k2:
        raise ValueError("tile element count does not match the bitrate")
    elements = [value & 0xFFFF for value in elements]
    ring = 16 * len(elements)
    states, end = [], 0
    for position in range(TILE_WEIGHTS):
        end += step_width(position, k2)
        state = 0
        for p in range(end - 16, end):
            state = (state << 1) | ring_bit(elements, p % ring)
        states.append(state)
    if end != ring:
        raise AssertionError("step widths do not fill the ring")
    return states


# Section 2.5: codebooks -----------------------------------------------------------------------


def codebook_value(state: int, codebook: str) -> int:
    """binary16 bits of the codebook value of one 16-bit state."""

    if codebook in ("3inst", "mcg"):
        if codebook == "3inst":
            x = (state * 89226354 + 64248484) % 2**32
        else:
            x = (state * 0xCBAC1FED) % 2**32
        x = (x & 0x8FFF8FFF) ^ 0x3B603B60
        return binary16_add(x & 0xFFFF, x >> 16)
    if codebook == "mul1":
        x = (state * 0x83DCD12D) % 2**32
        b = sum((x >> shift) & 0xFF for shift in (0, 8, 16, 24))
        return binary16_fma(0x6400 + b, 0x1EEE, 0xC931)
    raise ValueError(f"unknown EXL3 codebook {codebook!r}")


# Section 2.3: tiles ---------------------------------------------------------------------------


def element_coordinates(position: int) -> tuple[int, int]:
    t, j = position >> 3, position & 7
    r = 2 * (t & 3) + (j & 1) + 8 * ((j >> 1) & 1)
    c = (t >> 2) + 8 * ((j >> 2) & 1)
    return r, c


def decode_tile(elements: list[int], k2: int, codebook: str) -> list[list[int]]:
    """binary16 bits of one tile, [r][c]."""

    if codebook not in _CODEBOOKS:
        raise ValueError(f"unknown EXL3 codebook {codebook!r}")
    if k2 % 2 and (k2 not in (3, 5, 7) or codebook != "mul1"):
        raise ValueError("half-integer bitrates are 1.5, 2.5, 3.5 with the mul1 codebook")
    if k2 % 2 == 0 and not 2 <= k2 <= 16:
        raise ValueError("integer bitrates are 1..8")
    tile = [[0] * TILE for _ in range(TILE)]
    for position, state in enumerate(tile_states(elements, k2)):
        r, c = element_coordinates(position)
        tile[r][c] = codebook_value(state, codebook)
    return tile


def decode_inner(trellis: list[list[list[int]]], codebook: str) -> list[list[int]]:
    """`W_inner` bits [k][n] from trellis[a][b] = the 16 K elements of tile (a, b)."""

    rows, cols = len(trellis), len(trellis[0])
    k2 = len(trellis[0][0]) // 8
    inner = [[0] * (cols * TILE) for _ in range(rows * TILE)]
    for a in range(rows):
        for b in range(cols):
            tile = decode_tile(trellis[a][b], k2, codebook)
            for r in range(TILE):
                for c in range(TILE):
                    inner[TILE * a + r][TILE * b + c] = tile[r][c]
    return inner


# Section 2.1: the linear ----------------------------------------------------------------------


def hadamard_entry(i: int, j: int) -> int:
    """Sylvester H128 in natural order: (-1)^popcount(i AND j)."""

    return -1 if bin(i & j).count("1") % 2 else 1


def _block_hadamard(vector: list[float]) -> list[float]:
    """vector . Hn: block-diagonal copies of H128 / sqrt(128) (Hn is symmetric)."""

    if len(vector) % HADAMARD:
        raise ValueError("EXL3 dimensions are multiples of 128")
    norm = HADAMARD**-0.5
    out = [0.0] * len(vector)
    for base in range(0, len(vector), HADAMARD):
        block = vector[base : base + HADAMARD]
        for j in range(HADAMARD):
            out[base + j] = norm * sum(block[i] * hadamard_entry(i, j) for i in range(HADAMARD))
    return out


def linear_weight(inner: list[list[int]], suh: list[int], svh: list[int]) -> list[list[float]]:
    """W = diag(suh) Hn_k W_inner Hn_n diag(svh) in binary64, from binary16 bit patterns."""

    k, n = len(inner), len(inner[0])
    values = [[float(binary16_value(bits)) for bits in row] for row in inner]
    # Columns of Hn_k W_inner, then rows of (.) Hn_n.
    left = [[0.0] * n for _ in range(k)]
    for column in range(n):
        mixed = _block_hadamard([values[row][column] for row in range(k)])
        for row in range(k):
            left[row][column] = mixed[row]
    weight = []
    for row in range(k):
        mixed = _block_hadamard(left[row])
        u = float(binary16_value(suh[row]))
        weight.append([u * mixed[col] * float(binary16_value(svh[col])) for col in range(n)])
    return weight


def linear_forward(x: list[float], weight: list[list[float]]) -> list[float]:
    """y = x . W in binary64."""

    n = len(weight[0])
    return [sum(x[i] * weight[i][j] for i in range(len(x))) for j in range(n)]
