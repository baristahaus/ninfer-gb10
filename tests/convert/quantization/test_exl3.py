"""EXL3 codec oracles (docs/maintainer/exl3-trellis-contract.md 3.1, 3.3).

The fixture tests are the qualification: both Python implementations must reproduce the pinned
exllamav3's reconstruction bit for bit. They skip until `tests/fixtures/exl3/exllamav3-1.6.1.json`
has been generated. The agreement tests compare the vectorized oracle with the independent literal
reference; that parity is supplementary evidence, not the oracle's ground truth.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from tools.convert.quantization import exl3
from tools.convert.quantization import exl3_reference as reference

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "exl3" / "exllamav3-1.6.1.json"
RATES_K2 = (2, 4, 6, 8, 10, 12, 14, 16, 3, 5, 7)


def _u16(text: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(text), dtype="<u2").astype(np.uint16)


@pytest.fixture(scope="module")
def fixture() -> dict:
    if not FIXTURE.exists():
        pytest.skip(f"{FIXTURE} has not been generated (tests/fixtures/exl3/README.md)")
    value = json.loads(FIXTURE.read_text())
    assert value["format"] == "ninfer_exl3_fixture" and value["format_version"] == 1
    assert value["exllamav3"]["commit"] == "3a513ac9f91aae777f317ca2341b8e445b62c64e"
    return value


def test_fixture_tiles_decode_bit_exactly(fixture: dict) -> None:
    covered = set()
    for tile in fixture["tiles"]:
        k2, codebook = tile["k2"], tile["codebook"]
        elements = _u16(tile["trellis"])
        expected = _u16(tile["inner"]).reshape(16, 16)
        assert (exl3.decode_tiles(elements, k2, codebook) == expected).all(), (k2, codebook)
        literal = reference.decode_tile([int(v) for v in elements], k2, codebook)
        assert (np.array(literal, dtype=np.uint16) == expected).all(), (k2, codebook)
        covered.add((k2, codebook))
    required = {(k2, cb) for k2 in RATES_K2 for cb in exl3.CODEBOOKS if k2 % 2 == 0 or cb == "mul1"}
    assert covered == required


def test_fixture_coverage_pins_every_codebook_state(fixture: dict) -> None:
    for linear in fixture["coverage"]:
        k, n, k2 = linear["k"], linear["n"], linear["k2"]
        trellis = _u16(linear["trellis"]).reshape(k // 16, n // 16, 8 * k2)
        assert len(np.unique(exl3.tile_states(trellis, k2))) == 1 << 16
        expected = _u16(linear["inner"]).reshape(k, n)
        assert (exl3.decode_inner(trellis, linear["codebook"]) == expected).all()


def test_fixture_linear_inner_and_composition(fixture: dict) -> None:
    linear = fixture["linear"]
    k, n, k2 = linear["k"], linear["n"], linear["k2"]
    trellis = _u16(linear["trellis"]).reshape(k // 16, n // 16, 8 * k2)
    inner = exl3.decode_inner(trellis, linear["codebook"])
    assert (inner == _u16(linear["inner"]).reshape(k, n)).all()

    # exllamav3 composes in FP16; a wrong transform order or side is an O(1) error.
    weight = np.array(
        reference.linear_weight(inner.tolist(), _u16(linear["suh"]).tolist(),
                                _u16(linear["svh"]).tolist()))
    exllamav3 = _u16(linear["weight"]).view(np.float16).astype(np.float64).reshape(k, n)
    error = exllamav3 - weight
    relative_l2 = math.sqrt((error**2).sum() / (weight**2).sum())
    gross = np.abs(error).max() / np.abs(weight).max()
    assert relative_l2 <= 2e-3 and gross <= 4e-3, (relative_l2, gross)


def test_oracle_and_reference_agree_on_every_codebook_state() -> None:
    states = np.arange(1 << 16)
    for codebook in exl3.CODEBOOKS:
        oracle = exl3.codebook_bits(states, codebook)
        literal = np.array([reference.codebook_value(int(s), codebook) for s in states])
        assert (oracle == literal).all(), codebook


def test_oracle_and_reference_agree_on_random_tiles_at_every_rate() -> None:
    rng = np.random.default_rng(20261014)
    for k2 in RATES_K2:
        for codebook in exl3.CODEBOOKS if k2 % 2 == 0 else ("mul1",):
            elements = rng.integers(0, 1 << 16, size=8 * k2).astype(np.uint16)
            literal = reference.decode_tile([int(v) for v in elements], k2, codebook)
            assert (exl3.decode_tiles(elements, k2, codebook) == np.array(literal)).all()


def test_binary16_rounding_is_nearest_even_with_subnormals() -> None:
    patterns = np.arange(0x7C00, dtype=np.uint16)
    values = patterns.view(np.float16).astype(np.float64)
    midpoints = (values[:-1] + values[1:]) / 2
    expected = np.where(patterns[:-1] % 2 == 0, patterns[:-1], patterns[1:])
    for value, bits in zip(midpoints[::97], expected[::97]):
        assert reference.round_to_binary16(Fraction(float(value))) == bits
    assert reference.binary16_add(0x3C00, 0xBC00) == 0x0000  # exact cancellation is +0


def test_rates_and_codebooks_outside_the_contract_are_rejected() -> None:
    with pytest.raises(ValueError):
        exl3.validate_rate(9, "mul1")  # 4.5 bpw is not a v1.6.1 rate
    with pytest.raises(ValueError):
        exl3.validate_rate(5, "mcg")  # half-integer rates are MUL1 only
    with pytest.raises(ValueError):
        exl3.validate_rate(18, "mul1")


def test_packed_signs_expand_bit_b_of_element_w_to_entry_16w_plus_b() -> None:
    packed = np.array([0b1000_0000_0000_0001, 0], dtype=np.uint16).view(np.int16)
    expanded = exl3.expand_packed_signs(packed, 32)
    assert expanded[0] == 0xBC00 and expanded[15] == 0xBC00
    assert (expanded[1:15] == 0x3C00).all() and (expanded[16:] == 0x3C00).all()
