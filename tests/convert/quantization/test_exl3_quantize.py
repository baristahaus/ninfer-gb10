"""The SP-7 EXL3 quantization core (``docs/maintainer/exl3-trellis-contract.md`` 6.2 steps 2-3).

The fixture tests are the qualification: ``pack_trellis`` must invert the oracle's state
extraction bit-exactly on the pinned exllamav3 tiles (every admitted rate and codebook,
half-integer MUL1 included), and ``quantize_tiles`` must re-encode each integer-rate fixture
tile's inner weights to bit-identical decoded values -- the fixture's own ring proves a
zero-error path exists, so a correct Viterbi finds one. The synthetic tests check the same
properties on constructed rings and quantify the DP against a greedy feasible baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.convert.quantization import exl3
from tools.convert.quantization import exl3_quantize as q

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "exl3" / "exllamav3-1.6.1.json"
INTEGER_RATES_K2 = (2, 4, 6, 8, 10, 12, 14, 16)


def _u16(text: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(text), dtype="<u2").astype(np.uint16)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def fixture() -> dict:
    if not FIXTURE.exists():
        pytest.skip(f"{FIXTURE} has not been generated (tests/fixtures/exl3/README.md)")
    value = json.loads(FIXTURE.read_text())
    assert value["format"] == "ninfer_exl3_fixture" and value["format_version"] == 1
    assert value["exllamav3"]["commit"] == "3a513ac9f91aae777f317ca2341b8e445b62c64e"
    return value


def _tile_targets(tile: dict) -> torch.Tensor:
    """The fixture tile's inner weights in tensor-core sequence order, float32 [256]."""

    inner = _u16(tile["inner"]).reshape(16, 16).view(np.float16).astype(np.float32)
    rows, cols = exl3.tile_element_order()
    return torch.as_tensor(inner[rows, cols], device=_device())


def test_pack_round_trips_every_fixture_tile(fixture: dict) -> None:
    """pack_trellis inverts the oracle's state extraction on every admitted rate."""

    covered = set()
    for tile in fixture["tiles"]:
        k2, codebook = tile["k2"], tile["codebook"]
        elements = _u16(tile["trellis"])
        states = torch.as_tensor(exl3.tile_states(elements, k2), device=_device())
        packed = q.pack_trellis(states, k2)
        assert (packed.cpu().numpy().view(np.uint16) == elements).all(), (k2, codebook)
        covered.add((k2, codebook))
    required = {(k2, cb) for k2 in (2, 4, 6, 8, 10, 12, 14, 16, 3, 5, 7)
                for cb in exl3.CODEBOOKS if k2 % 2 == 0 or cb == "mul1"}
    assert covered == required


def test_quantize_tiles_reach_zero_error_on_fixture_tiles(fixture: dict) -> None:
    """The fixture's own ring decodes exactly to its inner weights, so a correct tail-biting
    Viterbi must find a zero-error ring for the same target on every integer-rate tile."""

    covered = set()
    for tile in fixture["tiles"]:
        k2, codebook = tile["k2"], tile["codebook"]
        if k2 % 2:
            continue  # the half-integer DP lands as the follow-up piece
        targets = _tile_targets(tile)
        values, states = q.quantize_tiles(targets[None], k2, codebook)
        # Bit-exact against the target: the decoded FP16 patterns in sequence order equal
        # the fixture's inner weights read through the tile element order.
        state_bits = exl3.codebook_bits(states[0].cpu().numpy(), codebook)
        rows, cols = exl3.tile_element_order()
        expected_seq = _u16(tile["inner"]).reshape(16, 16)[rows, cols]
        assert (state_bits == expected_seq).all(), (k2, codebook)
        covered.add((k2, codebook))
    assert covered == {(k2, cb) for k2 in INTEGER_RATES_K2 for cb in exl3.CODEBOOKS}


def test_emitted_states_are_consistent_rings(fixture: dict) -> None:
    """The Viterbi's states satisfy the section 2.4 shift constraint and close the ring:
    packing them and re-extracting through the oracle returns them unchanged."""

    for tile in fixture["tiles"]:
        k2, codebook = tile["k2"], tile["codebook"]
        if k2 % 2:
            continue
        targets = _tile_targets(tile)
        _, states = q.quantize_tiles(targets[None], k2, codebook)
        packed = q.pack_trellis(states, k2)
        reextracted = exl3.tile_states(packed[0].cpu().numpy(), k2)
        assert (reextracted == states[0].cpu().numpy()).all(), (k2, codebook)


def test_quantize_tiles_zero_error_on_constructed_rings() -> None:
    """A random valid ring's decoded weights are an achievable target by construction;
    the Viterbi must reach zero error on every integer rate and codebook."""

    rng = np.random.default_rng(20261015)
    for k2 in INTEGER_RATES_K2:
        for codebook in exl3.CODEBOOKS:
            ring_bits = rng.integers(0, 2, size=(1, 128 * k2), dtype=np.uint8)
            words = np.packbits(ring_bits.reshape(-1, 8), axis=-1)  # not the storage order; fix below
            # Build the trellis elements from the ring bits directly (MSB-first per word).
            elements = np.zeros((1, 8 * k2), dtype=np.uint16)
            for w in range(4 * k2):
                word = 0
                for b in range(32):
                    word |= int(ring_bits[0, 32 * w + b]) << (31 - b)
                elements[0, 2 * w + 1] = word & 0xFFFF
                elements[0, 2 * w] = (word >> 16) & 0xFFFF
            states = exl3.tile_states(elements, k2)
            bits = exl3.codebook_bits(states, codebook)
            targets = torch.as_tensor(bits.view(np.float16).astype(np.float32),
                                      device=_device())
            values, emitted = q.quantize_tiles(targets, k2, codebook)
            emitted_bits = exl3.codebook_bits(emitted[0].cpu().numpy(), codebook)
            assert (emitted_bits == bits).all(), (k2, codebook)


def test_quantize_tiles_beat_a_greedy_feasible_path() -> None:
    """The DP's total squared error must not exceed a greedy feasible baseline's: at each
    position, greedily take the best state consistent with the previous position's state
    under the shift constraint, ignoring the ring closure."""

    rng = np.random.default_rng(7)
    for k2 in (4, 8):
        for codebook in ("3inst", "mul1"):
            K = k2 // 2
            cb_values = exl3.codebook_bits(
                np.arange(1 << 16, dtype=np.uint16), codebook).view(np.float16).astype(np.float32)
            targets = rng.normal(0.0, 0.05, size=(1, 256)).astype(np.float32)
            targets_t = torch.as_tensor(targets, device=_device())
            values, _ = q.quantize_tiles(targets_t, k2, codebook)
            dp_error = float(((values[0].cpu().numpy() - targets[0]) ** 2).sum())

            # Greedy feasible: walk the ring, each step taking the best state whose top 16-K
            # bits equal the previous state's low 16-K bits (the shift constraint).
            greedy_states = np.zeros(256, dtype=np.uint16)
            prev = np.uint16(0)
            for i in range(256):
                candidates = np.arange(1 << 16, dtype=np.uint32)[
                    (np.arange(1 << 16, dtype=np.uint32) >> K) == prev]
                errs = (cb_values[candidates] - targets[0, i]) ** 2
                best = int(candidates[int(errs.argmin())])
                greedy_states[i] = np.uint16(best)
                prev = np.uint16(best & ((1 << (16 - K)) - 1))
            greedy_error = float(((cb_values[greedy_states] - targets[0]) ** 2).sum())
            assert dp_error <= greedy_error, (k2, codebook, dp_error, greedy_error)


# ---------------------------------------------------------------------------
# Incoherence transforms (step 2)
# ---------------------------------------------------------------------------


def test_hadamard_128_matches_the_sylvester_formula():
    h = q.hadamard_128(torch.device("cuda")).cpu().numpy()
    assert h.shape == (128, 128)
    idx = np.stack([np.random.randint(0, 128, size=64), np.random.randint(0, 128, size=64)])
    for i, j in idx.T:
        expected = 1 - 2 * (bin(i & j).count("1") % 2)  # (-1)^popcount(i AND j)
        assert h[i, j] == expected, (i, j, h[i, j], expected)


def test_incoherence_round_trips_the_original_weight():
    torch.manual_seed(11)
    k, n = 256, 512
    w = torch.randn(k, n, device="cuda", dtype=torch.float32) * 0.05
    su = torch.sign(torch.randn(k, device="cuda"))
    sv = torch.sign(torch.randn(n, device="cuda"))
    for g_scale in (1.0, 0.37, 2.9):
        suh, svh, w_inner, applied = q.incoherence(w, su, sv, g_scale=g_scale)
        assert applied, "random weight must apply output scales"
        w_back = q.decode_composition(suh, w_inner, svh)
        rel = float((w_back - w).norm() / w.norm())
        assert rel < 1e-5, f"g_scale={g_scale} round-trip relative error {rel}"

    # The composition against explicit full block-diagonal Hadamard matrices.
    h128 = q.hadamard_128(torch.device("cuda")) / (128 ** 0.5)
    hk = torch.block_diag(*[h128] * (k // 128))
    hn = torch.block_diag(*[h128] * (n // 128))
    suh, svh, w_inner, _ = q.incoherence(w, su, sv, g_scale=0.8)
    w_contract = suh.reshape(-1, 1) * (hk @ w_inner @ hn) * svh.reshape(1, -1)
    assert float((w_contract - w).norm() / w.norm()) < 1e-5


def test_incoherence_zero_output_channels():
    torch.manual_seed(13)
    k, n = 128, 256
    w = torch.randn(k, n, device="cuda", dtype=torch.float32) * 0.05
    w[:, [3, 70, 200]] = 0.0  # three dead output channels
    su = torch.sign(torch.randn(k, device="cuda"))
    sv = torch.sign(torch.randn(n, device="cuda"))
    suh, svh, w_inner, applied = q.incoherence(w, su, sv)
    assert applied
    assert float(svh[[3, 70, 200]].abs().max()) == 0.0, "dead channels must carry svh == 0"
    w_back = q.decode_composition(suh, w_inner, svh)
    rel = float((w_back - w).norm() / w.norm())
    assert rel < 1e-5, f"zero-channel round-trip relative error {rel}"

    # The fully-zero weight: the reference keeps the flag True and carries the zeros in svh.
    suh, svh, w_inner, applied = q.incoherence(torch.zeros(k, n, device="cuda"), su, sv)
    assert applied
    assert float(svh.abs().max()) == 0.0
    assert float(q.decode_composition(suh, w_inner, svh).abs().max()) == 0.0
