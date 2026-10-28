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
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.convert.quantization import exl3
from tools.convert.quantization import exl3_quantize as q

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "exl3" / "exllamav3-1.6.1.json"
INTEGER_RATES_K2 = (2, 4, 6, 8, 10, 12, 14, 16)
FRAC_RATES_K2 = (3, 5, 7)  # half-integer shifts, mul1 only


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
    Viterbi must find a zero-error ring for the same target on every admitted tile."""

    covered = set()
    for tile in fixture["tiles"]:
        k2, codebook = tile["k2"], tile["codebook"]
        targets = _tile_targets(tile)
        values, states = q.quantize_tiles(targets[None], k2, codebook)
        # Bit-exact against the target: the decoded FP16 patterns in sequence order equal
        # the fixture's inner weights read through the tile element order.
        state_bits = exl3.codebook_bits(states[0].cpu().numpy(), codebook)
        rows, cols = exl3.tile_element_order()
        expected_seq = _u16(tile["inner"]).reshape(16, 16)[rows, cols]
        assert (state_bits == expected_seq).all(), (k2, codebook)
        covered.add((k2, codebook))
    assert covered == ( {(k2, cb) for k2 in INTEGER_RATES_K2 for cb in exl3.CODEBOOKS}
                        | {(k2, "mul1") for k2 in FRAC_RATES_K2} )


def test_emitted_states_are_consistent_rings(fixture: dict) -> None:
    """The Viterbi's states satisfy the section 2.4 shift constraint and close the ring:
    packing them and re-extracting through the oracle returns them unchanged."""

    for tile in fixture["tiles"]:
        k2, codebook = tile["k2"], tile["codebook"]
        targets = _tile_targets(tile)
        _, states = q.quantize_tiles(targets[None], k2, codebook)
        packed = q.pack_trellis(states, k2)
        reextracted = exl3.tile_states(packed[0].cpu().numpy(), k2)
        assert (reextracted == states[0].cpu().numpy()).all(), (k2, codebook)


def test_quantize_tiles_zero_error_on_constructed_rings() -> None:
    """A random valid ring's decoded weights are an achievable target by construction;
    the Viterbi must reach zero error on every admitted rate and codebook."""

    rng = np.random.default_rng(20261015)
    cases = [(k2, cb) for k2 in INTEGER_RATES_K2 for cb in exl3.CODEBOOKS]
    cases += [(k2, "mul1") for k2 in FRAC_RATES_K2]
    for k2, codebook in cases:
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


# ---------------------------------------------------------------------------
# Block-LDLQ (step 3's error feedback)
# ---------------------------------------------------------------------------


def _coupled_metric(k: int, rho: float = 0.9) -> torch.Tensor:
    """A normalized SPD metric with strong adjacent-row coupling."""
    idx = torch.arange(k, device="cuda")
    cov = rho ** (idx.reshape(-1, 1) - idx.reshape(1, -1)).abs()
    cov = cov + torch.eye(k, device="cuda") * 0.01
    return cov / cov.diagonal().mean()


def test_block_ldl_identity_and_degenerate_metrics():
    torch.manual_seed(3)
    k = 256
    a = torch.randn(k, k, device="cuda") * 0.05
    h = a @ a.T + torch.eye(k, device="cuda") * 0.1
    L = q.block_ldl(h, 16)
    # H = L' * blockdiag(DL DL^T) * L'^T with the identity diagonal blocks restored.
    l_full = L.clone()
    dr = torch.arange(k, device="cuda")
    l_full[dr, dr] = 1.0
    l0 = torch.linalg.cholesky(h)
    dl = torch.diagonal(l0.reshape(k // 16, 16, k // 16, 16), dim1=0, dim2=2).permute(2, 0, 1).contiguous()
    dbig = torch.zeros(k, k, device="cuda")
    for i in range(k // 16):
        dbig[i*16:(i+1)*16, i*16:(i+1)*16] = dl[i] @ dl[i].T
    recon = l_full @ dbig @ l_full.T
    assert float((recon - h).norm() / h.norm()) < 1e-5
    # Identity and diagonal metrics carry no cross-row coupling: L must be zero.
    assert float(q.block_ldl(torch.eye(k, device="cuda"), 16).abs().max()) == 0.0
    d = torch.rand(k, device="cuda") + 0.5
    assert float(q.block_ldl(torch.diag(d), 16).abs().max()) == 0.0


def test_ldlq_with_zero_factor_equals_plain_quantization():
    torch.manual_seed(5)
    k, n = 256, 512
    weight = torch.randn(k, n, device="cuda") * 0.05
    plain_q, plain_states = q.quantize_weight(weight, 2, "mcg")
    ldlq_q, ldlq_states = q.ldlq(weight, torch.zeros(k, k, device="cuda"), 2, "mcg")
    assert bool((ldlq_q == plain_q).all()), "zero factor must reproduce the plain quantization"
    assert bool((ldlq_states == plain_states.to(ldlq_states.dtype)).all())

def test_ldlq_beats_plain_on_a_coupled_metric():
    torch.manual_seed(5)
    k, n = 256, 512
    weight = torch.randn(k, n, device="cuda") * 0.05
    h = _coupled_metric(k)
    plain_q, _ = q.quantize_weight(weight, 2, "mcg")
    ldlq_q, _ = q.ldlq(weight, q.block_ldl(h, 16), 2, "mcg")

    def hess_err(quantized):
        e = weight - quantized
        return float((e * (h @ e)).sum())

    e_plain, e_ldlq = hess_err(plain_q), hess_err(ldlq_q)
    assert e_ldlq < 0.9 * e_plain, f"ldlq {e_ldlq} did not beat plain {e_plain} by 10%"


def test_ldlq_matches_the_reference_kernel():
    """The reference ext's ldlq on the same weight and factor: the Hessian-metric errors must
    agree within the tie-breaking band (the codebook's tied rings propagate differently through
    the feedback), and the feedback must beat the plain quantization decisively at the codebook's
    operating scale."""
    try:
        sys.path.insert(0, "/home/barista/src/exllamav3")
        from exllamav3.modules.quant.exl3_lib.quantize import ldlq as ref_ldlq
    except Exception as exc:  # pragma: no cover - the reference ext is a local prerequisite
        pytest.skip(f"exllamav3 ext unavailable: {exc}")

    torch.manual_seed(5)
    k, n = 256, 512
    cb = q._codebook_values("mcg", torch.device("cuda"))
    rms = float(cb.square().mean().sqrt())
    weight = torch.randn(k, n, device="cuda") * rms
    h = _coupled_metric(k)
    factor = q.block_ldl(h, 16)
    quant_args = {"K": 1, "buf_size_k": 128, "devices": ["cuda:0"], "mcg": True,
                  "apply_out_scales": None}
    w_q_ref, _ = ref_ldlq(weight.clone(), factor.clone(), quant_args)
    w_q_mine, _ = q.ldlq(weight.clone(), factor.clone(), 2, "mcg")
    plain_q, _ = q.quantize_weight(weight, 2, "mcg")

    def hess_err(quantized):
        e = weight - quantized
        return float((e * (h @ e)).sum())

    e_ref, e_mine, e_plain = hess_err(w_q_ref), hess_err(w_q_mine), hess_err(plain_q)
    assert e_mine < 0.9 * e_plain, f"feedback improvement too small: {e_mine} vs plain {e_plain}"
    assert abs(e_mine / e_ref - 1.0) < 0.03, f"metric error diverged from the reference: {e_mine} vs {e_ref}"


# ---------------------------------------------------------------------------
# Global-scale search (step 2 completion)
# ---------------------------------------------------------------------------


def test_g_scale_search_beats_the_default_scale():
    """The searched scale must not lose to the default 1.0 on the full inner weight: the
    sample (wrapped diagonal plus RMS outliers) is representative, and the drift inflation
    lands the recursion's input at the codebook's operating point."""
    torch.manual_seed(9)
    k, n = 256, 512
    w = torch.randn(k, n, device="cuda") * 0.05
    su = torch.sign(torch.randn(k, device="cuda"))
    sv = torch.sign(torch.randn(n, device="cuda"))
    _, _, inner, _ = q.incoherence(w, su, sv)
    g, mse = q.g_scale_search(inner, 2, "mcg")
    assert 0.01 < g < 2.1, f"g_scale {g} outside the search range"

    def nmse_at(scale):
        values, _ = q.quantize_weight(inner * scale, 2, "mcg")
        e = values / scale - inner
        return float((e ** 2).mean() / (inner ** 2).mean())

    e_searched, e_default = nmse_at(g), nmse_at(1.0)
    assert e_searched < e_default, f"searched scale {g}: NMSE {e_searched} >= default {e_default}"
    # And the optimum the search claims is genuinely better than its coarse neighbors.
    for neighbor in (0.8, 1.3):
        assert e_searched < nmse_at(neighbor), f"searched scale loses to neighbor {neighbor}"


# ---------------------------------------------------------------------------
# Post-quantization scale refit (step 3 polish)
# ---------------------------------------------------------------------------


def test_refit_scales_polish_the_stored_scales():
    """The full pipeline -- incoherence, LDLQ with the per-tensor metric factor, refit --
    must end with the refit improving the original-basis Hessian error, and the returned
    scales must reproduce exactly the error the refit reports."""
    torch.manual_seed(17)
    k, n = 256, 512
    w = torch.randn(k, n, device="cuda") * 0.05
    a = torch.randn(k, k, device="cuda") * 0.05
    h = a @ a.T + torch.eye(k, device="cuda") * 0.1
    su = torch.sign(torch.randn(k, device="cuda"))
    sv = torch.sign(torch.randn(n, device="cuda"))
    suh, svh, inner, _ = q.incoherence(w, su, sv)
    factor = q.scaled_ldl(h, suh)
    inner_q, _ = q.ldlq(inner, factor, 2, "mcg")
    suh2, svh2, e_before, e_after = q.refit_scales(w, inner_q, suh, svh, h, rounds=2)
    assert e_after < e_before, f"refit worsened the error: {e_before} -> {e_after}"

    # The returned scales must reproduce the reported error exactly.
    def orig_err(reconstruction):
        e = w - reconstruction
        return float((e * (h @ e)).sum()) / float((w * (h @ w)).sum())

    assert abs(orig_err(q.decode_composition(suh2, inner_q, svh2)) - e_after) < 1e-6

    # And a zero-round refit must report the unrefit error for the given scales.
    _, _, e0, e0b = q.refit_scales(w, inner_q, suh, svh, h, rounds=0)
    assert abs(e0 - e0b) < 1e-9 and abs(e0 - e_before) < 1e-6


# ---------------------------------------------------------------------------
# Per-linear quantization route (step 4: the assembled tensor set)
# ---------------------------------------------------------------------------


def _inner_round_trip(trellis: torch.Tensor, k2: int, codebook: str, k: int, n: int) -> torch.Tensor:
    """The full artifact round trip: packed trellis -> oracle states -> codebook values ->
    ring (lane) positions -> the row-major inner weight [k, n]."""
    cb = q._codebook_values(codebook, trellis.device)
    rows, cols = exl3.tile_element_order()
    seq = torch.as_tensor(rows * 16 + cols, device=trellis.device)
    tiles_k, tiles_n = k // 16, n // 16
    inner = torch.zeros(tiles_k, tiles_n, 256, dtype=torch.int64, device=trellis.device)
    for tk in range(tiles_k):
        for tn in range(tiles_n):
            t = trellis[tk, tn].cpu().numpy().astype(np.uint16)
            inner[tk, tn] = torch.as_tensor(exl3.tile_states(t, k2), device=trellis.device)
    vals = cb[inner][..., torch.argsort(seq)]  # ring -> row-major: the inverse of seq
    return vals.reshape(tiles_k, tiles_n, 16, 16).permute(0, 2, 1, 3).reshape(k, n)


def test_quantize_linear_assembles_the_tensor_set():
    """Both paths produce the stored tensor set (fp16 scales, packed int16 trellis) whose
    artifact round trip reproduces the reported NMSE -- the fp16 scale storage alone may
    differ by roundoff."""
    torch.manual_seed(21)
    k, n = 256, 512
    w = torch.randn(k, n, device="cuda") * 0.05
    su = torch.sign(torch.randn(k, device="cuda"))
    sv = torch.sign(torch.randn(n, device="cuda"))
    a = torch.randn(k, k, device="cuda") * 0.05
    h_raw = a @ a.T + torch.eye(k, device="cuda") * 0.1

    for tag, out in (("fallback", q.quantize_linear(w, 2, "mcg", su, sv, hessian=None)),
                     ("hessian", q.quantize_linear(w, 2, "mcg", su, sv, hessian=h_raw))):
        assert out["suh"].dtype == torch.float16 and out["suh"].shape == (k,)
        assert out["svh"].dtype == torch.float16 and out["svh"].shape == (n,)
        assert out["trellis"].dtype == torch.int16
        assert tuple(out["trellis"].shape) == (k // 16, n // 16, 16)
        assert out["q_fallback"] == (tag == "fallback")
        assert out["nmse"] > 0
        inner_rt = _inner_round_trip(out["trellis"], 2, "mcg", k, n)
        recon = q.decode_composition(out["suh"].float(), inner_rt, out["svh"].float())
        nmse_rt = float(((recon - w) ** 2).sum() / (w ** 2).sum())
        assert abs(nmse_rt - out["nmse"]) < 1e-4, (tag, nmse_rt, out["nmse"])
        if tag == "hessian":
            assert out["hessian_err"] is not None and out["hessian_err"] > 0
        else:
            assert out["hessian_err"] is None


def test_quantize_linear_hessian_path_beats_fallback_in_its_metric():
    """The feedback and refit optimize the Hessian metric; measured in that metric (the
    damping share taken out, as the reported error does) the Hessian path must beat the
    fallback's plain quantization."""
    torch.manual_seed(21)
    k, n = 256, 512
    w = torch.randn(k, n, device="cuda") * 0.05
    su = torch.sign(torch.randn(k, device="cuda"))
    sv = torch.sign(torch.randn(n, device="cuda"))
    a = torch.randn(k, k, device="cuda") * 0.05
    h_raw = a @ a.T + torch.eye(k, device="cuda") * 0.1
    sigma = 0.025
    h_damped = h_raw.clone()
    h_damped.diagonal().add_(sigma * h_raw.diagonal().mean())
    h_err = h_damped.clone()
    h_err.diagonal().sub_((sigma / (1 + sigma)) * h_damped.diagonal().mean())

    def undamped_err(reconstruction):
        e = w - reconstruction
        return float((e * (h_err @ e)).sum() / (w * (h_err @ w)).sum())

    out_fb = q.quantize_linear(w, 2, "mcg", su, sv, hessian=None)
    out_h = q.quantize_linear(w, 2, "mcg", su, sv, hessian=h_raw)
    suh, svh, inner, _ = q.incoherence(w, su, sv)
    inner = inner * out_fb["g_scale"]
    suh = suh / out_fb["g_scale"]
    inner_q_fb, _ = q.quantize_weight(inner, 2, "mcg")
    e_fb = undamped_err(q.decode_composition(suh, inner_q_fb, svh))
    assert out_h["hessian_err"] < 0.9 * e_fb, \
        f"hessian path {out_h['hessian_err']} did not beat fallback {e_fb} by 10%"
