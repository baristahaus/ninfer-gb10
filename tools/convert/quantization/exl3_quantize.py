"""The SP-7 EXL3 quantization core (``docs/maintainer/exl3-trellis-contract.md`` 6.2, steps 2-3).

Three pieces, each verified against ``exl3.py`` (the exact decode oracle):

- ``pack_trellis`` inverts ``exl3.tile_states``: the per-position 16-bit states of one
  tail-biting ring, satisfying the section 2.4 bitshift constraint, pack into the tile's
  int16 elements bit-exactly.
- ``quantize_tiles`` is the tail-biting trellis Viterbi in the reference algorithm's two-pass
  form: pass one runs the DP over the ring rolled by half a period with no initial state and
  estimates the wrap boundary; pass two reruns it from position zero with that boundary
  constrained at both ends, so the traced path closes the ring. Every position evaluates all
  65,536 window states, grouped as ``2**(16-K)`` out-edges by ``2**K`` branch candidates whose
  in-edge is the state's top ``16-K`` bits (section 2.4's shift relation
  ``s_i = ((s_{i-1} << K) | new) & 0xFFFF``). Integer rates 1..8 for every codebook;
  the half-integer MUL1 rates land as the follow-up piece.
- ``incoherence`` is step 2's transform sequence: output-channel scales folded into ``svh``,
  the output-side Hadamard, block-RMS input-channel scales and the codebook scale folded into
  ``suh``, the input-side Hadamard, and the global scale folded into ``suh``. The division
  precedes the Hadamard on each side; because the decode applies ``Hn`` once and the encode
  once, ``Hn * Hn = I`` makes the round trip through section 2.1's
  ``W = diag(suh) * Hn_k * W_inner * Hn_n * diag(svh)`` exact.

The cost arithmetic accumulates in float32. The reference kernel uses float16 costs; the
contract pins the format and the decode, not the encoder's tie-breaking, so a wider cost
accumulator is a quality improvement, not a compatibility break.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from tools.convert.quantization import exl3

TILE_WEIGHTS = 256
STATE_BITS = 16
HADAMARD = 128  # section 2.1's Hn blocks are H128 / sqrt(128)
CODEBOOK_SCALE = 1.24371088  # exllamav3's constant: the codebooks' effective value scale


def _validate_pack_rate(k2: int) -> None:
    """Packing handles every admitted rate, integer and half-integer alike."""

    exl3.validate_rate(k2, "3inst" if k2 % 2 == 0 else "mul1")


def _validate_integer_rate(k2: int) -> None:
    if k2 % 2 or not 2 <= k2 <= 16:
        raise NotImplementedError(f"integer rates only for now: k2={k2} (half-integer MUL1 follows)")


def _codebook_values(codebook: str, device: torch.device) -> torch.Tensor:
    """float32 codebook values of all 65,536 states, [65536].

    The codebooks map some states to NaN bit patterns (the LOP3 fold can pair opposite-signed
    infinities). No usable ring visits them, but a NaN squared error would propagate through
    ``torch.min`` and poison the accumulated costs, so those states carry +inf instead and the
    DP excludes them naturally.
    """

    bits = exl3.codebook_bits(np.arange(1 << STATE_BITS, dtype=np.uint16), codebook)
    values = bits.view(np.float16).astype(np.float32)
    values[np.isnan(values)] = np.inf
    return torch.as_tensor(values, device=device)


# ---------------------------------------------------------------------------
# Ring packing: states -> trellis elements (the exact inverse of exl3.tile_states)
# ---------------------------------------------------------------------------


def pack_trellis(states: torch.Tensor, k2: int) -> torch.Tensor:
    """Pack per-position states ``[..., 256]`` (uint16) into trellis int16 elements ``[..., 8*k2]``.

    Position i's new bits are its state's low ``D(i)`` bits, most significant first: new bit
    ``b`` (``b = 0`` the state's bit ``D(i)-1``) lands on ring bit ``S(i-1) + b``. The ring
    packs into uint32 words MSB-first (ring bit ``p`` is bit ``31 - p mod 32`` of word
    ``p // 32``), and each word splits into its little-endian int16 pair (element ``2w+1`` the
    low half, element ``2w`` the high half; section 2.4's storage rule).
    """

    _validate_pack_rate(k2)
    if not isinstance(states, torch.Tensor):
        states = torch.as_tensor(np.asarray(states))
    if states.dtype != torch.uint16 or states.shape[-1] != TILE_WEIGHTS:
        raise ValueError("pack_trellis expects uint16 [..., 256] states")
    device = states.device
    widths = torch.as_tensor(exl3.step_widths(k2), device=device)  # [256] int64
    ends = torch.cumsum(widths, dim=0)
    starts = ends - widths
    max_d = int(widths.max())

    ring_bits = 128 * k2
    words = ring_bits // 32

    # The new bits of every position: [..., 256, max_d], invalid slots zero.
    b = torch.arange(max_d, device=device)
    shift = (widths[:, None] - 1 - b[None, :]).to(device)  # [256, max_d] int64
    valid = b[None, :] < widths[:, None]
    states64 = states.to(torch.int64)
    bits = ((states64[..., None] >> shift) & 1).to(torch.uint8)
    bits = bits * valid.to(torch.uint8)

    # Scatter the new bits into the ring. Only valid (position, b) slots exist: with
    # half-integer rates an invalid slot's index would collide with the next position's
    # first new bit, so the invalid slots must not be written at all. The valid slots'
    # ring positions are the same for every tile in the batch.
    ring = torch.zeros(*states.shape[:-1], ring_bits, dtype=torch.uint8, device=device)
    index_2d = (starts[:, None] + b[None, :]).to(device)  # [256, max_d]
    valid_1d = valid.to(device).reshape(-1)  # [256 * max_d]
    valid_positions = index_2d.reshape(-1)[valid_1d]  # [n_valid]
    bits_flat = bits.reshape(*states.shape[:-1], TILE_WEIGHTS * max_d)
    bits_v = bits_flat[..., valid_1d]  # [..., n_valid]
    ring.scatter_(-1, valid_positions.expand(*bits_v.shape), bits_v)

    # Pack MSB-first into uint32 words, then split into the int16 element pairs.
    weight = torch.as_tensor((1 << np.arange(31, -1, -1)), dtype=torch.int64, device=device)
    word_values = (ring.to(torch.int64).reshape(*ring.shape[:-1], words, 32) * weight).sum(dim=-1)
    elements = torch.zeros(*word_values.shape[:-1], words * 2, dtype=torch.int16, device=device)
    # Element 2w+1 holds the word's first 16 ring bits (its high half), element 2w the last
    # 16 (its low half) -- section 2.4's storage rule.
    elements[..., 1::2] = ((word_values >> 16) & 0xFFFF).to(torch.int16)
    elements[..., 0::2] = (word_values & 0xFFFF).to(torch.int16)
    return elements


# ---------------------------------------------------------------------------
# Tail-biting trellis Viterbi (integer rates)
# ---------------------------------------------------------------------------


def _viterbi_pass(
    targets: torch.Tensor,  # [B, 256] float32, the sequence this pass covers (rolled or natural)
    codebook_values: torch.Tensor,  # [65536] float32
    k2: int,
    wrap_state: torch.Tensor | None,  # [B] int64: position 0's constrained in-edge, or None
) -> tuple[torch.Tensor, torch.Tensor]:
    """One Viterbi pass. Returns (final accumulated costs [B, edges], argmin_k [256, B, edges] uint8)."""

    device = targets.device
    B = targets.shape[0]
    K = k2 // 2
    edges = 1 << (STATE_BITS - K)
    states_all = torch.arange(1 << STATE_BITS, device=device, dtype=torch.int64)
    state_in_edge = states_all >> K  # [65536]: the in-edge of every state

    argmin_k = torch.zeros(TILE_WEIGHTS, B, edges, dtype=torch.uint8, device=device)
    acc = torch.zeros(B, edges, dtype=torch.float32, device=device)

    for i in range(TILE_WEIGHTS):
        err = (codebook_values[None, :] - targets[:, i : i + 1]) ** 2  # [B, 65536]
        cost = err + acc[:, state_in_edge]  # gather each candidate's in-edge accumulated cost
        if wrap_state is not None and i == 0:
            # The closure constraint: only branch candidates whose in-edge equals the estimated
            # wrap boundary are admissible at position zero. (masked_fill, not an inf
            # multiply: 0 * inf is NaN and would poison every unblocked candidate.)
            blocked = state_in_edge[None, :] != wrap_state[:, None]  # [B, 65536]
            cost = cost.masked_fill(blocked, float("inf"))
        grouped = cost.reshape(B, 1 << K, edges)  # state = k * edges + e
        best, argmin = grouped.min(dim=1)  # [B, edges]
        argmin_k[i] = argmin.to(torch.uint8)
        acc = best
    return acc, argmin_k


def _backward(
    argmin_k: torch.Tensor,  # [256, B, edges] uint8, this pass's backpointers
    start_edge: torch.Tensor,  # [B] int64: the final position's out-edge
    k2: int,
    order: torch.Tensor | None = None,  # [256] int64: the ring position of pass index i, or None (identity)
    stop_at_zero: bool = False,  # pass 1's early stop at ring position 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trace back. Returns (states [B, 256] int64 (positions in ring order), the edge at ring position 0 [B])."""

    device = start_edge.device
    B = start_edge.shape[0]
    K = k2 // 2
    edges = 1 << (STATE_BITS - K)
    states = torch.zeros(B, TILE_WEIGHTS, dtype=torch.int64, device=device)
    edge = start_edge.clone()
    batch = torch.arange(B, device=device)
    wrap_edge = torch.zeros(B, dtype=torch.int64, device=device)
    for i in range(TILE_WEIGHTS - 1, -1, -1):
        ri = i if order is None else int(order[i].item())
        k_star = argmin_k[i, batch, edge].to(torch.int64)  # [B]
        prev_edge = (k_star * edges + edge) >> K  # the candidate's in-edge
        states[:, ri] = k_star * edges + edge
        edge = prev_edge
        if stop_at_zero and ri == 0:
            break
    wrap_edge = edge
    return states, wrap_edge


def tile_lane_perm(device: torch.device) -> torch.Tensor:
    """Ring position -> tile element index within a 16x16 tile, [256].

    The artifact's ring binds the trellis positions to the tensor-core lane order (the
    oracle's ``tile_element_order``, the same map as the reference's ``tensor_core_perm``):
    ring position l carries the tile element (rows[l], cols[l]), not the row-major element l.
    Tiles entering the Viterbi must be gathered into this order, and decoded values gathered
    back out of it; the emitted rings are already in it and pack unchanged.
    """

    rows, cols = exl3.tile_element_order()
    return torch.as_tensor(rows * 16 + cols, dtype=torch.int64, device=device)


def quantize_tiles(targets: torch.Tensor, k2: int, codebook: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize target tiles ``[B, 256]`` (tensor-core order, float) at integer rate ``k2/2``.

    Returns ``(decoded_values [B, 256] float32, states [B, 256] uint16)``. The states satisfy
    section 2.4's shift constraint and close the ring; ``exl3.codebook_bits`` maps them back to
    the decoded values bit-exactly.
    """

    _validate_integer_rate(k2)
    exl3.validate_rate(k2, codebook)
    if targets.shape[-1] != TILE_WEIGHTS:
        raise ValueError("quantize_tiles expects 256-position tiles")
    device = targets.device
    roll = TILE_WEIGHTS // 2
    cb = _codebook_values(codebook, device)

    # Pass 1: the ring rolled by half a period, no initial constraint. The DP's own transition
    # across the wrap (ring position 255 -> 0) enforces the shift relation, so its optimal path
    # is a valid ring and its boundary at ring position 0 estimates the wrap state.
    order_pass1 = torch.tensor([(i + roll) % TILE_WEIGHTS for i in range(TILE_WEIGHTS)],
                               device=device)
    targets_pass1 = targets[:, order_pass1]
    acc1, argmin1 = _viterbi_pass(targets_pass1, cb, k2, None)
    _, wrap_state = _backward(argmin1, acc1.argmin(dim=1), k2, order=order_pass1, stop_at_zero=True)

    # Pass 2: the natural order, position 0's in-edge constrained to the estimated wrap state.
    acc2, argmin2 = _viterbi_pass(targets, cb, k2, wrap_state)
    # The ring closes at both ends on the same boundary: the final position's out-edge is the
    # wrap state, and the traced path is the optimum under that closure.
    states, _ = _backward(argmin2, wrap_state, k2)
    values = cb[states]
    return values, states.to(torch.uint16)


# ---------------------------------------------------------------------------
# Incoherence transforms (step 2)
# ---------------------------------------------------------------------------


def hadamard_128(device: torch.device) -> torch.Tensor:
    """H128, the Sylvester Hadamard matrix in natural order (section 2.1), float32 [128, 128]."""

    h = torch.ones(1, 1, dtype=torch.float32, device=device)
    for _ in range(7):
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h


def block_hadamard_right(x: torch.Tensor) -> torch.Tensor:
    """``x * Hn`` along the output dimension: every 128-column block right-multiplied by
    ``H128 / sqrt(128)``."""

    k, n = x.shape
    if n % HADAMARD:
        raise ValueError(f"output dimension {n} is not a multiple of {HADAMARD}")
    h = hadamard_128(x.device) / (HADAMARD ** 0.5)
    blocks = x.reshape(k, n // HADAMARD, HADAMARD)
    return (blocks @ h).reshape(k, n)


def block_hadamard_left(x: torch.Tensor) -> torch.Tensor:
    """``Hn * x`` along the input dimension: every 128-row block left-multiplied by
    ``H128 / sqrt(128)``."""

    k, n = x.shape
    if k % HADAMARD:
        raise ValueError(f"input dimension {k} is not a multiple of {HADAMARD}")
    h = hadamard_128(x.device) / (HADAMARD ** 0.5)
    blocks = x.reshape(k // HADAMARD, HADAMARD, n)
    return (h @ blocks).reshape(k, n)


def incoherence(
    weight: torch.Tensor,  # [k, n] float32, both dimensions multiples of 128
    su_signs: torch.Tensor,  # [k] float32, +-1
    sv_signs: torch.Tensor,  # [n] float32, +-1
    *,
    codebook_scale: float = CODEBOOK_SCALE,
    g_scale: float = 1.0,
    apply_out_scales: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Step 2's transform sequence, the reference method's order.

    Returns ``(suh [k], svh [n], weight_inner [k, n], apply_out_scales)``. The divisions
    precede the Hadamards on each side, and the global scale multiplies the inner weight
    while dividing ``suh`` -- so the decode composition reproduces the original weight
    exactly (verified by the composition test). The epsilon guards and the zero-channel
    handling follow the reference: the output channels whose RMS underflows are zeroed in
    ``svh``, and ``apply_out_scales`` reports whether the output scales were normalized.
    """

    if weight.ndim != 2 or weight.shape[0] % HADAMARD or weight.shape[1] % HADAMARD:
        raise ValueError("incoherence expects a [k, n] weight with k, n multiples of 128")
    weight = weight.clone()
    su = su_signs.reshape(-1).to(weight.dtype).to(weight.device)
    sv = sv_signs.reshape(-1).to(weight.dtype).to(weight.device)

    # Output-channel scales: the RMS over the input dimension, normalized by its mean. Dead
    # channels (RMS underflow) divide by 0.1, not by their stored zero -- 0/0 would put NaN
    # into the transformed weight and poison the trellis quantization of healthy tiles sharing
    # the dead column. Their stored svh is zeroed only after the division, which the decode's
    # diag(svh) turns into an exact zero of the dead output.
    apply_out_scales = True if apply_out_scales is None else bool(apply_out_scales)
    out_scales = weight.square().mean(dim=0).sqrt()
    mean = float(out_scales.mean())
    if apply_out_scales and mean > 1e-30:
        out_scales = out_scales / mean
    if apply_out_scales:
        zero = out_scales.abs() < 1e-30
        out_scales = torch.where(zero, torch.full_like(out_scales, 0.1), out_scales)
        svh = sv * out_scales + 1e-10
    else:
        svh = sv.clone()
        zero = out_scales.abs() < 1e-30
    weight = weight / svh
    svh = torch.where(zero, torch.zeros_like(svh), svh)

    # Output-side Hadamard.
    weight = block_hadamard_right(weight)

    # Input-channel scales: the RMS over the output dimension; the codebook scale folds in
    # with a negation (the reference's -1/codebook_scale convention).
    in_scales = weight.square().mean(dim=1).sqrt()
    in_scales = torch.where(in_scales.abs() < 1e-30,
                            torch.full_like(in_scales, 0.1), in_scales)
    suh = su * in_scales / (-codebook_scale) + 1e-10
    weight = weight / suh.reshape(-1, 1)

    # Input-side Hadamard, then the global scale.
    weight = block_hadamard_left(weight)
    weight = weight * g_scale
    suh = suh / g_scale
    return suh, svh, weight, apply_out_scales


def decode_composition(suh: torch.Tensor, weight_inner: torch.Tensor, svh: torch.Tensor) -> torch.Tensor:
    """Section 2.1's composition: ``W = diag(suh) * Hn_k * W_inner * Hn_n * diag(svh)``."""

    w = block_hadamard_left(weight_inner)  # Hn_k * W_inner
    w = w * suh.reshape(-1, 1)  # diag(suh) applied after the left Hadamard
    w = block_hadamard_right(w)  # * Hn_n
    return w * svh.reshape(1, -1)  # * diag(svh)


# ---------------------------------------------------------------------------
# Block-LDLQ (step 3's error feedback)
# ---------------------------------------------------------------------------


def block_ldl(hessian: torch.Tensor, block: int = 16) -> torch.Tensor:
    """The block-LDL factor of a damped SPD Hessian.

    From the Cholesky factor ``L0`` (``H = L0 @ L0.T``), each block-column is right-multiplied
    by the inverse of its diagonal block ``DL[i]`` (16x16 lower-triangular), giving the
    block-unit-lower factor ``L'`` with identity diagonal blocks and the block-LDL identity
    ``H = L' @ blockdiag(DL[i] @ DL[i].T) @ L'.T``. The scalar diagonal is then zeroed as the
    reference's ``ldlq()`` expects. The input must already carry the capture's damping; a
    Cholesky failure retries with incremental diagonal damping exactly as the reference does.
    """

    n = hessian.shape[0]
    if n % block:
        raise ValueError(f"Hessian dimension {n} is not a multiple of the block size {block}")
    m = n // block
    h = hessian.clone()
    retries = 0
    while True:
        try:
            l0 = torch.linalg.cholesky(h)
            break
        except Exception:
            retries += 1
            if retries > 10:
                raise
            h.diagonal().add_(2.0 * 0.025 * h.diagonal().mean())
    # The diagonal blocks of L0: [m, block, block].
    dl = torch.diagonal(l0.reshape(m, block, m, block), dim1=0, dim2=2).permute(2, 0, 1).contiguous()
    dl_inv = torch.linalg.inv(dl)
    # Scale each block-column: L'[rows, block i] = L0[rows, block i] @ DL[i]^-1.
    l = torch.einsum("rib,ibj->rij", l0.reshape(n, m, block), dl_inv).reshape(n, n)
    # The diagonal blocks are identities up to roundoff; enforce them exactly, then zero the
    # scalar diagonal. (In the block-grid layout [m, m, b, b] the pair-indexed [i, i] selects
    # the diagonal blocks; in the raw [m, b, m, b] layout it would hit dims 0 and 1 instead.)
    l_blocks = l.reshape(m, block, m, block).permute(0, 2, 1, 3)
    idx = torch.arange(m, device=l.device)
    l_blocks[idx, idx] = torch.eye(block, device=l.device, dtype=l.dtype)
    dr = torch.arange(n, device=l.device)
    l[dr, dr] = 0
    return l


def scaled_ldl(hessian: torch.Tensor, suh: torch.Tensor, block: int = 16) -> torch.Tensor:
    """The per-tensor metric factor for the LDLQ of a regularized weight.

    The regularized weight's rows were divided by ``suh`` and block-Hadamard transformed, so
    its error metric is ``P @ diag(suh) @ H @ diag(suh) @ P`` with ``P`` the blockwise
    128-Hadamard (its own inverse) and the full ``suh`` (signs and scales -- the global scale
    cancels). The input Hessian must already be damped (the capture's ``sigma_reg`` share of
    its mean diagonal).
    """

    s = suh.reshape(-1).to(hessian.dtype).to(hessian.device)
    h = hessian * s.reshape(1, -1)  # H @ diag(suh)
    h = h * s.reshape(-1, 1)  # diag(suh) @ H
    h = block_hadamard_right(h)  # @ P
    h = block_hadamard_left(h)  # P @
    return block_ldl(h, block)


def quantize_weight(weight: torch.Tensor, k2: int, codebook: str = "mcg") -> tuple[torch.Tensor, torch.Tensor]:
    """The plain (no-feedback) whole-matrix tile quantization.

    Returns ``(values [k, n], states [k // 16, n // 16, 256])`` with the values in the weight's
    row-major layout and the rings in the storage's ring (lane) order.
    """

    size_k, size_n = weight.shape
    if size_k % 16 or size_n % 128:
        raise ValueError("quantize_weight expects k a multiple of 16 and n a multiple of 128")
    tiles_n = size_n // 16
    lane = tile_lane_perm(weight.device)
    lane_inv = torch.argsort(lane)
    tiles = weight.reshape(size_k // 16, 16, tiles_n, 16).permute(0, 2, 1, 3).reshape(-1, 256)
    tiles = tiles[:, lane]  # row-major -> ring (lane) order
    values, states = quantize_tiles(tiles, k2, codebook)
    values = values[:, lane_inv].reshape(size_k // 16, tiles_n, 16, 16).permute(0, 2, 1, 3).reshape(size_k, size_n)
    return values, states.reshape(size_k // 16, tiles_n, 256)


def ldlq(
    weight: torch.Tensor,  # [k, n] float32, the regularized weight; k a multiple of 16, n of 128
    L: torch.Tensor,  # [k, k] the block-LDL factor of the metric (zero diagonal)
    k2: int,  # tile bitrate
    codebook: str = "mcg",
    buf_size_k: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The block-LDLQ recursion: the reference's ``ldlq()`` on the fork's Viterbi.

    The input rows are quantized bottom-up in 16-row steps. Quantizing a step feeds its
    residual back through the strictly-block-lower factor: every not-yet-quantized row ``c``
    accumulates ``sum_{s > c} L[s, c] * err[s]`` (the buffered spans update the whole cache at
    once, the within-span steps add their slice), and the adjusted rows are what the trellis
    Viterbi quantizes. Returns ``(weight_q [k, n], states [k // 16, n // 16, 256])``.
    """

    size_k, size_n = weight.shape
    if size_k % 16 or size_n % 128:
        raise ValueError("ldlq expects k a multiple of 16 and n a multiple of 128")
    buf = max(buf_size_k, 16)
    if size_k % buf:
        raise ValueError(f"k={size_k} is not a multiple of the buffer size {buf}")
    tiles_n = size_n // 16
    device = weight.device
    cb = _codebook_values(codebook, device)
    lane_perm = tile_lane_perm(device)
    lane_perm_inv = torch.argsort(lane_perm)
    prod_cache = torch.zeros_like(weight)
    weight_q = torch.zeros_like(weight)
    states = torch.zeros(size_k // 16, tiles_n, 256, dtype=torch.int64, device=device)

    for j in range(size_k, 0, -buf):
        i = j - buf
        b_weight = weight[i:j]
        b_prod_cache = prod_cache[i:j]
        b_L = L[i:j]
        for bj in range(buf, 0, -16):
            bi = bj - 16
            # The residual of everything already quantized below this step, inside the span.
            bb_err = b_weight[bj:] - weight_q[i + bj:j]
            bb_L = b_L[bj:, i + bi:i + bj]  # absolute columns: this block's own rows
            compensation = b_prod_cache[bi:bj] + (bb_L.T @ bb_err)
            rows = b_weight[bi:bj] + compensation
            tiles = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)
            tiles = tiles[:, lane_perm]  # row-major -> ring (lane) order
            values, ring = quantize_tiles(tiles, k2, codebook)
            values = values[:, lane_perm_inv]  # ring order -> row-major
            quant_w = values.reshape(tiles_n, 16, 16).permute(1, 0, 2).reshape(16, size_n)
            weight_q[i + bi:i + bj] = quant_w
            states[(i + bi) // 16:(i + bj) // 16] = ring
        # Feed the whole span's residual to every row above it.
        b_err = b_weight - weight_q[i:j]
        prod_cache += b_L.T @ b_err
    return weight_q, states
