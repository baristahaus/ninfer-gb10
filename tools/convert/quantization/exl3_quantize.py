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
  ``s_i = ((s_{i-1} << K) | new) & 0xFFFF``). Integer rates 1..8 for every codebook; the
  half-integer MUL1 rates (k2 = 3, 5, 7) take the same two-pass structure with per-ring-
  position step widths, the accumulated costs indexed by each position's next out-edge.
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


def _validate_quantize_rate(k2: int, codebook: str) -> None:
    """Every admitted rate reaches the Viterbi: integer for all codebooks, half-integer MUL1."""

    if not 2 <= k2 <= 16:
        raise ValueError(f"k2={k2} outside the admitted 2..16 range")
    exl3.validate_rate(k2, codebook)


def _codebook_values(codebook: str, device: torch.device) -> torch.Tensor:
    """float32 codebook values of all 65,536 states, [65536]."""

    bits = exl3.codebook_bits(np.arange(1 << STATE_BITS, dtype=np.uint16), codebook)
    return torch.as_tensor(bits.view(np.float16).astype(np.float32), device=device)


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


def _frac_viterbi_pass(
    targets: torch.Tensor,  # [B, 256] float32, the sequence this pass covers
    codebook_values: torch.Tensor,  # [65536] float32
    ring_widths: torch.Tensor,  # [256] int64: the RING position's step width at each pass index
    wrap_state: torch.Tensor | None,  # [B] int64: position 0's constrained in-edge, or None
) -> tuple[torch.Tensor, torch.Tensor]:
    """One DP pass over positions with per-ring-position step widths.

    The state at position i is the 16-bit window ending at position i's last bit (section 2.4),
    so position i's d_i new bits enter at the state's low end, its in-edge shared with position
    i-1 is ``s >> d_i``, and its out-edge shared with position i+1 is ``s & (2^(16-d_{i+1}) -
    1)`` -- the accumulated costs after each position are indexed by that next edge, whose
    width follows the NEXT position's step. The out-edge group's free part is the state's HIGH
    bits, so the regroup is a reshape [B, 2^d_{i+1}, 2^(16-d_{i+1})] minimized along the high
    axis.

    Returns (final accumulated costs [B, 2^(16-d_wrap)], backpointers [256, B, E_max] uint8
    holding each out-edge group's argmin high part, valid on the per-position prefix).
    """

    device = targets.device
    B = targets.shape[0]
    d0 = int(ring_widths[0].item())
    acc = torch.zeros(B, 1 << (STATE_BITS - d0), dtype=torch.float32, device=device)
    e_max = 1 << (STATE_BITS - int(ring_widths.min().item()))
    states_all = torch.arange(1 << STATE_BITS, device=device, dtype=torch.int64)
    bp = torch.zeros(TILE_WEIGHTS, B, e_max, dtype=torch.uint8, device=device)
    for i in range(TILE_WEIGHTS):
        d_i = int(ring_widths[i].item())
        d_next = int(ring_widths[(i + 1) % TILE_WEIGHTS].item())
        in_edge = states_all >> d_i  # [65536]
        err = (codebook_values[None, :] - targets[:, i : i + 1]) ** 2  # [B, 65536]
        cost = err + acc[:, in_edge]
        if wrap_state is not None and i == 0:
            # The closure constraint, as in the integer pass: masked_fill, not an inf multiply.
            blocked = in_edge[None, :] != wrap_state[:, None]
            cost = cost.masked_fill(blocked, float("inf"))
        e_next = 1 << (STATE_BITS - d_next)
        grouped = cost.reshape(B, 1 << d_next, e_next)  # state = h * e_next + edge
        best, argmin = grouped.min(dim=1)  # [B, e_next], argmin = the high part h
        bp[i, :, :e_next] = argmin.to(torch.uint8)
        acc = best
    return acc, bp


def _frac_backward(
    bp: torch.Tensor,  # [256, B, E_max] uint8, this pass's backpointers
    start_edge: torch.Tensor,  # [B] int64: the final position's out-edge
    ring_widths: torch.Tensor,  # [256] int64, in this pass's order
    order: torch.Tensor | None = None,  # [256] int64: the ring position of pass index i
    stop_at_zero: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trace back through the frac backpointers: edge e (shared with i+1) -> h = bp[i, e] ->
    state = h * e_next + e -> next edge = state >> d_i (shared with i-1)."""

    device = start_edge.device
    B = start_edge.shape[0]
    states = torch.zeros(B, TILE_WEIGHTS, dtype=torch.int64, device=device)
    edge = start_edge.clone()
    batch = torch.arange(B, device=device)
    for i in range(TILE_WEIGHTS - 1, -1, -1):
        ri = i if order is None else int(order[i].item())
        d_next = int(ring_widths[(i + 1) % TILE_WEIGHTS].item())
        d_i = int(ring_widths[i].item())
        h = bp[i, batch, edge].to(torch.int64)  # [B] the high part
        state = h * (1 << (STATE_BITS - d_next)) + edge
        states[:, ri] = state
        edge = state >> d_i  # the in-edge shared with position i-1
        if stop_at_zero and ri == 0:
            break
    return states, edge


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

    _validate_quantize_rate(k2, codebook)
    if targets.shape[-1] != TILE_WEIGHTS:
        raise ValueError("quantize_tiles expects 256-position tiles")
    device = targets.device
    roll = TILE_WEIGHTS // 2
    cb = _codebook_values(codebook, device)

    # The backpointers dominate the pass memory: TILE_WEIGHTS * 2**(16-K) bytes per tile,
    # 8.4 MB at the widest shift. Large tile batches are quantized in memory-bounded chunks
    # without changing the callers' contracts (every tile's Viterbi is independent).
    if k2 % 2 == 0:
        bp_bytes_per_tile = TILE_WEIGHTS * (1 << (STATE_BITS - k2 // 2))
    else:
        bp_bytes_per_tile = TILE_WEIGHTS * (1 << (STATE_BITS - max(k2 // 2, 1)))
    chunk = max(1, 4_000_000_000 // bp_bytes_per_tile)
    batch = targets.shape[0]
    if batch > chunk:
        values = torch.empty(batch, TILE_WEIGHTS, dtype=torch.float32, device=device)
        states = torch.empty(batch, TILE_WEIGHTS, dtype=torch.uint16, device=device)
        for start in range(0, batch, chunk):
            end = min(start + chunk, batch)
            v, st = quantize_tiles(targets[start:end], k2, codebook)
            values[start:end] = v
            states[start:end] = st
        return values, states

    # Pass 1: the ring rolled by half a period, no initial constraint. The DP's own transition
    # across the wrap (ring position 255 -> 0) enforces the shift relation, so its optimal path
    # is a valid ring and its boundary at ring position 0 estimates the wrap state.
    order_pass1 = torch.tensor([(i + roll) % TILE_WEIGHTS for i in range(TILE_WEIGHTS)],
                               device=device)
    if k2 % 2 == 0:
        targets_pass1 = targets[:, order_pass1]
        acc1, argmin1 = _viterbi_pass(targets_pass1, cb, k2, None)
        _, wrap_state = _backward(argmin1, acc1.argmin(dim=1), k2, order=order_pass1,
                                  stop_at_zero=True)
        # Pass 2: the natural order, position 0's in-edge constrained to the estimated wrap
        # state. The ring closes at both ends on the same boundary: the final position's
        # out-edge is the wrap state, and the traced path is the optimum under that closure.
        acc2, argmin2 = _viterbi_pass(targets, cb, k2, wrap_state)
        states, _ = _backward(argmin2, wrap_state, k2)
    else:
        # The half-integer rates: alternating d and d+1-bit steps (mul1 only). The same
        # two-pass structure with the per-ring-position widths; the rolled pass keeps the
        # ring's own width at each rolled position (the even roll preserves the period-2
        # pattern, but the code follows the ring positions, not the pattern).
        widths = torch.as_tensor(exl3.step_widths(k2), dtype=torch.int64, device=device)
        targets_pass1 = targets[:, order_pass1]
        acc1, bp1 = _frac_viterbi_pass(targets_pass1, cb, widths[order_pass1], None)
        _, wrap_state = _frac_backward(bp1, acc1.argmin(dim=1), widths[order_pass1],
                                       order=order_pass1, stop_at_zero=True)
        _, bp2 = _frac_viterbi_pass(targets, cb, widths, wrap_state)
        states, _ = _frac_backward(bp2, wrap_state, widths)
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


# ---------------------------------------------------------------------------
# Global-scale search (step 2 completion)
# ---------------------------------------------------------------------------

# LDLQ error feedback inflates the tiles the quantizer actually sees relative to the pre-feedback
# weight (the accumulated compensation grows along the recursion). Mean stream RMS over the
# recursion relative to the first block, measured on Qwen3-8B (q/up/down, layer 10) by the
# reference; the scale search runs on the pre-feedback weight, so its sample tiles are inflated
# by the expected drift and the scale it picks lands the recursion's mean input at the
# codebook's operating point.
LDLQ_DRIFT = {1.0: 1.08, 1.5: 1.035, 2.0: 1.018, 2.5: 1.009, 3.0: 1.004}


def _sample_scale_tiles(weight: torch.Tensor, width: int = 3) -> torch.Tensor:
    """Sample tiles for the global-scale search, in ring (lane) order.

    A wrapped diagonal guarantees every tile row and column is sampled at least once (outliers
    are typically whole input or output channels), plus the highest- and lowest-RMS tiles as
    explicit outlier insurance for anything the diagonal misses.
    """

    tiles_k = weight.shape[0] // 16
    tiles_n = weight.shape[1] // 16
    w4 = weight.reshape(tiles_k, 16, tiles_n, 16)
    diag_len = max(tiles_k, tiles_n)
    ii = torch.arange(diag_len, device=weight.device).repeat_interleave(width)
    ww = torch.arange(width, device=weight.device).repeat(diag_len)
    kk = ii % tiles_k
    nn = (ii + ww) % tiles_n
    num_x = max(8, (diag_len * width) // 16)
    tile_ms = w4.square().mean(dim=(1, 3)).flatten()
    num_x = min(num_x, (tile_ms.shape[0] + 1) // 2)
    hi = torch.topk(tile_ms, num_x).indices
    lo = torch.topk(tile_ms, num_x, largest=False).indices
    xk = torch.cat((hi, lo)) // tiles_n
    xn = torch.cat((hi, lo)) % tiles_n
    tiles = w4[torch.cat((kk, xk)), :, torch.cat((nn, xn)), :].reshape(-1, 256)
    return tiles[:, tile_lane_perm(weight.device)].contiguous()


def g_scale_search(
    weight: torch.Tensor,  # [k, n] float32, the regularized (pre-feedback) inner weight
    k2: int,
    codebook: str = "mcg",
    width: int = 3,
) -> tuple[float, float]:
    """The global-scale search: the reference's coarse-then-fine grid with parabolic refinement.

    The sampled error curve is flat near the optimum but not perfectly unimodal, so a sequential
    golden-section search can converge on a local wiggle well away from the true minimum;
    instead a coarse grid over the full range on a subsample, then a fine grid around the coarse
    minimum on the full sample, refined by parabolic interpolation. The sample tiles carry the
    expected LDLQ drift so the chosen scale lands the recursion's mean input at the codebook's
    operating point. The objective is the inflated-domain mean squared error of the sample tiles
    quantized at the candidate scale and de-scaled back.

    Returns ``(g_scale, mse)``.
    """

    _validate_quantize_rate(k2, codebook)
    if weight.ndim != 2 or weight.shape[0] % 16 or weight.shape[1] % 16:
        raise ValueError("g_scale_search expects a [k, n] weight with k, n multiples of 16")
    tiles = _sample_scale_tiles(weight, width) * LDLQ_DRIFT.get(k2 / 2, 1.0)

    def mse_at(scale: float, batch: torch.Tensor) -> float:
        values, _ = quantize_tiles(batch * scale, k2, codebook)
        return float(((values / scale - batch) ** 2).mean())

    coarse = [0.1 + 0.2 * i for i in range(10)]
    mse_coarse = [mse_at(s, tiles[::3]) for s in coarse]
    center = coarse[int(np.argmin(mse_coarse))]
    step = 0.075
    fine = [center + step * (i - 2) for i in range(5)]
    mse_fine = [mse_at(s, tiles) for s in fine]
    best = int(np.argmin(mse_fine))
    offset = 0.0
    if 0 < best < 4:
        y0, y1, y2 = mse_fine[best - 1], mse_fine[best], mse_fine[best + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom > 0:
            offset = max(-0.5, min(0.5, 0.5 * (y0 - y2) / denom))
    g_scale = max(fine[best] + offset * step, 0.01)
    return g_scale, mse_fine[best]


# ---------------------------------------------------------------------------
# Post-quantization scale refit (step 3 polish)
# ---------------------------------------------------------------------------


def refit_scales(
    weight: torch.Tensor,  # [k, n] float32, the ORIGINAL-basis weight
    weight_inner_q: torch.Tensor,  # [k, n] float32, the quantized inner weight (trellis fixed)
    suh: torch.Tensor,  # [k] float32, the stored input scales
    svh: torch.Tensor,  # [n] float32, the stored output scales
    hessian: torch.Tensor,  # [k, k] float32, the damped ORIGINAL-basis Hessian
    *,
    rounds: int = 2,
    chunk: int = 16384,
    damping_share: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Refit the stored per-channel scales to the quantized weight in the Hessian metric,
    holding the trellis fixed -- free at inference, since the decode applies the same scales.

    The current reconstruction is ``Q = diag(suh) * Hn_k * weight_inner_q * Hn_n * diag(svh)``
    (section 2.1). Per output column the optimal scale is the closed form
    ``c_n = (q_n^T H w_n) / (q_n^T H q_n)``; per input row the scales solve
    ``((Q Q^T) ⊙ H) r = rowsum(Q ⊙ (H W))`` with a small diagonal lift for the solve.
    Alternating the two fits, each round rescales Q and the stored scale vectors in lockstep.

    ``damping_share`` is the share of the Hessian's mean diagonal that the capture added;
    it is taken back out of the diagonal for the reported error only. The fits themselves use
    the Hessian as given (damped).

    Returns ``(suh, svh, error_before, error_after)`` with the errors normalized as
    ``trace(E H E^T) / trace(W H W^T)``.
    """

    if weight.shape != weight_inner_q.shape:
        raise ValueError("refit_scales expects the original and inner weights to share shape")
    k, n = weight.shape
    device = weight.device
    h = hessian.to(device=device, dtype=torch.float32)
    if damping_share > 0.0:
        h_err = h.clone()
        h_err.diagonal().sub_(damping_share * h.diagonal().mean().item())
    else:
        h_err = h
    su = suh.reshape(k, 1).to(device=device, dtype=torch.float32).clone()
    sv = svh.reshape(1, n).to(device=device, dtype=torch.float32).clone()

    # H @ W once (streamed over columns), reused by both fits; the normalizer trace(W H W^T).
    hw = torch.empty((k, n), dtype=torch.float32, device=device)
    den = 0.0
    den_err = 0.0
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        wc = weight[:, a:b].to(device=device, dtype=torch.float32)
        hw[:, a:b] = h @ wc
        den += float((wc * hw[:, a:b]).sum())
        if h_err is not h:
            den_err += float((wc * (h_err @ wc)).sum())
    if h_err is h:
        den_err = den

    def err(reconstruction: torch.Tensor) -> float:
        total = 0.0
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            e = weight[:, a:b].to(device=device, dtype=torch.float32) - reconstruction[:, a:b]
            total += float((e * (h_err @ e)).sum())
        return total / max(den_err, 1e-30)

    q = decode_composition(su.reshape(-1), weight_inner_q, sv.reshape(-1))
    e_before = err(q)
    for _ in range(rounds):
        # Output-side scales: the closed form per column.
        num = torch.zeros(n, device=device)
        dn = torch.zeros(n, device=device)
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            hq = h @ q[:, a:b]
            num[a:b] = (q[:, a:b] * hw[:, a:b]).sum(dim=0)
            dn[a:b] = (q[:, a:b] * hq).sum(dim=0)
        c = torch.where(dn > 1e-30, num / dn.clamp(min=1e-30), torch.ones_like(num))
        q = q * c.view(1, n)
        sv = sv * c.view(1, n)
        # Input-side scales: the coupled solve, with a diagonal lift for stability.
        a_mat = (q @ q.T) * h
        b_vec = (q * hw).sum(dim=1)
        a_mat.diagonal().add_(1e-6 * a_mat.diagonal().mean())
        r = torch.linalg.solve(a_mat, b_vec)
        r = torch.where(torch.isfinite(r) & (r > 0), r, torch.ones_like(r))
        q = q * r.view(k, 1)
        su = su * r.view(k, 1)
    e_after = err(q)
    return su.reshape(-1), sv.reshape(-1), e_before, e_after


# ---------------------------------------------------------------------------
# Per-linear quantization route (step 4: the assembled tensor set)
# ---------------------------------------------------------------------------


def quantize_linear(
    weight: torch.Tensor,  # [k, n] float32, the ORIGINAL basis
    k2: int,
    codebook: str,
    su_signs: torch.Tensor,  # [k] float32, +-1: the shared-per-input signs
    sv_signs: torch.Tensor,  # [n] float32, +-1: this tensor's output signs
    hessian: torch.Tensor | None = None,  # [k, k] float32: the RAW (undamped) capture, or None
    *,
    sigma_reg: float = 0.025,
    g_scale: float | None = None,  # a given scale skips the search
    apply_out_scales: bool | None = None,
    refit_rounds: int = 2,  # 0 disables the scale polish
) -> dict:
    """The per-linear EXL3 quantization route, composing the verified pieces.

    With a Hessian: damp it (``sigma_reg`` times its mean diagonal, the capture's
    regularization), run the incoherence transforms with the global-scale search, factor the
    per-tensor metric ``P diag(suh) H diag(suh) P`` (the regularized weight's rows were
    divided by ``suh`` and Hadamard transformed, so the shared sign-only factorization would
    weight every row alike while an error in a heavily scaled row returns at full size),
    quantize with the block-LDLQ error feedback, then refit the stored scales in the
    original-basis Hessian metric. Without a Hessian (the uncalibrated fallback): the same
    incoherence and scale search, then the plain quantization with no feedback and no refit.

    Returns the stored tensor set and metrics:
    ``{"suh" fp16 [k], "svh" fp16 [n], "trellis" int16 [k // 16, n // 16, 8 * k2],
    "codebook", "k2", "g_scale", "q_fallback", "nmse", "hessian_err"}`` with ``nmse`` the
    plain original-basis ``|W_q - W|^2 / |W|^2`` and ``hessian_err`` the refit's reported
    error (the damping share taken back out) on the Hessian path, ``None`` on the fallback.
    """

    _validate_quantize_rate(k2, codebook)
    if weight.ndim != 2 or weight.shape[0] % 16 or weight.shape[1] % 16:
        raise ValueError("quantize_linear expects a [k, n] weight with k, n multiples of 16")
    device = weight.device
    k, n = weight.shape
    su = su_signs.reshape(-1).to(device=device, dtype=torch.float32)
    sv = sv_signs.reshape(-1).to(device=device, dtype=torch.float32)

    # The capture's damping, applied to the raw Hessian before any use.
    h_damped = None
    if hessian is not None:
        if hessian.shape != (k, k):
            raise ValueError("quantize_linear expects the Hessian to match the input dimension")
        h_damped = hessian.to(device=device, dtype=torch.float32).clone()
        h_damped.diagonal().add_(sigma_reg * h_damped.diagonal().mean())

    # Step 2: the incoherence transforms, then the global scale search on the inner weight.
    suh, svh, inner, applied = incoherence(
        weight, su, sv, apply_out_scales=apply_out_scales)
    if g_scale is None:
        g_scale, _ = g_scale_search(inner, k2, codebook)
    inner = inner * g_scale
    suh = suh / g_scale

    # Step 3: the quantization. The Hessian path feeds back through the per-tensor metric
    # factor; the fallback quantizes plainly with no feedback.
    q_fallback = h_damped is None
    if q_fallback:
        inner_q, states = quantize_weight(inner, k2, codebook)
        hessian_err = None
    else:
        factor = scaled_ldl(h_damped, suh)
        inner_q, states = ldlq(inner, factor, k2, codebook)
        hessian_err = None  # set by the refit below

    # Step 3 polish: the stored scales refit, holding the trellis fixed.
    if not q_fallback and refit_rounds > 0:
        damping_share = sigma_reg / (1.0 + sigma_reg)
        suh, svh, _, hessian_err = refit_scales(
            weight, inner_q, suh, svh, h_damped,
            rounds=refit_rounds, damping_share=damping_share)

    # Step 4: the packed tensor set and the metrics.
    trellis = pack_trellis(states.reshape(-1, TILE_WEIGHTS).to(torch.uint16), k2)
    trellis = trellis.reshape(k // 16, n // 16, 8 * k2)
    reconstruction = decode_composition(suh, inner_q, svh)
    nmse = float(((reconstruction - weight) ** 2).sum() / (weight ** 2).sum())
    return {
        "suh": suh.to(torch.float16).contiguous(),
        "svh": svh.to(torch.float16).contiguous(),
        "trellis": trellis,
        "codebook": codebook,
        "k2": k2,
        "g_scale": float(g_scale),
        "q_fallback": q_fallback,
        "nmse": nmse,
        "hessian_err": hessian_err,
    }
