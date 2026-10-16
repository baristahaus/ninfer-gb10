"""The SP-7 EXL3 calibration capture tests (``tools/convert/quantization/exl3_calibrate.py``).

The calibration forward is a doc-faithful torch port of the Qwen3.5 Dense text stack; its
pieces are verified against the fork's own C++ Op contracts and the sequential mathematical
oracle:

- The chunked GDN recurrence (parallel state passing) against the doc's per-token oracle,
  unbatched and batched, at several chunk widths. The full forward was additionally verified
  end-to-end against the fork's own engine on the real 27B checkpoint: per-token logprobs
  matched within the quantization noise and BF16 perplexity 4.04 vs the quantized artifact's
  4.96 on the same corpus (the engine's ninfer-perplexity).
- The partial RoPE against ``include/ninfer/ops/rope.h``: split-half NeoX pairs
  ``(i, i + rotary_dim/2)`` rotating by ``pos * theta^(-2i/rotary_dim)``.
- The depthwise causal conv against ``include/ninfer/ops/causal_conv1d_silu.h``:
  ``y[c,t] = sum_j weight[c,j] * x[c, t - taps + 1 + j]``.
"""

from __future__ import annotations

import math

import torch

from tools.convert.quantization import exl3_calibrate as cal


def _gdn_case(device: torch.device, t: int = 200, kg: int = 4, vg: int = 12,
              dk: int = 128, seed: int = 42):
    torch.manual_seed(seed)
    q = torch.randn(t, kg, dk, device=device, dtype=torch.bfloat16)
    k = torch.randn(t, kg, dk, device=device, dtype=torch.bfloat16)
    v = torch.randn(t, vg, dk, device=device, dtype=torch.bfloat16)
    a = torch.randn(t, vg, device=device, dtype=torch.float32) * 0.5
    b = torch.randn(t, vg, device=device, dtype=torch.float32)
    a_log = torch.randn(vg, device=device, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(vg, device=device, dtype=torch.float32) * 0.1
    return (q, k, v), a, b, a_log, dt_bias


def test_gdn_chunked_matches_the_sequential_oracle():
    """The UT-transform chunked recurrence must reproduce the doc's per-token oracle at every
    chunk width -- including the whole sequence as one chunk, where the triangular solve alone
    carries the answer."""
    device = torch.device("cuda")
    qkv, a, b, a_log, dt_bias = _gdn_case(device)
    ref = cal.gdn_sequential(qkv, a, b, a_log, dt_bias, qkv[0].shape[1])
    scale = float(ref.abs().max())
    # The chunked pass materializes its output in BF16 (the C++ Op's storage boundary), so
    # the criterion carries the storage rounding; the FP32 recurrence itself is exact to ~2e-6.
    for chunk in (16, 64, 200):
        out, _ = cal.gdn_chunked(qkv, a, b, a_log, dt_bias, qkv[0].shape[1], chunk=chunk)
        rel = float((out.float() - ref).abs().max()) / scale
        assert rel < 5e-3, f"chunk={chunk} relative error {rel}"


def test_gdn_chunked_batched_matches_per_sequence():
    """The batched chunked form processes every sequence independently: a flipped batch member
    must match its own sequential oracle."""
    device = torch.device("cuda")
    qkv, a, b, a_log, dt_bias = _gdn_case(device)
    q, k, v = qkv
    qb = torch.stack([q, q.flip(0)])
    kb = torch.stack([k, k.flip(0)])
    vb = torch.stack([v, v.flip(0)])
    ab = torch.stack([a, a.flip(0)])
    bb = torch.stack([b, b.flip(0)])
    out_b, state_b = cal.gdn_chunked((qb, kb, vb), ab, bb, a_log, dt_bias,
                                     q.shape[1], chunk=64)
    ref_flip = cal.gdn_sequential((q.flip(0), k.flip(0), v.flip(0)),
                                  a.flip(0), b.flip(0), a_log, dt_bias, q.shape[1])
    rel = float((out_b[1].float() - ref_flip).abs().max()) / float(ref_flip.abs().max())
    assert rel < 5e-3, f"batched flipped relative error {rel}"
    # The final state is chunking-invariant.
    _, state_single = cal.gdn_chunked(qkv, a, b, a_log, dt_bias, q.shape[1], chunk=200)
    assert float((state_b[0].float() - state_single).abs().max()) < 1e-4


def test_partial_rope_matches_the_split_half_contract():
    """``include/ninfer/ops/rope.h``: pair i spans dims (i, i + rotary_dim/2) with
    phi = pos * theta^(-2i/rotary_dim); the unrotated dimensions stay bit-exact."""
    device = torch.device("cuda")
    torch.manual_seed(7)
    t, heads, head_dim, rotary = 32, 2, 256, 64
    theta = 1e7
    q = torch.randn(1, heads, t, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, heads, t, head_dim, device=device, dtype=torch.bfloat16)
    positions = torch.arange(t, device=device)
    qr, kr = cal.partial_rope(q, k, theta, rotary, positions)
    pairs = rotary // 2
    freqs = theta ** (-2 * torch.arange(pairs, device=device, dtype=torch.float64) / rotary)
    for i in range(pairs):
        angle = positions.double() * freqs[i]
        cos, sin = angle.cos(), angle.sin()
        expected_first = q[0, 0, :, i].double() * cos - q[0, 0, :, i + pairs].double() * sin
        expected_second = q[0, 0, :, i + pairs].double() * cos + q[0, 0, :, i].double() * sin
        assert float((qr[0, 0, :, i].double() - expected_first).abs().max()) < 2e-2
        assert float((qr[0, 0, :, i + pairs].double() - expected_second).abs().max()) < 2e-2
    assert bool((qr[..., rotary:] == q[..., rotary:]).all()), "unrotated dims must be exact"


def test_gdn_conv_matches_the_causal_contract():
    """``include/ninfer/ops/causal_conv1d_silu.h``: y[c,t] = sum_j weight[c,j] * x[c, t - 3 + j]
    with a zero left tail (the calibration sequences each start fresh)."""
    device = torch.device("cuda")
    torch.manual_seed(11)
    bsz, t, channels, taps = 2, 40, 96, 4
    x = torch.randn(bsz, t, channels, device=device, dtype=torch.bfloat16)
    weight = torch.randn(channels, 1, taps, device=device, dtype=torch.bfloat16)
    out = cal.gdn_conv(x, weight, taps)
    for pos in range(t):
        acc = torch.zeros(bsz, channels, device=device, dtype=torch.float32)
        for tap in range(taps):
            source = pos - taps + 1 + tap
            if source >= 0:
                acc += weight[:, 0, tap].float() * x[:, source, :].float()
        # BF16 inputs and accumulation: the criterion carries the storage rounding.
        assert float((out[:, pos, :].float() - acc).abs().max()) < 0.05 * float(acc.abs().max())


def test_layer_tensor_inventory_groups():
    """Every layer linear maps to its capture Hessian group, with the a/b controls excluded
    (BF16 pass-through per the C++ Op contract)."""
    config = {
        "layer_types": ["linear_attention", "linear_attention", "linear_attention",
                        "full_attention"],
        "heads": 24, "head_dim": 256,
        "gdn_key_heads": 16, "gdn_value_heads": 48, "gdn_head_dim": 128,
    }
    from tools.convert.quantization import exl3_convert as conv
    gdn = {name: (src, group, rows)
           for name, src, group, rows in conv.layer_tensor_inventory(config, 0)}
    assert set(gdn) == {"gdn_query", "gdn_key", "gdn_value", "z", "out", "gate", "up", "down"}
    assert gdn["gdn_query"][2] == (0, 2048) and gdn["gdn_key"][2] == (2048, 4096)
    assert gdn["gdn_value"][2] == (4096, 10240) and gdn["out"][1] == "mixer_out"
    assert gdn["gate"][1] == "mlp" and gdn["down"][1] == "down"
    attn = {name: (src, group, rows)
            for name, src, group, rows in conv.layer_tensor_inventory(config, 3)}
    assert set(attn) == {"query", "attention_gate", "key", "value", "output",
                          "gate", "up", "down"}
    assert attn["query"][1] == "mixer" and attn["output"][1] == "mixer_out"
    assert all(entry[2] is None for entry in attn.values())


def test_split_q_gate_interleaves_per_head():
    """The packed q_proj holds each head's Q rows then its gate rows; the split extracts
    them in the converter's row order."""
    device = torch.device("cuda")
    from tools.convert.quantization import exl3_convert as conv
    heads, d, hidden = 4, 8, 16
    packed = torch.arange(heads * 2 * d * hidden, device=device,
                          dtype=torch.float32).reshape(heads * 2 * d, hidden)
    config = {"heads": heads, "head_dim": d}
    q, gate = conv._split_q_gate(packed, config)
    assert q.shape == (heads * d, hidden) and gate.shape == (heads * d, hidden)
    for h in range(heads):
        assert torch.equal(q[h * d:(h + 1) * d], packed[h * 2 * d: h * 2 * d + d])
        assert torch.equal(gate[h * d:(h + 1) * d], packed[h * 2 * d + d: (h + 1) * 2 * d])
