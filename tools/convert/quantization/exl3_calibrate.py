"""The Hessian capture for the EXL3 conversion (SP-7 contract 6.2 step 1).

A doc-faithful torch forward of the Qwen3.5 Dense text stack
(``docs/maintainer/qwen3_5-model.md``), layer-streamed so the full checkpoint never needs to be
resident, with the per-input-group Hessian accumulation that feeds ``quantize_linear``: the mixer
input (the attention or GDN projections), the MLP input (gate/up), and the down-projection input.

The forward is a calibration instrument, not a product inference route: it exists to produce the
activation Hessians the EXL3 LDLQ needs. The GDN recurrence uses the doc-sanctioned chunked form
(parallel state passing) verified against the sequential oracle; full attention runs as one causal
prefill over each sequence.

Layer streaming keeps one layer's weights resident while every calibration sequence flows through
it, accumulating that layer's Hessians on the GPU and transferring them to the host once. The
per-layer Hessian footprint is 2 * H^2 + I^2 floats (about 1.4 GB at the 27B geometry), and the
per-batch activations are only [tokens, hidden], so nothing large is ever duplicated.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

CHUNK = 64  # GDN chunk width: the within-chunk parallel state passing


# ---------------------------------------------------------------------------
# Checkpoint access
# ---------------------------------------------------------------------------


class ShardReader:
    """Lazy per-tensor reads across a sharded safetensors checkpoint."""

    def __init__(self, checkpoint: Path):
        self.shards = sorted(Path(checkpoint).glob("model-*.safetensors"))
        if not self.shards:
            raise FileNotFoundError(f"no model shards under {checkpoint}")
        self.index: dict[str, Path] = {}
        for shard in self.shards:
            with safe_open(shard, framework="pt") as f:
                for key in f.keys():
                    self.index[key] = shard

    def get(self, key: str, device: torch.device) -> torch.Tensor:
        shard = self.index[key]
        with safe_open(shard, framework="pt") as f:
            tensor = f.get_tensor(key)
        return tensor.to(device)


def load_text_config(checkpoint: Path) -> dict:
    """The text stack's geometry from the checkpoint config."""

    config = json.loads((Path(checkpoint) / "config.json").read_text())
    text = config["text_config"]
    return {
        "hidden": text["hidden_size"],
        "layers": text["num_hidden_layers"],
        "layer_types": text["layer_types"],
        "heads": text["num_attention_heads"],
        "kv_heads": text["num_key_value_heads"],
        "head_dim": text["head_dim"],
        "intermediate": text.get("intermediate_size") or text["moe_intermediate_size"],
        "gdn_key_heads": text["linear_num_key_heads"],
        "gdn_value_heads": text["linear_num_value_heads"],
        "gdn_head_dim": text["linear_value_head_dim"],
        "gdn_conv_width": text["linear_conv_kernel_dim"],
        "eps": text["rms_norm_eps"],
        "rope_theta": text["rope_parameters"]["rope_theta"],
        "partial_rotary_factor": text["partial_rotary_factor"],
        "vocab": text["vocab_size"],
        "num_experts": text.get("num_experts"),
        "moe_intermediate": text.get("moe_intermediate_size"),
    }


def _layer_prefix(layer: int) -> str:
    return f"model.language_model.layers.{layer}."


# ---------------------------------------------------------------------------
# Text block mathematics (qwen3_5-model.md)
# ---------------------------------------------------------------------------


def offset_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """(1 + w) * x / sqrt(mean(x^2) + eps), feature dim last."""

    variance = x.float().square().mean(dim=-1, keepdim=True)
    return ((1.0 + weight).to(x.dtype)) * x / (variance + eps).sqrt().to(x.dtype)


def plain_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """w * x / sqrt(mean(x^2) + eps), feature dim last."""

    variance = x.float().square().mean(dim=-1, keepdim=True)
    return weight.to(x.dtype) * x / (variance + eps).sqrt().to(x.dtype)


def partial_rope(q: torch.Tensor, k: torch.Tensor, theta: float,
                 rotary_dim: int, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Partial split-half NeoX RoPE (``include/ninfer/ops/rope.h``): pair i spans dims
    ``(i, i + rotary_dim/2)``, rotating by ``pos * theta^(-i / pairs)``. Text-only positions
    make the three MRoPE axes equal, so the multimodal form reduces to this ordinary partial
    RoPE; the config's ``mrope_interleaved`` names the pair-axis assignment, not the
    dimension pairing."""

    pairs = rotary_dim // 2
    freqs = theta ** (-torch.arange(pairs, device=q.device, dtype=torch.float32) / pairs)
    angles = positions.float()[:, None] * freqs[None, :]  # [T, pairs]
    cos = angles.cos().to(q.dtype)
    sin = angles.sin().to(q.dtype)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        first = x[..., 0:pairs]
        second = x[..., pairs:rotary_dim]
        out = x.clone()
        out[..., 0:pairs] = first * cos - second * sin
        out[..., pairs:rotary_dim] = second * cos + first * sin
        return out

    return rotate(q), rotate(k)


def causal_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Causal attention, q [B, Hq, T, D], k/v [B, Hkv, T, D], each KV head serving a
    consecutive group of query heads."""

    b, hq, t, d = q.shape
    hkv = k.shape[1]
    group = hq // hkv
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    scores = (q @ k.transpose(-1, -2)) * scale
    mask = torch.full((t, t), float("-inf"), device=q.device).triu(1)
    scores = scores + mask
    return scores.softmax(dim=-1) @ v


def _head_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Offset RMSNorm over the head dimension; x [T, heads, D], weight [D]."""

    variance = x.float().square().mean(dim=-1, keepdim=True)
    return ((1.0 + weight).to(x.dtype)) * x / (variance + eps).sqrt().to(x.dtype)


# ---------------------------------------------------------------------------
# Gated DeltaNet: the sequential oracle and the chunked form
# ---------------------------------------------------------------------------


def gdn_sequential(qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                   a: torch.Tensor, b: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor,
                   key_heads: int) -> torch.Tensor:
    """The doc's per-token recurrence, one token at a time -- the mathematical oracle for the
    chunked form. q/k [T, Kg, Dk], v [T, Vg, Dv], a/b [T, Vg]. Returns o [T, Vg, Dv]."""

    q, k, v = qkv
    t, kg, dk = q.shape
    vg, dv = v.shape[1], v.shape[2]
    g = -torch.exp(a_log.float()) * torch.nn.functional.softplus(a.float() + dt_bias.float())
    alpha = torch.exp(g)
    beta = torch.sigmoid(b.float())
    k_n = k.float() / (k.float().square().sum(dim=-1, keepdim=True) + 1e-6).sqrt()
    q_n = q.float() / (q.float().square().sum(dim=-1, keepdim=True) + 1e-6).sqrt()

    state = torch.zeros(vg, dv, dk, device=q.device, dtype=torch.float32)
    out = torch.zeros(t, vg, dv, device=q.device, dtype=torch.float32)
    for i in range(t):
        # Value head j uses key head j // (value_heads / key_heads).
        k_i = k_n[i].repeat_interleave(vg // kg, dim=0)  # [Vg, Dk]
        q_i = q_n[i].repeat_interleave(vg // kg, dim=0)
        s_bar = alpha[i].reshape(vg, 1, 1) * state
        u = beta[i].reshape(vg, 1) * (v[i].float() - (s_bar @ k_i.unsqueeze(-1)).squeeze(-1))
        state = s_bar + u.unsqueeze(-1) * k_i.unsqueeze(1)
        out[i] = (state @ q_i.unsqueeze(-1)).squeeze(-1) / math.sqrt(dk)
    return out


def gdn_chunked(qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                a: torch.Tensor, b: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor,
                key_heads: int, chunk: int = CHUNK,
                state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The chunked delta-rule recurrence with parallel state passing.

    Per token: ``g = -exp(A_log) * softplus(a + dt_bias)``, ``alpha = exp(g)``,
    ``beta = sigmoid(b)``, k/q L2-normalized, ``Sbar = alpha S``,
    ``u = beta (v - Sbar k)``, ``S = Sbar + u outer k``, ``o = S q / sqrt(Dk)``.

    Within a chunk the innovations ``u`` obey a strictly-lower triangular system: with the
    cumulative decay ``A_i`` from the chunk entry and ``G[i,m] = beta_i (A_i/A_m) (k_m . k_i)``
    for ``m < i``,

        u_i = beta_i (v_i - A_i S_0 k_i) - sum_{m<i} G[i,m] u_m      (I + G) u = b

    which a batched unit-lower triangular solve resolves for every head at once; then

        o_i = A_i S_0 q_i + sum_{m<=i} (A_i/A_m)(k_m . q_i) u_m
        S_end = A_C S_0 + sum_m (A_C/A_m) u_m outer k_m

    All decay ratios are exp of nonpositive log differences (m <= i), so nothing overflows.
    Returns (o [T, Vg, Dv], the final state [Vg, Dv, Dk]).
    """

    q, k, v = qkv
    t, kg, dk = q.shape[-3], q.shape[-2], q.shape[-1]
    vg, dv = v.shape[-2], v.shape[-1]
    group = vg // kg
    batched = q.ndim == 4
    if not batched:
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        a, b = a.unsqueeze(0), b.unsqueeze(0)
    bsz = q.shape[0]
    g = -torch.exp(a_log.float()) * torch.nn.functional.softplus(a.float() + dt_bias.float())  # [B, T, Vg]
    beta = torch.sigmoid(b.float())  # [B, T, Vg]
    k_n = (k.float() / (k.float().square().sum(dim=-1, keepdim=True) + 1e-6).sqrt()
           ).repeat_interleave(group, dim=-2)  # [B, T, Vg, Dk]
    q_n = (q.float() / (q.float().square().sum(dim=-1, keepdim=True) + 1e-6).sqrt()
           ).repeat_interleave(group, dim=-2)  # [B, T, Vg, Dk]
    v_f = v.float()

    if state is None:
        state = torch.zeros(bsz, vg, dv, dk, device=q.device, dtype=torch.float32)
    log_a = g.cumsum(dim=1)  # [B, T, Vg] log of the cumulative decay A
    # The output materializes in BF16, matching the C++ Op's storage boundary; the state,
    # controls and chunk temporaries stay FP32 per the contract.
    out = torch.zeros(bsz, t, vg, dv, device=q.device, dtype=torch.bfloat16)

    for start_pos in range(0, t, chunk):
        end_pos = min(start_pos + chunk, t)
        length = end_pos - start_pos
        sl = slice(start_pos, end_pos)
        # The within-chunk cumulative decay: the global cumsum rebased at the chunk entry.
        # The forcing and the direct state term need A_i = (global A at i) / (global A at
        # the chunk entry); the ratios and the state update only use differences, which the
        # rebase leaves unchanged.
        log_entry = log_a[:, start_pos - 1] if start_pos > 0 else torch.zeros(bsz, vg, device=q.device)
        log_c = log_a[:, sl].permute(0, 2, 1) - log_entry[:, :, None]  # [B, Vg, C]
        a_c = torch.exp(log_c)  # [B, Vg, C]
        q_c, k_c, v_c = q_n[:, sl], k_n[:, sl], v_f[:, sl]  # [B, C, Vg, D]
        beta_c = beta[:, sl]  # [B, C, Vg]
        causal = torch.tril(torch.ones(length, length, device=q.device, dtype=torch.bool))
        strictly = causal.tril(-1)

        # Decay ratios exp(logA_i - logA_m) for m <= i: nonpositive. The non-causal
        # entries are masked in LOG space first -- exp of a positive log difference would
        # overflow long before the boolean masks could zero it.
        log_diff = (log_c[:, :, :, None] - log_c[:, :, None, :]).masked_fill(~causal, 0.0)
        ratios = torch.exp(log_diff)  # [B, Vg, C(i), C(m)]
        dots_kk = torch.einsum("b m v y, b i v y -> b v i m", k_c, k_c)
        dots_kq = torch.einsum("b m v y, b i v y -> b v i m", k_c, q_c)

        # The forcing b_i = beta_i (v_i - A_i S_0 k_i).
        s0_k = torch.einsum("b v x y, b c v y -> b c v x", state, k_c)  # [B, C, Vg, Dv]
        b_force = beta_c[:, :, :, None] * (v_c - a_c.permute(0, 2, 1)[:, :, :, None] * s0_k)
        # The kernel G[i, m] = beta_i (A_i/A_m)(k_m . k_i), m < i.
        kernel = beta_c.permute(0, 2, 1)[:, :, :, None] * ratios * dots_kk * strictly
        # (I + G) u = b, unit-lower triangular solve per head.
        eye = torch.eye(length, device=q.device, dtype=torch.float32)
        u = torch.linalg.solve_triangular(kernel + eye, b_force.permute(0, 2, 1, 3),
                                          upper=False, unitriangular=True).permute(0, 2, 1, 3)
        # The output: A_i S_0 q_i + sum_{m<=i} (A_i/A_m)(k_m . q_i) u_m.
        s0_q = torch.einsum("b v x y, b c v y -> b c v x", state, q_c)  # [B, C, Vg, Dv]
        attn = ratios * dots_kq * causal
        out[:, sl] = ((a_c.permute(0, 2, 1)[:, :, :, None] * s0_q
                       + torch.einsum("b v i m, b m v x -> b i v x", attn, u))
                      / math.sqrt(dk)).to(torch.bfloat16)
        # The chunk's final state: A_C S_0 + sum_m (A_C/A_m) u_m outer k_m.
        u_w = torch.exp(log_c[:, :, -1, None] - log_c).permute(0, 2, 1)[:, :, :, None] * u
        state = (a_c[:, :, -1].reshape(bsz, vg, 1, 1) * state
                 + torch.einsum("b m v x, b m v y -> b v x y", u_w, k_c))
    if not batched:
        return out[0], state[0]
    return out, state


# ---------------------------------------------------------------------------
# Layer forward
# ---------------------------------------------------------------------------


def load_layer(reader: ShardReader, layer: int, config: dict, device: torch.device) -> dict:
    """One layer's weights, BF16 (controls fp32), with the full-attention Q/gate rows split."""

    prefix = _layer_prefix(layer)
    w = {
        "input_norm": reader.get(prefix + "input_layernorm.weight", device),
        "post_norm": reader.get(prefix + "post_attention_layernorm.weight", device),
        "gate": reader.get(prefix + "mlp.gate_proj.weight", device),
        "up": reader.get(prefix + "mlp.up_proj.weight", device),
        "down": reader.get(prefix + "mlp.down_proj.weight", device),
    }
    if config["layer_types"][layer] == "full_attention":
        q_gate = reader.get(prefix + "self_attn.q_proj.weight", device)  # [heads*2D, H]
        d = config["head_dim"]
        heads = config["heads"]
        rows = torch.arange(heads, device=device)[:, None] * 2 * d + torch.arange(d, device=device)[None, :]
        w["q"] = q_gate[rows.reshape(-1)]  # [heads*D, H]
        gate_rows = rows + d
        w["attn_gate"] = q_gate[gate_rows.reshape(-1)]  # [heads*D, H]
        w["k"] = reader.get(prefix + "self_attn.k_proj.weight", device)
        w["v"] = reader.get(prefix + "self_attn.v_proj.weight", device)
        w["o"] = reader.get(prefix + "self_attn.o_proj.weight", device)
        w["q_norm"] = reader.get(prefix + "self_attn.q_norm.weight", device)
        w["k_norm"] = reader.get(prefix + "self_attn.k_norm.weight", device)
    else:
        w["qkv"] = reader.get(prefix + "linear_attn.in_proj_qkv.weight", device)
        w["z"] = reader.get(prefix + "linear_attn.in_proj_z.weight", device)
        w["a"] = reader.get(prefix + "linear_attn.in_proj_a.weight", device)
        w["b"] = reader.get(prefix + "linear_attn.in_proj_b.weight", device)
        w["a_log"] = reader.get(prefix + "linear_attn.A_log", device).float()
        w["dt_bias"] = reader.get(prefix + "linear_attn.dt_bias", device).float()
        w["conv"] = reader.get(prefix + "linear_attn.conv1d.weight", device)  # [C, 1, taps]
        w["gdn_norm"] = reader.get(prefix + "linear_attn.norm.weight", device)
        w["out"] = reader.get(prefix + "linear_attn.out_proj.weight", device)
    return w


def full_attention_layer(h: torch.Tensor, w: dict, config: dict) -> torch.Tensor:
    """The doc's gated full attention on a batch: q/gate split, q/k norms, partial interleaved
    RoPE, causal GQA, ``y = output(a * sigmoid(gate))``. h [B, T, H]."""

    bsz, t, hidden = h.shape
    heads, kv_heads, d = config["heads"], config["kv_heads"], config["head_dim"]
    scale = 1.0 / math.sqrt(d)
    q = (h @ w["q"].T).reshape(bsz, t, heads, d).transpose(1, 2)  # [B, H, T, D]
    gate = (h @ w["attn_gate"].T).reshape(bsz, t, heads, d).transpose(1, 2)
    k = (h @ w["k"].T).reshape(bsz, t, kv_heads, d).transpose(1, 2)
    v = (h @ w["v"].T).reshape(bsz, t, kv_heads, d).transpose(1, 2)
    q = _head_rmsnorm(q, w["q_norm"], config["eps"])
    k = _head_rmsnorm(k, w["k_norm"], config["eps"])
    positions = torch.arange(t, device=h.device)
    rotary = int(d * config["partial_rotary_factor"])
    q, k = partial_rope(q, k, config["rope_theta"], rotary, positions)
    k = k.repeat_interleave(heads // kv_heads, dim=1)
    v = v.repeat_interleave(heads // kv_heads, dim=1)
    a = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)
    a = a * torch.sigmoid(gate)  # [B, H, T, D]
    a = a.transpose(1, 2).reshape(bsz, t, heads * d)
    return a @ w["o"].T, a  # (the block output, the o_proj input)


def gdn_conv(qkv: torch.Tensor, conv_weight: torch.Tensor, taps: int) -> torch.Tensor:
    """The depthwise causal conv over the token axis with a zero left tail, batched.
    qkv [B, T, C], conv_weight [C, 1, taps]: y[b, t, c] = sum_w weight[c, 0, w] * x[b, t - taps + 1 + w, c].
    """

    bsz, t, channels = qkv.shape
    padded = torch.zeros(bsz, t + taps - 1, channels, device=qkv.device, dtype=qkv.dtype)
    padded[:, taps - 1:] = qkv
    out = torch.zeros_like(qkv)
    for tap in range(taps):
        out = out + conv_weight[:, 0, tap].reshape(1, 1, channels) * padded[:, tap:tap + t]
    return out


def gdn_layer(h: torch.Tensor, w: dict, config: dict) -> torch.Tensor:
    """The doc's Gated DeltaNet on a batch: fused QKV/z/a/b projections, the causal conv and
    SiLU on Q/K/V, the chunked delta-rule recurrence, the plain head-norm and SiLU(z) gate,
    the output projection. h [B, T, H]."""

    bsz, t, hidden = h.shape
    kg = config["gdn_key_heads"]
    vg = config["gdn_value_heads"]
    dk = config["gdn_head_dim"]
    taps = config["gdn_conv_width"]
    qkv = h @ w["qkv"].T  # [B, T, 2*kg*dk + vg*dk]
    # The conv covers the whole fused QKV block: channels are laid out [Q, K, V].
    qkv_conv = gdn_conv(qkv, w["conv"], taps)
    qkv_conv = torch.nn.functional.silu(qkv_conv)
    q = qkv_conv[..., : kg * dk].reshape(bsz, t, kg, dk)
    k = qkv_conv[..., kg * dk: 2 * kg * dk].reshape(bsz, t, kg, dk)
    v = qkv_conv[..., 2 * kg * dk:].reshape(bsz, t, vg, dk)
    a = h @ w["a"].T  # [B, T, Vg]
    b = h @ w["b"].T
    o, _ = gdn_chunked((q, k, v), a, b, w["a_log"], w["dt_bias"], kg, chunk=CHUNK)
    o = plain_rmsnorm(o, w["gdn_norm"], config["eps"])
    z = torch.nn.functional.silu(h @ w["z"].T).reshape(bsz, t, vg, dk)
    mixed = (o * z).reshape(bsz, t, vg * dk).to(h.dtype)
    return mixed @ w["out"].T, mixed  # (the block output, the out_proj input)


def mlp_layer(h2: torch.Tensor, w: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """down(SiLU(gate(h2)) * up(h2)). Returns (the block output, the down-projection input
    [T, intermediate] -- the third Hessian group)."""

    gate = torch.nn.functional.silu(h2 @ w["gate"].T)
    up = h2 @ w["up"].T
    hidden = gate * up
    return hidden @ w["down"].T, hidden


def layer_forward(x: torch.Tensor, w: dict, layer: int, config: dict,
                  capture: list[torch.Tensor] | None = None) -> torch.Tensor:
    """One Text block on a batch. Returns ``(x, mixer_output)``; when ``capture`` is a list it
    receives the four Hessian groups in order: the mixer input (the attention/GDN projection
    input), the mixer output (the o_proj/out_proj input), the MLP input (gate/up), and the
    down-projection input."""

    bsz = x.shape[0]
    h = offset_rmsnorm(x, w["input_norm"], config["eps"])
    if capture is not None:
        capture.append(h)  # the mixer-input group: the attention/GDN projection input
    if config["layer_types"][layer] == "full_attention":
        m, m_pre = full_attention_layer(h, w, config)
    else:
        m, m_pre = gdn_layer(h, w, config)
    if capture is not None:
        capture.append(m_pre)  # the mixer-output group: the o_proj / out_proj input
    x = x + m
    h2 = offset_rmsnorm(x, w["post_norm"], config["eps"])
    if capture is not None:
        capture.append(h2)
    y, mlp_hidden = mlp_layer(h2, w)
    if capture is not None:
        capture.append(mlp_hidden)  # the down-projection input, [T, intermediate]
    return x + y, m


# ---------------------------------------------------------------------------
# Corpus, tokenization, and the calibration driver
# ---------------------------------------------------------------------------


def tokenize_corpus(checkpoint: Path, corpus_path: Path, total_tokens: int,
                     seq_len: int) -> torch.Tensor:
    """Tokenize a UTF-8 text corpus into fixed-length sequences [n, seq_len]."""

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(Path(checkpoint) / "tokenizer.json"))
    text = Path(corpus_path).read_text(encoding="utf-8", errors="ignore")
    ids = tokenizer.encode(text).ids
    n_sequences = total_tokens // seq_len
    if len(ids) < n_sequences * seq_len + 1:
        raise ValueError(
            f"corpus yields {len(ids)} tokens, need {n_sequences * seq_len + 1}")
    flat = torch.tensor(ids[: n_sequences * seq_len + 1], dtype=torch.long)
    # Input tokens [n, seq_len], teacher-forced targets shifted by one.
    inputs = flat[: n_sequences * seq_len].reshape(n_sequences, seq_len)
    targets = flat[1: n_sequences * seq_len + 1].reshape(n_sequences, seq_len)
    return inputs, targets


def calibrate(checkpoint: Path, corpus_path: Path, total_tokens: int, seq_len: int,
              device: torch.device, out_dir: Path) -> dict:
    """The full Hessian capture: the layer-streamed forward over the corpus with the
    per-input-group accumulation.

    ``out_dir`` receives one ``layer_NN.pt`` per layer holding ``mixer [H, H]``, ``mlp [H, H]``,
    ``down [I, I]``, the token count and the layer type -- the raw (undamped) accumulations
    ``sum_t x_t x_t^T`` the quantize route damps with its sigma_reg -- plus ``residual.pt`` with
    the residual stream after the latest completed layer. Completed layer files are resumed as-is:
    a restart reloads the residual and continues from the next layer, so an interrupted run loses
    at most the in-flight layer.

    Per layer the GDN dominates the transient memory (the normalized q/k, values and output are
    FP32/RF16 [B, T, Vg, Dv] tensors plus the BF16 QKV/conv/gate projections), so the sub-batch
    bound follows the value-head geometry. The Hessians accumulate across sub-batches on the GPU
    and move to the host once per layer.
    """

    config = load_text_config(Path(checkpoint))
    reader = ShardReader(Path(checkpoint))
    inputs, targets = tokenize_corpus(Path(checkpoint), corpus_path, total_tokens, seq_len)
    n_sequences = inputs.shape[0]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gdn_bytes = 16 * seq_len * config["gdn_value_heads"] * config["gdn_head_dim"]
    sub_batch = min(n_sequences, max(1, 4_000_000_000 // gdn_bytes))
    print(f"calibration: {n_sequences} sequences x {seq_len} tokens, "
          f"{config['layers']} layers, sub-batch {sub_batch}")

    # Resume: count the completed layer files and reload the residual stream.
    done = 0
    for layer in range(config["layers"]):
        if not (out_dir / f"layer_{layer:02d}.pt").exists():
            break
        done = layer + 1
    if done:
        x = torch.load(out_dir / "residual.pt", map_location=device)
        print(f"resuming after layer {done - 1}")
    else:
        embed = reader.get("model.language_model.embed_tokens.weight", device)
        x = embed[inputs.to(device)]  # [n, T, H] BF16

    for layer in range(done, config["layers"]):
        w = load_layer(reader, layer, config, device)
        h_mixer = torch.zeros(config["hidden"], config["hidden"], device=device)
        h_mixer_out = torch.zeros(config["gdn_value_heads"] * config["gdn_head_dim"],
                                  config["gdn_value_heads"] * config["gdn_head_dim"],
                                  device=device)
        h_mlp = torch.zeros(config["hidden"], config["hidden"], device=device)
        h_down = torch.zeros(config["intermediate"], config["intermediate"], device=device)
        x_next = torch.empty_like(x)
        for start in range(0, n_sequences, sub_batch):
            end = min(start + sub_batch, n_sequences)
            captured = []
            x_next[start:end], _ = layer_forward(x[start:end], w, layer, config,
                                                  capture=captured)
            for acc, act in zip((h_mixer, h_mixer_out, h_mlp, h_down), captured):
                flat = act.float().reshape(-1, act.shape[-1])
                acc += flat.T @ flat
            del captured
        x = x_next
        diag = float(h_mixer.diagonal().mean())
        torch.save({
            "mixer": h_mixer.cpu(),
            "mixer_out": h_mixer_out.cpu(),
            "mlp": h_mlp.cpu(),
            "down": h_down.cpu(),
            "tokens": torch.tensor(n_sequences * seq_len),
            "layer_type": config["layer_types"][layer],
        }, out_dir / f"layer_{layer:02d}.pt")
        torch.save(x.cpu(), out_dir / "residual.pt")
        del w, h_mixer, h_mlp, h_down
        print(f"  layer {layer:2d} ({config['layer_types'][layer]}): "
              f"mixer diag mean {diag:.3f}", flush=True)
        torch.cuda.empty_cache()
    return config
    return config
