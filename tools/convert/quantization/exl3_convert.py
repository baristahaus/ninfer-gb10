"""The model-level EXL3 conversion driver (SP-7 contract 6.2 step 4 completion).

Orchestrates ``quantize_linear`` over a Qwen3.5 Dense checkpoint's linears with the captured
per-layer Hessians (``exl3_calibrate``), producing the complete EXL3 tensor set per layer:
every mixer projection, the MLP projections, the packed trellises and the stored fp16 scales.
Norms, embeddings and the output matrix stay BF16 (the reference's practice).

Sign vectors: ``su`` is shared per (layer, Hessian group) -- the reference ties it to the
capture's input group -- and ``sv`` is fresh per tensor; both come from one seeded generator
so a conversion is reproducible.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from tools.convert.quantization import exl3_calibrate as cal
from tools.convert.quantization import exl3_quantize as q


def layer_tensor_inventory(
    config: dict, layer: int
) -> list[tuple[str, str, str, tuple[int, int] | None]]:
    """(tensor name, source suffix, Hessian group, optional source row range).

    The Hessian groups follow the capture: ``mixer`` (the attention/GDN projection input),
    ``mixer_out`` (the o_proj/out_proj input), ``mlp`` (gate/up) and ``down``. The weight's
    quantize orientation is [k = input, n = output]; the checkpoint stores [out, in].
    """

    prefix = cal._layer_prefix(layer)
    if config["layer_types"][layer] == "full_attention":
        tensors = [
            ("query", "self_attn.q_proj.weight", "mixer"),
            ("attention_gate", "self_attn.q_proj.weight", "mixer"),
            ("key", "self_attn.k_proj.weight", "mixer"),
            ("value", "self_attn.v_proj.weight", "mixer"),
            ("output", "self_attn.o_proj.weight", "mixer_out"),
        ]
    else:
        # The a/b control projections [48, 5120] stay BF16: the C++ Op contract
        # (gdn_gating_proj.h) consumes them as BF16, and their width is not a multiple
        # of the Hadamard block, so the incoherence transforms do not apply. The fused
        # in_proj_qkv is quantized as the model's three separate query/key/value
        # parameters (an EXL3 object cannot be row-sliced at load), in the fused row
        # order q, k, v with widths kg*dk, kg*dk, vg*dk.
        kg = config["gdn_key_heads"] * config["gdn_head_dim"]
        vg = config["gdn_value_heads"] * config["gdn_head_dim"]
        tensors = [
            ("gdn_query", (f"linear_attn.in_proj_qkv.weight", 0, kg), "mixer"),
            ("gdn_key", (f"linear_attn.in_proj_qkv.weight", kg, 2 * kg), "mixer"),
            ("gdn_value", (f"linear_attn.in_proj_qkv.weight", 2 * kg, 2 * kg + vg), "mixer"),
            ("z", "linear_attn.in_proj_z.weight", "mixer"),
            ("out", "linear_attn.out_proj.weight", "mixer_out"),
        ]
    tensors += [
        ("gate", "mlp.gate_proj.weight", "mlp"),
        ("up", "mlp.up_proj.weight", "mlp"),
        ("down", "mlp.down_proj.weight", "down"),
    ]
    return [(name, prefix + (item if isinstance(item, str) else item[0]), group,
             None if isinstance(item, str) else item[1:])
            for name, item, group in tensors]


def _split_q_gate(weight: torch.Tensor, config: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """The packed q_proj [heads*2D, H] holds each head's Q rows then its gate rows."""

    heads, d = config["heads"], config["head_dim"]
    rows = torch.arange(heads, device=weight.device)[:, None] * 2 * d
    cols = torch.arange(d, device=weight.device)[None, :]
    q_rows = (rows + cols).reshape(-1)
    gate_rows = (rows + d + cols).reshape(-1)
    return weight[q_rows], weight[gate_rows]


def convert_layer(reader: cal.ShardReader, config: dict, layer: int,
                  hessian_layer: dict, k2: int, codebook: str,
                  device: torch.device, generator: torch.Generator) -> dict:
    """One layer's EXL3 conversion. Returns {tensor name: the quantize_linear output}."""

    out = {}
    inventory = layer_tensor_inventory(config, layer)
    # su is shared per Hessian group within the layer.
    su_cache: dict[str, torch.Tensor] = {}
    for name, source, group, rows in inventory:
        weight = reader.get(source, device).float()
        if rows is not None:
            weight = weight[rows[0]:rows[1]]
        if name == "query":
            weight, gate_w = _split_q_gate(weight, config)
        elif name == "attention_gate":
            _, weight = _split_q_gate(reader.get(source, device).float(), config)
        if group not in su_cache:
            su_cache[group] = torch.sign(
                torch.randn(weight.shape[1], device=device, generator=generator))
        sv = torch.sign(torch.randn(weight.shape[0], device=device, generator=generator))
        hessian = hessian_layer[group].to(device) if hessian_layer is not None else None
        t0 = time.time()
        result = q.quantize_linear(weight.T.contiguous(), k2, codebook,
                                   su_cache[group], sv, hessian=hessian)
        elapsed = time.time() - t0
        result["weight_shape"] = tuple(weight.shape)  # [out, in] as stored
        result["quantize_seconds"] = elapsed
        out[name] = result
        print(f"  layer {layer:2d} {name:14s} {tuple(weight.shape)}: "
              f"nmse {result['nmse']:.4f}  "
              f"{'hessian_err ' + format(result['hessian_err'], '.4f') if result['hessian_err'] is not None else 'fallback':>24s}  "
              f"{elapsed:.0f}s", flush=True)
    return out


def convert_model(checkpoint: Path, hessian_dir: Path | None, out_dir: Path,
                  k2: int, codebook: str = "mcg", layers: list[int] | None = None,
                  seed: int = 20261016, device: torch.device | None = None) -> dict:
    """The model-level EXL3 conversion over the selected layers (None = every layer).

    ``out_dir`` receives one ``layer_NN.pt`` per converted layer with every tensor's EXL3
    payload (fp16 suh/svh, the packed int16 trellis) and metrics. With ``hessian_dir`` the
    per-layer captured Hessians drive the calibrated path; without it the conversion runs the
    uncalibrated fallback. Completed layer files are resumed as-is.
    """

    device = device or torch.device("cuda")
    checkpoint = Path(checkpoint)
    config = cal.load_text_config(checkpoint)
    reader = cal.ShardReader(checkpoint)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = list(range(config["layers"])) if layers is None else list(layers)
    generator = torch.Generator(device=device).manual_seed(seed)

    for layer in layers:
        target = out_dir / f"layer_{layer:02d}.pt"
        if target.exists():
            print(f"layer {layer}: already converted, resuming")
            continue
        hessian_layer = None
        if hessian_dir is not None:
            hessian_layer = torch.load(Path(hessian_dir) / f"layer_{layer:02d}.pt")
        t0 = time.time()
        result = convert_layer(reader, config, layer, hessian_layer, k2, codebook,
                               device, generator)
        torch.save(result, target)
        print(f"layer {layer} complete in {time.time() - t0:.0f}s", flush=True)
    return config
