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


def _fused_group(result: dict, name: str, weights: list[torch.Tensor], group: str,
                 su_cache: dict, generator: torch.Generator, hessian_layer: dict | None,
                 k2: int, codebook: str, device: torch.device) -> None:
    """Quantize one fused group (one linear over the stacked binding-order rows)."""

    weight = torch.cat(weights, dim=0).float()  # [out, in]
    if group not in su_cache:
        su_cache[group] = torch.sign(
            torch.randn(weight.shape[1], device=device, generator=generator))
    sv = torch.sign(torch.randn(weight.shape[0], device=device, generator=generator))
    hessian = hessian_layer[group].to(device) if hessian_layer is not None else None
    t0 = time.time()
    out = q.quantize_linear(weight.T.contiguous(), k2, codebook, su_cache[group], sv,
                            hessian=hessian)
    out["weight_shape"] = tuple(weight.shape)  # [out, in] as stored
    out["quantize_seconds"] = time.time() - t0
    result[name] = out
    print(f"  {name:16s} {tuple(weight.shape)}: nmse {out['nmse']:.4f}  "
          f"{'hessian_err ' + format(out['hessian_err'], '.4f') if out['hessian_err'] is not None else 'fallback':>24s}  "
          f"{time.time() - t0:.0f}s", flush=True)


def convert_model_fused(checkpoint: Path, hessian_dir: Path | None, out_dir: Path,
                        k2: int, codebook: str = "mcg", layers: list[int] | None = None,
                        seed: int = 20261016, device: torch.device | None = None) -> dict:
    """The fused-group EXL3 conversion for the dense route (SP-7).

    Every mixer/FFN group quantizes as ONE linear in the binding row order — attention
    (query, key, gate, value) with the interleaved q_proj rows split, GDN (query, key,
    value, z), MLP (gate, up) — sharing the group's input-side transform; the output and
    down projections stay their own linears. ``out_dir`` receives one ``layer_NN.pt`` per
    layer with the keys ``attention_fused``/``gdn_fused``, ``gate_up``, ``down`` and
    ``output``/``out``. Completed layer files are resumed as-is.
    """

    device = device or torch.device("cuda")
    checkpoint = Path(checkpoint)
    config = cal.load_text_config(checkpoint)
    reader = cal.ShardReader(checkpoint)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = list(range(config["layers"])) if layers is None else list(layers)
    generator = torch.Generator(device=device).manual_seed(seed)
    heads, d = config["heads"], config["head_dim"]
    kg = config["gdn_key_heads"] * config["gdn_head_dim"]
    vg = config["gdn_value_heads"] * config["gdn_head_dim"]

    for layer in layers:
        target = out_dir / f"layer_{layer:02d}.pt"
        if target.exists():
            print(f"layer {layer}: already converted, resuming", flush=True)
            continue
        hessian_layer = None
        if hessian_dir is not None:
            hessian_layer = torch.load(Path(hessian_dir) / f"layer_{layer:02d}.pt")
        t0 = time.time()
        prefix = cal._layer_prefix(layer)
        result: dict = {}
        su_cache: dict[str, torch.Tensor] = {}
        if config["layer_types"][layer] == "full_attention":
            q_gate = reader.get(prefix + "self_attn.q_proj.weight", device)
            q_rows, gate_rows = _split_q_gate(q_gate, config)
            _fused_group(result, "attention_fused",
                         [q_rows, reader.get(prefix + "self_attn.k_proj.weight", device),
                          gate_rows, reader.get(prefix + "self_attn.v_proj.weight", device)],
                         "mixer", su_cache, generator, hessian_layer, k2, codebook, device)
            single = "output"
        else:
            qkv = reader.get(prefix + "linear_attn.in_proj_qkv.weight", device)
            _fused_group(result, "gdn_fused",
                         [qkv[0:kg], qkv[kg:2 * kg], qkv[2 * kg:2 * kg + vg],
                          reader.get(prefix + "linear_attn.in_proj_z.weight", device)],
                         "mixer", su_cache, generator, hessian_layer, k2, codebook, device)
            single = "out"
        _fused_group(result, "gate_up",
                     [reader.get(prefix + "mlp.gate_proj.weight", device),
                      reader.get(prefix + "mlp.up_proj.weight", device)],
                     "mlp", su_cache, generator, hessian_layer, k2, codebook, device)
        for name in ("down", single):
            source, group, rows = next(
                (entry[1], entry[2], entry[3] if len(entry) > 3 else None)
                for entry in layer_tensor_inventory(config, layer) if entry[0] == name)
            weight = reader.get(source, device).float()
            if rows is not None:
                weight = weight[rows[0]:rows[1]]
            _fused_group(result, name, [weight], group, su_cache, generator, hessian_layer,
                         k2, codebook, device)
        torch.save(result, target)
        print(f"layer {layer} complete in {time.time() - t0:.0f}s", flush=True)
    return config


def convert_model_banks(checkpoint: Path, out_dir: Path, k2: int, codebook: str = "mcg",
                        layers: list[int] | None = None, seed: int = 20261016,
                        device: torch.device | None = None,
                        roles: tuple[str, ...] = ("routed_gate_up", "routed_down")) -> dict:
    """The routed-bank EXL3 conversion (SP-7 contract 6.2): every routed expert's gate, up
    and down linear quantizes uncalibrated (per-expert calibration data is too sparse to
    inform a Hessian), with one su per (layer, projection) group and per-expert sv.

    ``out_dir`` receives one ``layer_NN.pt`` per MoE layer with keys ``routed_gate_up``
    (one packed linear per expert over the checkpoint's [2*intermediate, hidden] block,
    the engine's gate/up bank convention) and ``routed_down``, each a list of
    per-expert ``quantize_linear`` results in expert-id order. Completed layer files
    are resumed as-is. ``roles`` selects which projections run (the packed gate_up and
    the down are independent; a pass may fill either).
    """

    device = device or torch.device("cuda")
    checkpoint = Path(checkpoint)
    config = cal.load_text_config(checkpoint)
    reader = cal.ShardReader(checkpoint)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = list(range(config["layers"])) if layers is None else list(layers)
    generator = torch.Generator(device=device).manual_seed(seed)
    experts = config["num_experts"]
    intermediate = config["moe_intermediate"]
    hidden = config["hidden"]

    for layer in layers:
        target = out_dir / f"layer_{layer:02d}.pt"
        if target.exists():
            print(f"layer {layer}: already converted, resuming", flush=True)
            continue
        t0 = time.time()
        prefix = cal._layer_prefix(layer)
        result: dict = {}
        # The checkpoint stores the routed tensors batched [E, ...]; the packed
        # gate_up block per expert is one linear over [2*intermediate, hidden].
        gate_up = reader.get(prefix + "mlp.experts.gate_up_proj", device).float()
        down_source = prefix + "mlp.experts.down_proj"
        shapes = {"routed_gate_up": (2 * intermediate, hidden),
                  "routed_down": (hidden, intermediate)}
        for role in roles:
            out_n, out_k = shapes[role]
            su = torch.sign(torch.randn(out_k, device=device, generator=generator))
            per_expert = []
            for expert in range(experts):
                weight = (gate_up[expert] if role == "routed_gate_up"
                          else reader.get(down_source, device).float()[expert])
                sv = torch.sign(
                    torch.randn(weight.shape[0], device=device, generator=generator))
                out = q.quantize_linear(weight.T.contiguous(), k2, codebook, su, sv,
                                        hessian=None)
                out["weight_shape"] = tuple(weight.shape)
                per_expert.append(out)
            result[role] = per_expert
            print(f"  layer {layer:2d} {role:14s} {experts} experts "
                  f"[{out_n},{out_k}]: done {time.time() - t0:.0f}s", flush=True)
        torch.save(result, target)
        print(f"layer {layer} complete in {time.time() - t0:.0f}s", flush=True)
    return config


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
