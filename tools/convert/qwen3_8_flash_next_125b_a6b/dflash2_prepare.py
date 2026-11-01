"""Prepare the published DFlash2 companion for the Qwen4Exp convert route.

The checkpoint ships separate `self_attn.{q,k,v}_proj` and `mlp.{gate,up}_proj`
tensors; the route's runtime consumes one fused `self_attn.query_key_value.weight`
(row order q, k, v) and one packed `mlp.gate_up_proj.weight` (gate rows first).
Everything else is copied unchanged, including config.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def prepare(source: Path, output: Path) -> None:
    tensors = load_file(source / "model.safetensors")
    prepared: dict[str, torch.Tensor] = {}
    for name, value in tensors.items():
        if name.endswith("self_attn.k_proj.weight") or name.endswith(
            "self_attn.v_proj.weight"
        ) or name.endswith("mlp.up_proj.weight"):
            continue
        if name.endswith("self_attn.q_proj.weight"):
            p = name[: -len("self_attn.q_proj.weight")]
            prepared[p + "self_attn.query_key_value.weight"] = torch.cat(
                [value, tensors[p + "self_attn.k_proj.weight"],
                 tensors[p + "self_attn.v_proj.weight"]], dim=0
            )
        elif name.endswith("mlp.gate_proj.weight"):
            p = name[: -len("mlp.gate_proj.weight")]
            prepared[p + "mlp.gate_up_proj.weight"] = torch.cat(
                [value, tensors[p + "mlp.up_proj.weight"]], dim=0
            )
        else:
            prepared[name] = value
    assert len(prepared) == len(tensors) - len(
        [n for n in tensors if n.endswith("self_attn.k_proj.weight")
         or n.endswith("self_attn.v_proj.weight") or n.endswith("mlp.up_proj.weight")]
    ), (len(prepared), len(tensors))
    output.mkdir(parents=True, exist_ok=True)
    save_file(prepared, output / "model.safetensors")
    config = json.loads((source / "config.json").read_text())
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"prepared {len(prepared)} tensors -> {output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="the published DFlash2 checkpoint directory")
    parser.add_argument("output", type=Path, help="the prepared companion directory")
    args = parser.parse_args(argv)
    prepare(args.source, args.output)


if __name__ == "__main__":
    main()
