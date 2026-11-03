#!/usr/bin/env python3
"""Convert our exl3_rfn_probe output into his sc_optimize --rfn_ref anchor format.

His format: {"results": [{"key": <HF tensor name>, "K": <bits>, "rfn": <error>}]}.
A fused leaf (one artifact object covering several parameters, e.g. the GDN
in_proj_qkv group or his fused attention parent) anchors every member parameter
with the leaf's measured (K, rfn).

Usage:
    python tools/convert/quantization/rfn_to_ref.py profiles/rfn/rfn_2bpw.json \
        out/rfn_ref_2bpw.json [profiles/rfn/rfn_sc385_df2.json ...]
"""

import json
import sys
from pathlib import Path


def main() -> int:
    out = Path(sys.argv[2])
    # His measurement keys are exllamav3 module names with fused parents
    # (linear_attn.qkvz, self_attn.qkv, mlp.gate_up); our probe records carry the
    # per-role parameters. Map each parameter to its member HF tensor, group the
    # members per module, and combine via err_sq = sum((rfn*ref_norm)^2),
    # ref_sq = sum(ref_norm^2) - the weighted form his grouped-member logic uses.
    # The first input file wins per member; the 2.00bpw anchor is primary.
    role_member = {("linear_attn", "query"): "in_proj_qkv",
                   ("linear_attn", "key"): "in_proj_qkv",
                   ("linear_attn", "value"): "in_proj_qkv",
                   ("linear_attn", "z"): "in_proj_z",
                   ("linear_attn", "out"): "out_proj",
                   ("linear_attn", "output"): "out_proj",
                   ("self_attn", "query"): "q_proj", ("self_attn", "gate"): "q_proj",
                   ("self_attn", "attention_gate"): "q_proj",
                   ("self_attn", "key"): "k_proj", ("self_attn", "value"): "v_proj",
                   ("self_attn", "output"): "o_proj",
                   ("mlp", "gate"): "gate_proj", ("mlp", "gate_proj"): "gate_proj",
                   ("mlp", "up"): "up_proj", ("mlp", "up_proj"): "up_proj",
                   ("mlp", "down"): "down_proj", ("mlp", "down_proj"): "down_proj"}
    module_suffix = {
        ("linear_attn", "in_proj_qkv"): "linear_attn.qkvz",
        ("linear_attn", "in_proj_z"): "linear_attn.qkvz",
        ("linear_attn", "out_proj"): "linear_attn.out_proj",
        ("self_attn", "q_proj"): "self_attn.qkv",
        ("self_attn", "k_proj"): "self_attn.qkv",
        ("self_attn", "v_proj"): "self_attn.qkv",
        ("self_attn", "o_proj"): "self_attn.o_proj",
        ("mlp", "gate_proj"): "mlp.gate_up",
        ("mlp", "up_proj"): "mlp.gate_up",
        ("mlp", "down_proj"): "mlp.down_proj",
    }
    members: dict[str, list[tuple[int, float, float]]] = {}  # hf tensor -> entries
    inputs = [sys.argv[1]] + sys.argv[3:]  # the first input is the primary anchor
    for path in inputs:
        data = json.load(open(path))
        for r in data["results"]:
            for param in r["parameters"]:
                parts = param.split("/")  # text / layers / N / <group> / <role>
                layer, role = int(parts[2]), parts[-1]
                stem = ("self_attn" if "/attention/" in param else
                        "linear_attn" if "/gdn/" in param else "mlp")
                if (stem, role) not in role_member:
                    continue
                member = role_member[(stem, role)]
                hf = f"model.language_model.layers.{layer}.{stem}.{member}.weight"
                members.setdefault(hf, []).append((r["k2"], r["rfn"],
                                                   r.get("ref_norm", 0.0)))
    # Combine the members into his module keys; his lm_head is anchored globally
    # (the head bitrate is a separate sc_optimize knob and our head is not EXL3).
    results = [dict(key="lm_head", K=2, rfn=0.292)]
    by_module: dict[str, list[tuple[int, float, float]]] = {}
    for hf, entries in members.items():
        # model . language_model . layers . N . stem . member . weight
        _, _, _, layer, stem, member, _ = hf.split(".")
        module = f"model.language_model.layers.{layer}.{module_suffix[(stem, member)]}"
        by_module.setdefault(module, []).extend(entries)
    for module, entries in sorted(by_module.items()):
        if len({e[0] for e in entries}) == 1:
            k_common = entries[0][0]
        else:
            k_common = max({e[0] for e in entries},
                           key=lambda k: sum(1 for e in entries if e[0] == k))
        err_sq = sum((rfn * ref) ** 2 for _, rfn, ref in entries)
        ref_sq = sum(ref ** 2 for _, _, ref in entries)
        rfn = (err_sq / ref_sq) ** 0.5 if ref_sq else min(e[1] for e in entries)
        results.append(dict(key=module, K=k_common, rfn=rfn))
    json.dump(dict(results=results), open(out, "w"), indent=1)
    print(f"{len(results)} anchors -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
