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
    # His measurement keys are the BF16 model's HF tensor names; the GDN's fused
    # in_proj_qkv is one such tensor, as is his interleaved q_proj (the query and
    # gate leaves of one module). Multiple leaves mapping to one key combine via
    # err_sq = sum((rfn*ref_norm)^2) and ref_sq = sum(ref_norm^2), the same
    # weighted form his grouped-member logic uses.
    role_tensor = {"query": "q_proj", "gate": "q_proj", "attention_gate": "q_proj",
                   "key": "k_proj", "value": "v_proj", "output": "o_proj",
                   "z": "in_proj_z", "out": "out_proj", "gate_proj": "gate_proj",
                   "up": "up_proj", "up_proj": "up_proj", "down": "down_proj",
                   "down_proj": "down_proj"}
    inputs = [sys.argv[1]] + sys.argv[3:]  # the first input is the primary anchor
    modules: dict[str, list[tuple[int, float, float]]] = {}
    for path in inputs:
        data = json.load(open(path))
        for r in data["results"]:
            for param in r["parameters"]:
                parts = param.split("/")  # text / layers / N / <group> / <role>
                layer, role = int(parts[2]), parts[-1]
                if role not in role_tensor:
                    continue
                stem = ("self_attn" if "/attention/" in param else
                        "linear_attn" if "/gdn/" in param else "mlp")
                key = f"model.language_model.layers.{layer}.{stem}.{role_tensor[role]}.weight"
                modules.setdefault(key, []).append((r["k2"], r["rfn"],
                                                    r.get("ref_norm", 0.0)))
    results = []
    for key, entries in sorted(modules.items()):
        if len({e[0] for e in entries}) == 1:
            k_common = entries[0][0]
        else:
            k_common = max({e[0] for e in entries}, key=lambda k: sum(1 for e in entries if e[0] == k))
        err_sq = sum((rfn * ref) ** 2 for _, rfn, ref in entries)
        ref_sq = sum(ref ** 2 for _, _, ref in entries)
        rfn = (err_sq / ref_sq) ** 0.5 if ref_sq else min(e[1] for e in entries)
        results.append(dict(key=key, K=k_common, rfn=rfn))
    json.dump(dict(results=results), open(out, "w"), indent=1)
    print(f"{len(results)} anchors -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
