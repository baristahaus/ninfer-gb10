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
    # in_proj_qkv is one such tensor, as is his interleaved q_proj (query + gate).
    # The first input file wins per key: pass the 2.00bpw anchor first.
    role_tensor = {"query": "q_proj", "gate": "q_proj", "key": "k_proj",
                   "value": "v_proj", "output": "o_proj", "z": "in_proj_z",
                   "out": "out_proj", "down": "down_proj"}
    inputs = [sys.argv[1]] + sys.argv[3:]  # the first input is the primary anchor
    anchors = {}
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
                anchors.setdefault(key, (r["k2"], r["rfn"]))
    results = [dict(key=k, K=kv[0], rfn=kv[1]) for k, kv in sorted(anchors.items())]
    json.dump(dict(results=results), open(out, "w"), indent=1)
    print(f"{len(results)} anchors -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
