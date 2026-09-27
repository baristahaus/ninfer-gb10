"""Write the leading bytes of representative Flash-Next weights for the memory probe.

Each sample is one decode-path tensor class as stored in the artifact. The probe loads a sample
into plain and compression-requested allocations and reads both, which answers whether GB10
generic compression helps real weight bytes rather than zero-filled buffers.
Prints one `label=path` line per sample.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.artifact.reader import Artifact  # noqa: E402
from tools.artifact.schema import binding_parts  # noqa: E402

LAYER = "model.language_model.layers."
SAMPLES = (
    ("bf16-gdn-in_proj_qkv", LAYER + "0.linear_attn.in_proj_qkv.weight"),
    ("bf16-hc-input_mix_down", LAYER + "0.attn_hyper_connection.input_mix_weight_down.weight"),
    ("bf16-lm_head", "lm_head.weight"),
    ("nvfp4-experts-gate_up", LAYER + "0.mlp.experts.gate_up"),
    ("fp8-ple-table", LAYER + "1.ple.ple_embedding.ngram_embedding.weight"),
    ("q4-proposal-head", "ninfer.optimized_proposal_head.weight"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--mib", type=int, default=64, help="bytes per sample (default 64 MiB)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with Artifact(args.artifact) as artifact:
        bindings = artifact.directory.bindings
        for label, parameter in SAMPLES:
            if parameter not in bindings:
                print(f"skipping {label}: no binding {parameter}", file=sys.stderr)
                continue
            object_id = binding_parts(bindings[parameter], artifact.by_id, parameter)[0][0]
            obj = artifact.object(object_id)
            count = min(obj.bytes, args.mib << 20)
            path = args.out_dir / f"{label}.bin"
            with path.open("wb") as out:
                for chunk in artifact.iter_range(obj.offset, count):
                    out.write(chunk)
            print(f"{label}={path}")


if __name__ == "__main__":
    main()
