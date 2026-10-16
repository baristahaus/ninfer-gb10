"""The real-converted-tensor check for the dense EXL3 Linear route (SP-7 dense step).

Packs one converted tensor of a real 27B quantization into a `.ninfer` container as an
`exl3_dense_v1` object (the production writer), alongside the composed FP32 expected weight
(section 2.1's `diag(suh) Hn_k W_hat Hn_n diag(svh)`, transposed to the logical `[n, k]`
orientation) as a raw_bytes resource. The C++ checker binary
(`ninfer_linear_exl3_dense_a16_test <container>`) then runs the kernel against that matrix.

Usage:
    python tools/convert/quantization/exl3_real_check.py \
        LAYER_PT TENSOR_NAME OUT_CONTAINER [CHECK_BINARY]

    LAYER_PT     a convert_model layer file (e.g. /home/barista/calib/exl3_subset_k4/layer_00.pt)
    TENSOR_NAME  the tensor key inside it (e.g. "out")
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from tools.artifact.schema import TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert.quantization import exl3
from tools.convert.quantization.exl3_pack import _quantize_payload
from tools.convert.quantization.exl3_quantize import decode_composition


def compose_expected_fp32(result: dict, codebook: str) -> np.ndarray:
    """The composed logical weight [n, k] in FP32 from a quantize_linear result."""

    trellis_bytes, k, n, k2 = _quantize_payload(result)
    del trellis_bytes
    trellis = np.frombuffer(
        result["trellis"].detach().cpu().numpy().tobytes(), dtype=np.int16
    ).reshape(k // 16, n // 16, 8 * k2)
    inner = exl3.decode_inner(trellis, codebook)  # [k, n] fp16 bit patterns
    inner_t = torch.from_numpy(inner.astype(np.uint16)).view(torch.float16).float()
    suh = result["suh"].detach().cpu().float()
    svh = result["svh"].detach().cpu().float()
    weight = decode_composition(suh, inner_t, svh)  # [k, n]
    return weight.T.to(torch.float32).contiguous().numpy()  # logical [n, k]


def main() -> int:
    layer_path = Path(sys.argv[1])
    tensor_name = sys.argv[2]
    out_container = Path(sys.argv[3])
    result = torch.load(layer_path, map_location="cpu", weights_only=False)[tensor_name]
    codebook = "mcg"
    payload = _pack(result, codebook)
    expected = compose_expected_fp32(result, codebook)
    n, k = expected.shape
    with ArtifactWriter(
        out_container,
        [TensorSpec("exl3_dense", (n, k), "exl3", "exl3_dense_v1", bytes=len(payload)),
         TensorSpec("expected", (n, k), "fp32", "contiguous_le_v1")],
        components={"text": {"config": {}}},
        bindings={"dense": {"object": "exl3_dense"}},
    ) as writer:
        writer.write_object("exl3_dense", payload)
        writer.write_object("expected", expected.tobytes())
    print(f"wrote {out_container}: [{n},{k}] payload {len(payload)} bytes, "
          f"expected {expected.size * 4} bytes")
    return 0


def _pack(result: dict, codebook: str) -> bytes:
    from tools.convert.quantization.exl3_pack import pack_dense

    return pack_dense(result, codebook)


if __name__ == "__main__":
    raise SystemExit(main())
