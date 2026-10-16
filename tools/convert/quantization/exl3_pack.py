"""The EXL3 persistent-object packer (SP-7 contract section 4).

Serializes a ``quantize_linear`` result into the registered EXL3 layouts' byte payloads:

- ``pack_dense``: one dense linear as ``exl3_dense_v1`` — trellis at offset 0, the FP16
  ``suh``/``svh`` planes, and the trailing codebook id word.
- ``pack_bank``: E experts of one projection as ``exl3_expert_bank`` — the per-expert
  directory, the packed trellis extents, the ``suh``/``svh`` planes, and the codebook word.

The trellis bytes are the quantizer's packed int16 ring, little-endian; each extent is
zero-padded to its 256-byte aligned plane size.
"""

from __future__ import annotations

import struct

import torch

from tools.artifact.layouts import (
    EXL3_CODEBOOK_IDS,
    exl3_bank_geometry,
    exl3_dense_geometry,
)


def _quantize_payload(result: dict) -> tuple[bytes, int, int, int]:
    """The quantize output -> (packed trellis bytes, k, n, 2K) in the quantize orientation."""

    trellis = result["trellis"].detach().cpu().contiguous()
    if trellis.dtype != torch.int16:
        raise ValueError("the quantize trellis must be int16")
    if trellis.ndim != 3:
        raise ValueError("the quantize trellis must be [k/16, n/16, 16K]")
    tiles_k, tiles_n, steps = trellis.shape
    if steps % 8:
        raise ValueError("the trellis step dimension must be a multiple of 8")
    # The trellis carries 16·K = 8·k2 int16 elements per tile (contract section 2.4).
    return trellis.numpy().tobytes(), tiles_k * 16, tiles_n * 16, steps // 8


def pack_dense(result: dict, codebook: str = "mcg") -> bytes:
    """One dense linear's ``exl3_dense_v1`` object bytes from a quantize_linear result.

    ``result`` carries ``trellis`` [k/16, n/16, 16K] int16, ``suh`` [k] fp16 and ``svh``
    [n] fp16 (the quantize orientation, input-major). The logical weight is [n, k].
    """

    trellis_bytes, k, n, k2 = _quantize_payload(result)
    geometry = exl3_dense_geometry(n, k, k2, codebook)
    suh = result["suh"].detach().cpu().to(torch.float16).numpy().tobytes()
    svh = result["svh"].detach().cpu().to(torch.float16).numpy().tobytes()
    if len(suh) != geometry.suh_bytes or len(svh) != geometry.svh_bytes:
        raise ValueError("the suh/svh planes do not match the trellis shape")
    payload = bytearray(geometry.payload_bytes)
    payload[geometry.trellis_offset:geometry.trellis_offset + len(trellis_bytes)] = trellis_bytes
    payload[geometry.suh_offset:geometry.suh_offset + len(suh)] = suh
    payload[geometry.svh_offset:geometry.svh_offset + len(svh)] = svh
    struct.pack_into(
        "<I", payload, geometry.codebook_offset, EXL3_CODEBOOK_IDS[geometry.codebook])
    return bytes(payload)


def pack_bank(results: list[dict], codebook: str = "mcg") -> bytes:
    """One projection's ``exl3_expert_bank`` object bytes, expert id in list order.

    Each result carries that expert's own ``trellis``, ``suh`` and ``svh``; all experts
    share one codebook and may carry their own bitrates (mixed-width banks).
    """

    if not results:
        raise ValueError("an EXL3 bank requires at least one expert")
    trellis_bytes, k, n, _ = _quantize_payload(results[0])
    k2 = []
    packed = []
    for result in results:
        bytes_, expert_k, expert_n, expert_k2 = _quantize_payload(result)
        if (expert_k, expert_n) != (k, n):
            raise ValueError("an EXL3 bank's experts must share one [n, k] shape")
        k2.append(expert_k2)
        packed.append(bytes_)
    geometry = exl3_bank_geometry(n, k, k2, codebook)
    payload = bytearray(geometry.payload_bytes)
    offset = geometry.trellis_offset
    for byte_offset, extent, _ in geometry.directory_entries():
        struct.pack_into("<Q", payload, offset, byte_offset)
        struct.pack_into("<I", payload, offset + 8, extent)
        offset += 16
    for bytes_, extent in zip(packed, geometry.trellis_extents, strict=True):
        if len(bytes_) > extent:
            raise ValueError("a trellis extent exceeds its aligned plane size")
    for bytes_, (byte_offset, _, _) in zip(
            packed, geometry.directory_entries(), strict=True):
        payload[byte_offset:byte_offset + len(bytes_)] = bytes_
    suh = b"".join(
        result["suh"].detach().cpu().to(torch.float16).numpy().tobytes()
        for result in results)
    svh = b"".join(
        result["svh"].detach().cpu().to(torch.float16).numpy().tobytes()
        for result in results)
    if len(suh) != geometry.suh_bytes or len(svh) != geometry.svh_bytes:
        raise ValueError("the suh/svh planes do not match the bank shape")
    payload[geometry.suh_offset:geometry.suh_offset + len(suh)] = suh
    payload[geometry.svh_offset:geometry.svh_offset + len(svh)] = svh
    struct.pack_into(
        "<I", payload, geometry.codebook_offset, EXL3_CODEBOOK_IDS[geometry.codebook])
    return bytes(payload)
