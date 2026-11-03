#!/usr/bin/env python3
"""W2.2 rfn probe: per-tensor weight-space error of the SC artifact vs the BF16 source.

rfn_t = ||W_q - W_ref||_F / ||W_ref||_F per EXL3 leaf, measured against the unquantized
Qwen3.8-27B checkpoint. Leaf provenance (his source tensor, row bands/spans, parameter
names) comes from the conversion report written next to the artifact; the trellis is
decoded with the same numpy decoder the converter validates. CPU only.

Usage:
    python tools/convert/quantization/exl3_rfn_probe.py \
        out/qwen3_8_27b_exl3_sc385_df2.ninfer out/rfn_sc385_df2.json [--limit N]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from safetensors import safe_open

from tools.artifact.reader import Artifact
from tools.convert.quantization import exl3

HADAMARD = 128


def hadamard_128() -> torch.Tensor:
    idx = torch.arange(HADAMARD)
    and_table = (idx[:, None] & idx[None, :]).numpy()
    parity = np.vectorize(lambda v: bin(int(v)).count("1") % 2)(and_table)
    h = torch.from_numpy((1 - 2 * parity).astype(np.float32))
    return h * HADAMARD**-0.5


_H = hadamard_128()


def binary16_values(bits: torch.Tensor) -> torch.Tensor:
    """Exact fp32 values of binary16 bit patterns (uint16 -> float)."""
    bits = bits.long()
    signs = torch.where(bits & 0x8000 != 0, -1.0, 1.0)
    expo = ((bits >> 10) & 0x1F).float()
    mant = (bits & 0x3FF).float()
    subnormal = (bits >> 10) & 0x1F == 0
    values = signs * torch.where(subnormal, mant * 2.0**-24,
                                 (1.0 + mant / 1024.0) * 2.0 ** (expo - 15.0))
    return values * ((bits & 0x7FFF) != 0)


def parse_leaf(payload: bytes, n: int, k: int) -> tuple[torch.Tensor, int, str]:
    """Decode one exl3_dense_v1 payload to the logical [n, k] fp32 weight.

    Layout (contract section 4): the trellis at 0 (k*n*K2/8 bytes; n, k are
    128-multiples so the extent is already 256-aligned), the suh fp16 [k], the svh
    fp16 [n], then the 4-byte codebook id. W = diag(suh) Hn_k W_inner Hn_n diag(svh),
    transposed to the logical [n, k] orientation.
    """

    codebook = {0: "3inst", 1: "mcg", 2: "mul1"}.get(int.from_bytes(payload[-4:], "little"))
    if codebook is None:
        raise ValueError(f"unknown codebook id {int.from_bytes(payload[-4:], 'little')}")
    suh_offset = len(payload) - 4 - n * 2 - k * 2
    k2 = suh_offset * 8 // (n * k)
    if suh_offset * 8 != n * k * k2 or not 2 <= k2 <= 16:
        raise ValueError(f"trellis extent {suh_offset} matches no bitrate for {n}x{k}")
    trellis = np.frombuffer(payload, dtype=np.uint16, count=suh_offset // 2, offset=0)
    inner = exl3.decode_inner(trellis.reshape(k // 16, n // 16, 16 * k2), codebook)  # [k, n] bits
    w = binary16_values(torch.from_numpy(inner.astype(np.int64)))
    suh = torch.from_numpy(np.frombuffer(payload, np.uint16, k, suh_offset).copy())
    svh = torch.from_numpy(np.frombuffer(payload, np.uint16, n, suh_offset + k * 2).copy())
    k_blocks = w.reshape(k // 128, 128, n).transpose(1, 2) @ _H      # Hn_k per k-block
    w = k_blocks.transpose(1, 2).reshape(k, n)
    w = w.reshape(k, n // 128, 128) @ _H                             # Hn_n per n-block
    w = w.reshape(k, n)
    w = binary16_values(suh)[:, None] * w * binary16_values(svh)[None, :]
    return w.T.contiguous(), k2, codebook  # [n, k]


class BF16Source:
    """Row reads from the BF16 checkpoint's safetensors shards."""

    def __init__(self, checkpoint: Path):
        index_file = checkpoint / "model.safetensors.index.json"
        self.shards: dict[str, Path] = {}
        if index_file.exists():
            weight_map = json.load(open(index_file))["weight_map"]
            for name, shard in weight_map.items():
                self.shards[name] = checkpoint / shard
        else:  # a single-file checkpoint
            self.shards = {p.stem: p for p in checkpoint.glob("model.safetensors")}
        self._handles: dict[Path, any] = {}

    def rows(self, name: str, row_begin: int, rows: int) -> torch.Tensor:
        path = self.shards[name]
        if path not in self._handles:
            self._handles[path] = safe_open(path, framework="pt", device="cpu")
        tensor = self._handles[path].get_slice(name)
        return tensor[row_begin:row_begin + rows, :].float()


def source_tensor(source_ref: str) -> str:
    """HF weight name from a source ref (the last `:...weight` before any span tail)."""

    return re.search(r":([^:,]+\.weight)(?:,|$)", source_ref).group(1)


def leaf_reference_rows(source: "BF16Source", record: dict, leaf_n: int,
                        heads: int, head_dim: int) -> torch.Tensor:
    """Reference rows [leaf_n, k] for one leaf, per its parameter/source pairs.

    Banded parameters (the attention query/gate rows of his interleaved q_proj)
    select their 16-row bands: band index b covers rows b*16..b*16+16. Spanned
    sources take contiguous [begin, begin+count) ranges accumulated per tensor
    (the GDN fused qkv); plain sources take the parameter's own full rows, sized
    from the leaf total minus the counted spans.
    """

    mp = record.get("method_parameters", {})
    if "row_bands" in mp:  # the SC import form: one tensor, explicit band list
        return torch.cat([source.rows(mp["tensor"] + ".weight", b * 16, 16)
                          for b in mp["row_bands"]])
    stride = 2 * head_dim // 16
    bands_per_role = head_dim // 16
    blocks, begin_by_tensor = [], {}
    for param, ref in zip(record["parameters"], record["sources"]):
        tensor = source_tensor(ref)
        role = param.rsplit("/", 1)[-1]
        if role in ("query", "gate") and "q_proj" in tensor:
            offset = head_dim // 16 if role == "gate" else 0
            bands = [head * stride + offset + band
                     for head in range(heads) for band in range(bands_per_role)]
            blocks.append((tensor, "bands", bands))
            continue
        m = re.search(r", (\d+) rows\)$", ref)
        if m is None:
            blocks.append((tensor, "plain", None))
        else:
            count = int(m.group(1))
            begin = begin_by_tensor.get(tensor, 0)
            blocks.append((tensor, "span", (begin, count)))
            begin_by_tensor[tensor] = begin + count
    counted = sum(spec[1] for _, kind, spec in blocks if kind == "span")
    plain_blocks = [b for b in blocks if b[1] == "plain"]
    plain_count = (leaf_n - counted) // len(plain_blocks) if plain_blocks else 0
    parts = []
    for tensor, kind, spec in blocks:
        if kind == "bands":
            parts.append(torch.cat([source.rows(tensor, b * 16, 16) for b in spec]))
        elif kind == "span":
            parts.append(source.rows(tensor, spec[0], spec[1]))
        else:
            parts.append(source.rows(tensor, 0, plain_count))
    ref = torch.cat(parts)
    assert ref.shape[0] == leaf_n, f"reference rows {ref.shape[0]} != leaf {leaf_n}"
    return ref


def main() -> int:
    artifact_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    report = json.load(open(str(artifact_path) + ".conversion.json"))
    base = Path(report["provenance"]["sources"]["base"]["path"])
    source = BF16Source(base)
    config = json.load(open(base / "config.json"))["text_config"]
    heads, head_dim = config["num_attention_heads"], config["head_dim"]
    methods = {m["object"]: m for m in report["methods"] if "exl3" in m["method"]}

    artifact = Artifact.open(artifact_path)
    results, failures = [], []
    for obj in artifact.objects:
        if type(obj).__name__ != "TensorObject" or obj.format != "exl3":
            continue
        record = methods.get(obj.id)
        if record is None:
            continue
        try:
            mp = record["method_parameters"]
            n, k = obj.shape
            payload = artifact.read_range(obj.offset, obj.bytes)
            w_q, k2, codebook = parse_leaf(payload, n, k)

            # The reference rows: banded leaves select his interleaved q_proj bands;
            # grouped leaves take contiguous role spans; plain leaves the full tensor.
            ref = leaf_reference_rows(source, record, n, heads, head_dim)
            ref = ref[:n].contiguous()  # [n, k], matching w_q
            if ref.shape != w_q.shape:
                raise ValueError(f"reference shape {tuple(ref.shape)} != {tuple(w_q.shape)}")
            diff = (w_q - ref)
            ref_sq = ref.square().sum().item()
            rfn = (diff.square().sum().item() / ref_sq) ** 0.5
            results.append(dict(object=obj.id, parameters=record["parameters"],
                                source=record["sources"][0], n=n, k=k, k2=k2, codebook=codebook,
                                rfn=rfn, ref_norm=ref_sq ** 0.5))
        except Exception as exc:  # noqa: BLE001 - per-leaf failures recorded, not fatal
            failures.append(dict(object=obj.id, error=repr(exc)[:200]))
        if limit and len(results) >= limit:
            break

    by_k: dict[int, list[float]] = {}
    for r in results:
        by_k.setdefault(r["k2"], []).append(r["rfn"])
    summary = {str(k): dict(n=len(v), median=sorted(v)[len(v) // 2],
                            min=min(v), max=max(v)) for k, v in sorted(by_k.items())}
    json.dump(dict(artifact=str(artifact_path), base=str(base), results=results,
                   failures=failures, summary=summary),
              open(out_path, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    if failures:
        print(f"{len(failures)} leaf failures; first: {failures[0]}")
    print(f"-- {len(results)} leaves -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
