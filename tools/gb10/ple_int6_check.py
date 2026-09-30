#!/usr/bin/env python3
"""Plan 7f side-check: int6 group-32 PLE rows against our FP8 rows, both measured against BF16.

Usage: ple_int6_check.py BF16_CHECKPOINT FP8_CHECKPOINT [--blocks 400] [--block-rows 256]

BF16_CHECKPOINT is the original Qwen3.8-Flash-Next BF16 release (the PLE n-gram table in BF16);
FP8_CHECKPOINT is the RadixArk NVFP4 source our artifact copies the FP8 table from without
requantizing (128 shards of 2,500,012 x 160 FP8 E4M3FN plus one `weight_scale`). Both are
safetensors directories with an index. Needs torch and safetensors (the converter environment).

Samples `--blocks` random blocks of `--block-rows` contiguous rows (fixed seed) and reports, for
the represented FP8 table (code times weight_scale) and for int6 with one FP16 scale per 32
values (symmetric, scale = max|x|/31 rounded to FP16, codes rounded to nearest even and clamped
to [-31, 31]): mean per-row relative L2, the global relative L2, and max row relative L2, plus
the bytes per row. tcclaviger's card reports 0.0218 (int6) against 0.0267 (their FP8).
"""
import argparse
import json
import pathlib
import random

import torch
from safetensors import safe_open

PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
ROWS = 320_001_536
SHARD_ROWS = 2_500_012
WIDTH = 160


class Table:
    """Row access to the PLE table, stored either whole or in the 128 source shards."""

    def __init__(self, directory):
        root = pathlib.Path(directory)
        index = json.loads(next(root.glob("*.safetensors.index.json")).read_text())["weight_map"]
        self.root, self.index, self.handles = root, index, {}
        self.whole = PREFIX + "weight" in index
        self.scale = None
        if PREFIX + "weight_scale" in index:
            self.scale = self.tensor(PREFIX + "weight_scale").float().item()

    def handle(self, name):
        file = self.index[name]
        if file not in self.handles:
            self.handles[file] = safe_open(str(self.root / file), framework="pt")
        return self.handles[file]

    def tensor(self, name):
        return self.handle(name).get_tensor(name)

    def rows(self, begin, count):
        if self.whole:
            name, local = PREFIX + "weight", begin
        else:
            shard, local = divmod(begin, SHARD_ROWS)
            count = min(count, SHARD_ROWS - local)
            name = PREFIX + f"shard_{shard}.weight"
        block = self.handle(name).get_slice(name)[local:local + count]
        values = block.float()
        return values * self.scale if self.scale is not None else values


def int6_group32(x):
    groups = x.reshape(-1, 32)
    scale = (groups.abs().amax(dim=1, keepdim=True) / 31).half().float()
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    codes = torch.clamp(torch.round(groups / scale), -31, 31)  # torch.round is half-to-even
    return (codes * scale).reshape(x.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bf16")
    ap.add_argument("fp8")
    ap.add_argument("--blocks", type=int, default=400)
    ap.add_argument("--block-rows", type=int, default=256)
    args = ap.parse_args()
    reference, ours = Table(args.bf16), Table(args.fp8)
    if ours.scale is None:
        raise SystemExit("FP8 checkpoint has no ngram_embedding.weight_scale")
    rng = random.Random(20260929)
    sums = {"fp8": [], "int6": []}
    num = {"fp8": 0.0, "int6": 0.0}
    den = 0.0
    for _ in range(args.blocks):
        begin = rng.randrange(0, ROWS - args.block_rows)
        want = ours.rows(begin, args.block_rows)
        ref = reference.rows(begin, want.shape[0])
        norms = ref.norm(dim=1)
        keep = norms > 0
        den += float((ref ** 2).sum())
        for label, approx in (("fp8", want), ("int6", int6_group32(ref))):
            err = (approx - ref)
            num[label] += float((err ** 2).sum())
            sums[label].extend((err.norm(dim=1)[keep] / norms[keep]).tolist())
    print(f"{len(sums['fp8'])} rows in {args.blocks} blocks; FP8 weight_scale {ours.scale:g}\n")
    print("| format | bytes/row | mean row rel L2 | max row rel L2 | global rel L2 |")
    print("|---|---:|---:|---:|---:|")
    for label, size in (("FP8 E4M3 + table scale (ours)", WIDTH),
                        ("int6, FP16 scale per 32", WIDTH * 6 // 8 + WIDTH // 32 * 2)):
        key = "fp8" if label.startswith("FP8") else "int6"
        rel = sums[key]
        print(f"| {label} | {size} | {sum(rel) / len(rel):.4f} | {max(rel):.4f} "
              f"| {(num[key] / den) ** 0.5:.4f} |")


if __name__ == "__main__":
    main()
