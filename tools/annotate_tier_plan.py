#!/usr/bin/env python3
"""Annotate a NInfer v3 artifact's directory metadata with an expert tier plan (strategy SP-4).

Run: python3 annotate_tier_plan.py INPUT.ninfer OUTPUT.ninfer PLAN.json

PLAN.json names routed expert banks by object id and assigns every expert of each bank to the
device (hot), pinned-host (warm), or file-backed (cold) tier:

    {
      "banks": [
        {"object": "text/layers/0/mlp.experts.gate_up_proj.weight",
         "experts": 256, "rows_per_expert": 1024,
         "tiers": {"hot": [0, 1], "warm": [], "cold": [2, 3]}}
      ],
      "activation_profiles": {"<object id>": [count, ...]}   # optional, one count per expert
    }

Version 1 plans are declarative: the loader validates them and reports them in the load
summary; the materialization honors them only when the tiered execution path consumes them.
The container's payload bytes are preserved exactly; only file zero is rewritten (its payload
region shifts when the directory JSON changes length) and continuation shards are copied
byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

HEADER = struct.Struct("<8sQ16s")
MAGIC = b"NINFER\x00\x03"
PART_MAGIC = b"NINFER\x00\x03P"[:8]  # placeholder; continuation magic read per file
ALIGNMENT = 4096
IO_CHUNK = 8 * 1024 * 1024


def align_up(value: int, amount: int = ALIGNMENT) -> int:
    return (value + amount - 1) // amount * amount


def read_entry_header(path: Path) -> tuple[bytes, int, bytes, int, int]:
    """Returns (magic, json_bytes, artifact_id, json, payload_start) for an entry file."""
    with path.open("rb") as stream:
        header = stream.read(HEADER.size)
        if len(header) != HEADER.size:
            raise SystemExit(f"{path}: truncated entry header")
        magic, json_bytes, artifact_id = HEADER.unpack(header)
        if magic != MAGIC:
            raise SystemExit(f"{path}: not a NInfer v3 entry file")
        json_text = stream.read(json_bytes)
        if len(json_text) != json_bytes:
            raise SystemExit(f"{path}: truncated directory JSON")
        return magic, json_bytes, artifact_id, json_text, align_up(HEADER.size + json_bytes)


def continuation_files(path: Path) -> list[Path]:
    """Continuation shard paths, read from the directory's files[1:] records."""
    _, _, _, json_text, _ = read_entry_header(path)
    directory = json.loads(json_text)
    shards = []
    for index, file in enumerate(directory.get("files", [])):
        record_path = file.get("path")
        if index > 0:
            if not record_path:
                raise SystemExit("continuation shard is missing its path record")
            shards.append(path.parent / record_path)
    return shards


def build_tier_plan(plan: dict) -> dict:
    if plan.get("version") not in (None, 1):
        raise SystemExit("tier plan version must be 1")
    banks = plan.get("banks")
    if not isinstance(banks, list) or not banks:
        raise SystemExit("tier plan must declare a nonempty banks array")
    profiles = plan.get("activation_profiles", {})
    if not isinstance(profiles, dict):
        raise SystemExit("activation_profiles must be an object keyed by object id")
    out_banks = []
    for bank in banks:
        if not isinstance(bank, dict):
            raise SystemExit("each tier plan bank must be an object")
        object_id = bank.get("object")
        if not isinstance(object_id, str) or not object_id:
            raise SystemExit("tier plan bank is missing its object id")
        experts = bank.get("experts")
        rows = bank.get("rows_per_expert")
        tiers = bank.get("tiers", {})
        if not isinstance(experts, int) or experts <= 0:
            raise SystemExit(f"{object_id}: experts must be a positive integer")
        if not isinstance(rows, int) or rows <= 0:
            raise SystemExit(f"{object_id}: rows_per_expert must be a positive integer")
        if not isinstance(tiers, dict):
            raise SystemExit(f"{object_id}: tiers must be an object")
        hot = tiers.get("hot", [])
        warm = tiers.get("warm", [])
        cold = tiers.get("cold", [])
        for name, ids in (("hot", hot), ("warm", warm), ("cold", cold)):
            if not isinstance(ids, list) or any(
                not isinstance(i, int) or i < 0 or i >= experts for i in ids
            ):
                raise SystemExit(f"{object_id}: tier {name} has invalid expert ids")
        union = sorted([*hot, *warm, *cold])
        if union != list(range(experts)):
            raise SystemExit(f"{object_id}: tiers must cover every expert exactly once")
        entry = {
            "object": object_id,
            "experts": experts,
            "rows_per_expert": rows,
            "hot": hot,
            "warm": warm,
            "cold": cold,
        }
        profile = profiles.get(object_id)
        if profile is not None:
            if not isinstance(profile, list) or len(profile) != experts or any(
                not isinstance(c, int) or c < 0 for c in profile
            ):
                raise SystemExit(
                    f"{object_id}: activation_profile must have one non-negative count per expert"
                )
            entry["activation_profile"] = profile
        out_banks.append(entry)
    return {"version": 1, "banks": out_banks}


def annotate(input_path: Path, output_path: Path, plan_path: Path) -> None:
    if output_path.exists():
        raise SystemExit(f"output already exists: {output_path}")
    plan = build_tier_plan(json.loads(plan_path.read_text()))

    magic, _, artifact_id, json_text, old_payload_start = read_entry_header(input_path)
    directory = json.loads(json_text)
    objects = directory.get("objects") or []
    by_id = {obj.get("id"): obj for obj in objects if isinstance(obj, dict)}

    # The plan's banks must name real tensor objects, and each bank's row count must be exactly
    # experts * rows_per_expert so the runtime's semantic check passes.
    for bank in plan["banks"]:
        obj = by_id.get(bank["object"])
        if obj is None:
            raise SystemExit(f"tier plan names an absent object: {bank['object']}")
        if obj.get("kind") != "tensor":
            raise SystemExit(f"tier plan bank is not a tensor: {bank['object']}")
        shape = obj.get("shape") or []
        expected = bank["experts"] * bank["rows_per_expert"]
        if not shape or shape[0] != expected:
            raise SystemExit(
                f"{bank['object']}: shape {shape} does not match "
                f"experts * rows_per_expert = {expected}"
            )

    metadata = directory.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        directory["metadata"] = metadata
    metadata["tier_plan"] = plan

    encoded = json.dumps(directory, ensure_ascii=False, separators=(",", ": ")).encode("utf-8")
    new_payload_start = align_up(HEADER.size + len(encoded))

    files = directory.get("files") or []
    first_payload_bytes = int(files[0]["payload_bytes"])
    old_size = input_path.stat().st_size
    old_payload_bytes = old_size - old_payload_start
    if old_payload_bytes != first_payload_bytes:
        raise SystemExit(
            f"entry file length does not match files[0].payload_bytes: "
            f"{old_payload_bytes} != {first_payload_bytes}"
        )

    with output_path.open("wb") as out:
        out.write(HEADER.pack(magic, len(encoded), artifact_id))
        out.write(encoded)
        pad = new_payload_start - HEADER.size - len(encoded)
        out.write(b"\x00" * pad)
        copied = 0
        with input_path.open("rb") as source:
            source.seek(old_payload_start)
            while copied < first_payload_bytes:
                chunk = source.read(min(IO_CHUNK, first_payload_bytes - copied))
                if not chunk:
                    raise SystemExit("entry payload ended early")
                out.write(chunk)
                copied += len(chunk)

    for shard in continuation_files(input_path):
        target = output_path.parent / shard.name
        with shard.open("rb") as source, target.open("wb") as sink:
            while chunk := source.read(IO_CHUNK):
                sink.write(chunk)

    print(
        f"annotated {input_path} -> {output_path}: "
        f"{len(plan['banks'])} banks planned, "
        f"payload shifted {new_payload_start - old_payload_start:+d} bytes",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    annotate(args.input, args.output, args.plan)


if __name__ == "__main__":
    main()
