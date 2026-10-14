#!/usr/bin/env python3
"""Annotate a NInfer v3 artifact's directory metadata with an expert tier plan (strategy SP-4).

Run: python3 annotate_tier_plan.py INPUT.ninfer OUTPUT.ninfer PLAN.json

PLAN.json names routed expert banks by object id and assigns every expert of each bank to the
device (hot), pinned-host (warm), or file-backed (cold) tier:

    {
      "banks": [
        {"object": "text/layers/0/moe/routed_gate_up",
         "experts": 256, "rows_per_expert": 1024,
         "tiers": {"hot": [0, 1], "warm": [], "cold": [2, 3]}}
      ],
      "activation_profiles": {"<object id>": [count, ...]}   # optional, one count per expert
    }

Ranked mode (strategy SP-4 phase 2b): when the plan carries "residency" (a hot-expert
budget) and per-bank activation profiles, the tool ranks each layer's experts by measured
activation count, keeps the top budget hot, and PERMUTES the artifact so the hot tier is
the positional expert prefix [0, budget): every layer's routed_gate_up and routed_down banks
have their per-expert row blocks reordered identically per plane (codes, high, scales), and
the layer's router_shared_gate rows [0, experts) are permuted to match. The model is
permutation-symmetric, so generation is bit-identical; the emitted tier plan, activation
profiles, and the <output>.permutation.json sidecar all use the new expert numbering:

    {
      "version": 1,
      "activation_profiles": {"<gate_up id>": [count, ...], ...},
      "residency": {"hot_experts": 204}
    }

Version 1 plans are declarative: the loader validates them and reports them in the load
summary. Since SP-4 phase 2b the tiered materialization consumes the hot/warm lists directly
and per expert (arbitrary scattered hot sets, no permutation required), so the ranked-mode
permutation is an optional artifact-level optimization, not a correctness requirement. The
container's payload bytes are preserved exactly except for the tiered permutation; only file
zero is rewritten (its payload region shifts when the directory JSON changes length) and
continuation shards are copied byte-for-byte.
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


def plane_row_bytes(fmt: str, cols: int) -> tuple[int, int, int]:
    """Per-row (codes, high, scales) byte widths for the row-split quantized layouts."""
    if fmt == "q4_g64_fp16":
        return cols // 2, 0, cols // 64 * 2
    if fmt == "q5_g64_fp16":
        return cols // 2, cols // 64 * 8, cols // 64 * 2
    if fmt == "q6_g64_fp16":
        return cols // 2, cols // 64 * 16, cols // 64 * 2
    if fmt == "q8_g32_fp16":
        return cols, 0, cols // 32 * 2
    raise SystemExit(f"unsupported tiered bank format: {fmt}")


def rank_residency(counts: list, hot_experts: int) -> tuple[list, list]:
    """The hot tier is the top `hot_experts` experts by activation count; ties break by
    expert id so the ranking is deterministic. Both tiers keep expert id order."""
    if not 0 < hot_experts < len(counts):
        raise SystemExit(f"hot budget {hot_experts} must be a strict expert prefix of {len(counts)}")
    order = sorted(range(len(counts)), key=lambda e: (-counts[e], e))
    return sorted(order[:hot_experts]), sorted(order[hot_experts:])


def permute_row_split(payload: bytes, fmt: str, shape: list, rows_per_expert: int,
                      new_order: list) -> bytes:
    """Reorder one row-split bank payload by whole-expert blocks, per plane. The layout is
    [codes | high | scales] with each plane row-major over all rows, so every expert owns one
    contiguous block per plane."""
    rows, cols = shape
    code_rb, high_rb, scale_rb = plane_row_bytes(fmt, cols)
    code_plane, high_plane = rows * code_rb, rows * high_rb
    if code_plane + high_plane + rows * scale_rb != len(payload):
        raise SystemExit(f"plane layout does not cover the bank payload ({len(payload)} bytes)")
    out = bytearray(len(payload))
    for new_e, old_e in enumerate(new_order):
        old_base, new_base = old_e * rows_per_expert, new_e * rows_per_expert
        size = rows_per_expert * code_rb
        out[new_base * code_rb:new_base * code_rb + size] = payload[old_base * code_rb:old_base * code_rb + size]
        if high_rb:
            size = rows_per_expert * high_rb
            src = code_plane + old_base * high_rb
            dst = code_plane + new_base * high_rb
            out[dst:dst + size] = payload[src:src + size]
        size = rows_per_expert * scale_rb
        src = code_plane + high_plane + old_base * scale_rb
        dst = code_plane + high_plane + new_base * scale_rb
        out[dst:dst + size] = payload[src:src + size]
    return bytes(out)


def permute_router(payload: bytes, row_bytes: int, experts: int, new_order: list) -> bytes:
    """Reorder a bf16 router bank's expert rows [0, experts); trailing rows stay in place."""
    if len(payload) % row_bytes:
        raise SystemExit("router payload is not a whole number of rows")
    if len(payload) < (experts + 1) * row_bytes:
        raise SystemExit("router payload is shorter than its expert rows")
    out = bytearray(len(payload))
    for new_e, old_e in enumerate(new_order):
        out[new_e * row_bytes:(new_e + 1) * row_bytes] = payload[old_e * row_bytes:(old_e + 1) * row_bytes]
    tail = experts * row_bytes
    out[tail:] = payload[tail:]
    return bytes(out)


def derive_ranked_plan(plan_raw: dict, by_id: dict) -> tuple[dict, dict, list]:
    """Build the positional tier plan for a profile-ranked residency, and return the payload
    permutation specs: {relative payload offset: (object id, kind, format, shape,
    rows_per_expert, new_order)} plus the sidecar mapping {layer: old_to_new}."""
    profiles = plan_raw.get("activation_profiles")
    residency = plan_raw.get("residency")
    if not isinstance(residency, dict) or "hot_experts" not in residency:
        raise SystemExit("residency must name hot_experts")
    hot_experts = residency["hot_experts"]
    if not isinstance(hot_experts, int) or hot_experts <= 0:
        raise SystemExit("residency.hot_experts must be a positive integer")
    if not isinstance(profiles, dict) or not profiles:
        raise SystemExit("ranked residency requires activation_profiles keyed by object id")
    gate_up_ids = sorted(k for k in profiles if k.endswith("/moe/routed_gate_up"))
    if not gate_up_ids:
        raise SystemExit("activation_profiles name no routed_gate_up banks")

    banks, specs, sidecar = [], {}, {}
    hit_total, selection_total = 0, 0
    for gate_up_id in gate_up_ids:
        counts = profiles[gate_up_id]
        layer = gate_up_id[: -len("/moe/routed_gate_up")]
        down_id, router_id = layer + "/moe/routed_down", layer + "/moe/router_shared_gate"
        gate_up, down, router = by_id.get(gate_up_id), by_id.get(down_id), by_id.get(router_id)
        if gate_up is None or down is None or router is None:
            raise SystemExit(f"{gate_up_id}: layer is missing its bank or router object")
        if down_id in profiles and profiles[down_id] != counts:
            raise SystemExit(f"{gate_up_id}: gate_up and down activation profiles differ")
        experts = router["shape"][0] - 1
        if len(counts) != experts:
            raise SystemExit(
                f"{gate_up_id}: profile has {len(counts)} counts, router names {experts} experts")
        rows_per_expert, rem = divmod(gate_up["shape"][0], experts)
        down_rows_per_expert, down_rem = divmod(down["shape"][0], experts)
        if rem or down_rem:
            raise SystemExit(f"{gate_up_id}: bank rows do not divide into experts")
        hot, warm = rank_residency(counts, hot_experts)
        new_order = hot + warm  # hot prefix, then warm suffix, expert order in each
        old_to_new = [0] * experts
        for new_e, old_e in enumerate(new_order):
            old_to_new[old_e] = new_e

        for obj in (gate_up, down):
            code_rb, high_rb, scale_rb = plane_row_bytes(obj["format"], obj["shape"][1])
            if obj["shape"][0] * (code_rb + high_rb + scale_rb) != obj["bytes"]:
                raise SystemExit(f"{obj['id']}: plane layout does not cover the object bytes")
            specs[obj["offset"]] = (obj["id"], "bank", obj["format"], obj["shape"],
                                   obj["shape"][0] // experts, new_order)
        if router["format"] != "bf16" or router["shape"][0] != experts + 1:
            raise SystemExit(f"{router_id}: router is not a bf16 [experts+1, hidden] bank")
        if router["bytes"] != router["shape"][0] * router["shape"][1] * 2:
            raise SystemExit(f"{router_id}: router bytes do not match its shape")
        specs[router["offset"]] = (router_id, "router", router["format"], router["shape"], 1,
                                   new_order)

        permuted_counts = [counts[old] for old in new_order]
        hot_ids = list(range(hot_experts))
        warm_ids = list(range(hot_experts, experts))
        banks.append({"object": gate_up_id, "experts": experts, "rows_per_expert": rows_per_expert,
                      "hot": hot_ids, "warm": warm_ids, "cold": [],
                      "activation_profile": permuted_counts})
        banks.append({"object": down_id, "experts": experts,
                      "rows_per_expert": down_rows_per_expert,
                      "hot": hot_ids, "warm": warm_ids, "cold": [],
                      "activation_profile": list(permuted_counts)})
        hit_total += sum(counts[e] for e in hot)
        selection_total += sum(counts)
        sidecar[layer] = old_to_new

    hit_rate = hit_total / selection_total if selection_total else 0.0
    print(f"ranked residency: {hot_experts}/{experts} experts per layer, "
          f"measured hot-set hit rate {hit_rate * 100:.2f}% "
          f"({hit_total}/{selection_total} selections)", flush=True)
    return {"version": 1, "banks": banks}, specs, {"hot_experts": hot_experts,
                                                    "hit_rate": hit_rate, "layers": sidecar}


def annotate(input_path: Path, output_path: Path, plan_path: Path) -> None:
    if output_path.exists():
        raise SystemExit(f"output already exists: {output_path}")
    plan_raw = json.loads(plan_path.read_text())

    magic, _, artifact_id, json_text, old_payload_start = read_entry_header(input_path)
    directory = json.loads(json_text)
    objects = directory.get("objects") or []
    by_id = {obj.get("id"): obj for obj in objects if isinstance(obj, dict)}

    # Ranked mode (SP-4 phase 2b): the tiers come from the measured profiles and the payload
    # is permuted so the hot tier is the positional prefix. Explicit mode keeps the input's
    # declared tiers and byte-identical payloads.
    specs, sidecar = {}, {}
    if plan_raw.get("residency") is not None:
        plan, specs, sidecar = derive_ranked_plan(plan_raw, by_id)
    else:
        plan = build_tier_plan(plan_raw)

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
        pending = sorted(specs)
        pending_index = 0
        with input_path.open("rb") as source:
            source.seek(old_payload_start)
            while copied < first_payload_bytes:
                next_stop = first_payload_bytes
                if pending_index < len(pending):
                    next_stop = min(next_stop, pending[pending_index])
                if next_stop > copied:
                    chunk = source.read(min(IO_CHUNK, next_stop - copied))
                    if not chunk:
                        raise SystemExit("entry payload ended early")
                    out.write(chunk)
                    copied += len(chunk)
                if pending_index < len(pending) and copied == pending[pending_index]:
                    obj_id, kind, fmt, shape, unit_rows, new_order = specs[copied]
                    length = by_id[obj_id]["bytes"]
                    payload = source.read(length)
                    if len(payload) != length:
                        raise SystemExit(f"{obj_id}: entry payload ended early")
                    if kind == "bank":
                        permuted = permute_row_split(payload, fmt, shape, unit_rows, new_order)
                    else:
                        permuted = permute_router(payload, shape[1] * 2, shape[0] - 1, new_order)
                    out.write(permuted)
                    copied += length
                    pending_index += 1

    for shard in continuation_files(input_path):
        target = output_path.parent / shard.name
        with shard.open("rb") as source, target.open("wb") as sink:
            while chunk := source.read(IO_CHUNK):
                sink.write(chunk)

    if sidecar.get("layers"):
        sidecar_path = output_path.with_suffix(".permutation.json")
        sidecar_path.write_text(json.dumps(sidecar, indent=1))
        print(f"permutation sidecar: {sidecar_path}", flush=True)

    print(
        f"annotated {input_path} -> {output_path}: "
        f"{len(plan['banks'])} banks planned, "
        f"payload shifted {new_payload_start - old_payload_start:+d} bytes"
        + (", payload permuted to the ranked expert order" if sidecar else ""),
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
