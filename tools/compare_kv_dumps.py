#!/usr/bin/env python3
"""Compare two KV page-dump directories over the committed region of each plane.

A page dump carries `committed_columns` plus a per-plane layout: every plane has its own offset, head
count, head payload size and leading extent. The committed KV of a plane is `committed_columns` columns
of EACH head, at the plane's own offset - not a prefix fraction of the page buffer, because planes are
padded and have different widths.

A raw byte comparison reports about a hundred thousand differing bytes even when the valid KV is
identical, since everything past the committed columns is whatever the pool last held. That produced
two wrong conclusions during the 125B investigation, so the scoping is a tool rather than a habit.

    compare_kv_dumps.py DIR_A DIR_B [--json]

Dumps pair by (frontier, page) and must also agree on phase, lane, rank and physical page: two dumps
can share a frontier and page while belonging to different sequences.
"""
import argparse
import json
import pathlib
import sys

DTYPE_BYTES = {0: 2, 1: 2, 2: 4, 3: 1, 4: 1, 5: 1, 6: 4, 7: 1}


def committed_spans(metadata):
    """Byte spans holding valid KV, derived from the plane layout the dump records."""
    committed = int(metadata.get("committed_columns") or 0)
    if not committed:
        return []
    spans = []
    for plane in metadata.get("planes") or []:
        offset = int(plane.get("offset") or 0)
        heads = int(plane.get("head_extent") or 1)
        head_bytes = int(plane.get("head_payload_bytes") or 0)
        leading = int(plane.get("leading_extent") or 0)
        rows = int(plane.get("tokens_per_row") or 1)
        element = DTYPE_BYTES.get(int(plane.get("dtype") or 0), 2)
        column_bytes = leading * element * rows
        if not head_bytes or not column_bytes:
            continue
        valid = min(committed * column_bytes, head_bytes)
        for head in range(heads):
            start = offset + head * head_bytes
            spans.append((start, start + valid))
    return spans


def load(directory):
    dumps = {}
    for metadata_path in sorted(pathlib.Path(directory).glob("*.json")):
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, ValueError):
            continue
        payload = metadata_path.with_suffix(".bin")
        if not payload.is_file():
            continue
        dumps.setdefault((metadata.get("frontier"), metadata.get("page")), []).append(
            (metadata_path.stem, metadata, payload))
    return dumps


def compare(payload_a, payload_b, metadata):
    a = payload_a.read_bytes()
    b = payload_b.read_bytes()
    if len(a) != len(b):
        return None
    spans = committed_spans(metadata)
    inside = outside = valid = 0
    covered = set()
    for start, end in spans:
        valid += end - start
        for i in range(start, min(end, len(a))):
            covered.add(i)
            if a[i] != b[i]:
                inside += 1
    for i in range(len(a)):
        if i not in covered and a[i] != b[i]:
            outside += 1
    return {"committed_bytes": valid, "differing_committed": inside,
            "differing_uncommitted": outside}


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    dumps_a = load(args.a)
    dumps_b = load(args.b)
    shared = sorted(set(dumps_a) & set(dumps_b))
    pairs = []
    for key in shared:
        for stem_a, meta_a, payload_a in dumps_a[key]:
            for stem_b, meta_b, payload_b in dumps_b[key]:
                if any(meta_a.get(f) != meta_b.get(f) for f in ("phase", "lane", "rank", "physical")):
                    continue
                result = compare(payload_a, payload_b, meta_a)
                if result is not None:
                    pairs.append({"frontier": key[0], "page": key[1], "a": stem_a, "b": stem_b,
                                  **result})

    if args.json:
        print(json.dumps({"pairs": pairs}, indent=2))
        return 0
    if not pairs:
        print(f"no comparable pairs ({len(dumps_a)} and {len(dumps_b)} dumps, "
              f"{len(shared)} shared frontier/page keys)")
        return 0
    clean = sum(1 for p in pairs if p["differing_committed"] == 0)
    print(f"comparable pairs: {len(pairs)}  committed-identical: {clean}  "
          f"committed-differing: {len(pairs) - clean}")
    for p in pairs:
        if p["differing_committed"]:
            print(f"  frontier {p['frontier']} page {p['page']} {p['a']} vs {p['b']}: "
                  f"{p['differing_committed']} of {p['committed_bytes']} committed bytes differ")
    if clean == len(pairs):
        print("  committed KV identical in every comparable pair; any other byte differences are "
              "in the uncommitted tail and are not KV for the sequence")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
