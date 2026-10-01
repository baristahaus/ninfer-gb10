#!/usr/bin/env python3
"""Compare the C1 vs C4 route-diagnostic captures.

Usage: compare_routediag.py <c1-logits-dir> <c4-logits-dir> <c1-state-dir> <c4-state-dir>

Logits: per-round (serial) captures with .json metadata (route, positions,
batch, width, vocab, kv_table_rows) and a .bf16 payload. Rounds are compared
in serial order; within the first differing round, rows are aligned by token
position and the first differing position with its max |delta| is reported.

State: one capture per config at the prefill frontier: .json (route, lane,
physical_slot, ledger, layers) plus per-layer .conv.<L>.bf16 and
.recurrent.<L>.fp32 files, compared file by file.
"""
import glob
import json
import os
import struct
import sys



def main() -> None:
    c1l, c4l, c1s, c4s = sys.argv[1:5]

    def serials(d):
        return sorted(int(os.path.basename(p).split(".")[0])
                      for p in glob.glob(os.path.join(d, "*.json")))
    s1, s4 = serials(c1l), serials(c4l)
    print(f"logits captures: C1={len(s1)} C4={len(s4)} (serials 0..{max(s1)}/{max(s4)})")
    common = [i for i in s1 if i in s4]
    n_diff = sum(open(os.path.join(c1l, f"{i}.bf16"), "rb").read() !=
                 open(os.path.join(c4l, f"{i}.bf16"), "rb").read() for i in common)
    first_diff = None
    for i in common:
        j1 = json.load(open(os.path.join(c1l, f"{i}.json")))
        j4 = json.load(open(os.path.join(c4l, f"{i}.json")))
        b1 = open(os.path.join(c1l, f"{i}.bf16"), "rb").read()
        b4 = open(os.path.join(c4l, f"{i}.bf16"), "rb").read()
        meta_same = all(j1[k] == j4[k] for k in
                        ("route", "positions", "batch", "width", "vocab",
                         "kv_table_rows"))
        if b1 == b4 and meta_same:
            continue
        first_diff = i
        mdiff = [k for k in ("route", "positions", "batch", "width", "vocab",
                             "kv_table_rows") if j1[k] != j4[k]]
        print(f"first differing round serial={i}")
        print(f"  C1 meta: route={j1['route']} positions={j1['positions']} "
              f"batch={j1['batch']} width={j1['width']} vocab={j1['vocab']} "
              f"kv_table_rows={j1['kv_table_rows']}")
        print(f"  C4 meta: route={j4['route']} positions={j4['positions']} "
              f"batch={j4['batch']} width={j4['width']} vocab={j4['vocab']} "
              f"kv_table_rows={j4['kv_table_rows']}")
        print(f"  metadata fields differing: {mdiff or 'none'}; payload identical: {b1 == b4}")
        if b1 != b4 and len(b1) == len(b4) and j1["vocab"] == j4["vocab"]:
            v = j1["vocab"]
            n = j1["width"] * j1["batch"]
            if len(b1) == v * n * 2:
                # payload is flat [vocab, n]: value (v_, c) at flat index v_ * n + c
                stat = {c: [0, 0.0, None] for c in range(n)}
                for v_ in range(v):
                    for c in range(n):
                        o = (v_ * n + c) * 2
                        u1 = struct.unpack_from("<H", b1, o)[0]
                        u4 = struct.unpack_from("<H", b4, o)[0]
                        if u1 != u4:
                            s = stat[c]
                            s[0] += 1
                            if s[2] is None:
                                s[2] = v_
                            f1 = struct.unpack("<f", struct.pack("<I", u1 << 16))[0]
                            f4 = struct.unpack("<f", struct.pack("<I", u4 << 16))[0]
                            s[1] = max(s[1], abs(f1 - f4))
                for c in range(n):
                    ndiff, worst, first_v = stat[c]
                    pos = j1["positions"][c] if c < len(j1["positions"]) else "?"
                    print(f"  position {pos} (column {c}): {ndiff}/{v} logits differ"
                          + (f", first vocab index={first_v}, max|delta|={worst:.6g}"
                             if ndiff else ""))
        break
    if first_diff is None:
        print("all common rounds bitwise identical; serial sets:",
              "identical" if s1 == s4 else f"differ (C1 {len(s1)} vs C4 {len(s4)})")
    print(f"payloads differing: {n_diff} of {len(common)} common serials; "
          f"capture counts C1={len(s1)} C4={len(s4)}")

    # State comparison
    def state_files(d):
        if not os.path.isdir(d):
            return None
        return {os.path.basename(p): p for p in glob.glob(os.path.join(d, "*"))}

    f1, f4 = state_files(c1s), state_files(c4s)
    if f1 is None or f4 is None:
        print("state: capture missing on one side")
        return
    j1 = json.load(open(f1["0.json"]))
    j4 = json.load(open(f4["0.json"]))
    print(f"state: C1 route={j1['route']} lane={j1['lane']} slot={j1['physical_slot']} "
          f"frontier={j1['frontier']} ledger={j1['ledger']} layers={j1['layers']}")
    print(f"state: C4 route={j4['route']} lane={j4['lane']} slot={j4['physical_slot']} "
          f"frontier={j4['frontier']} ledger={j4['ledger']} layers={j4['layers']}")
    names = sorted(set(f1) | set(f4))
    diff = []
    for name in names:
        if name not in f1 or name not in f4:
            diff.append(f"{name} (missing one side)")
            continue
        if open(f1[name], "rb").read() != open(f4[name], "rb").read():
            diff.append(name)
    print(f"state files compared: {len(names)}; differing: {diff or 'NONE - bitwise identical'}")


if __name__ == "__main__":
    main()
