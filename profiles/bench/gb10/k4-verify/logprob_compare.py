#!/usr/bin/env python3
"""Compare the three logprobs runs: first divergent token + top-2 logprobs there.

Usage: logprob_compare.py <k4a-c4.json> <k4a-c1.json> <k4b-c4.json>
"""
import json
import sys


def load(path, label, expected_len, expected_sha):
    with open(path) as f:
        d = json.load(f)
    toks = [e["tok"] for e in d["logprobs"]]
    match = "MATCH" if (d["text_len"] == expected_len and d["text_sha"] == expected_sha) \
        else "MISMATCH"
    print(f"{label}: len={d['text_len']} sha={d['text_sha']} ({match} vs plain run "
          f"{expected_len}/{expected_sha}) positions={len(toks)} finish={d['finish']}")
    return d, toks


def first_div(ta, tb):
    for i in range(min(len(ta), len(tb))):
        if ta[i] != tb[i]:
            return i
    return min(len(ta), len(tb)) if len(ta) != len(tb) else None


def show(d, i, label):
    e = d["logprobs"][i]
    top2 = e["top2"] or []
    gap = None
    if len(top2) >= 2:
        gap = round(top2[0]["logprob"] - top2[1]["logprob"], 4)
    print(f"  {label}: pos {i} chosen={e['tok']!r} logprob={e['logprob']:.6f} "
          f"top2={[ (t['tok'], round(t['logprob'], 4)) for t in top2 ]} gap={gap}")


def main():
    pairs = [
        (sys.argv[1], "K4a C4", 1234, "847314b2"),
        (sys.argv[2], "K4a C1", 1212, "9f1893c5"),
        (sys.argv[3], "K4b C4", 1172, "86bd9884"),
    ]
    runs = {}
    for path, label, elen, esha in pairs:
        d, toks = load(path, label, elen, esha)
        runs[label] = (d, toks)

    for la, lb in (("K4a C4", "K4a C1"), ("K4a C4", "K4b C4"), ("K4a C1", "K4b C4")):
        da, ta = runs[la]
        db, tb = runs[lb]
        i = first_div(ta, tb)
        if i is None:
            print(f"\n{la} vs {lb}: identical token sequences")
            continue
        print(f"\n{la} vs {lb}: first divergence at position {i} of {min(len(ta), len(tb))} "
              f"(context: {ta[max(0, i - 3):i]!r})")
        show(da, i, la)
        show(db, i, lb)
        # does each run's chosen token appear in the other's top-2?
        for (ds, ts, ls), (do, to, lo) in (
                ((da, ta, la), (db, tb, lb)), ((db, tb, lb), (da, ta, la))):
            others = [t["tok"] for t in (ds["logprobs"][i]["top2"] or [])]
            other_tok = to[i]
            mark = "in" if other_tok in others else "NOT in"
            print(f"    {lo} chooses {other_tok!r}: {mark} {ls}'s top-2 {others}")


if __name__ == "__main__":
    main()
