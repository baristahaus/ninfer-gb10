#!/usr/bin/env python3
"""Served token-logprobs invariant checker (GB10 verification campaign, Block A2).

Reads a JSONL of request records; each line is an object with:
  name          label for the record
  logprobs      bool, whether the request asked for logprobs
  top_logprobs  int, the top_logprobs value requested (0 = absent)
  response      the raw /v1/chat/completions response body

Checks:
  1. exactly one logprob report per completion token
  2. every logprob is <= 0
  3. each top_logprobs list is sorted descending
  4. the sampled token equals top-1 at every position
  5. for records with top_logprobs >= 20: the top-20 probabilities sum to <= 1 at
     every position

Prints one PASS/FAIL line per check with the first few violations and exits
non-zero if any check fails.
"""
import json
import math
import sys

LIMIT = 5  # violations shown per check


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <responses.jsonl>", file=sys.stderr)
        return 2
    path = sys.argv[1]

    checks = {
        "one report per completion token": [],
        "all logprobs <= 0": [],
        "top lists sorted descending": [],
        "sampled token == top-1": [],
        "top-20 probabilities sum <= 1": [],
    }
    n_records = 0
    n_checked = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_records += 1
            name = rec.get("name", f"#{n_records}")
            if not rec.get("logprobs"):
                continue
            n_checked += 1
            resp = rec["response"]
            choice = resp["choices"][0]
            usage = resp.get("usage", {})
            lp = choice.get("logprobs")
            if lp is None:
                checks["one report per completion token"].append(
                    f"{name}: logprobs missing although requested")
                continue
            content = lp.get("content") or []
            n_tok = usage.get("completion_tokens")
            if n_tok is not None and len(content) != n_tok:
                checks["one report per completion token"].append(
                    f"{name}: {len(content)} reports for {n_tok} completion tokens")
            want20 = rec.get("top_logprobs", 0) >= 20
            for i, entry in enumerate(content):
                if entry["logprob"] > 0:
                    checks["all logprobs <= 0"].append(
                        f"{name}[{i}]: logprob {entry['logprob']:.6f} > 0")
                tops = entry.get("top_logprobs") or []
                vals = [t["logprob"] for t in tops]
                for j in range(1, len(vals)):
                    if vals[j] > vals[j - 1]:
                        checks["top lists sorted descending"].append(
                            f"{name}[{i}]: top[{j}] {vals[j]:.4f} above top[{j - 1}] "
                            f"{vals[j - 1]:.4f}")
                        break
                if tops and tops[0]["token"] != entry["token"]:
                    checks["sampled token == top-1"].append(
                        f"{name}[{i}]: sampled {entry['token']!r} != top-1 "
                        f"{tops[0]['token']!r}")
                if want20:
                    s = sum(math.exp(t["logprob"]) for t in tops[:20])
                    if s > 1.0 + 1e-9:
                        checks["top-20 probabilities sum <= 1"].append(
                            f"{name}[{i}]: sum {s:.6f} > 1")

    rc = 0
    for label, violations in checks.items():
        if violations:
            rc = 1
            print(f"FAIL  {label}: {len(violations)} violation(s)")
            for v in violations[:LIMIT]:
                print(f"      {v}")
        else:
            print(f"PASS  {label}")
    print(f"(records: {n_records}, with logprobs checked: {n_checked})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
