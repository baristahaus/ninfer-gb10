#!/usr/bin/env python3
"""Compare two `ninfer-perplexity --token-scores` runs by position within each window.

Both runs must score the same corpus with the same context and stride, so every scored token
has the same (stream, window, position) and token id in each. Every window starts from empty
state, so a token's offset from its window's first input token is how far the recurrent state
and KV have been carried when it is scored. A candidate whose quantized weights feed the FP32
GDN state accumulates error with that offset; the per-bin mean NLL difference and its
least-squares slope show whether it does.
"""

import argparse
import json
import math
from pathlib import Path


def load(directory: Path) -> dict:
    report = json.loads((directory / "report.json").read_text())
    begins = {}
    for stream in report["streams"]:
        for window in stream["windows"]:
            begins[stream["id"], window["index"]] = window["input_begin"]
    scores = {}
    with (directory / "token_scores.jsonl").open() as lines:
        for line in lines:
            row = json.loads(line)
            key = (row["stream"], row["window"], row["position"])
            scores[key] = (row["token"], row["logprob"], row["position"] - begins[key[:2]])
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="output directory of the reference run")
    parser.add_argument("candidate", type=Path, help="output directory of the compared run")
    parser.add_argument("--bin", type=int, default=4096, help="offset bin width in tokens")
    args = parser.parse_args()

    baseline = load(args.baseline)
    candidate = load(args.candidate)
    if baseline.keys() != candidate.keys():
        raise SystemExit("the runs scored different token positions")

    bins: dict[int, list[float]] = {}
    sums = [0.0, 0.0, 0.0, 0.0, 0.0]  # n, sum x, sum y, sum xx, sum xy
    for key, (token, base_logprob, offset) in baseline.items():
        other_token, other_logprob, _ = candidate[key]
        if other_token != token:
            raise SystemExit(f"token ids differ at {key}")
        delta = base_logprob - other_logprob  # candidate NLL minus baseline NLL
        bins.setdefault(offset // args.bin, []).append(delta)
        for i, value in enumerate((1.0, offset, delta, offset * offset, offset * delta)):
            sums[i] += value

    n, sx, sy, sxx, sxy = sums
    print(f"{'offset':>17} {'tokens':>9} {'mean dNLL':>11} {'mean |dNLL|':>12}")
    for index in sorted(bins):
        deltas = bins[index]
        mean = sum(deltas) / len(deltas)
        mean_abs = sum(abs(d) for d in deltas) / len(deltas)
        begin, end = index * args.bin, (index + 1) * args.bin
        print(f"{begin:>8}-{end:<8} {len(deltas):>9} {mean:>+11.5f} {mean_abs:>12.5f}")
    denominator = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denominator if denominator > 0 else math.nan
    print(f"overall mean dNLL {sy / n:+.5f} over {int(n)} tokens "
          f"(PPL ratio {math.exp(sy / n):.4f}); slope {slope * 10000:+.6f} nats per 10K tokens")


if __name__ == "__main__":
    main()
