#!/usr/bin/env python3
"""Per-request decode attribution from ninfer-serve request logs.

Usage: request_log_summary.py REQUEST_JSONL [REQUEST_JSONL ...]

For every `request_done` record: prompt and completion tokens, prefill and decode seconds, decode
tok/s (completion tokens over decode seconds, excluding prefill and transport), MTP rounds and
tokens per round, and the decode round split into Device wait and Host exposure from
`engine_timing.decode`. Prints one Markdown table per file plus the file's totals, so a served
rate can be compared with ninfer_bench's decode rate on the same terms.
"""
import json
import sys


def rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("event") == "request_done":
                yield rec


def summarize(path):
    print(f"### {path}\n")
    print("| req | prompt | computed prefill | completion | prefill s | decode s | decode tok/s "
          "| rounds | tok/round | device wait ms/round | host exposed ms/round |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    total = {"completion": 0, "decode": 0.0, "prefill": 0.0, "rounds": 0, "wait": 0.0, "host": 0.0}
    for rec in rows(path):
        result = rec["result"]
        completion = int(result["completion_tokens"])
        prompt = int(result["prompt_tokens"])
        computed = result["computed_prefill_tokens"]
        prefill = float(rec["timings_seconds"]["prefill"])
        decode = float(rec["timings_seconds"]["decode"])
        dec = rec["engine_timing"]["decode"]
        rounds = int(dec["rounds"])
        wait = float(dec["device_wait_exposed_seconds"])
        host = float(dec["host_exposed_seconds"])
        rate = completion / decode if decode else float("nan")
        per_round = completion / rounds if rounds else float("nan")
        print(f"| {rec['request']['request_id']} | {prompt} | {computed} | {completion} "
              f"| {prefill:.2f} | {decode:.2f} | {rate:.1f} | {rounds} | {per_round:.2f} "
              f"| {1e3 * wait / rounds if rounds else float('nan'):.1f} "
              f"| {1e3 * host / rounds if rounds else float('nan'):.2f} |")
        total["completion"] += completion
        total["decode"] += decode
        total["prefill"] += prefill
        total["rounds"] += rounds
        total["wait"] += wait
        total["host"] += host
    r = total["rounds"]
    print(f"\nTotals: {total['completion']} tokens, decode {total['decode']:.1f} s "
          f"({total['completion'] / total['decode'] if total['decode'] else float('nan'):.1f} tok/s), "
          f"prefill {total['prefill']:.1f} s, rounds {r}, "
          f"{total['completion'] / r if r else float('nan'):.2f} tok/round, "
          f"device wait {1e3 * total['wait'] / r if r else float('nan'):.1f} ms/round, "
          f"host exposed {1e3 * total['host'] / r if r else float('nan'):.2f} ms/round\n")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for path in sys.argv[1:]:
        summarize(path)


if __name__ == "__main__":
    main()
