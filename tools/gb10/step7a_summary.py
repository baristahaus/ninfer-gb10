#!/usr/bin/env python3
"""Append the step 7a baseline result sections to summary.md."""
import json
import sys


def json_load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def tail_lines(path, n):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()[-n:]
    except Exception:
        return ["(missing)"]


def main():
    d = sys.argv[1]

    print("\n## 1. Perplexity token scores (fixed corpus, default protocol)\n")
    report = json_load(f"{d}/perplexity/report.json")
    if report and isinstance(report.get("overall"), dict):
        o = report["overall"]
        print(f"- overall mean NLL: {o.get('mean_nll')}")
        print(f"- overall perplexity: {o.get('perplexity')}")
        print(f"- scored tokens: {o.get('scored_tokens')}")
    else:
        print("- report.json: missing or no overall section")
    for key in ("domains", "streams"):
        if report and isinstance(report.get(key), dict):
            print(f"\n| {key} | mean NLL | PPL |")
            print("|---|---|---|")
            for name, agg in sorted(report[key].items()):
                if isinstance(agg, dict):
                    print(f"| {name} | {agg.get('mean_nll', '-')} | "
                          f"{agg.get('perplexity', '-')} |")
    print("\nProduct table (stdout):\n")
    print("```")
    print("\n".join(tail_lines(f"{d}/perplexity-stdout.log", 14)))
    print("```\n")

    long_report = json_load(f"{d}/perplexity-64k/report.json")
    if long_report and isinstance(long_report.get("overall"), dict):
        o = long_report["overall"]
        print("## 1b. Perplexity token scores (65536/32768 windows, for drift)\n")
        print(f"- overall mean NLL: {o.get('mean_nll')}")
        print(f"- overall perplexity: {o.get('perplexity')}")
        print(f"- scored tokens: {o.get('scored_tokens')}")
        print("- compare two artifacts with `tools/bench/compare_token_drift.py "
              "<baseline>/perplexity-64k <candidate>/perplexity-64k`\n")

    print("## 2. MTP acceptance (serving run, greedy, 1024 tokens/stream)\n")
    acc = json_load(f"{d}/acceptance.json")
    if acc:
        print(f"- requests: {acc['requests']}, rounds: {acc['rounds']}")
        print(f"- drafted: {acc['drafted_tokens']}, accepted: {acc['accepted_tokens']}")
        print(f"- acceptance ratio (accepted/drafted): {acc['acceptance_ratio']}")
        print(f"- fallback steps: {acc['fallback_steps']}")
        print(f"- per-position per-round: {acc['accepted_per_position_per_round']}")
        toks = [s["generated_tokens"] for s in acc.get("streams", [])]
        secs = [s["seconds"] for s in acc.get("streams", [])]
        if toks:
            print(f"- generated tokens: {sum(toks)} total, "
                  f"{min(toks)}-{max(toks)} per stream")
            print(f"- wall time: {sum(secs):.0f}s total")
    else:
        print("- acceptance.json missing; see acceptance.log")
    print()

    print("## 3. TEB hardmode (--seed 42; single trial by default, TEB_TRIALS overrides)\n")
    print("Report dir: `teb/`. Console output (tail):\n")
    print("```")
    print("\n".join(tail_lines(f"{d}/teb.log", 40)))
    print("```")


if __name__ == "__main__":
    main()
