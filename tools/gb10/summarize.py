#!/usr/bin/env python3
"""Condense GB10 step outputs into short Markdown for tools/gb10 summaries.

Standard library only. Each subcommand prints Markdown to stdout.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path


def is_json_value(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return False
    try:
        json.loads(stripped)
    except json.JSONDecodeError:
        return False
    return True


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def clip(text: str, limit: int = 300) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + "…"


def cmd_ctest(path: Path) -> None:
    lines = path.read_text(errors="replace").splitlines()
    totals = [line for line in lines if re.search(r"\d+% tests passed", line)]
    failed, in_failed = [], False
    for line in lines:
        if line.startswith("The following tests FAILED"):
            in_failed = True
            continue
        if in_failed:
            if not line.strip() or line.startswith("Errors while running"):
                in_failed = False
            else:
                failed.append(line.strip())
    skipped = sorted({m.group(1) for line in lines
                      if (m := re.search(r"Test\s+#\d+:\s+(\S+)\s.*\*\*\*(Skipped|Not Run)", line))})
    print(f"- Result: {totals[-1].strip() if totals else 'no ctest summary line found'}")
    print(f"- Failed ({len(failed)}): {', '.join(failed) if failed else 'none'}")
    print(f"- Skipped ({len(skipped)}): {', '.join(skipped) if skipped else 'none'}")


def cmd_stream(path: Path) -> None:
    events, done = [], False
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            done = True
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            events.append({"unparsed": payload})
    errors = [event for event in events if "error" in event or "unparsed" in event]
    contents, reasoning, finish = [], 0, None
    for event in events:
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                contents.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning += 1
            finish = choice.get("finish_reason") or finish
    content = "".join(contents)
    ok = done and not errors and len(contents) == 1 and is_json_value(content) \
        and content == content.strip()
    print(f"- Streamed JSON check: **{verdict(ok)}**")
    print(f"- Events: {len(events)}; reasoning deltas: {reasoning}; content deltas: {len(contents)}; "
          f"finish_reason: {finish}; [DONE]: {done}")
    print(f"- Error events: {len(errors)}" + (f" — {clip(json.dumps(errors[0]))}" if errors else ""))
    print(f"- Content: `{clip(content)}`")


def cmd_chat(path: Path) -> None:
    try:
        body = json.loads(path.read_text(errors="replace"))
    except json.JSONDecodeError:
        print(f"- Non-streamed check: **FAIL** — response is not JSON: `{clip(path.read_text(errors='replace'))}`")
        return
    if "error" in body:
        print(f"- Non-streamed check: **FAIL** — error: `{clip(json.dumps(body['error']))}`")
        return
    choice = (body.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    ok = is_json_value(content)
    print(f"- Non-streamed check: **{verdict(ok)}**; finish_reason: {choice.get('finish_reason')}")
    print(f"- Content: `{clip(content)}`")


def fmt(mean, stddev) -> str:
    if mean is None:
        return "—"
    return f"{mean:,.1f} ± {stddev:,.1f}" if stddev is not None else f"{mean:,.1f}"


def cmd_bench(paths: list[Path]) -> None:
    print("| Config | Test | Prefill tok/s | Decode output tok/s | Mean accepted length |")
    print("|---|---|---:|---:|---:|")
    for path in paths:
        report = json.loads(path.read_text())
        config = report.get("config", {})
        spec = config.get("speculative_backend", "?")
        name = spec if spec == "none" else f"{spec} K={config.get('draft_tokens')}"
        name += f", KV {config.get('kv_cache')}"
        for test in report.get("tests", []):
            speculative = test.get("speculative") or {}
            accepted = speculative.get("acceptance_length") if speculative.get("enabled") else None
            print(f"| {name} | {test.get('label')} "
                  f"| {fmt(test.get('prefill_tok_s_mean'), test.get('prefill_tok_s_stddev'))} "
                  f"| {fmt(test.get('decode_output_tok_s_mean'), test.get('decode_output_tok_s_stddev'))} "
                  f"| {'—' if accepted is None else f'{accepted:.2f}'} |")
    first = json.loads(paths[0].read_text()) if paths else {}
    memory, env = first.get("memory", {}), first.get("environment", {})
    if env:
        print(f"\n- Device: {env.get('gpu_name')}; CUDA runtime {env.get('cuda_runtime_version')}, "
              f"driver {env.get('cuda_driver_version')}")
    if memory:
        print(f"- KV capacity: {memory.get('kv_capacity')} tokens ({memory.get('kv_cache')}); "
              f"available after startup: {memory.get('available_after_startup_bytes', 0) / 2**30:.1f} GiB")


def mean(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def cmd_serving(path: Path) -> None:
    rows: dict[str, list[dict]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows.setdefault(row["case"], []).append(row)
    print("| Case | Rows | Mean wall s | Mean aggregate tok/s | Mean TTFT s | Mean decode tok/s |")
    print("|---|---:|---:|---:|---:|---:|")
    for case, group in rows.items():
        requests = [request for row in group for request in row.get("requests", [])]
        cells = [mean(row.get("wall_seconds") for row in group),
                 mean(row.get("aggregate_tps") for row in group),
                 mean(request.get("ttft") for request in requests),
                 mean(request.get("decode_tps") for request in requests)]
        print(f"| {case} | {len(group)} | " + " | ".join("—" if c is None else f"{c:,.3f}" for c in cells) + " |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ctest", "stream", "chat", "serving"):
        sub.add_parser(name).add_argument("path", type=Path)
    sub.add_parser("bench").add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "bench":
        cmd_bench(args.paths)
    else:
        {"ctest": cmd_ctest, "stream": cmd_stream, "chat": cmd_chat,
         "serving": cmd_serving}[args.command](args.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
