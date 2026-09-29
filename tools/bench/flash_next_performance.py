"""Correlate compiled-in Flash-Next work scopes with Nsight Systems GPU activities.

Read-only SQLite input; no third-party Python dependencies. See tools/bench/README.md.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sqlite3


WORK_PREFIX = "ninfer.work/1|"
MEASURED = "ninfer.region/1|measured"


@dataclass(frozen=True)
class Work:
    stage: str
    tokens: int
    batch: int
    context: int
    bytes_min: int
    bytes_max: int
    bf16_flops: int
    nvfp4_flops: int
    fp32_flops: int

    @classmethod
    def parse(cls, text: str) -> Work:
        fields = text.removeprefix(WORK_PREFIX).split("|")
        if not text.startswith(WORK_PREFIX) or len(fields) != 9:
            raise ValueError("invalid work annotation: " + text)
        values = [int(value) for value in fields[1:]]
        if min(values) < 0 or values[0] == 0 or values[3] > values[4]:
            raise ValueError("invalid work counts: " + text)
        return cls(fields[0], *values)


@dataclass(frozen=True)
class Hardware:
    name: str
    reference: str
    kind: str
    dram_gbps: float
    bf16_tflops: float
    nvfp4_tflops: float
    fp32_tflops: float

    def __post_init__(self) -> None:
        if not self.name or not self.reference or self.kind not in {"theoretical", "sustained"}:
            raise ValueError("hardware needs name, reference and theoretical/sustained kind")
        for value in (self.dram_gbps, self.bf16_tflops, self.nvfp4_tflops, self.fp32_tflops):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("hardware rates must be finite and positive; use dense compute rates")

    def estimate_ms(self, work: Work) -> tuple[float, float]:
        # Different instruction resources may overlap. Do not sum their peak-time floors.
        compute = max(work.bf16_flops / (self.bf16_tflops * 1e9),
                      work.nvfp4_flops / (self.nvfp4_tflops * 1e9),
                      work.fp32_flops / (self.fp32_tflops * 1e9))
        return (max(compute, work.bytes_min / (self.dram_gbps * 1e6)),
                max(compute, work.bytes_max / (self.dram_gbps * 1e6)))


@dataclass(frozen=True)
class Range:
    id: int
    start: int
    end: int
    tid: int
    text: str


class RangeIndex:
    """Thread-local nested scopes; disjoint scopes are skipped with a prefix-max end index."""
    def __init__(self, ranges: list[Range]):
        self.rows = defaultdict(list)
        for row in ranges:
            self.rows[row.tid].append(row)
        self.starts = {}
        self.ends = {}
        for tid, rows in self.rows.items():
            rows.sort(key=lambda row: (row.start, -row.end))
            self.starts[tid] = [row.start for row in rows]
            maximum = 0
            ends = []
            for row in rows:
                maximum = max(maximum, row.end)
                ends.append(maximum)
            self.ends[tid] = ends

    def containing(self, tid: int, start: int, end: int) -> list[Range]:
        rows = self.rows.get(tid, [])
        i = bisect_right(self.starts.get(tid, []), start) - 1
        found = []
        while i >= 0 and self.ends[tid][i] >= end:
            row = rows[i]
            if row.start <= start and row.end >= end:
                found.append(row)
            i -= 1
        return sorted(found, key=lambda row: (row.end - row.start, -row.start))


def process_id(global_id: int) -> int:
    # Nsight serializes PID above the low 24 TID bits, in both globalTid and globalPid.
    return global_id >> 24


def union_ns(intervals: list[tuple[int, int]]) -> int:
    total = 0
    end = -1
    for begin, stop in sorted(intervals):
        total += max(0, stop - max(begin, end))
        end = max(end, stop)
    return total


def read_report(path: Path, hardware: Hardware, sample: int = 0) -> dict:
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}

        def rows(table: str) -> list[dict]:
            if table not in tables:
                return []
            return [dict(row) for row in db.execute(f'SELECT rowid AS _id, * FROM "{table}"')]

        strings = {row["id"]: row["value"] for row in rows("StringIds")}
        ranges = []
        for row in rows("NVTX_EVENTS"):
            text = row.get("text") or strings.get(row.get("textId"), "")
            if row.get("end") is not None and row["end"] > row["start"]:
                ranges.append(Range(row["_id"], row["start"], row["end"], row["globalTid"], text))
        measurements = sorted((r for r in ranges if r.text == MEASURED), key=lambda r: r.start)
        if sample < 0 or sample >= len(measurements):
            raise ValueError("no selected measured range; build with NINFER_PERFORMANCE_TRACE=ON "
                             "and capture a ninfer_bench measured repetition")
        measured = measurements[sample]
        pid = process_id(measured.tid)
        ranges = [r for r in ranges if process_id(r.tid) == pid]
        work_ranges = {r.id: r for r in ranges if r.text.startswith(WORK_PREFIX)}
        works = {key: Work.parse(r.text) for key, r in work_ranges.items()}
        work_index = RangeIndex(list(work_ranges.values()))
        phase_ranges = [r for r in ranges if r.text in {"prefill", "decode"}]
        phase_index = RangeIndex(phase_ranges)
        role_index = RangeIndex([r for r in ranges if r.text.startswith("ninfer.region/1|target.")
                                or r.text == "ninfer.region/1|predictor"])
        apis = rows("CUPTI_ACTIVITY_KIND_RUNTIME") + rows("CUPTI_ACTIVITY_KIND_DRIVER")
        apis = [r for r in apis if process_id(r["globalTid"]) == pid]
        by_correlation = defaultdict(list)
        for api in apis:
            if api.get("correlationId") is not None:
                by_correlation[api["correlationId"]].append(api)

        # Node creation and clone lineage are needed even when graphs were captured before
        # the measurement. A graph launch's CPU range is NOT the range of each replayed Op.
        nodes = defaultdict(list)
        for node in rows("CUDA_GRAPH_NODE_EVENTS"):
            if node.get("globalTid") is not None and process_id(node["globalTid"]) == pid:
                nodes[node["graphNodeId"]].append(node)
        for events in nodes.values():
            events.sort(key=lambda r: r["start"])

        def node_origin(node_id: int, before: int) -> tuple[Range | None, int]:
            visited = set()
            while node_id and node_id not in visited:
                visited.add(node_id)
                candidates = [r for r in nodes.get(node_id, []) if r["start"] <= before]
                if not candidates:
                    return None, node_id
                # Nsight emits a creation row with no parent, followed by a separate clone
                # relationship row for the same ID (including cudaGraphInstantiate clones).
                parents = [r for r in candidates if r.get("originalGraphNodeId")]
                if parents:
                    node_id = parents[-1]["originalGraphNodeId"]
                    continue
                node = candidates[0]
                enclosing = work_index.containing(node["globalTid"], node["start"], node["start"])
                return (enclosing[0] if enclosing else None), node_id
            return None, node_id

        expected_nodes = defaultdict(set)
        for node_id in nodes:
            owner, origin = node_origin(node_id, measured.end)
            if owner:
                expected_nodes[owner.id].add(origin)

        activities = []
        for table, kind in (("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"),
                            ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
                            ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
            for activity in rows(table):
                if process_id(activity.get("globalPid") or 0) != pid:
                    continue
                if activity["end"] <= measured.start or activity["start"] >= measured.end:
                    continue
                if activity["start"] < measured.start or activity["end"] > measured.end:
                    raise ValueError("GPU activity crosses the measured boundary; capture a complete repetition")
                activity["kind"] = kind
                activities.append(activity)
        node_launches = {a.get("correlationId") for a in activities if a.get("graphNodeId")}
        for graph in rows("CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
            if (process_id(graph.get("globalPid") or 0) == pid and
                    graph["start"] < measured.end and graph["end"] > measured.start and
                    graph.get("correlationId") not in node_launches):
                raise ValueError("graph-only GPU activity would omit decode work; recapture with --cuda-graph-trace=node")
        if not activities:
            raise ValueError("no GPU node activities in measurement; use --trace=cuda,nvtx --cuda-graph-trace=node")
        if len({a["deviceId"] for a in activities}) != 1:
            raise ValueError("report requires one GPU per measured repetition")

        groups = {}
        unattributed = []
        phase_intervals = defaultdict(list)
        for activity in activities:
            candidates = [a for a in by_correlation.get(activity.get("correlationId"), [])
                          if a["start"] <= activity["start"]]
            # A runtime call may enclose a driver call. Prefer the innermost matching API.
            api = max(candidates, key=lambda a: a["start"]) if candidates else None
            phases = phase_index.containing(api["globalTid"], api["start"], api["end"]) if api else []
            if api and not phases:
                # Engine and Program may run on different host threads. Only use a process
                # interval when its phase is unambiguous; concurrent phases remain "other".
                surrounding = [r for r in phase_ranges if r.start <= api["start"] and r.end >= api["end"]]
                if len({r.text for r in surrounding}) == 1:
                    phases = surrounding
            phase = phases[0].text if phases else "other"
            phase_intervals[phase].append((activity["start"], activity["end"]))
            node_id = activity.get("graphNodeId") or 0
            origin = 0
            if node_id:
                owner, origin = node_origin(node_id, api["start"] if api else activity["start"])
            else:
                enclosing = work_index.containing(api["globalTid"], api["start"], api["end"]) if api else []
                owner = enclosing[0] if enclosing else None
            if owner is None:
                unattributed.append(activity)
                continue
            roles = role_index.containing(owner.tid, owner.start, owner.end)
            role = roles[0].text.split("|")[1] if roles else "other"
            # Count work once per Op invocation, not once per kernel or once per capture.
            # Separate each graph replay using its actual launch correlation and start time.
            launch = (api["correlationId"], api["start"]) if node_id and api else None
            if node_id and launch is None:
                unattributed.append(activity)
                continue
            key = (owner.id, launch)
            group = groups.setdefault(key, {"work": works[owner.id], "phase": phase,
                                            "role": role, "activities": [], "nodes": set(),
                                            "owner": owner.id, "graph": bool(node_id)})
            group["activities"].append(activity)
            if node_id:
                group["nodes"].add(origin)

        result_rows = {}
        partial = 0
        for group in groups.values():
            work = group["work"]
            key = (group["phase"], group["role"], work)
            row = result_rows.setdefault(key, {"phase": group["phase"], "role": group["role"],
                **asdict(work), "calls": 0, "gpu_work_ms": 0.0, "gpu_busy_ms": 0.0,
                "estimated_ms_min": 0.0, "estimated_ms_max": 0.0, "partial_calls": 0})
            duration = sum(a["end"] - a["start"] for a in group["activities"]) / 1e6
            row["gpu_work_ms"] += duration
            row["gpu_busy_ms"] += union_ns([(a["start"], a["end"]) for a in group["activities"]]) / 1e6
            row["calls"] += 1
            if group["graph"] and group["nodes"] != expected_nodes[group["owner"]]:
                row["partial_calls"] += 1
                partial += 1
            else:
                low, high = hardware.estimate_ms(work)
                row["estimated_ms_min"] += low
                row["estimated_ms_max"] += high
        total_work = sum(a["end"] - a["start"] for a in activities) / 1e6
        output_rows = []
        for row in result_rows.values():
            row["gpu_work_share_pct"] = 100 * row["gpu_work_ms"] / total_work
            row["phase_gpu_work_share_pct"] = 100 * row["gpu_work_ms"] / (
                sum(end - start for start, end in phase_intervals[row["phase"]]) / 1e6)
            if row["partial_calls"]:
                row["estimated_ms_min"] = row["estimated_ms_max"] = None
                row["estimate_efficiency_pct_min"] = row["estimate_efficiency_pct_max"] = None
            else:
                for suffix in ("min", "max"):
                    row["estimate_efficiency_pct_" + suffix] = (
                        100 * row["estimated_ms_" + suffix] / row["gpu_busy_ms"]
                        if row["gpu_busy_ms"] else None)
            output_rows.append(row)
        host_rows = []
        for label in sorted({r.text for r in ranges if r.text.startswith("ninfer.host/1|")}):
            selected = [r for r in ranges if r.text == label and measured.start <= r.start
                        and r.end <= measured.end]
            host_rows.append({"stage": label.split("|")[1], "calls": len(selected),
                              "host_work_ms": sum(r.end - r.start for r in selected) / 1e6})
        unknown_ms = sum(a["end"] - a["start"] for a in unattributed) / 1e6
        unknown_rows = {}
        for activity in unattributed:
            name = strings.get(activity.get("shortName"), activity["kind"])
            key = (activity["kind"], name)
            entry = unknown_rows.setdefault(key, {"kind": key[0], "name": name,
                                                  "activities": 0, "gpu_work_ms": 0.0})
            entry["activities"] += 1
            entry["gpu_work_ms"] += (activity["end"] - activity["start"]) / 1e6
        warnings = ["Estimates are partial cold useful-work models, not distance from optimal.",
                    "Token counts are execution envelopes; padded columns are not committed outputs.",
                    "QSA estimates cover projections only; device-dependent scoring/attention is unmodeled.",
                    "GDN prefill estimates omit chunked recurrence compute; PLE estimates omit convolution/state traffic.",
                    "DRAM cache reuse, scratch traffic, nonlinear math and instruction geometry are unmodeled.",
                    "Profiler overhead affects timings; use unprofiled paired benchmarks for speed claims."]
        if partial:
            warnings.append("Incomplete graph Op instances have no estimate; collect graph construction and all nodes.")
        if unknown_ms:
            warnings.append("Unattributed GPU work is retained; no whole-model efficiency is reported.")
        if any((r["estimate_efficiency_pct_max"] or 0) > 100 for r in output_rows):
            warnings.append("An estimate exceeds measured time: inspect cache reuse, envelope work and hardware rates; percentages are not clamped.")
        return {"artifact_type": "ninfer_flash_next_performance", "schema_version": 1,
                "trace": str(path), "sample": sample, "hardware": asdict(hardware),
                "measurement": {"wall_ms": (measured.end - measured.start) / 1e6,
                                "gpu_work_ms": total_work,
                                "gpu_busy_ms": union_ns([(a["start"], a["end"]) for a in activities]) / 1e6,
                                "attributed_gpu_work_pct": 100 * (total_work - unknown_ms) / total_work,
                                "unattributed_gpu_work_ms": unknown_ms,
                                "unattributed_graph_activities": sum(bool(a.get("graphNodeId")) for a in unattributed)},
                "phases": [{"phase": name, "gpu_work_ms": sum(b-a for a,b in times) / 1e6,
                            "gpu_busy_ms": union_ns(times) / 1e6} for name, times in sorted(phase_intervals.items())],
                "stages": sorted(output_rows, key=lambda r: -r["gpu_work_ms"]),
                "unattributed": sorted(unknown_rows.values(), key=lambda r: -r["gpu_work_ms"]),
                "host": host_rows, "warnings": warnings}


def markdown(report: dict) -> str:
    m = report["measurement"]
    lines = ["# Flash-Next GPU work report", "",
             f"Measured wall: {m['wall_ms']:.3f} ms; GPU busy: {m['gpu_busy_ms']:.3f} ms; "
             f"attributed GPU work: {m['attributed_gpu_work_pct']:.2f}%.", "",
             "Estimates describe partial modeled work, not an achievable optimum. "
             "Bytes/FLOPs below are per call; time and share are aggregated.", "",
             "| Phase / role | Stage | T / B / context envelope | Calls | GPU work ms | Phase share | Useful bytes min–max | BF16 / NVFP4 / FP32 FLOPs | Estimate ms min–max | Estimate efficiency |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in report["stages"]:
        estimate = efficiency = "unavailable"
        if r["estimated_ms_min"] is not None:
            estimate = f"{r['estimated_ms_min']:.3f}–{r['estimated_ms_max']:.3f}"
            efficiency = f"{r['estimate_efficiency_pct_min']:.1f}–{r['estimate_efficiency_pct_max']:.1f}%"
        lines.append(f"| {r['phase']} / {r['role']} | {r['stage']} | {r['tokens']} / {r['batch']} / {r['context']} | "
                     f"{r['calls']} | {r['gpu_work_ms']:.3f} | {r['phase_gpu_work_share_pct']:.1f}% | "
                     f"{r['bytes_min']}–{r['bytes_max']} | {r['bf16_flops']} / {r['nvfp4_flops']} / {r['fp32_flops']} | "
                     f"{estimate} | {efficiency} |")
    lines += ["", f"Unattributed GPU work: {m['unattributed_gpu_work_ms']:.3f} ms.", ""]
    if report["host"]:
        lines += ["Host lookup work (may overlap GPU execution):", ""]
        lines += [f"- {r['stage']}: {r['host_work_ms']:.3f} ms across {r['calls']} calls."
                  for r in report["host"]]
        lines.append("")
    lines += [f"- {message}" for message in report["warnings"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Nsight Systems SQLite export, including graph construction")
    parser.add_argument("--hardware", required=True, type=Path, help="explicit theoretical or sustained rate profile JSON")
    parser.add_argument("--benchmark", required=True, type=Path, help="ninfer_bench JSON from the same capture")
    parser.add_argument("--sample", default=0, type=int, help="zero-based measured repetition in the trace")
    parser.add_argument("--output", required=True, type=Path, help="JSON output; sibling .md is also written")
    args = parser.parse_args()
    try:
        hardware = Hardware(**json.loads(args.hardware.read_text()))
        benchmark = json.loads(args.benchmark.read_text())
        if (benchmark.get("artifact_type") != "ninfer_bench_report" or
                benchmark.get("schema_version") != 16 or
                benchmark.get("load", {}).get("architecture") != "Qwen3_8FlashNextForCausalLM" or
                "nvfp4" not in benchmark.get("load", {}).get("formats", []) or
                len(benchmark.get("tests", [])) != 1):
            raise ValueError("benchmark must contain one v3 Flash-Next NVFP4 benchmark test")
        if hardware.name != benchmark.get("environment", {}).get("gpu_name"):
            raise ValueError("hardware profile name does not match benchmark GPU")
        report = read_report(args.trace, hardware, args.sample)
        report["benchmark"] = benchmark
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        args.output.with_suffix(".md").write_text(markdown(report))
    except (ValueError, TypeError, OSError, sqlite3.Error) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
