#!/usr/bin/env python3
"""Measure serial vLLM prefill and decode throughput over an exact-token context matrix."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence
import urllib.error
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path.home() / "models/RadixArk/Qwen3.8-Flash-Next-NVFP4"
DEFAULT_RUNTIME = Path.home() / ".local/share/vllm-qwen38next-55f69ea17-offload/dist-packages"
DEFAULT_DEPENDENCY_RUNTIME = Path.home() / ".local/share/vllm-qwen38-native/dist-packages"
DEFAULT_PYTHON = Path.home() / ".venv/bin/python"
DEFAULT_CORPUS = REPO_ROOT / "bench/fixtures/qwen3_8_flash_next_context.ids"
DEFAULT_OUTPUT = REPO_ROOT / "profiles/bench/qwen3_8_flash_next_vllm_baseline.json"
PROMPT_LENGTHS = (1024, 8192, 32768, 65536, 131072, 196608, 261632)
OUTPUT_TOKENS = 512
MODEL_NAME = "Qwen/Qwen3.8-Flash-Next"
METRIC_NAMES = (
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_prefill_time_seconds_count",
    "vllm:request_decode_time_seconds_sum",
    "vllm:request_decode_time_seconds_count",
    "vllm:request_prefill_kv_computed_tokens_sum",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)
METRIC_PATTERN = re.compile(
    r"^(?P<name>[^\s{]+)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+)$"
)


class CampaignError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class Profile:
    name: str
    speculative_tokens: int


PROFILES = (Profile("mtp0", 0), Profile("mtp3", 3))


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_ids(path: Path) -> list[int]:
    values = path.read_text(encoding="utf-8").split()
    if not values or any(not value.isdigit() for value in values):
        raise CampaignError(f"invalid token corpus: {path}")
    result = [int(value) for value in values]
    if len(result) < max(PROMPT_LENGTHS) + 128:
        raise CampaignError(
            f"corpus has {len(result)} tokens; need {max(PROMPT_LENGTHS) + 128}"
        )
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_json(url: str, timeout: float = 10.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def post_json(url: str, payload: dict[str, Any], timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def fetch_metrics(base_url: str) -> dict[str, float]:
    with urllib.request.urlopen(base_url + "/metrics", timeout=30.0) as response:
        text = response.read().decode("utf-8")
    result: dict[str, float] = {}
    wanted = set(METRIC_NAMES)
    for line in text.splitlines():
        match = METRIC_PATTERN.match(line)
        if match is None or match.group("name") not in wanted:
            continue
        labels = match.group("labels") or ""
        if "model_name=" in labels and f'model_name="{MODEL_NAME}"' not in labels:
            continue
        name = match.group("name")
        result[name] = result.get(name, 0.0) + float(match.group("value"))
    return {name: result.get(name, 0.0) for name in METRIC_NAMES}


def delta(after: dict[str, float], before: dict[str, float], name: str) -> float:
    return after.get(name, 0.0) - before.get(name, 0.0)


def process_rollup(pid: int) -> dict[str, int]:
    totals = {"rss_kib": 0, "swap_kib": 0, "read_bytes": 0, "major_faults": 0}
    pending = [pid]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        children = Path(f"/proc/{current}/task/{current}/children")
        try:
            pending.extend(int(item) for item in children.read_text().split())
        except (FileNotFoundError, PermissionError, ValueError):
            pass
        try:
            status = Path(f"/proc/{current}/status").read_text().splitlines()
            for line in status:
                if line.startswith("VmRSS:"):
                    totals["rss_kib"] += int(line.split()[1])
                elif line.startswith("VmSwap:"):
                    totals["swap_kib"] += int(line.split()[1])
            io_lines = Path(f"/proc/{current}/io").read_text().splitlines()
            for line in io_lines:
                if line.startswith("read_bytes:"):
                    totals["read_bytes"] += int(line.split()[1])
            stat = Path(f"/proc/{current}/stat").read_text().split()
            totals["major_faults"] += int(stat[11]) + int(stat[12])
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            pass
    return totals


class VllmServer:
    def __init__(self, args: argparse.Namespace, profile: Profile, log_path: Path):
        self.args = args
        self.profile = profile
        self.log_path = log_path
        self.process: subprocess.Popen[bytes] | None = None
        self.log_handle: Any = None

    def command(self) -> list[str]:
        command = [
            str(self.args.python),
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            str(self.args.model),
            "--served-model-name",
            MODEL_NAME,
            "--host",
            self.args.host,
            "--port",
            str(self.args.port),
            "--max-model-len",
            "262144",
            "--gpu-memory-utilization",
            str(self.args.gpu_memory_utilization),
            "--tensor-parallel-size",
            "1",
            "--distributed-executor-backend",
            "mp",
            "--max-num-seqs",
            "1",
            "--max-num-batched-tokens",
            "8192",
            "--kv-cache-dtype",
            "auto",
            "--no-enable-prefix-caching",
            "--enable-prompt-tokens-details",
            "--no-enable-flashinfer-autotune",
        ]
        if self.profile.speculative_tokens:
            command.extend(
                [
                    "--speculative-config",
                    json.dumps(
                        {
                            "method": "mtp",
                            "num_speculative_tokens": self.profile.speculative_tokens,
                        },
                        separators=(",", ":"),
                    ),
                ]
            )
        return command

    def __enter__(self) -> "VllmServer":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("ab", buffering=0)
        env = os.environ.copy()
        env["PYTHONPATH"] = ":".join(
            (
                str(self.args.runtime),
                str(self.args.dependency_runtime / "nvidia_cutlass_dsl/dsl_packages"),
                str(self.args.dependency_runtime),
            )
        )
        env["VLLM_PLE_CPU_OFFLOAD"] = "1"
        env["VLLM_PLE_FP8_CHECKPOINT"] = "1"
        env["TORCH_CUDA_ARCH_LIST"] = "12.0f"
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        self.process = subprocess.Popen(
            self.command(),
            cwd=REPO_ROOT,
            env=env,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.args.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise CampaignError(
                    f"vLLM exited with {self.process.returncode}; see {self.log_path}"
                )
            try:
                get_json(self.args.base_url + "/v1/models", timeout=5.0)
                return self
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                time.sleep(2.0)
        raise CampaignError(f"vLLM did not become ready; see {self.log_path}")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=240.0)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        if self.log_handle is not None:
            self.log_handle.close()

    @property
    def pid(self) -> int:
        if self.process is None:
            raise CampaignError("vLLM process is not running")
        return self.process.pid


def request_payload(tokens: Sequence[int], output_tokens: int) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "prompt": list(tokens),
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "seed": 7632647173703958409,
    }


def measure_request(
    args: argparse.Namespace,
    server: VllmServer,
    tokens: Sequence[int],
    profile: Profile,
    length: int,
    regime: str,
    repetition: int,
) -> dict[str, Any]:
    before_metrics = fetch_metrics(args.base_url)
    before_process = process_rollup(server.pid)
    started = time.monotonic()
    response = post_json(
        args.base_url + "/v1/completions",
        request_payload(tokens, OUTPUT_TOKENS),
        args.request_timeout,
    )
    wall_seconds = time.monotonic() - started
    after_process = process_rollup(server.pid)
    after_metrics = fetch_metrics(args.base_url)
    count = delta(
        after_metrics,
        before_metrics,
        "vllm:request_prefill_time_seconds_count",
    )
    decode_count = delta(
        after_metrics,
        before_metrics,
        "vllm:request_decode_time_seconds_count",
    )
    if count != 1.0 or decode_count != 1.0:
        raise CampaignError(
            f"expected one request metric delta, got prefill={count} decode={decode_count}"
        )
    usage = response.get("usage", {})
    prompt_tokens = int(usage.get("prompt_tokens", -1))
    completion_tokens = int(usage.get("completion_tokens", -1))
    if prompt_tokens != length or completion_tokens != OUTPUT_TOKENS:
        raise CampaignError(
            f"usage mismatch: prompt={prompt_tokens}/{length}, "
            f"completion={completion_tokens}/{OUTPUT_TOKENS}"
        )
    prefill_seconds = delta(
        after_metrics, before_metrics, "vllm:request_prefill_time_seconds_sum"
    )
    decode_seconds = delta(
        after_metrics, before_metrics, "vllm:request_decode_time_seconds_sum"
    )
    computed = delta(
        after_metrics,
        before_metrics,
        "vllm:request_prefill_kv_computed_tokens_sum",
    )
    if computed and int(computed) != length:
        raise CampaignError(f"vLLM computed {computed} prefill tokens, expected {length}")
    return {
        "profile": profile.name,
        "prompt_tokens": length,
        "output_tokens": completion_tokens,
        "regime": regime,
        "repetition": repetition,
        "recorded_at": utc_now(),
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "prefill_tok_s": length / prefill_seconds,
        "decode_tok_s": (completion_tokens - 1) / decode_seconds,
        "wall_seconds": wall_seconds,
        "ttft_seconds": delta(
            after_metrics, before_metrics, "vllm:time_to_first_token_seconds_sum"
        ),
        "e2e_seconds": delta(
            after_metrics, before_metrics, "vllm:e2e_request_latency_seconds_sum"
        ),
        "computed_prefill_tokens": computed,
        "drafts": delta(
            after_metrics, before_metrics, "vllm:spec_decode_num_drafts_total"
        ),
        "accepted_draft_tokens": delta(
            after_metrics,
            before_metrics,
            "vllm:spec_decode_num_accepted_tokens_total",
        ),
        "rss_kib_after": after_process["rss_kib"],
        "swap_kib_after": after_process["swap_kib"],
        "read_bytes_delta": after_process["read_bytes"] - before_process["read_bytes"],
        "major_faults_delta": (
            after_process["major_faults"] - before_process["major_faults"]
        ),
    }


def key(profile: str, length: int, regime: str, repetition: int) -> tuple[Any, ...]:
    return profile, length, regime, repetition


def save_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def summarize(samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for sample in samples:
        group = (sample["profile"], sample["prompt_tokens"], sample["regime"])
        grouped.setdefault(group, []).append(sample)
    rows: list[dict[str, Any]] = []
    for group, records in sorted(grouped.items()):
        row: dict[str, Any] = {
            "profile": group[0],
            "prompt_tokens": group[1],
            "regime": group[2],
            "repetitions": len(records),
        }
        for metric in ("prefill_tok_s", "decode_tok_s", "ttft_seconds", "e2e_seconds"):
            values = [float(record[metric]) for record in records]
            row[metric + "_mean"] = statistics.mean(values)
            row[metric + "_stddev"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def save_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    corpus = read_ids(args.corpus)
    report: dict[str, Any]
    if args.output.exists() and args.resume:
        report = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        version = subprocess.check_output(
            [
                str(args.python),
                "-c",
                "import vllm; print(vllm.__version__)",
            ],
            env={
                **os.environ,
                "PYTHONPATH": ":".join(
                    (str(args.runtime), str(args.dependency_runtime))
                ),
            },
            text=True,
        ).strip()
        report = {
            "artifact_type": "ninfer_vllm_context_baseline",
            "schema_version": 1,
            "created_at": utc_now(),
            "model": str(args.model),
            "served_model": MODEL_NAME,
            "vllm_version": version,
            "corpus": str(args.corpus),
            "corpus_sha256": sha256(args.corpus),
            "prompt_lengths": list(PROMPT_LENGTHS),
            "output_tokens": OUTPUT_TOKENS,
            "cache_regime": "warm; one unmeasured identical request precedes five measured requests at each length",
            "samples": [],
            "summary": [],
        }
    samples: list[dict[str, Any]] = report["samples"]
    complete = {
        key(
            str(item["profile"]),
            int(item["prompt_tokens"]),
            str(item["regime"]),
            int(item["repetition"]),
        )
        for item in samples
    }
    for profile in PROFILES:
        missing = [
            (length, repetition)
            for length in PROMPT_LENGTHS
            for repetition in range(args.warm_repetitions)
            if key(profile.name, length, "warm", repetition) not in complete
        ]
        if not missing:
            continue
        log_path = args.output.parent / "vllm-logs" / f"{profile.name}.log"
        print(f"starting {profile.name}", flush=True)
        with VllmServer(args, profile, log_path) as server:
            for length in PROMPT_LENGTHS:
                target_tokens = corpus[:length]
                if not any(item[0] == length for item in missing):
                    continue
                print(f"warming {profile.name} length={length}", flush=True)
                post_json(
                    args.base_url + "/v1/completions",
                    request_payload(target_tokens, OUTPUT_TOKENS),
                    args.request_timeout,
                )
                for repetition in range(args.warm_repetitions):
                    warm_key = key(profile.name, length, "warm", repetition)
                    if warm_key in complete:
                        continue
                    sample = measure_request(
                        args,
                        server,
                        target_tokens,
                        profile,
                        length,
                        "warm",
                        repetition,
                    )
                    samples.append(sample)
                    complete.add(warm_key)
                    report["summary"] = summarize(samples)
                    save_report(args.output, report)
                    save_csv(args.output, report["summary"])
                    print(
                        f"warm {repetition + 1}/{args.warm_repetitions} "
                        f"pp={sample['prefill_tok_s']:.2f} "
                        f"tg={sample['decode_tok_s']:.2f}",
                        flush=True,
                    )
    report["completed_at"] = utc_now()
    report["summary"] = summarize(samples)
    save_report(args.output, report)
    save_csv(args.output, report["summary"])
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--dependency-runtime", type=Path, default=DEFAULT_DEPENDENCY_RUNTIME)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18003)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.965)
    parser.add_argument("--warm-repetitions", type=int, default=5)
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=86400.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    args.base_url = f"http://{args.host}:{args.port}"
    if args.warm_repetitions < 1:
        parser.error("--warm-repetitions must be positive")
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except (CampaignError, OSError, urllib.error.URLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
