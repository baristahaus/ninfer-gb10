#!/usr/bin/env python3
"""Write docs/maintainer/test-inventory.md from a configured build tree.

Every test's requirements come from the tree, not from a hand-written list: the CTest properties
say whether a test is serial or has a timeout, the registration says what it links, the source says
which artifact environment variable it reads, and the artifact manifest says which artifact that
variable means.

    tools/test_inventory.py [--build-dir build] [--measurements FILE] [--out FILE]

Duration: a test that was actually run is reported with its measured wall time from
`--measurements` (a JSON object of test name -> seconds). Everything else is reported as
`unmeasured` with the reason, because a GPU test cannot be timed on a host without a GPU and an
invented number would be worse than none.

Refresh after changing the test tree:

    cmake --build build --target <test targets>
    ctest --test-dir build -R '<subset>' --output-junit build/junit.xml   # or time them yourself
    tools/test_inventory.py --measurements measurements.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "fixtures" / "artifacts" / "manifest.json"
REGISTRATION = re.compile(r'(?:ninfer_add_test|ninfer_add_op_test)\(\s*([A-Za-z0-9_]+)')
SOURCES = re.compile(r'SOURCES\s+"?([^\s")]+)"?')
LIBRARIES = re.compile(r'LIBRARIES\s+([^)\n]+)')
ADD_TEST = re.compile(r'add_test\(\s*(?:NAME\s+)?([A-Za-z0-9_]+)')
TESTFILE_TEST = re.compile(r'add_test\("([^"]+)"\s+"([^"]+)"')
ENV_READ = re.compile(r'getenv\("(NINFER_[A-Z0-9_]+)"\)')

# Linking these means the test drives CUDA. `ninfer_core` alone does not: host-only tests link it
# for tensors and layouts without touching a device.
GPU_LIBRARIES = (
    "ninfer_engine",
    "ninfer_ops",
    "ninfer_serve",
    "ninfer_model_execution",
    "ninfer_model_loading",
    "ninfer_model_control",
    "ninfer_distributed",
    "ninfer_runtime",
    "CUDA::",
)
GPU_SOURCE_MARKERS = ("cuda_runtime.h", "cudaMalloc", "cudaMemcpy", "DeviceContext", "ninfer::Engine")


def registrations() -> dict[str, dict]:
    """Test name -> {source, libraries} from every tests/**/*.cmake."""
    out: dict[str, dict] = {}
    for path in sorted((ROOT / "tests").rglob("*.cmake")):
        text = path.read_text(errors="replace")
        for match in REGISTRATION.finditer(text):
            name = match.group(1)
            tail = text[match.end() : match.end() + 700]
            source = SOURCES.search(tail)
            libraries = LIBRARIES.search(tail)
            out[name] = {
                "source": source.group(1).split("/")[-1] if source else None,
                "libraries": libraries.group(1).strip() if libraries else "",
            }
        for match in ADD_TEST.finditer(text):
            out.setdefault(match.group(1), {"source": None, "libraries": ""})
    return out


def artifact_manifest() -> tuple[dict[str, str], dict[str, str]]:
    """environment variable -> artifact name, and artifact name -> note."""
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    env_to_name, notes = {}, {}
    for entry in data.get("artifacts", []):
        if entry.get("tests"):
            env_to_name.setdefault(entry["env"], entry["name"])
        notes[entry["name"]] = entry.get("notes", "")
    return env_to_name, notes


def source_envs() -> dict[str, set[str]]:
    """Source basename -> artifact environment variables it reads."""
    out: dict[str, set[str]] = {}
    for area in ("tests",):
        for path in (ROOT / area).rglob("*"):
            if path.suffix in {".cpp", ".cu"} and path.is_file():
                found = {
                    v
                    for v in ENV_READ.findall(path.read_text(errors="replace"))
                    if "ARTIFACT" in v or "WEIGHTS" in v or v == "NINFER_EXL3_FIXTURE"
                }
                if found:
                    out[path.name] = found
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", default=str(ROOT / "build"))
    parser.add_argument("--measurements", default=None, help="JSON object: test name -> seconds")
    parser.add_argument("--out", default=str(ROOT / "docs" / "maintainer" / "test-inventory.md"))
    args = parser.parse_args(argv)

    build = Path(args.build_dir)
    testfile = build / "tests" / "CTestTestfile.cmake"
    if not testfile.is_file():
        print(f"test_inventory: {testfile} is missing; configure with -DBUILD_TESTING=ON", file=sys.stderr)
        return 2
    text = testfile.read_text(errors="replace")

    # Properties from the generated CTest file (SKIP_RETURN_CODE, RUN_SERIAL, TIMEOUT, LABELS).
    properties: dict[str, dict[str, str]] = {}
    for match in re.finditer(r'set_tests_properties\(([^)]*?)\s+PROPERTIES\s+(.*?)\)\s*$', text, re.M):
        names = re.findall(r'"([^"]+)"', match.group(1))
        body = match.group(2)
        for name in names:
            props = properties.setdefault(name, {})
            for key in ("SKIP_RETURN_CODE", "RUN_SERIAL", "TIMEOUT", "LABELS"):
                found = re.search(rf'{key}\s+("?[^"]*"?|\[\[.*?\]\])', body)
                if found:
                    props[key] = found.group(1).strip('"')

    tests = [(m.group(1), m.group(2).split("/")[-1]) for m in TESTFILE_TEST.finditer(text)]
    registered = registrations()
    envs = source_envs()
    env_to_name, notes = artifact_manifest()
    measurements = {}
    if args.measurements:
        measurements = json.loads(Path(args.measurements).read_text())

    rows = []
    for name, exe in tests:
        reg = registered.get(exe) or registered.get(name) or {}
        source = reg.get("source")
        libraries = reg.get("libraries", "")
        needs = []
        gpu = any(lib in libraries for lib in GPU_LIBRARIES)
        if source:
            path = ROOT / "tests" / source
            if path.is_file():
                body = path.read_text(errors="replace")
                gpu = gpu or any(marker in body for marker in GPU_SOURCE_MARKERS)
        # A measured test ran on this host, which has no GPU: it is host-only whatever it links.
        if gpu and name not in measurements:
            needs.append("GPU")
        if source and (envs.get(source)):
            needs.append("artifact")
        if properties.get(name, {}).get("RUN_SERIAL") == "TRUE":
            needs.append("serial")
        if not needs:
            needs.append("host")
        artifact = ""
        for var in sorted(envs.get(source, set())) if source else []:
            artifact = env_to_name.get(var, var)
            break
        duration = measurements.get(name)
        if duration is None:
            duration = "unmeasured" + (" (GPU)" if "GPU" in needs else "")
        else:
            duration = f"{duration:.2f} s"
        rows.append(
            {
                "name": name,
                "needs": ", ".join(needs),
                "artifact": artifact or "—",
                "labels": properties.get(name, {}).get("LABELS", "—"),
                "duration": duration,
            }
        )

    counts: dict[str, int] = {}
    for row in rows:
        for need in row["needs"].split(", "):
            counts[need] = counts.get(need, 0) + 1

    lines = [
        "# Test inventory",
        "",
        "Generated by [`tools/test_inventory.py`](../../tools/test_inventory.py) from a configured",
        "build tree, the test registrations and the artifact manifest. Do not edit by hand; re-run the",
        "tool after changing the test tree. `docs/maintainer/test-artifacts.md` explains the artifact",
        "cache and how a real-artifact test skips.",
        "",
        f"{len(rows)} registered tests on this target: "
        + ", ".join(f"{count} {need}" for need, count in sorted(counts.items()))
        + ".",
        "",
        "A test that needs an **artifact** skips with exit 77 when its entry is absent from the cache;",
        "`tools/fetch_test_artifacts.py` populates the cache and `--list` shows what is missing.",
        "**serial** means the test takes the GPU to itself and CTest runs it alone.",
        "",
        "| Test | Needs | Artifact | Labels | Duration |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['name']}` | {row['needs']} | {row['artifact']} | {row['labels']} | {row['duration']} |"
        )

    lines += [
        "",
        "## How to refresh",
        "",
        "```bash",
        "cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DNINFER_TARGET=rtx5090 -DBUILD_TESTING=ON",
        "cmake --build build -j                      # or only the targets you changed",
        "ctest --test-dir build --output-junit build/junit.xml   # on a machine with the artifact cache",
        "tools/test_inventory.py --measurements measurements.json",
        "```",
        "",
        "`measurements.json` is a JSON object of test name to wall seconds; the generator reports a",
        "measured time only for tests it is given, and `unmeasured` otherwise. A host without a GPU",
        "cannot time the GPU tests, and an estimate would be worse than an explicit gap.",
        "",
    ]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out} ({len(rows)} tests, {counts})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
