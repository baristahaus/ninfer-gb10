#!/usr/bin/env python3
"""Formatting guard: every C++ source file must match the checked-in `.clang-format`.

A formatter configuration that nothing enforces describes an intention, not the tree. When this
guard was added, 520 of 1,900 files under `src` did not match it, so a reader could not tell
deliberate style from drift and every diff carried unrelated whitespace.

The check runs `clang-format --dry-run --Werror` on each tracked C++ file and reports the files that
do not match. The binary's major version is printed, because clang-format output changes between
major versions: the tree and the tool must move together, and a version mismatch is reported as its
own gap rather than as a thousand formatting failures.

Set `CLANG_FORMAT` to name a specific binary.

Usage: check_format.py [ROOT] [--self-test]
Exit 1 and list every file that does not match, or 0 with a summary.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("src", "include", "apps", "bench", "tests")
SUFFIXES = (".cpp", ".cu", ".h", ".cuh")


class Gaps:
    """Collects (kind, message) pairs; the guard prints them and exits non-zero when non-empty."""

    def __init__(self) -> None:
        self.items: list[tuple[str, str]] = []

    def add(self, kind: str, message: str) -> None:
        self.items.append((kind, message))

    def __bool__(self) -> bool:
        return bool(self.items)


def clang_format_binary() -> str:
    return os.environ.get("CLANG_FORMAT") or "clang-format"


def clang_format_version(binary: str) -> str:
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
    except OSError as error:
        return f"unavailable: {error}"
    text = (out.stdout or out.stderr).strip()
    return text.splitlines()[0] if text else "unavailable"


def major_version(version: str) -> int | None:
    match = re.search(r"(\d+)\.\d+", version)
    return int(match.group(1)) if match else None


def tracked_sources(root: Path) -> list[Path]:
    """The tracked C++ files under SOURCE_DIRS, in a stable order.

    `git ls-files` is used when the root is a checkout, so an untracked scratch file cannot fail the
    guard. A plain walk is the fallback for a copied tree.
    """
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "--", *SOURCE_DIRS],
                             capture_output=True, text=True, check=True, timeout=120)
        files = [root / name for name in out.stdout.split()
                 if name.endswith(SUFFIXES) and (root / name).is_file()]
        if files:
            return sorted(files)
    except (OSError, subprocess.SubprocessError):
        pass
    files = []
    for directory in SOURCE_DIRS:
        base = root / directory
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.suffix in SUFFIXES and path.is_file():
                files.append(path)
    return sorted(files)


def file_matches(binary: str, path: Path) -> tuple[bool, str]:
    try:
        result = subprocess.run([binary, "--dry-run", "-Werror", str(path)],
                                capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"clang-format failed on this file: {error}"
    if result.returncode == 0:
        return True, ""
    return False, "does not match .clang-format"


def run(root: Path) -> tuple[Gaps, dict[str, int]]:
    gaps = Gaps()
    counts: dict[str, int] = {}
    binary = clang_format_binary()
    version = clang_format_version(binary)
    counts["version"] = 0
    print(f"clang-format: {version}")
    if shutil.which(binary) is None and not Path(binary).exists():
        gaps.add("clang-format-missing",
                 f"`{binary}` was not found. Install clang-format, or set CLANG_FORMAT to name the "
                 "binary. A formatting guard that passes because the tool is absent checks nothing.")
        return gaps, counts
    if major_version(version) is None:
        gaps.add("clang-format-version",
                 f"could not read a major version from `{version}`. The tree and the tool must move "
                 "together; name a versioned binary with CLANG_FORMAT.")
        return gaps, counts
    config = root / ".clang-format"
    if not config.is_file():
        gaps.add("config-missing", f"{config} is missing; the guard has nothing to check against.")
        return gaps, counts

    files = tracked_sources(root)
    counts["files"] = len(files)
    if not files:
        gaps.add("no-files", f"no C++ source files were found under {', '.join(SOURCE_DIRS)}")
        return gaps, counts

    workers = max(1, min(32, os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda path: (path, *file_matches(binary, path)), files))
    for path, matched, message in results:
        if matched:
            continue
        gaps.add("format", f"{path.relative_to(root)}: {message}")
    counts["violating"] = sum(1 for _, matched, _ in results if not matched)
    return gaps, counts


def summary(counts: dict[str, int]) -> str:
    return (f"ok: {counts.get('files', 0)} C++ files match .clang-format "
            f"({counts.get('violating', 0)} violating)")


SELF_TEST_BAD = """#include <cstdio>
int main ( ) {
    int  x=1;
  if(x){return 0;}
        return x;
}
"""

SELF_TEST_GOOD = """#include <cstdio>

int main() {
    const int x = 1;
    if (x) { return 0; }
    return x;
}
"""


def self_test() -> int:
    """Plant a misformatted file in a temporary tree and assert the guard reports it."""
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        shutil.copyfile(REPO / ".clang-format", root / ".clang-format")
        (root / "src").mkdir()
        (root / "src" / "bad.cpp").write_text(SELF_TEST_BAD)
        gaps, _ = run(root)
        if not any(kind == "format" and "bad.cpp" in message for kind, message in gaps.items):
            failures.append("a misformatted file was not reported")
        (root / "src" / "bad.cpp").write_text(SELF_TEST_GOOD)
        gaps, _ = run(root)
        if gaps.items:
            failures.append(f"a formatted file was reported: {gaps.items[0][1]}")
    if failures:
        print(f"self-test failed: {len(failures)} case(s)")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("self-test ok: a misformatted file is reported and a formatted file passes")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", default=str(REPO),
                        help="repository root (default: this script's checkout)")
    parser.add_argument("--self-test", action="store_true",
                        help="plant a misformatted file in a temporary tree and assert the guard "
                             "reports it")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    gaps, counts = run(Path(args.root).resolve())
    if gaps.items:
        print(f"{len(gaps.items)} formatting gap(s):")
        for kind, message in gaps.items[:200]:
            print(f"  [{kind}] {message}")
        if len(gaps.items) > 200:
            print(f"  ... and {len(gaps.items) - 200} more")
        return 1
    print(summary(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
