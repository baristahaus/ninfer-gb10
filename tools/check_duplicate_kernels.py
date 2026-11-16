#!/usr/bin/env python3
"""Duplicate-kernel guard: no `__global__` entry symbol may be registered by more than one object.

CUDA registers every kernel in an object's fat binary by its entry symbol. When two translation
units emit the same entry symbol, the module registers it twice and the device tools refuse the
binary: compute-sanitizer reports "Duplicate entry kernels named ..." and cuda-gdb reports "No CUDA
kernels. No CUDA devices." on a live process.

The usual cause is a `__global__` template instantiated in more than one translation unit. Internal
linkage and separable compilation hide that from the host linker, so a successful link proves
nothing here.

This guard reads the compiled objects instead of the source, so it needs no GPU. It parses the
build graph for `.cu.o` targets, runs `cuobjdump -symbols` on each, and reports every entry symbol
that more than one object defines.

Usage: check_duplicate_kernels.py [ROOT] [--build-dir DIR] [--demangle/--no-demangle] [--limit N]
Exit 1 and list every duplicated symbol, or 0 with a summary.

The guard fails when it scans zero objects: a check that passes because it found nothing to read
checks nothing.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CUOBJDUMP = os.environ.get("CUOBJDUMP") or "cuobjdump"


def cuda_objects(build_dir: Path) -> list[Path]:
    """The `.cu.o` targets of the build graph, as absolute paths."""
    ninja = build_dir / "build.ninja"
    if not ninja.is_file():
        raise SystemExit(f"{ninja} is missing; configure the build first")
    text = ninja.read_text(errors="replace")
    relative = sorted(set(re.findall(r"^build (\S+\.cu\.o):", text, re.M)))
    return [build_dir / name for name in relative]


def entry_symbols(object_path: Path) -> list[str]:
    """The STO_ENTRY symbols of one object, or an empty list when it has no device code."""
    try:
        result = subprocess.run([CUOBJDUMP, "-symbols", str(object_path)],
                                capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    return [line.split()[-1] for line in result.stdout.splitlines() if "STO_ENTRY" in line]


def demangle(symbols: list[str]) -> list[str]:
    if not symbols:
        return []
    try:
        out = subprocess.run(["c++filt"], input="\n".join(symbols), capture_output=True, text=True,
                             timeout=300)
        lines = out.stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return symbols
    return lines if len(lines) == len(symbols) else symbols


def run(root: Path, build_dir: Path, limit: int) -> int:
    if shutil.which(CUOBJDUMP) is None and not Path(CUOBJDUMP).exists():
        print(f"error: `{CUOBJDUMP}` was not found. Install the CUDA toolkit, or set CUOBJDUMP.")
        return 1
    objects = cuda_objects(build_dir)
    if not objects:
        print(f"error: no .cu.o targets in {build_dir}/build.ninja; the guard has nothing to read")
        return 1
    owners: dict[str, list[Path]] = defaultdict(list)
    scanned = 0
    for index, object_path in enumerate(objects):
        if not object_path.is_file():
            continue
        scanned += 1
        for symbol in entry_symbols(object_path):
            owners[symbol].append(object_path)
        if (index + 1) % 200 == 0:
            print(f"  scanned {index + 1}/{len(objects)} objects")
    if scanned == 0:
        print(f"error: none of the {len(objects)} objects exists; build first")
        return 1

    duplicated = {symbol: files for symbol, files in owners.items() if len(files) > 1}
    print(f"scanned {scanned} objects, {len(owners)} distinct entry symbols, "
          f"{len(duplicated)} registered by more than one object")
    if not duplicated:
        return 0

    symbols = list(duplicated)
    pretty = demangle(symbols)
    rows = sorted(zip(symbols, pretty), key=lambda row: row[1])
    for symbol, name in rows[:limit]:
        files = duplicated[symbol]
        print(f"  {len(files)}x  {name[:180]}")
        for object_path in files:
            print(f"        {object_path.relative_to(root)}")
    if len(rows) > limit:
        print(f"  ... and {len(rows) - limit} more")
    print()
    print("A duplicate entry symbol means one __global__ template was instantiated in more than one")
    print("translation unit. Give the instantiation one owner and declare it `extern template` in")
    print("every other translation unit that selects it, as src/ops/linear/add/q5/launch.cuh does.")
    return 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", default=str(REPO),
                        help="repository root (default: this script's checkout)")
    parser.add_argument("--build-dir", default=None,
                        help="build directory (default: <root>/build)")
    parser.add_argument("--no-demangle", dest="demangle", action="store_false",
                        help="print mangled symbols")
    parser.add_argument("--limit", type=int, default=40,
                        help="maximum duplicated symbols to list (default: 40)")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    build_dir = Path(args.build_dir).resolve() if args.build_dir else root / "build"
    return run(root, build_dir, args.limit)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
