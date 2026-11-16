#!/usr/bin/env python3
"""Duplicate-kernel guard: no linked artifact may register the same kernel twice.

CUDA registers every kernel by its entry symbol when a module loads. When one artifact carries the
same entry symbol twice, the module registers it twice and the device tools refuse the binary:
compute-sanitizer reports "Duplicate entry kernels named ..." and cuda-gdb reports "No CUDA
kernels. No CUDA devices." on a live process.

The check reads the linked artifacts, because that is the level the driver loads. A scan of
individual objects overstates the defect: with separable compilation the device linker merges
duplicate weak template instantiations, so an object-level duplicate may never reach a module. The
kernels that do reach it are the ones in a non-RDC target, where each object's fat binary is carried
into the final module unchanged.

`--objects` runs the object-level scan as a diagnostic. Treat its count as an upper bound and the
artifact count as the defect.

Usage: check_duplicate_kernels.py [ROOT] [--build-dir DIR] [--objects] [--limit N]
Exit 1 and list every duplicated symbol, or 0 with a summary.

The guard fails when it scans zero artifacts: a check that passes because it found nothing to read
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


def artifact_paths(build_dir: Path) -> list[Path]:
    """Every shared library and executable under the build directory."""
    found = []
    for pattern in ("lib/*.so", "lib/*.so.*", "apps/*", "tests/*", "bench/*"):
        for path in sorted(build_dir.glob(pattern)):
            if path.is_file() and os.access(path, os.X_OK):
                found.append(path)
    return found


def cuda_objects(build_dir: Path) -> list[Path]:
    ninja = build_dir / "build.ninja"
    if not ninja.is_file():
        raise SystemExit(f"{ninja} is missing; configure the build first")
    text = ninja.read_text(errors="replace")
    return [build_dir / name for name in sorted(set(re.findall(r"^build (\S+\.cu\.o):", text, re.M)))]


def entry_symbols(path: Path) -> list[str]:
    """The STO_ENTRY symbols of one artifact, or an empty list when it has no device code."""
    try:
        result = subprocess.run([CUOBJDUMP, "-symbols", str(path)],
                                capture_output=True, text=True, timeout=600)
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


def scan(paths: list[Path], root: Path, limit: int, label: str) -> tuple[int, int]:
    """Return (duplicated symbol count, scanned artifact count) and print the duplicated symbols."""
    owners: dict[str, list[Path]] = defaultdict(list)
    scanned = 0
    for path in paths:
        symbols = entry_symbols(path)
        if not symbols:
            continue
        scanned += 1
        for symbol in symbols:
            owners[symbol].append(path)
    duplicated = {symbol: files for symbol, files in owners.items() if len(files) > 1}
    print(f"{label}: scanned {scanned}, {len(owners)} distinct entry symbols, "
          f"{len(duplicated)} registered more than once")
    if not duplicated:
        return 0, scanned
    symbols = list(duplicated)
    pretty = demangle(symbols)
    for symbol, name in sorted(zip(symbols, pretty), key=lambda row: row[1])[:limit]:
        print(f"  {len(duplicated[symbol])}x  {name[:180]}")
        for path in duplicated[symbol]:
            print(f"        {path.relative_to(root)}")
    if len(symbols) > limit:
        print(f"  ... and {len(symbols) - limit} more")
    return len(duplicated), scanned


def run(root: Path, build_dir: Path, objects: bool, limit: int) -> int:
    if shutil.which(CUOBJDUMP) is None and not Path(CUOBJDUMP).exists():
        print(f"error: `{CUOBJDUMP}` was not found. Install the CUDA toolkit, or set CUOBJDUMP.")
        return 1

    artifacts = artifact_paths(build_dir)
    if not artifacts:
        print(f"error: no shared libraries or executables under {build_dir}; build first")
        return 1
    duplicated, scanned = scan(artifacts, root, limit, "linked artifacts")
    if scanned == 0:
        print(f"error: none of the {len(artifacts)} artifacts carries device code; build the CUDA "
              "targets first")
        return 1

    if objects:
        object_paths = [path for path in cuda_objects(build_dir) if path.is_file()]
        if object_paths:
            print()
            object_duplicates, _ = scan(object_paths, root, limit, "objects (diagnostic only)")
            if object_duplicates:
                print()
                print("Object-level duplicates are an upper bound, not the defect: the device linker")
                print("merges weak template instantiations in an RDC target. Only a duplicate that")
                print("survives into a linked artifact reaches the driver.")

    if duplicated:
        print()
        print("A duplicate entry symbol in a linked artifact means the same __global__ function was")
        print("emitted twice into it. For an RDC target the device linker did not merge it, which")
        print("means the two copies are not weak instantiations of one template. For a non-RDC")
        print("target nothing merges them, so the same template instantiated in two translation")
        print("units is enough. Give the instantiation one owner, as")
        print("src/ops/linear/kernels/nvfp4/nvfp4_a4_tma_shared.h does.")
        return 1
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", default=str(REPO),
                        help="repository root (default: this script's checkout)")
    parser.add_argument("--build-dir", default=None,
                        help="build directory (default: <root>/build)")
    parser.add_argument("--objects", action="store_true",
                        help="also scan individual objects as a diagnostic")
    parser.add_argument("--limit", type=int, default=40,
                        help="maximum duplicated symbols to list (default: 40)")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    build_dir = Path(args.build_dir).resolve() if args.build_dir else root / "build"
    return run(root, build_dir, args.objects, args.limit)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
