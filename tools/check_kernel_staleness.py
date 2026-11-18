#!/usr/bin/env python3
"""Fail when a kernel-shape source no longer matches its compiled object.

Background (ninfer-gb10-1r0): tar-based source syncs with preserved mtimes set source
mtimes OLD while the content is NEW. make/ninja then see the object as up to date,
silently skip the recompile, and the engine runs pre-pin kernels producing bit-subtly
wrong numerics. The failure is invisible in results; only a content diff or a
determinism test catches it. A plain mtime comparison cannot work -- the mtime is
exactly what lies.

The authoritative signal is content: cmake/NinferPlatform.cmake (when
NINFER_TRACK_KERNEL_SOURCES=ON, the default) wraps every nvcc invocation with
tools/kernel_source_hash_launcher.py, which writes SHA256(source) into a
<object>.srcsha sidecar at compile time. This guard compares each sidecar against the
current source and fails on mismatch. A skipped recompile leaves the old sidecar in
place, which is exactly what makes the staleness visible.

For objects built before tracking was enabled (no sidecar), the guard falls back to
the best mtime evidence and reports the pair as "unverified" rather than failed; a
source provably newer than its object is still a failure.

Guarded set: src/ops/**/shapes/*.cu plus every .cu under the
linear_attention/softmax_attention/normalization op families (306 files). Objects are
located through build/compile_commands.json.

Usage:
    python3 tools/check_kernel_staleness.py [ROOT] [--build-dir DIR] [--self-test]

Exit 0 with a summary, or exit 1 listing every stale or failed pair. --self-test
runs the synthetic tar-trap cases with no build directory and no CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

GUARDED_SUFFIXES = (".cu",)
GUARDED_DIR_PARTS = ("shapes", "linear_attention", "softmax_attention", "normalization")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_guarded(source: Path) -> bool:
    if source.suffix not in GUARDED_SUFFIXES:
        return False
    return any(part in GUARDED_DIR_PARTS for part in source.parts)


def sidecar_for(object_path: Path) -> Path:
    return object_path.with_name(object_path.name + ".srcsha")


def verify_pair(source: Path, object_path: Path) -> tuple[str, str]:
    """Classify one source/object pair.

    Returns (verdict, detail) where verdict is "ok", "stale", or "unverified".
    """
    sidecar = sidecar_for(object_path)
    if not sidecar.exists():
        if source.stat().st_mtime > object_path.stat().st_mtime:
            return "stale", "no sidecar and the source is newer than the object"
        return "unverified", "object was built before source tracking was enabled"
    recorded = sidecar.read_text().strip()
    current = sha256_of(source)
    if recorded == current:
        return "ok", ""
    return (
        "stale",
        f"object was compiled from different source content "
        f"(recorded {recorded[:12]}, current {current[:12]})",
    )


def write_sidecar(object_path: Path, source: Path) -> None:
    sidecar_for(object_path).write_text(sha256_of(source) + "\n")


def guarded_sources(root: Path) -> list[Path]:
    ops = root / "src" / "ops"
    if not ops.is_dir():
        return []
    out = []
    for path in sorted(ops.rglob("*.cu")):
        if is_guarded(path):
            out.append(path)
    return out


def objects_from_compile_commands(build_dir: Path) -> dict[Path, Path]:
    """Map absolute source path -> object path from compile_commands.json."""
    manifest = build_dir / "compile_commands.json"
    if not manifest.is_file():
        return {}
    entries = json.loads(manifest.read_text())
    out: dict[Path, Path] = {}
    for entry in entries:
        command = entry.get("command", "")
        if "-c" not in command:
            continue
        pieces = command.split()
        try:
            index = pieces.index("-c")
        except ValueError:
            continue
        if index + 1 >= len(pieces) or "-o" not in pieces:
            continue
        try:
            output = pieces[pieces.index("-o") + 1]
        except IndexError:
            continue
        source = Path(entry.get("directory", ".")) / pieces[index + 1]
        out[source.resolve()] = Path(entry.get("directory", ".")) / output
    return out


def run_guard(root: Path, build_dir: Path) -> int:
    sources = [s for s in guarded_sources(root) if is_guarded(s)]
    if not sources:
        print("kernel staleness: no guarded sources found")
        return 0
    mapping = objects_from_compile_commands(build_dir)
    if not mapping:
        print(
            "kernel staleness: no compile_commands.json under "
            f"{build_dir}; configure the build with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
        )
        return 0
    failures = []
    unverified = 0
    checked = 0
    for source in sources:
        resolved = source.resolve()
        if resolved not in mapping:
            continue
        checked += 1
        verdict, detail = verify_pair(resolved, mapping[resolved])
        if verdict == "stale":
            failures.append(f"  [stale] {source}: {detail}")
        elif verdict == "unverified":
            unverified += 1
    print(
        f"kernel staleness: {checked} guarded objects checked, "
        f"{len(failures)} stale, {unverified} unverified (pre-tracking builds)"
    )
    for failure in failures:
        print(failure)
    return 1 if failures else 0


def self_test() -> int:
    """Synthetic tar-trap cases; no build directory and no CUDA required."""
    cases = [
        # (name, source_content, object_content_built_from, source_mtime_delta,
        #  sidecar_present, expected)
        (
            "honest build passes",
            "kernel v1", "kernel v1", 0, True, "ok",
        ),
        (
            "tar trap: content changed under an old mtime with a skipped recompile",
            "kernel v2", "kernel v1", -6 * 365 * 24 * 3600, True, "stale",
        ),
        (
            "mtime-only change with identical content passes",
            "kernel v1", "kernel v1", 3600, True, "ok",
        ),
        (
            "pre-tracking: source newer than the object with no sidecar is caught",
            "kernel v2", "kernel v1", 3600, False, "stale",
        ),
        (
            "pre-tracking: source older than the object is unverified, not failed",
            "kernel v1", "kernel v1", -3600, False, "unverified",
        ),
    ]
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for name, source_text, built_text, mtime_delta, sidecar, expected in cases:
            source = base / f"source_{abs(hash(name)) % 10000}.cu"
            object_path = base / f"object_{abs(hash(name)) % 10000}.o"
            source.write_text(source_text)
            object_path.write_text(built_text)
            stamp = object_path.stat().st_mtime + mtime_delta
            import os

            os.utime(source, (stamp, stamp))
            if sidecar:
                # The launcher records the hash of the content AS COMPILED; the
                # later content change is what the guard must catch.
                object_path.with_name(object_path.name + ".srcsha").write_text(
                    hashlib.sha256(built_text.encode()).hexdigest() + "\n"
                )
            verdict, _ = verify_pair(source, object_path)
            if verdict != expected:
                print(f"self-test FAIL: {name}: expected {expected}, got {verdict}")
                failures += 1
            else:
                print(f"self-test ok: {name}")
    if failures:
        print(f"self-test: {failures} failure(s)")
        return 1
    print("self-test: all cases passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", default=str(REPO))
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        return self_test()
    return run_guard(Path(arguments.root).resolve(), Path(arguments.build_dir).resolve())


if __name__ == "__main__":
    sys.exit(main())
