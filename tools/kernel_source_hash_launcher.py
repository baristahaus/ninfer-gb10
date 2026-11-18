#!/usr/bin/env python3
"""CUDA compiler launcher that records the source content each object was built from.

CMake prepends this script to every nvcc invocation (CMAKE_CUDA_COMPILER_LAUNCHER,
set in cmake/NinferPlatform.cmake when NINFER_TRACK_KERNEL_SOURCES is ON). For
kernel-shape sources -- the dispatch tables that pin kernel regimes -- it writes
an `<object>.srcsha` sidecar next to the object containing the SHA-256 of the
source as compiled:

    src/ops/**/shapes/*.cu
    src/ops/linear_attention/**/*.cu
    src/ops/softmax_attention/**/*.cu
    src/ops/normalization/**/*.cu

tools/check_kernel_staleness.py compares those records against the current
sources. The point is that mtimes cannot be trusted: a tar-based source sync
with preserved timestamps sets source mtimes OLD while the content is NEW, so
ninja sees the object as up to date, skips the recompile, and a plain mtime
check sees nothing wrong. The sidecar is written only when the object is
actually compiled, so a skipped recompile leaves the old record in place and
the content mismatch is caught.

The launcher must never break a build: every failure path falls through to
executing the real compiler command untouched.
"""

from __future__ import annotations

import hashlib
import os
import sys

GUARDED_DISPATCH_DIRS = (
    "src/ops/linear_attention/",
    "src/ops/softmax_attention/",
    "src/ops/normalization/",
)


def is_guarded(source: str) -> bool:
    """True for the kernel-shape sources whose objects carry a .srcsha record."""
    path = os.path.normpath(source).replace(os.sep, "/")
    for marker in GUARDED_DISPATCH_DIRS:
        if marker in path:
            return True
    return "/src/ops/" in f"/{path}" and "/shapes/" in f"{path}/"


def track(argv: list[str]) -> None:
    """Write <object>.srcsha for a guarded compile; raise on any problem."""
    source = None
    obj = None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "-o" and index + 1 < len(argv):
            obj = argv[index + 1]
            index += 2
            continue
        if arg.startswith("-o") and len(arg) > 2 and not arg.startswith("--"):
            obj = arg[2:]
        if arg.endswith(".cu"):
            source = arg
        index += 1
    if not source or not obj or not is_guarded(source):
        return
    digest = hashlib.sha256(open(source, "rb").read()).hexdigest()
    with open(obj + ".srcsha", "w", encoding="utf-8") as sidecar:
        sidecar.write(f"{digest}  {os.path.normpath(source).replace(os.sep, '/')}\n")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    command = sys.argv[1:]
    try:
        track(command)
    except Exception:  # noqa: BLE001 - staleness tracking must never fail a build
        pass
    os.execvp(command[0], command)  # pragma: no cover - replaces the process


if __name__ == "__main__":
    sys.exit(main())
