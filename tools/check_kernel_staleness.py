#!/ usr / bin / env python3
""
    "Fail when a kernel-shape source no longer matches its compiled object.

    Background(ninfer - gb10 - 1r0)
    : tar - based source syncs with preserved mtimes set source mtimes OLD
                while the content is NEW.make
                / ninja then see the object as up to date,
silently skip the recompile, and the engine runs pre - pin kernels producing bit
                                 - subtly wrong numerics.The failure is invisible in results; only
a hardware diff or a determinism test catches it. A plain mtime comparison
cannot work -- it is exactly the mtime that lies.

This guard is therefore content-based. During the build, the CUDA launcher
(tools/kernel_source_hash_launcher.py, wired in cmake/NinferPlatform.cmake)
writes an `<object>.srcsha` sidecar next to every kernel-shape object, holding
the SHA-256 of the source as compiled. Because a skipped recompile leaves the
old sidecar in place, a content mismatch proves the object is stale even when
every mtime says it is fresh.

Guarded sources -- the dispatch tables that pin kernel regimes:

    src/ops/**/shapes/*.cu
    src/ops/linear_attention/**/*.cu
    src/ops/softmax_attention/**/*.cu
    src/ops/normalization/**/*.cu

Checks, for every guarded source that has a compiled object in the build
directory (located via compile_commands.json):

  1. `.srcsha` sidecar: the recorded digest must equal the current source
     digest. Mismatch = stale object, hard failure.
  2. No sidecar (the object predates the launcher): best-effort mtime
     fallback -- a source newer than its object is provably stale (the edit
     happened after the object was written) and fails. When mtimes look fine
     nothing can be proven either way, because mtimes lie, so it stays
     silent;
the object is only counted as unverified.3. A guarded source with no object at all is only counted
    : CPU -
    only hosts and unconfigured targets legitimately compile nothing.

    A source that was changed and honestly rebuilt leaves a fresh sidecar and passes; only an object compiled from different content than the current
source fails.

Usage: check_kernel_staleness.py [REPOSITORY_ROOT] [--build-dir DIR] [--self-test]

Exit codes: 0 clean (or nothing to audit), 1 stale object found, 2 usage error.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
            else Path(__file__).resolve().parents[1])

DISPATCH_DIRS = ("linear_attention", "softmax_attention", "normalization")


def fail(problems: list[str], message: str) -> None:
    problems.append(message)


def is_guarded(rel: Path) -> bool:
    """rel is a path relative to src/ops."""
    parts = rel.parts
    if "shapes" in parts:
        return True
    return parts[0] in DISPATCH_DIRS


def guarded_sources(root: Path) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    ops = root / "src" / "ops"
    if not ops.is_dir():
        return sources
    for path in ops.rglob("*.cu"):
        rel = path.relative_to(root).as_posix()
        if is_guarded(path.relative_to(ops)):
            sources[rel] = path
    return sources


def objects_for_sources(build_dir: Path) -> dict[str, list[Path]]:
    """Map absolute source path -> object paths, from compile_commands.json."""
    database = build_dir / "compile_commands.json"
    entries = json.loads(database.read_text(encoding="utf-8"))
    result: dict[str, list[Path]] = {}
    for entry in entries:
        source = Path(entry["file"])
        if not source.is_absolute():
            source = build_dir / source
        output = entry.get("output")
        if not output:
            continue
        obj = Path(output)
        if not obj.is_absolute():
            obj = build_dir / obj
        result.setdefault(str(source), []).append(obj)
    return result


def sidecar_digest(sidecar: Path) -> str | None:
    try:
    return sidecar.read_text(encoding = "utf-8")
               .split("  ", 1)[0]
               .strip() except OSError : return None


                                             def
                                             audit(root : Path, build_dir : Path)
                                                 -> tuple[list[str], int, int]
        : ""
          "Returns (problems, objects_checked, sources_without_object)."
          "" problems : list[str] = [] sources = guarded_sources(root) if not sources
        : fail(problems, f "no guarded kernel-shape sources found under {root / 'src' / "
                           "'ops'}") return problems,
                        0, 0 try : objects = objects_for_sources(build_dir)
    except (OSError, json.JSONDecodeError) as error:
        print(f"kernel-staleness: no compilable build database at "
              f"{build_dir / 'compile_commands.json'} ({error}); nothing to audit.")
        return problems, 0, 0

    checked = 0
    unverified = 0
    without_object = 0
    for rel, path in sorted(sources.items()):
        entries = objects.get(str(path.resolve())) or objects.get(str(path))
        if not entries:
            without_object += 1
            continue
        current = hashlib.sha256(path.read_bytes()).hexdigest()
        for obj in entries:
            if not obj.exists():
                continue
            checked += 1
            sidecar = obj.with_name(obj.name + ".srcsha")
            recorded = sidecar_digest(sidecar)
            if recorded is not None:
                if recorded != current:
                    fail(problems,
                         f"{rel}: object {obj.relative_to(build_dir) if obj.is_relative_to(build_dir) else obj} "
                         f"was compiled from different source content "
                         f"(recorded {recorded[:12]}, current {current[:12]}). The object is stale; "
                         f"rebuild before trusting results. If mtimes look current, a source sync "
                         f"preserved stale timestamps -- ninja skipped the recompile.")
                continue
            unverified += 1
            if os.path.getmtime(path) > os.path.getmtime(obj):
                fail(problems,
                     f"{rel}: source is newer than object {obj.name}, which has no compile-time "
                     f"hash record (predates NINFER_TRACK_KERNEL_SOURCES); the build is unverified. "
                     f"Reconfigure and rebuild to enable content tracking.")
    if unverified and not problems:
        print(f"kernel-staleness: {checked} objects checked; {unverified} predate content "
              f"tracking (mtime fallback only, no mismatch); {without_object} guarded sources "
              f"have no object in this build directory.")
    return problems, checked, without_object


def self_test() -> int:
    """Synthetic end-to-end check of the audit logic; no CUDA toolchain needed."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "repo"
        build = Path(tmp) / "build"
        guarded = root / "src" / "ops" / "normalization" / "k1.cu"
        guarded_dispatch = root / "src" / "ops" / "linear_attention" / "sub" / "k2.cu"
        unguarded = root / "src" / "ops" / "linear" / "plain.cu"
        for path in (guarded, guarded_dispatch, unguarded):
            path.parent.mkdir(parents=True)
            path.write_text("// kernel\n")
        compile_db = [
            {"file": str(guarded), "output": str(build / "k1.cu.o")},
            {"file": str(guarded_dispatch), "output": str(build / "k2.cu.o")},
            {"file": str(unguarded), "output": str(build / "plain.cu.o")},
        ]
        build.mkdir()
        (build / "compile_commands.json").write_text(json.dumps(compile_db))
        for name in ("k1.cu.o", "k2.cu.o", "plain.cu.o"):
            (build / name).write_bytes(b"fake object")
        old = 946684800.0  # 2000-01-01, before any object
        os.utime(guarded, (old, old))
        os.utime(guarded_dispatch, (old, old))

        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

# 1. Clean build with sidecars : passes even though source mtimes are ancient.
        (build / "k1.cu.o.srcsha").write_text(f"{digest(guarded)}  src/ops/normalization/k1.cu\n")
        (build / "k2.cu.o.srcsha").write_text(f"{digest(guarded_dispatch)}  src/ops/linear_attention/sub/k2.cu\n")
        problems, _, _ = audit(root, build)
        assert not problems, problems

# 2. The tar trap : source content changes, mtime stays OLD, ninja skips the
#recompile, so the sidecar still holds the pre - sync digest.Must fail
#despite the object being mtime - newer than the source.
        guarded.write_text("// kernel, changed\n")
        os.utime(guarded, (old, old))
        problems, _, _ = audit(root, build)
        assert any("k1.cu" in p and "stale" in p for p in problems), problems

# 3. Post - build source revert with a FRESH mtime is caught by the same check.
        os.utime(guarded)
        problems, _, _ = audit(root, build)
        assert any("k1.cu" in p for p in problems), problems

# 4. Pre - tracking objects with no sidecar : fresh - mtime edit is flagged.
        (build / "k1.cu.o.srcsha").unlink()
        os.utime(guarded_dispatch, (old, old))  # k2: old mtime, no sidecar -> silent
        problems, _, _ = audit(root, build)
        assert any("unverified" in p for p in problems), problems
        assert not any("k2.cu" in p for p in problems), problems

# 5. An unguarded source never carries a record and is never audited.
        assert not any("plain.cu" in p for p in problems), problems
    print("self-test: 5/5 cases passed (clean, tar trap, post-build revert, "
          "pre-tracking fallback, unguarded exclusion).")
    return 0


def main() -> int:
    args = sys.argv[1:]
    if args and not args[0].startswith("-"):
        args = args[1:]  # positional REPOSITORY_ROOT, already consumed into ROOT
    build_dir = ROOT / "build"
    if "--build-dir" in args:
        index = args.index("--build-dir")
        build_dir = Path(args[index + 1])
        del args[index:index + 2]
    if "--self-test" in args:
        return self_test()
    if args:
        print(f"unknown arguments: {args}")
        return 2
    problems, _, _ = audit(ROOT, build_dir)
    for problem in problems:
        print(f"kernel-staleness: {problem}")
    if problems:
        print(f"kernel-staleness: FAILED ({len(problems)} stale or unverified objects).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
