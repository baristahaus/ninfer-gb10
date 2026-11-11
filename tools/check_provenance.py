#!/usr/bin/env python3
"""Provenance guard: a file that claims a port must have a ledger row.

Every file that carries third-party code says so in its header ("Ported from capy-flash ...",
"Adapted from ..."). PROVENANCE.md is the maintained record of those claims, and a claim with no
row is how attribution rots: the header keeps crediting a project the ledger has never heard of,
and a licence audit reads only the ledger.

The guard has two checks, and they catch the two directions the ledger drifts:

  - `check_claim_has_row`: a `Ported from <project>` or `Adapted from <project>` claim in a file
    whose path PROVENANCE.md does not name is a gap. The ledger is read for its path anchors
    (backticked paths, with `{h,cpp}` alternatives expanded), so a new row is all it takes to
    satisfy the check — there is no hand-maintained list to update.
  - `check_row_paths_exist`: a source path named in PROVENANCE.md that does not exist in the tree
    is a stale row (a file was renamed or deleted, or the row cites an upstream path as if it were
    local). This is the other half of "the ledger stays true".

A claim is read from the file's header window only. The header is where attribution lives, and a
mid-file mention (one option default borrowed from a project) is not a file-level claim. The
exceptions are listed in `DEEP_CLAIMS` with their line, so a file whose claim sits deeper is still
covered without scanning every line of every file.

The ledger writes a component's files in one cell (`src/core/disk_kv_cache.{h,cpp}`); the brace
form is expanded before matching, so a row covering the header covers its .cpp too.

Usage: check_provenance.py [ROOT] [--self-test]. Exit 1 and list every gap, or 0 with a summary.
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEDGER = "PROVENANCE.md"
# The header is the first 60 lines: enough for the licence block and the port note, short enough
# that a body comment cannot masquerade as a file-level claim.
HEADER_WINDOW = 60
CLAIM = re.compile(r"\b(?:Ported|Adapted) from ([A-Za-z0-9_.-]+)")
# Files whose claim sits in the body rather than the header. Each entry is the line the guard
# should read, so the guard stays a header check for every other file.
DEEP_CLAIMS = {
    "include/ninfer/types.h": 152,
}
SOURCE_SUFFIXES = (".h", ".cuh", ".cu", ".cpp", ".inc", ".hpp")
SOURCE_ROOTS = ("src", "include", "apps", "bench", "tests")


class Gaps:
    def __init__(self) -> None:
        self.items: list[tuple[str, str]] = []

    def add(self, check: str, message: str) -> None:
        self.items.append((check, message))


def source_files(root: Path):
    for base in SOURCE_ROOTS:
        directory = root / base
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix in SOURCE_SUFFIXES:
                yield path


def _expand(token: str) -> list[str]:
    """Expand a `{h,cpp}` alternative inside one backticked path."""
    match = re.search(r"\{([^{}]+)\}", token)
    if not match:
        return [token]
    out: list[str] = []
    for alternative in match.group(1).split(","):
        out.extend(_expand(token[: match.start()] + alternative + token[match.end():]))
    return out


def ledger_paths(root: Path) -> set[str]:
    """Every source path named in PROVENANCE.md, with brace alternatives expanded."""
    text = (root / LEDGER).read_text(errors="replace")
    named: set[str] = set()
    for token in re.findall(r"`([^`]+)`", text):
        if "/" not in token:
            continue
        for expanded in _expand(token):
            if expanded.endswith(SOURCE_SUFFIXES):
                named.add(expanded)
    return named


def claim_line(path: Path, relative: str) -> int | None:
    """The line of the file's port claim, or None when it makes no claim."""
    text = path.read_text(errors="replace").splitlines()
    deep = DEEP_CLAIMS.get(relative)
    window = text if deep is not None else text[:HEADER_WINDOW]
    for index, line in enumerate(window):
        if CLAIM.search(line):
            return deep if deep is not None else index + 1
    return None


def check_claim_has_row(root: Path, gaps: Gaps) -> int:
    named = ledger_paths(root)
    checked = 0
    for path in source_files(root):
        relative = path.relative_to(root).as_posix()
        line = claim_line(path, relative)
        if line is None:
            continue
        checked += 1
        if relative in named:
            continue
        gaps.add("port without ledger row",
                 f"{relative}:{line}: the header claims a port but {LEDGER} has no row naming "
                 f"this path; add a row to §4 (code taken) or §6 (technique only) and name the "
                 f"file in it")
    return checked


def check_row_paths_exist(root: Path, gaps: Gaps) -> int:
    named = ledger_paths(root)
    checked = 0
    for relative in sorted(named):
        checked += 1
        if not (root / relative).exists():
            gaps.add("ledger row without file",
                     f"{LEDGER}: names `{relative}`, which does not exist in this tree; fix the "
                     f"path or drop the row (an upstream path belongs in prose, not in a "
                     f"backticked local path)")
    return checked


CHECKS = (("claim has row", check_claim_has_row), ("row paths exist", check_row_paths_exist))


def run(root: Path) -> tuple[Gaps, dict[str, int]]:
    gaps = Gaps()
    counts: dict[str, int] = {}
    for name, check in CHECKS:
        counts[name] = check(root, gaps)
    return gaps, counts


def summary(counts: dict[str, int]) -> str:
    return (f"ok: {counts['claim has row']} port claims covered by the ledger, "
            f"{counts['row paths exist']} ledger paths present")


FIXTURE_LEDGER = """# Provenance

## 4. Vendored source

| component | origin | licence | scope |
|---|---|---|---|
| ported kernel | capy-flash | Apache-2.0 | `src/a.cu`, `src/b.{h,cpp}`, `src/adapted.cuh` |
| ported test | capy-flash | Apache-2.0 | `tests/t.cpp` |
"""

FIXTURE_FILES = {
    "src/a.cu": "// Ported from capy-flash src/a.cu.\nint a;\n",
    "src/b.h": "// Ported from capy-flash src/b.{h,cpp}.\nint b;\n",
    "src/b.cpp": "#include \"b.h\"\n",
    "src/adapted.cuh": "// Adapted from capy-flash src/adapted.cuh.\nint c;\n",
    "tests/t.cpp": "// Ported from capy-flash tests/t.cpp.\nint d;\n",
    "src/original.cpp": "// This file is original to this tree.\nint e;\n",
}


def write_fixture(root: Path) -> None:
    (root / LEDGER).write_text(FIXTURE_LEDGER)
    for relative, text in FIXTURE_FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def self_test() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_fixture(root)
        gaps, counts = run(root)
        if gaps.items:
            print("self-test FAILED on the clean fixture:")
            for _, message in gaps.items:
                print(f"  {message}")
            return 1
        # Plant a claim with no ledger row: the guard must report exactly that file.
        planted = root / "src" / "unrecorded.cu"
        planted.write_text("// Ported from capy-flash src/unrecorded.cu.\nint f;\n")
        gaps, _ = run(root)
        reported = [message for check, message in gaps.items if "unrecorded.cu" in message]
        if len(reported) != 1:
            print(f"self-test FAILED: planted claim reported {len(reported)} times")
            return 1
        planted.unlink()
        # Plant a stale ledger row: a named path that does not exist.
        ledger = root / LEDGER
        ledger.write_text(ledger.read_text() + "\n| stale | capy-flash | Apache-2.0 | `src/gone.cu` |\n")
        gaps, _ = run(root)
        reported = [message for check, message in gaps.items if "src/gone.cu" in message]
        if len(reported) != 1:
            print(f"self-test FAILED: planted stale row reported {len(reported)} times")
            return 1
        ledger.write_text(FIXTURE_LEDGER)
        gaps, _ = run(root)
        if gaps.items:
            print("self-test FAILED: the clean fixture does not pass after the planted cases")
            return 1
    print("self-test ok: covered claims pass; an unrecorded claim and a stale row are caught")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", type=Path, default=REPO)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    root = args.root.resolve()
    if not (root / LEDGER).is_file():
        print(f"{LEDGER} is missing under {root}", file=sys.stderr)
        return 2
    gaps, counts = run(root)
    if gaps.items:
        print(f"{len(gaps.items)} provenance gap(s):")
        for check, message in gaps.items:
            print(f"  [{check}] {message}")
        return 1
    print(summary(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
