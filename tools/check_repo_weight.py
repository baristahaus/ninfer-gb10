#!/usr/bin/env python3
"""Fail when the tracked tree grows a file it should not carry.

The repository carries source, fixtures and small corpora. It does not carry profiler output, model
weights or build products: the ignore rules already exclude them, but rules only stop NEW files, and
a large blob that reaches history cannot be removed without rewriting every commit. So the guard
checks three things and refuses to guess about any of them:

  * a tracked file larger than the per-file limit, unless it is listed in the allowlist with a reason;
  * a tracked file in a raw profiler format, whatever its size;
  * the total tracked size above a ceiling, so slow accretion is caught as well as single large files.

The allowlist is a checked-in text file: one path prefix per line, `#` comments, and each entry is a
deliberate exception someone had to write down.
"""
import argparse
import pathlib
import subprocess
import sys

PER_FILE_LIMIT_MB = 4.0
TOTAL_LIMIT_MB = 100.0
PROFILER_SUFFIXES = (".sqlite", ".sqlite3", ".nsys-rep", ".ncu-rep", ".qdrep", ".nvvp", ".rep")
ALLOWLIST = "tools/repo_weight_allowlist.txt"


def tracked_files(root):
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True)
    return [p for p in out.stdout.decode().split("\0") if p]


def allowlist(root):
    path = pathlib.Path(root) / ALLOWLIST
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            entries.append(line)
    return entries


def allowed(rel, entries):
    return any(rel == e or rel.startswith(e.rstrip("/") + "/") for e in entries)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--per-file-mb", type=float, default=PER_FILE_LIMIT_MB)
    parser.add_argument("--total-mb", type=float, default=TOTAL_LIMIT_MB)
    args = parser.parse_args(argv)

    root = pathlib.Path(args.root).resolve()
    entries = allowlist(root)
    files = tracked_files(root)
    problems = []
    total = 0
    for rel in files:
        path = root / rel
        if not path.is_file():
            continue
        size = path.stat().st_size
        total += size
        mb = size / (1024 * 1024)
        if path.suffix.lower() in PROFILER_SUFFIXES:
            problems.append(f"{rel}: raw profiler output ({mb:.1f} MB) is not tracked; "
                            f"commit the summary, not the trace")
        elif mb > args.per_file_mb and not allowed(rel, entries):
            problems.append(f"{rel}: {mb:.1f} MB exceeds the {args.per_file_mb:.0f} MB per-file limit "
                            f"and is not in {ALLOWLIST}")
    total_mb = total / (1024 * 1024)
    if total_mb > args.total_mb:
        problems.append(f"tracked tree is {total_mb:.1f} MB, above the {args.total_mb:.0f} MB ceiling")

    if problems:
        for problem in problems:
            print(f"repo weight: {problem}")
        return 1
    print(f"ok: {len(files)} tracked files, {total_mb:.1f} MB total, "
          f"{len(entries)} allowlisted prefixes, no profiler output")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
