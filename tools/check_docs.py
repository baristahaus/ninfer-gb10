#!/usr/bin/env python3
"""Documentation guard: what the code accepts must be in the user documentation.

Checks:
  - every `--flag` that `src/serve/serve_options.cpp` parses appears in an option table of
    `docs/serving.md` (a table row naming it in backticks);
  - every `NINFER_*` environment variable read with getenv() under `src/` is named somewhere under
    `docs/`.

Usage: check_docs.py [repository root]. Exit 1 and list what is missing.

Run it from CI or by hand; it is the guard against the class of drift where a flag or an environment
variable ships without a line in the docs.
"""
import re
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])


def read(path):
    return (root / path).read_text(encoding="utf-8")


def source_files():
    for directory in ("src", "include"):
        for path in (root / directory).rglob("*"):
            if path.suffix in {".cpp", ".h", ".cu", ".cuh", ".inc"}:
                yield path


missing = []

# 1. Serve flags must be documented in a docs/serving.md option table.
serving = read("docs/serving.md")
table_rows = "\n".join(line for line in serving.splitlines() if line.startswith("|"))
flags = sorted(set(re.findall(r'arg == "(--[a-z0-9-]+)"', read("src/serve/serve_options.cpp"))))
missing += [
    f"serve flag {flag} is parsed but no docs/serving.md option table names it"
    for flag in flags
    if f"`{flag}`" not in table_rows and f"`{flag} " not in table_rows
]

# 2. Environment variables must be documented somewhere under docs/.
documented_text = "\n".join(
    path.read_text(encoding="utf-8", errors="replace") for path in (root / "docs").rglob("*.md")
)
environment = set()
for path in source_files():
    environment |= set(
        re.findall(r'getenv\(\s*"(NINFER_[A-Z0-9_]+)"', path.read_text(encoding="utf-8", errors="replace"))
    )
missing += [
    f"environment variable {name} is read but no docs/*.md names it"
    for name in sorted(environment)
    if name not in documented_text
]

if missing:
    print(f"{len(missing)} documentation gap(s):")
    for entry in missing:
        print(f"  {entry}")
    sys.exit(1)

print(f"ok: {len(flags)} serve flags and {len(environment)} environment variables are documented")
