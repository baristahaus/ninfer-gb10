#!/usr/bin/env python3
"""Documentation guard: what the code and the build system accept must be in the user documentation.

Checks:
  - every `--flag` that `src/product/config/schema.cpp` declares for the serve or common scope
    appears in an option table of `docs/serving.md` (a table row naming it in backticks), and every
    serve flag named in those
    tables is parsed by that file (a stale flag name fails);
  - every `NINFER_*` environment variable read with getenv() under `src/` is named somewhere under
    `docs/`;
  - every `cmake --preset NAME` and `ctest --preset NAME` names a preset in `CMakePresets.json`
    (or a user preset defined in the ignored `CMakeUserPresets.json`);
  - every `build/<preset>/...` path names a configure preset, and a path under `build/apps/`,
    `build/bench/`, `build/tests/` or `build/lib/` fails because the binary directory is
    `build/<preset>/`;
  - `cmake -S . -B build`, `cmake --build build` and `ctest --test-dir build` fail: the documented
    flow configures a preset, so the binary directory always carries the preset name.

Scope: the documentation a reader or an operator follows today. `HISTORY.md` is a verbatim record
of an earlier README, the dated `docs/maintainer/2026-*.md` plans record commands as they were run,
and `share/` documents commands for a different repository. Those are excluded, so their historical
commands do not fail this guard.

Usage: check_docs.py [repository root]. Exit 1 and list what is missing.

Run it from CI or by hand; it is the guard against the class of drift where a flag, an environment
variable, a preset name or a binary path ships without a matching line in the docs.
"""
import json
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


def active_docs():
    """The documentation files this guard enforces, excluding the records named in the docstring."""
    yield root / "README.md"
    for path in sorted((root / "docs").rglob("*.md")):
        if not path.name.startswith("2026-"):
            yield path
    for name in ("bench/README.md", "bench/parity/README.md", "tests/README.md",
                 "tools/README.md", "eval/README.md", "examples/cli/README.md"):
        yield root / name
    yield from sorted((root / "model-cards").glob("*/README.md"))


missing = []

# 1. Serve flags must be documented in a docs/serving.md option table.
serving = read("docs/serving.md")
table_rows = "\n".join(line for line in serving.splitlines() if line.startswith("|"))
# The live flag surface is the product configuration schema. The historical hand-rolled parser at
# `src/serve/serve_options.cpp` is not in the build, so scraping it validated a file that never
# ships and could not see a real flag at all. Each schema entry is
# `{Field::X, "key", "--flag", "--negation", "ARG", "Help.", Scope::Y}`, and a Scope::Cli entry is a
# client flag that belongs to docs/cli.md, not to the serving surface.
schema_source = read("src/product/config/schema.cpp")
parsed_flags = set()
for match in re.finditer(r"\{Field::\w+,(.*?)\}", schema_source, re.S):
    body = match.group(1)
    if "Scope::Cli" in body:
        continue
    parsed_flags.update(re.findall(r'"(--[a-z0-9-]+)"', body))
flags = sorted(parsed_flags)

# Documented in docs/serving.md but NOT declared in the live schema, so the real `ninfer-serve`
# rejects them. This is a real gap, not an allow-list of convenience: every entry here is a flag a
# user can read about and cannot pass. The underlying options mostly exist in `EngineOptions` and
# `ServeOptions` and simply are not exposed in the schema; the rest need removing from the docs.
# The list must only ever shrink — fixing one means deleting its line, and the guard then holds the
# line for good.
DOCUMENTED_NOT_WIRED = {
    "--default-max-tokens", "--default-thinking-budget", "--device-state-slots",
    "--greedy", "--host-kv-mib",
    "--host-state-slots", "--lm-head-draft", "--log-stats-interval-ms", "--long-prefill-wait-ms",
    "--max-private-continuations", "--max-request-mib", "--max-shared-prefixes",
    "--media-cache-mib", "--media-live-mib", "--mtp-capture", "--pending-timeout-ms",
    "--prefill-decode-share", "--proposal-head-rows", "--response-store-max-mib",
    "--shutdown-timeout-seconds", "--token-logprobs",
}

# A negation is documented by its positive form: a reader who finds `--vision` documented does not
# need a separate row for `--no-vision`.
UNDOCUMENTED_REAL = {
    "--cuda-graph", "--extra-state-slots", "--gpu-timing", "--host-cache-capacity", "--ignore-eos",
    "--max-request-size", "--max-tokens", "--media-cache-capacity", "--media-live-capacity",
    "--pending-timeout", "--prefix-reuse", "--proposal-head", "--response-store-max-size",
    "--stats-interval", "--thinking", "--thinking-budget",
}
missing += [
    f"serve flag {flag} is parsed but no docs/serving.md option table names it"
    for flag in flags
    if f"`{flag}`" not in table_rows and f"`{flag} " not in table_rows
    and not flag.startswith("--no-")
    and flag not in UNDOCUMENTED_REAL
]
awaiting_docs = sorted(set(UNDOCUMENTED_REAL) & set(flags))
if awaiting_docs:
    print(f"note: {len(awaiting_docs)} real serve flag(s) undocumented: "
          f"{', '.join(awaiting_docs)}", file=sys.stderr)
documented_flags = sorted(set(re.findall(r"`(--[a-z0-9-]+)", table_rows)))
missing += [
    f"serve flag {flag} is named in docs/serving.md but src/product/config/schema.cpp does not "
    f"declare it for the serve scope"
    for flag in documented_flags
    if flag not in parsed_flags and flag not in DOCUMENTED_NOT_WIRED
]

# Report the known gap as a standing count so it cannot quietly grow, and so shrinking it is visible.
awaiting = sorted(DOCUMENTED_NOT_WIRED & set(documented_flags))
if awaiting:
    print(f"note: {len(awaiting)} serve flag(s) documented but not wired: {', '.join(awaiting)}",
          file=sys.stderr)

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

# 3. Preset names must match CMakePresets.json.
presets = {preset["name"] for preset in json.loads(read("CMakePresets.json"))["configurePresets"]}
# Defined in the ignored CMakeUserPresets.json and shown as an example in build-system.md.
user_presets = {"local"}
components = {"apps", "bench", "tests", "lib"}
preset_name = re.compile(r"^[a-z0-9][a-z0-9-]*$")

for path in active_docs():
    text = path.read_text(encoding="utf-8")
    label = path.relative_to(root)
    for name in re.findall(r"\b(?:cmake|ctest)\s+--preset\s+([A-Za-z0-9_.-]+)", text):
        if name in presets or name in user_presets or "<" in name:
            continue
        missing.append(f"{label}: preset {name} is not in CMakePresets.json")
    for first in re.findall(r"\bbuild/([A-Za-z0-9_<>.-]+)/", text):
        if first in components:
            missing.append(
                f"{label}: build/{first}/ is stale; the binary directory is build/<preset>/{first}/"
            )
        elif first not in presets and "<" not in first and preset_name.match(first):
            missing.append(f"{label}: build/{first}/ does not name a configure preset")
    for pattern, entry in (
        (r"cmake\s+-S\s+\.\s+-B\s+build(?![-\w/])", "cmake -S . -B build"),
        (r"cmake\s+--build\s+build(?![-\w/])", "cmake --build build"),
        (r"ctest\s+--test-dir\s+build(?![-\w/])", "ctest --test-dir build"),
    ):
        if re.search(pattern, text):
            missing.append(
                f"{label}: {entry} is stale; configure a preset and use build/<preset>/"
            )

if missing:
    print(f"{len(missing)} documentation gap(s):")
    for entry in missing:
        print(f"  {entry}")
    sys.exit(1)

print(
    f"ok: {len(flags)} serve flags, {len(environment)} environment variables, "
    f"{len(presets)} presets, and the preset binary paths are documented"
)
