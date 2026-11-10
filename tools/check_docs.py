#!/usr/bin/env python3
"""Documentation and source guard: what the code and the build system accept must be documented,
and what the tree contains must be compiled.

Every check is one function that takes the repository root and a `Gaps` collector. A gap message
names the file to edit and what to write. `run()` calls every check and collects the gaps.
`--self-test` builds a small fixture tree, plants one defect of each class, and asserts the guard
reports it.

Checks:
  - `check_serve_flags`: the serve surface is the schema's serve and common scopes. Every serve
    flag needs an option-table row in `docs/serving.md`, and every serve flag named there must be
    declared in `src/product/config/schema.cpp`;
  - `check_cli_flags`: every flag named in `docs/cli.md` must be declared for the cli or common
    scope, or be one of the parser's own process options;
  - `check_bench_flags`: the benchmark's `kOptions` extras and its hand-written usage block in
    `bench/inference/options.cpp` must name the same flags;
  - `check_environment_variables`: every `NINFER_*` variable read with `getenv()` under `src/` or
    `include/` must be named under `docs/`;
  - `check_presets`: every `cmake --preset` or `ctest --preset` in the active docs must name a
    preset in `CMakePresets.json` (or the user preset `local`);
  - `check_binary_paths`: every `build/...` path in the active docs must carry the preset
    directory, and the pre-preset forms (`build/apps/`, `cmake --build build`, ...) fail;
  - `check_source_ownership`: every `.cpp` under `src/`, `apps/` and `bench/` must be named by a
    CMake file; a source no CMake file names is dead code.

Scope: the documentation a reader follows today. `HISTORY.md` is a verbatim record of an earlier
README, the dated `docs/maintainer/2026-*.md` plans record commands as they were run, and `share/`
documents commands for a different repository. Those are excluded, so their historical commands do
not fail this guard.

Usage: check_docs.py [ROOT] [--self-test]. Exit 1 and list every gap, or 0 with a summary.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Directories whose contents are generated, vendored or historical; never scanned for sources.
SKIPPED_PARTS = {"build", "third_party", ".git", "out", "profiles"}
# The binary subdirectories a documented path may name below `build/<preset>/`.
BINARY_COMPONENTS = {"apps", "bench", "tests", "lib"}
# Defined in the ignored CMakeUserPresets.json and shown as an example in build-system.md.
USER_PRESETS = {"local"}


class Gaps:
    """Collects every failed check as (check name, actionable message)."""

    def __init__(self) -> None:
        self.items: list[tuple[str, str]] = []

    def add(self, check: str, message: str) -> None:
        self.items.append((check, message))


def read(root: Path, relative: str) -> str:
    return (root / relative).read_text(encoding="utf-8")


def active_docs(root: Path):
    """The documentation files this guard enforces, excluding the records named in the docstring."""
    yield root / "README.md"
    for path in sorted((root / "docs").rglob("*.md")):
        if not path.name.startswith("2026-"):
            yield path
    for name in ("bench/README.md", "bench/parity/README.md", "tests/README.md",
                 "tools/README.md", "eval/README.md", "examples/cli/README.md"):
        yield root / name
    yield from sorted((root / "model-cards").glob("*/README.md"))


def cmake_files(root: Path):
    """Every checked-in CMake file, excluding the skipped directories."""
    for path in root.rglob("*"):
        if any(part in SKIPPED_PARTS for part in path.relative_to(root).parts):
            continue
        if path.name == "CMakeLists.txt" or path.suffix == ".cmake":
            yield path


def schema_scopes(root: Path) -> tuple[set[str], set[str]]:
    """The serve and cli flag surfaces declared by the product configuration schema."""
    source = read(root, "src/product/config/schema.cpp")
    serve: set[str] = set()
    cli: set[str] = set()
    for match in re.finditer(r"\{Field::\w+,(.*?)\}", source, re.S):
        body = match.group(1)
        names = set(re.findall(r'"(--[a-z0-9-]+)"', body))
        if "Scope::Serve" in body:
            serve.update(names)
        elif "Scope::Cli" in body:
            cli.update(names)
        else:
            # Scope::Common applies to both entry points.
            serve.update(names)
            cli.update(names)
    return serve, cli


def check_serve_flags(root: Path, gaps: Gaps) -> dict[str, int]:
    serve, _ = schema_scopes(root)
    table_rows = "\n".join(
        line for line in read(root, "docs/serving.md").splitlines() if line.startswith("|")
    )
    for flag in sorted(serve):
        if flag.startswith("--no-"):
            continue
        if f"`{flag}`" in table_rows or f"`{flag} " in table_rows:
            continue
        gaps.add("serve flags",
                 f"docs/serving.md: add an option-table row that names `{flag}`; "
                 f"src/product/config/schema.cpp declares it for the serve or common scope")
    for flag in sorted(set(re.findall(r"`(--[a-z0-9-]+)", table_rows))):
        if flag not in serve:
            gaps.add("serve flags",
                     f"docs/serving.md: remove the row for `{flag}`, or declare it in "
                     f"src/product/config/schema.cpp for the serve or common scope")
    return {"serve flags": len(serve)}


def check_cli_flags(root: Path, gaps: Gaps) -> dict[str, int]:
    _, cli = schema_scopes(root)
    surface = cli | {"--config", "--help", "--json-object", "--clear-stops"}
    for flag in sorted(set(re.findall(r"`(--[a-z0-9-]+)", read(root, "docs/cli.md")))):
        if flag not in surface:
            gaps.add("cli flags",
                     f"docs/cli.md: remove `{flag}`, or declare it in "
                     f"src/product/config/schema.cpp for the cli or common scope")
    return {"cli flags": len(surface)}


def check_bench_flags(root: Path, gaps: Gaps) -> dict[str, int]:
    source = read(root, "bench/inference/options.cpp")
    extras_match = re.search(r"constexpr product::CommandLineOption kOptions\[\] = \{(.*?)\};",
                             source, re.S)
    if extras_match is None:
        gaps.add("bench flags",
                 "bench/inference/options.cpp: restore the kOptions table (the benchmark's extra "
                 "options), or update tools/check_docs.py if the options moved")
        return {"bench extras": 0}
    extras = set(re.findall(r'"(--[a-z0-9-]+)"', extras_match.group(1)))
    usage_at = source.find("std::string usage_text")
    if usage_at < 0:
        gaps.add("bench flags",
                 "bench/inference/options.cpp: restore usage_text (the benchmark's help), or update "
                 "tools/check_docs.py if the help moved")
        return {"bench extras": len(extras)}
    documented: set[str] = set()
    for line in source[usage_at:].splitlines():
        if re.match(r"^\s+--[a-z0-9-]", line):
            documented.update(re.findall(r"--[a-z0-9-]+", line))
    serve, cli = schema_scopes(root)
    for flag in sorted(extras):
        if flag not in documented:
            gaps.add("bench flags",
                     f"bench/inference/options.cpp: name `{flag}` in the usage_text block; the "
                     f"kOptions table parses it")
    for flag in sorted(documented):
        if flag not in extras | (serve & cli):
            gaps.add("bench flags",
                     f"bench/inference/options.cpp: remove `{flag}` from usage_text, or add it to "
                     f"kOptions; the schema declares no such common flag")
    return {"bench extras": len(extras)}


def check_environment_variables(root: Path, gaps: Gaps) -> dict[str, int]:
    documented = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (root / "docs").rglob("*.md")
    )
    variables: set[str] = set()
    for directory in ("src", "include"):
        for path in (root / directory).rglob("*"):
            if path.suffix in {".cpp", ".h", ".cu", ".cuh", ".inc"}:
                variables |= set(
                    re.findall(r'getenv\(\s*"(NINFER_[A-Z0-9_]+)"',
                               path.read_text(encoding="utf-8", errors="replace"))
                )
    for name in sorted(variables):
        if name not in documented:
            gaps.add("environment variables",
                     f"docs/: name `{name}` in a docs page; it is read with getenv() under src/")
    return {"environment variables": len(variables)}


def check_presets(root: Path, gaps: Gaps) -> dict[str, int]:
    presets = {preset["name"] for preset in
               json.loads(read(root, "CMakePresets.json"))["configurePresets"]}
    for path in active_docs(root):
        label = path.relative_to(root)
        for name in re.findall(r"\b(?:cmake|ctest)\s+--preset\s+([A-Za-z0-9_.-]+)",
                               path.read_text(encoding="utf-8")):
            if name in presets or name in USER_PRESETS or "<" in name:
                continue
            gaps.add("presets",
                     f"{label}: `--preset {name}` names no preset; add it to CMakePresets.json, or "
                     f"use one of: {', '.join(sorted(presets))}")
    return {"presets": len(presets)}


def check_binary_paths(root: Path, gaps: Gaps) -> dict[str, int]:
    presets = {preset["name"] for preset in
               json.loads(read(root, "CMakePresets.json"))["configurePresets"]}
    preset_name = re.compile(r"^[a-z0-9][a-z0-9-]*$")
    for path in active_docs(root):
        text = path.read_text(encoding="utf-8")
        label = path.relative_to(root)
        for first in re.findall(r"\bbuild/([A-Za-z0-9_<>.-]+)/", text):
            if first in BINARY_COMPONENTS:
                gaps.add("binary paths",
                         f"{label}: `build/{first}/` is stale; the binary directory is "
                         f"`build/<preset>/{first}/`")
            elif first not in presets and "<" not in first and preset_name.match(first):
                gaps.add("binary paths",
                         f"{label}: `build/{first}/` names no configure preset; use "
                         f"`build/<preset>/`")
        for pattern, entry in (
            (r"cmake\s+-S\s+\.\s+-B\s+build(?![-\w/])", "cmake -S . -B build"),
            (r"cmake\s+--build\s+build(?![-\w/])", "cmake --build build"),
            (r"ctest\s+--test-dir\s+build(?![-\w/])", "ctest --test-dir build"),
        ):
            if re.search(pattern, text):
                gaps.add("binary paths",
                         f"{label}: `{entry}` is stale; configure a preset and use "
                         f"`build/<preset>/` paths")
    return {}


def source_registrations(root: Path) -> tuple[set[str], list[re.Pattern]]:
    """Repository-relative source paths that some CMake file names.

    Handles the constructs this tree uses: literal paths, `${CMAKE_CURRENT_LIST_DIR}` and
    `${PROJECT_SOURCE_DIR}` anchors, and single- or multi-segment variables such as
    `${NINFER_TARGET}`, `${NINFER_OPS_DIR}` and `${isa}`. A variable inside a path is matched as a
    wildcard, so a per-target registration covers every target's file.
    """
    candidate = re.compile(r"[A-Za-z0-9_./${}-]+")
    wildcard = "\x00"

    def is_path_token(token: str) -> bool:
        if token.endswith((".cpp", ".cu")):
            return True
        if not token.endswith("}") or "/" not in token:
            return False
        segments = [segment for segment in token.split("/") if segment]
        return len(segments) >= 2 and any("${" not in segment for segment in segments)

    def resolve(token: str, cmake_dir: Path):
        token = token.replace("${CMAKE_CURRENT_LIST_DIR}", str(cmake_dir))
        token = token.replace("${CMAKE_CURRENT_SOURCE_DIR}", str(cmake_dir))
        token = token.replace("${PROJECT_SOURCE_DIR}", str(root))
        token = token.replace("${CMAKE_SOURCE_DIR}", str(root))
        token = re.sub(r"\$\{[^}]+\}", wildcard, token)
        if wildcard in token:
            absolute = token.startswith("/")
            parts: list[str] = []
            for segment in token.split("/"):
                if segment in ("", "."):
                    continue
                if segment == "..":
                    if parts:
                        parts.pop()
                else:
                    parts.append(segment)
            token = ("/" if absolute else "") + "/".join(parts)
            if token.startswith(str(root)):
                token = token[len(str(root)) + 1:]
            regex = ""
            for index, segment in enumerate(token.split("/")):
                if wildcard in segment:
                    regex += ".*" if index == 0 else "[^/]*"
                else:
                    regex += re.escape(segment)
                if index < len(token.split("/")) - 1:
                    regex += "/"
            return re.compile("^" + regex + "$")
        path = Path(token) if Path(token).is_absolute() else cmake_dir / token
        try:
            return str(path.resolve().relative_to(root))
        except ValueError:
            return None

    exact: set[str] = set()
    patterns: list[re.Pattern] = []
    for path in cmake_files(root):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().startswith("#"):
                continue
            for token in candidate.findall(line):
                if not is_path_token(token):
                    continue
                resolved = resolve(token, path.parent)
                if resolved is None:
                    continue
                if isinstance(resolved, str):
                    exact.add(resolved)
                else:
                    patterns.append(resolved)
    return exact, patterns


def check_source_ownership(root: Path, gaps: Gaps) -> dict[str, int]:
    exact, patterns = source_registrations(root)
    sources: set[str] = set()
    for directory in ("src", "apps", "bench"):
        for path in (root / directory).rglob("*.cpp"):
            relative = path.relative_to(root)
            if any(part in SKIPPED_PARTS for part in relative.parts):
                continue
            sources.add(str(relative))
    unregistered = sorted(
        source for source in sources
        if source not in exact and not any(pattern.match(source) for pattern in patterns)
    )
    for source in unregistered:
        gaps.add("source ownership",
                 f"{source}: no CMake target compiles this file; add it to the owning CMake "
                 f"manifest, or delete it if it is dead code")
    return {"sources": len(sources)}


CHECKS = (
    check_serve_flags,
    check_cli_flags,
    check_bench_flags,
    check_environment_variables,
    check_presets,
    check_binary_paths,
    check_source_ownership,
)


def run(root: Path) -> tuple[Gaps, dict[str, int]]:
    gaps = Gaps()
    counts: dict[str, int] = {}
    for check in CHECKS:
        counts.update(check(root, gaps))
    return gaps, counts


def summary(counts: dict[str, int]) -> str:
    return (
        f"ok: {counts.get('serve flags', 0)} serve flags, "
        f"{counts.get('cli flags', 0)} cli flags, "
        f"{counts.get('bench extras', 0)} bench extras, "
        f"{counts.get('environment variables', 0)} environment variables, "
        f"{counts.get('presets', 0)} presets, "
        f"{counts.get('sources', 0)} sources registered, "
        f"and the preset binary paths are documented"
    )


# --- self-test -------------------------------------------------------------------------------
# A minimal repository that every check accepts. Each case mutates one file and asserts the guard
# reports the defect class, so the guard's teeth are testable rather than demonstrated by hand.
FIXTURE = {
    "CMakePresets.json":
        '{"configurePresets": [{"name": "rtx5090-dev"}, {"name": "rtx5090-release"}]}',
    "CMakeLists.txt": "add_library(probe src/used.cpp src/uses_env.cpp "
                      "bench/inference/options.cpp src/product/config/schema.cpp)\n",
    "src/used.cpp": "int used() { return 0; }\n",
    "src/uses_env.cpp": '#include <cstdlib>\n'
                        'const char* probe() { return std::getenv("NINFER_PROBE"); }\n',
    "src/product/config/schema.cpp":
        "constexpr FieldSpec kFields[] = {\n"
        '    {Field::A, "a", "--common-flag", "", "N", "Common.", Scope::Common},\n'
        '    {Field::B, "b", "--serve-flag", "", "N", "Serve.", Scope::Serve},\n'
        '    {Field::C, "c", "--cli-flag", "", "N", "Cli.", Scope::Cli},\n'
        "};\n",
    "docs/serving.md": "# serving\n\n| Option | Meaning |\n|---|---|\n"
                       "| `--common-flag N` | common |\n| `--serve-flag N` | serve |\n",
    "docs/cli.md": "# cli\n\n`--common-flag`, `--cli-flag`, `--config`, `--json-object`.\n",
    "docs/README.md": "# docs\n\n`NINFER_PROBE` is documented here.\n",
    "bench/inference/options.cpp":
        "constexpr product::CommandLineOption kOptions[] = {\n"
        '    {"--prompt"},\n'
        "};\n\n"
        "std::string usage_text(std::string_view executable) {\n"
        '    return R"(\n'
        "  --prompt TEXT\n"
        "  --common-flag N\n"
        ')" ;\n'
        "}\n",
    "README.md": "# readme\n\ncmake --preset rtx5090-dev\n\nbuild/rtx5090-dev/apps/ninfer\n",
    "bench/README.md": "# bench\n",
    "bench/parity/README.md": "# parity\n",
    "tests/README.md": "# tests\n",
    "tools/README.md": "# tools\n",
    "eval/README.md": "# eval\n",
    "examples/cli/README.md": "# examples\n",
}


def write_fixture(root: Path) -> None:
    for relative, text in FIXTURE.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def append(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")


def replace(root: Path, relative: str, old: str, new: str) -> None:
    path = root / relative
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


SELF_TEST_CASES = (
    ("serve flag undocumented",
     lambda root: replace(root, "docs/serving.md", "| `--serve-flag N` | serve |\n", ""),
     ("docs/serving.md", "--serve-flag")),
    ("phantom serve flag",
     lambda root: append(root, "docs/serving.md", "| `--ghost-serve` | ghost |\n"),
     ("docs/serving.md", "--ghost-serve", "schema.cpp")),
    ("phantom cli flag",
     lambda root: append(root, "docs/cli.md", "\n`--ghost-cli`\n"),
     ("docs/cli.md", "--ghost-cli")),
    ("bench extra undocumented",
     lambda root: replace(root, "bench/inference/options.cpp", '    {"--prompt"},',
                          '    {"--prompt"}, {"--ghost-bench"},'),
     ("bench/inference/options.cpp", "--ghost-bench")),
    ("bench help names an unknown flag",
     lambda root: replace(root, "bench/inference/options.cpp", "  --common-flag N\n",
                          "  --common-flag N\n  --ghost-help N\n"),
     ("bench/inference/options.cpp", "--ghost-help")),
    ("environment variable undocumented",
     lambda root: append(root, "src/used.cpp", 'const char* ghost() { return getenv("NINFER_GHOST"); }\n'),
     ("NINFER_GHOST",)),
    ("unknown preset",
     lambda root: append(root, "README.md", "\ncmake --preset ghost-preset\n"),
     ("README.md", "ghost-preset")),
    ("stale binary path",
     lambda root: append(root, "README.md", "\nbuild/apps/ninfer\n"),
     ("README.md", "build/apps/")),
    ("pre-preset build command",
     lambda root: append(root, "README.md", "\ncmake --build build -j\n"),
     ("README.md", "cmake --build build")),
    ("source no target compiles",
     lambda root: (root / "src" / "orphan.cpp").write_text("int orphan() { return 0; }\n",
                                                           encoding="utf-8"),
     ("src/orphan.cpp", "no CMake target")),
)


def self_test() -> int:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_fixture(root)
        gaps, _ = run(root)
        if gaps.items:
            failures.append(f"clean fixture reported {len(gaps.items)} gap(s): "
                            + gaps.items[0][1])
    for name, mutate, expected in SELF_TEST_CASES:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            mutate(root)
            gaps, _ = run(root)
            reported = "\n".join(message for _, message in gaps.items)
            if not all(fragment in reported for fragment in expected):
                failures.append(f"{name}: guard did not report {expected}; "
                                f"reported: {reported or 'nothing'}")
    if failures:
        print(f"self-test failed: {len(failures)} of {len(SELF_TEST_CASES) + 1} case(s)")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(f"self-test ok: {len(SELF_TEST_CASES)} defect classes caught, clean fixture passes")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", default=str(REPO),
                        help="repository root (default: this script's checkout)")
    parser.add_argument("--self-test", action="store_true",
                        help="plant one defect of each class in a temporary tree and assert the "
                             "guard reports it")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    gaps, counts = run(Path(args.root).resolve())
    if gaps.items:
        print(f"{len(gaps.items)} documentation gap(s):")
        for check, message in gaps.items:
            print(f"  [{check}] {message}")
        return 1
    print(summary(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
