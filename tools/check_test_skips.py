#!/usr/bin/env python3
"""Test-skip guard: a skip must be visible as a skip, in both directions.

The suite has found three variants of one defect: a skip counted as a failure (no
`SKIP_RETURN_CODE`), a skip counted as a pass (a message then `return 0`), and a skip with no
message or the wrong one (`return 77` with nothing printed). A reader of `LastTest.log` must be
able to tell why a test did not run, and `ctest` must count it as skipped. This guard covers the
two that live in the test sources:

  - `check_skip_has_message`: every `return 77;` has a `SKIP:` message shortly before it, on stderr;
  - `check_skip_is_not_success`: a `SKIP:` message is never followed by a `return 0;` in the same
    block, which would make `ctest` count the skip as a pass;
  - `check_skip_return_code`: a test whose source can `return 77` is registered with
    `SKIP_RETURN_CODE 77`, or is listed as an artifact consumer in the manifest (which applies the
    property at configure time). Without it, ctest counts a real skip as a failure.

Every check takes the repository root and a `Gaps` collector. `--self-test` plants one defect of
each class in a temporary tree and asserts the guard reports it.

Usage: check_test_skips.py [ROOT] [--self-test]. Exit 1 and list every gap, or 0 with a summary.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# How far above a `return 77;` the message may sit: a guard block is a few lines, and a longer gap
# means the message belongs to something else.
MESSAGE_WINDOW = 12


class Gaps:
    def __init__(self) -> None:
        self.items: list[tuple[str, str]] = []

    def add(self, check: str, message: str) -> None:
        self.items.append((check, message))


def test_sources(root: Path):
    tests = root / "tests"
    if not tests.is_dir():
        return
    for path in sorted(tests.rglob("*.cpp")):
        yield path


def check_skip_has_message(root: Path, gaps: Gaps) -> int:
    checked = 0
    for path in test_sources(root):
        lines = path.read_text(errors="replace").splitlines()
        for index, line in enumerate(lines):
            if "return 77" not in line:
                continue
            checked += 1
            window = lines[max(0, index - MESSAGE_WINDOW):index]
            if not any("SKIP:" in candidate for candidate in window):
                gaps.add("skip message",
                         f"{path.relative_to(root)}:{index + 1}: `return 77` without a preceding "
                         f"`SKIP:` message; print why the test did not run")
    return checked


def check_skip_is_not_success(root: Path, gaps: Gaps) -> int:
    checked = 0
    for path in test_sources(root):
        lines = path.read_text(errors="replace").splitlines()
        for index, line in enumerate(lines):
            if "SKIP:" not in line:
                continue
            checked += 1
            if "std::cout" in line:
                gaps.add("skip stream",
                         f"{path.relative_to(root)}:{index + 1}: the skip message goes to stdout; "
                         f"use std::cerr so it lands in LastTest.log")
            # Walk forward to the end of the enclosing block and reject a plain `return 0`.
            for following in lines[index + 1:index + 8]:
                if "return 0" in following:
                    gaps.add("skip counted as pass",
                             f"{path.relative_to(root)}:{index + 1}: the skip message is followed by "
                             f"`return 0`; return 77 so ctest reports a skip")
                    break
                if "return 77" in following or "}" == following.strip():
                    break
    return checked


def _registered_tests(root: Path) -> dict[str, str]:
    """Test name -> source file name, from every tests/**/*.cmake `ninfer_add_test` block."""
    registered: dict[str, str] = {}
    for cmake in sorted((root / "tests").rglob("*.cmake")):
        text = cmake.read_text(errors="replace")
        for match in re.finditer(r"ninfer_add_test\(([\w]+)\s*\n?\s*SOURCES\s+([^\n]+)", text):
            source = match.group(2).strip().rstrip(")").strip().strip('"')
            registered[match.group(1)] = source.split("/")[-1]
    return registered


def _skip_coded_names(root: Path) -> set[str]:
    """Names that get SKIP_RETURN_CODE 77: explicit properties, list variables, and manifest
    consumers (the artifact plumbing applies the property to every consumer)."""
    names: set[str] = set()
    for cmake in sorted((root / "tests").rglob("*.cmake")):
        text = cmake.read_text(errors="replace")
        # `set(<list> name name ...)` lets a registration set properties on a group.
        groups: dict[str, list[str]] = {}
        for group in re.finditer(r"\nset\(([\w]+)\s*\n?((?:\s*ninfer_[\w]+)*)\s*\)", text):
            groups[group.group(1)] = re.findall(r"ninfer_[\w]+", group.group(2))
        for match in re.finditer(r"set_tests_properties\(([^)]*?)SKIP_RETURN_CODE", text, re.S):
            subject = match.group(1)
            names.update(re.findall(r"ninfer_[\w]+", subject))
            for variable in re.findall(r"\$\{([\w]+)\}", subject):
                names.update(groups.get(variable, []))
    manifest = root / "tests" / "fixtures" / "artifacts" / "manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        for artifact in data.get("artifacts", []):
            names.update(artifact.get("tests") or [])
    return names


def check_skip_return_code(root: Path, gaps: Gaps) -> int:
    coded = _skip_coded_names(root)
    checked = 0
    for name, source in _registered_tests(root).items():
        candidates = [p for p in test_sources(root) if p.name == source]
        if not candidates:
            continue
        if "return 77" not in candidates[0].read_text(errors="replace"):
            continue
        checked += 1
        if name not in coded:
            gaps.add("skip return code",
                     f"tests/cmake: {name} can return 77 but is registered without "
                     f"SKIP_RETURN_CODE 77; ctest would count the skip as a failure")
    return checked


def run(root: Path) -> int:
    gaps = Gaps()
    sites = check_skip_has_message(root, gaps)
    messages = check_skip_is_not_success(root, gaps)
    registrations = check_skip_return_code(root, gaps)
    if gaps.items:
        for check, message in gaps.items:
            print(f"[{check}] {message}")
        print(f"{len(gaps.items)} test-skip gap(s)")
        return 1
    print(f"ok: {sites} `return 77` sites, {messages} skip messages, and "
          f"{registrations} skip-capable registrations are visible skips")
    return 0


# --- self-test -------------------------------------------------------------------------------
SELF_TEST_CASES = [
    # (file body, check name that must fire)
    ('int main() { return 77; }\n', "skip message"),
    ('int main() {\n    std::cout << "SKIP: no device\\n";\n    return 77;\n}\n', "skip stream"),
    ('int main() {\n    std::cerr << "SKIP: no device\\n";\n    return 0;\n}\n',
     "skip counted as pass"),
]
GOOD = ('int main() {\n    std::cerr << "SKIP: no device\\n";\n    return 77;\n}\n')


def self_test() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        tests = root / "tests"
        tests.mkdir()
        for index, (body, expected) in enumerate(SELF_TEST_CASES):
            path = tests / f"case{index}.cpp"
            path.write_text(body)
            gaps = Gaps()
            check_skip_has_message(root, gaps)
            check_skip_is_not_success(root, gaps)
            names = {check for check, _ in gaps.items}
            if expected not in names:
                print(f"self-test: case {index} did not report {expected!r} (got {sorted(names)})")
                failures += 1
            path.unlink()
        (tests / "good.cpp").write_text(GOOD)
        gaps = Gaps()
        check_skip_has_message(root, gaps)
        check_skip_is_not_success(root, gaps)
        if gaps.items:
            print(f"self-test: a correct file reported {gaps.items}")
            failures += 1

        # A skip-capable test registered without SKIP_RETURN_CODE would be counted as a failure.
        (tests / "cmake").mkdir()
        (tests / "cmake" / "CaseTests.cmake").write_text(
            'ninfer_add_test(ninfer_case_test\n  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../good.cpp")\n')
        gaps = Gaps()
        check_skip_return_code(root, gaps)
        if "skip return code" not in {check for check, _ in gaps.items}:
            print("self-test: a registration without SKIP_RETURN_CODE was not reported")
            failures += 1
        (tests / "cmake" / "CaseTests.cmake").write_text(
            'ninfer_add_test(ninfer_case_test\n  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../good.cpp")\n'
            "set_tests_properties(ninfer_case_test PROPERTIES SKIP_RETURN_CODE 77)\n")
        gaps = Gaps()
        check_skip_return_code(root, gaps)
        if gaps.items:
            print(f"self-test: a correct registration reported {gaps.items}")
            failures += 1
    if failures:
        print(f"self-test failed: {failures} case(s)")
        return 1
    print(f"self-test ok: {len(SELF_TEST_CASES) + 1} defect classes and their clean cases")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=str(REPO))
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        return self_test()
    return run(Path(arguments.root))


if __name__ == "__main__":
    sys.exit(main())
