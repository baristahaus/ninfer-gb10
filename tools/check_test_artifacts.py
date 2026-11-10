#!/usr/bin/env python3
"""Check the test-artifact manifest against the tests and sources it describes.

The manifest (tests/fixtures/artifacts/manifest.json) is checked in, so it can rot: a test can be
renamed, an environment variable can disappear, a committed fixture can be regenerated with another
digest. This guard fails on all three. It is a host-only CTest (ninfer_test_artifacts_manifest_test)
and needs no GPU and no artifact.

Checks:
  1. schema, required keys, unique names, cache block;
  2. every `env` is actually read by at least one source under tests/;
  3. every name in `tests` is registered in a tests/**/*.cmake;
  4. a committed source exists and matches its pinned bytes and sha256;
  5. bytes and sha256 are both pinned or both null.

Usage: check_test_artifacts.py [REPOSITORY_ROOT]
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
MANIFEST = ROOT / "tests" / "fixtures" / "artifacts" / "manifest.json"
ENV_READ = re.compile(r'getenv\("(NINFER_[A-Z0-9_]+)"\)')
TEST_DECL = re.compile(r'\b(?:ninfer_add_test|ninfer_add_op_test|add_test)\(\s*(?:NAME\s+)?([A-Za-z0-9_]+)')
SOURCES = ("tests", "tools", "src", "apps")


def fail(problems: list[str], message: str) -> None:
    problems.append(message)


def main() -> int:
    problems: list[str] = []
    try:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"manifest: cannot read {MANIFEST}: {error}")
        return 1

    if data.get("schema") != 1:
        fail(problems, f"schema must be 1, is {data.get('schema')!r}")
    cache = data.get("cache") or {}
    for key in ("variable", "default"):
        if not cache.get(key):
            fail(problems, f"cache.{key} is missing")
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        fail(problems, "artifacts must be a non-empty list")
        artifacts = []

    # Every environment variable any test reads, and every registered test name.
    env_read = set()
    for area in SOURCES:
        for path in (ROOT / area).rglob("*"):
            if path.suffix in {".cpp", ".cu", ".h", ".cuh"} and path.is_file():
                env_read |= set(ENV_READ.findall(path.read_text(errors="replace")))
    registered = set()
    for path in (ROOT / "tests").rglob("*.cmake"):
        registered |= set(TEST_DECL.findall(path.read_text(errors="replace")))

    names: set[str] = set()
    pinned = 0
    for entry in artifacts:
        name = entry.get("name")
        if not name:
            fail(problems, "an entry has no name")
            continue
        if name in names:
            fail(problems, f"{name}: duplicate name")
        names.add(name)
        for key in ("env", "file", "source"):
            if not entry.get(key):
                fail(problems, f"{name}: missing {key}")

        env = entry.get("env")
        if env and env not in env_read:
            fail(problems, f"{name}: no source under tests/ reads {env}")

        tests = entry.get("tests") or []
        if not isinstance(tests, list):
            fail(problems, f"{name}: tests must be a list")
            tests = []
        for test in tests:
            if test not in registered:
                fail(problems, f"{name}: test {test} is not registered in tests/**/*.cmake")

        size, sha = entry.get("bytes"), entry.get("sha256")
        if (size is None) != (sha is None):
            fail(problems, f"{name}: bytes and sha256 must be pinned together")
        source = entry.get("source") or {}
        if source.get("kind") == "committed":
            path = ROOT / source.get("path", "")
            if not path.is_file():
                fail(problems, f"{name}: committed source {path} does not exist")
            elif sha is not None:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != sha:
                    fail(problems, f"{name}: committed sha256 {digest[:12]} != pinned {sha[:12]}")
                if size is not None and path.stat().st_size != size:
                    fail(problems, f"{name}: committed bytes {path.stat().st_size} != pinned {size}")
        if sha is not None:
            pinned += 1

    # Committed fixtures are always present, so their digests must hold on every host.
    fixtures = data.get("fixtures") or []
    for fixture in fixtures:
        path = ROOT / fixture.get("path", "")
        if not path.is_file():
            fail(problems, f"fixture {fixture.get('name')}: {path} does not exist")
            continue
        size, digest = path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()
        if size != fixture.get("bytes"):
            fail(problems, f"fixture {fixture.get('name')}: bytes {size} != {fixture.get('bytes')}")
        if digest != fixture.get("sha256"):
            fail(problems, f"fixture {fixture.get('name')}: sha256 {digest[:12]} != pinned")

    for problem in problems:
        print(f"  {problem}")
    if problems:
        print(f"test artifacts: FAIL ({len(problems)} problem(s))")
        return 1

    print(
        f"test artifacts: OK ({len(artifacts)} entries, {pinned} pinned, "
        f"{len(artifacts) - pinned} awaiting a pin, {len(fixtures)} committed fixtures, "
        f"{len(env_read)} variables in sources)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
