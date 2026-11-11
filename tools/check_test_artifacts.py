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
  5. bytes and sha256 are both pinned or both null;
  6. docs/test-artifacts.md lists every artifact with its environment variable and cache file, and
     every row in that table is a manifest artifact.

It also reports how many tests will skip on this host, with the same resolution rule the configure
step uses (explicit environment value, then the cache file). A green run and a run where nothing ran
must not look the same, and that number is what tells them apart.

Usage: check_test_artifacts.py [REPOSITORY_ROOT] [--self-test]
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
MANIFEST = ROOT / "tests" / "fixtures" / "artifacts" / "manifest.json"
ENV_READ = re.compile(r'getenv\("(NINFER_[A-Z0-9_]+)"\)')
TEST_DECL = re.compile(r'\b(?:ninfer_add_test|ninfer_add_op_test|add_test)\(\s*(?:NAME\s+)?([A-Za-z0-9_]+)')
SOURCES = ("tests", "tools", "src", "apps")
DOC = ROOT / "docs" / "test-artifacts.md"
DOC_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|")


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

    # Every artifact-shaped variable a test reads must be a manifest entry: an unlisted one is an
    # undocumented dependency, which is exactly the gap this inventory closes.
    manifest_envs = {entry.get("env") for entry in artifacts if entry.get("env")}
    artifact_like = {
        variable
        for variable in env_read
        if variable.endswith(("_WEIGHTS", "_ARTIFACT", "_FIXTURE"))
    }
    for variable in sorted(artifact_like - manifest_envs):
        fail(problems, f"{variable} is read by a test but is not a manifest artifact")

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

    # The inventory page must name every artifact, its variable and its cache file; a row with no
    # manifest entry is a stale table, and an entry with no row is an undocumented dependency.
    doc_rows: dict[str, tuple[str, str]] = {}
    if not DOC.is_file():
        fail(problems, f"inventory page {DOC} does not exist")
    else:
        for line in DOC.read_text(encoding="utf-8").splitlines():
            match = DOC_ROW.match(line)
            if match:
                doc_rows[match.group(1)] = (match.group(2), match.group(3))
        for entry in artifacts:
            name = entry.get("name")
            if not name:
                continue
            if name not in doc_rows:
                fail(problems, f"{name}: no row in docs/test-artifacts.md")
                continue
            env, file = doc_rows[name]
            if env != entry.get("env"):
                fail(problems, f"{name}: docs row names {env}, manifest names {entry.get('env')}")
            if file != entry.get("file"):
                fail(problems, f"{name}: docs row names {file}, manifest names {entry.get('file')}")
        for name in doc_rows:
            if name not in names:
                fail(problems, f"docs/test-artifacts.md row {name} is not a manifest artifact")

    # Skip counting, with the configure step's resolution rule. Distinct tests, because a test that
    # consumes two artifacts skips once.
    cache_value = os.environ.get("NINFER_TEST_ARTIFACT_CACHE") or (data.get("cache") or {}).get(
        "default", "")
    cache = Path(os.path.expanduser(cache_value)) if cache_value else None

    def present(entry: dict) -> bool:
        explicit = os.environ.get(entry.get("env", ""))
        if explicit and Path(explicit).is_file():
            return True
        source = entry.get("source") or {}
        if source.get("kind") == "committed":
            return (ROOT / source.get("path", "")).is_file()
        return cache is not None and (cache / entry.get("file", "")).is_file()

    skipping: set[str] = set()
    resolved = 0
    for entry in artifacts:
        if not entry.get("name"):
            continue
        tests = [test for test in (entry.get("tests") or []) if test in registered]
        if present(entry):
            resolved += 1
        else:
            skipping |= set(tests)
    absent = len([entry for entry in artifacts if entry.get("name")]) - resolved

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
    print(
        f"test artifacts: {resolved} resolved, {absent} absent, {len(skipping)} test(s) will skip "
        f"(cache {cache if cache is not None else 'unknown'})"
    )
    return 0


def self_test() -> int:
    """Plant the defects this guard must catch in a temporary tree and assert it reports each.

    The check is run as a subprocess against the tree, so the self-test exercises the same code path
    the CTest does rather than a second copy of the logic.
    """
    import json as _json

    def run_tree(build) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "tests" / "fixtures" / "artifacts").mkdir(parents=True)
            (root / "tests" / "cmake").mkdir(parents=True)
            (root / "docs").mkdir()
            manifest = {
                "schema": 1,
                "cache": {"variable": "NINFER_TEST_ARTIFACT_CACHE", "default": "~/.cache/x"},
                "artifacts": [
                    {
                        "name": "demo",
                        "env": "NINFER_DEMO_ARTIFACT",
                        "file": "demo.ninfer",
                        "bytes": None,
                        "sha256": None,
                        "source": {"kind": "file", "path": "/nfs/demo.ninfer"},
                        "tests": ["demo_test"],
                    }
                ],
                "fixtures": [],
            }
            (root / "tests" / "fixtures" / "artifacts" / "manifest.json").write_text(
                _json.dumps(manifest))
            (root / "tests" / "cmake" / "demo.cmake").write_text("ninfer_add_test(demo_test)\n")
            (root / "tests" / "demo.cpp").write_text('getenv("NINFER_DEMO_ARTIFACT")\n')
            (root / "docs" / "test-artifacts.md").write_text(build)
            result = subprocess.run(
                [sys.executable, "-B", __file__, str(root)], capture_output=True, text=True)
            return result.returncode, result.stdout + result.stderr

    good = "| Artifact | Env | File |\n|---|---|---|\n| `demo` | `NINFER_DEMO_ARTIFACT` | `demo.ninfer` |\n"
    cases = [
        ("the inventory row is present", good, 0),
        ("the inventory row is missing", "| Artifact | Env | File |\n|---|---|---|\n", 1),
        ("the inventory row names the wrong variable",
         good.replace("NINFER_DEMO_ARTIFACT", "NINFER_WRONG"), 1),
        ("the inventory page does not exist", None, 1),
    ]
    failures = 0
    for label, body, expected in cases:
        if body is None:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "tests" / "fixtures" / "artifacts").mkdir(parents=True)
                (root / "tests" / "cmake").mkdir(parents=True)
                (root / "tests" / "fixtures" / "artifacts" / "manifest.json").write_text(
                    _json.dumps({"schema": 1, "cache": {"variable": "V", "default": "d"},
                                 "artifacts": [{"name": "demo", "env": "NINFER_DEMO_ARTIFACT",
                                                "file": "demo.ninfer", "bytes": None,
                                                "sha256": None,
                                                "source": {"kind": "file", "path": "/nfs/d.ninfer"},
                                                "tests": ["demo_test"]}], "fixtures": []}))
                (root / "tests" / "cmake" / "demo.cmake").write_text("ninfer_add_test(demo_test)\n")
                (root / "tests" / "demo.cpp").write_text('getenv("NINFER_DEMO_ARTIFACT")\n')
                result = subprocess.run(
                    [sys.executable, "-B", __file__, str(root)], capture_output=True, text=True)
                code, output = result.returncode, result.stdout + result.stderr
        else:
            code, output = run_tree(body)
        if code != expected:
            failures += 1
            print(f"self-test: {label}: expected exit {expected}, got {code}\n{output}")
    if failures:
        print(f"self-test: FAIL ({failures} case(s))")
        return 1
    print(f"self-test: OK ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        sys.exit(self_test())
    sys.exit(main())
