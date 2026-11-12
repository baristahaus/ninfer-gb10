#!/usr/bin/env python3
"""Populate the test-artifact cache from tests/fixtures/artifacts/manifest.json.

The manifest names each artifact a test can consume: the environment variable the test reads, the
file name in the cache, its size and sha256 when pinned, and where to obtain it. This tool is the
only thing that writes the cache; the cache itself is never in git.

    tools/fetch_test_artifacts.py                     list the entries and their cache state
    tools/fetch_test_artifacts.py --check             verify every cached entry (bytes + sha256)
    tools/fetch_test_artifacts.py --fetch NAME ...    populate the cache (repeatable, or --all)
    tools/fetch_test_artifacts.py --pin NAME --from PATH
                                                      copy an artifact into the cache and record its
                                                      bytes and sha256 in the manifest
    tools/fetch_test_artifacts.py --print-env         shell exports for every cached entry
    tools/fetch_test_artifacts.py --self-test         exercise the --link decision and --check

`--link` is an optimization, never the documented route: a hard link shares the source's inode, so
it is only usable when the source and the cache are on the same device and that device's filesystem
is one the engine's direct-I/O loader accepts. Every other case copies, because a cache entry the
loader refuses is worse than a slow population. `--check` reports such an entry as UNUSABLE.

The cache directory is --cache, else NINFER_TEST_ARTIFACT_CACHE, else the manifest default
(~/.cache/ninfer/test-artifacts). A test whose entry is absent from the cache skips with exit 77;
it does not fail.

Source kinds in the manifest:
    committed  a file already in the repository; the cache is not used, the path is checked in
    file       a local path, copied into the cache (hard-linked with --link only when the
               source and the cache share one accepted local filesystem)
    https      a URL, downloaded
    hf         a Hugging Face repository and file, fetched with the `hf` CLI if it is installed
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "fixtures" / "artifacts" / "manifest.json"
CHUNK = 1 << 22

# Filesystems whose direct reads the engine refuses. The authority is
# io::unsupported_direct_io_filesystem (src/core/io/file.cpp, declared in src/core/io/file.h), and
# the artifact loader applies it at open for ReadMode::Direct (src/artifact/reader.cpp): a direct
# read on one of these can wait uninterruptibly inside the kernel, so the engine refuses the file
# instead of hanging. The magic numbers are the kernel's superblock types, the same values that
# function switches on. Keep this table in step with it; a filesystem missing here is one the loader
# would refuse while this tool accepted.
REFUSED_FILESYSTEMS = {
    0x6969: "NFS",       # NFS_SUPER_MAGIC, NFSv3 and NFSv4
    0xFF534D42: "CIFS",  # CIFS_MAGIC_NUMBER
    0xFE534D42: "SMB",   # SMB2_MAGIC_NUMBER, SMB2 and SMB3
    0x65735546: "FUSE",  # FUSE_SUPER_MAGIC
    0x01021997: "9P",    # V9FS_MAGIC
}

# statfs(2) writes a struct statfs, which is well under this on every supported platform. Only
# f_type, its first member, is read.
_STATFS_BYTES = 4096

# A state that describe() reports and that --check must fail: a wrong digest or a cache entry the
# loader cannot open.
FAILURE_PREFIXES = ("MISMATCH", "UNUSABLE")


def fail(message: str) -> "NoReturn":  # noqa: F821 - typing only
    print(f"fetch_test_artifacts: {message}", file=sys.stderr)
    raise SystemExit(2)


def filesystem_type(path: Path) -> int:
    """The kernel's f_type for the filesystem holding `path`, from statfs(2).

    The engine reads the same value with io::filesystem_type (src/core/io/file.cpp).
    """
    libc = ctypes.CDLL(None, use_errno=True)
    buffer = ctypes.create_string_buffer(_STATFS_BYTES)
    if libc.statfs(os.fsencode(path), buffer) != 0:
        error = ctypes.get_errno()
        fail(f"cannot read the filesystem type of {path}: {os.strerror(error)}")
    return int(ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ulong)).contents.value)


def refused_filesystem(path: Path) -> str | None:
    """The name of the refused filesystem holding `path`, or None when the loader accepts it."""
    return REFUSED_FILESYSTEMS.get(filesystem_type(path))


def link_plan(source: Path, cache: Path) -> tuple[bool, str]:
    """Whether --link may hard-link `source` into `cache`, and the reason for that answer.

    A hard link shares the source's inode, so it is usable only when the source and the cache are on
    one device and that device's filesystem is one the loader accepts. A refused cache filesystem
    cannot be worked around: a real copy would land on it too, so the fetch is refused instead.
    """
    refused = refused_filesystem(cache)
    if refused is not None:
        fail(f"cache {cache} is on a {refused} filesystem, which the engine's direct-I/O loader "
             "refuses; point the cache at local storage")
    refused = refused_filesystem(source)
    if refused is not None:
        return False, f"the source is on {refused}, which the loader refuses"
    if os.stat(source).st_dev != os.stat(cache).st_dev:
        return False, "the source and the cache are on different filesystems"
    return True, "same local filesystem"


def load(manifest_path: Path) -> dict:
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read {manifest_path}: {error}")
    if data.get("schema") != 1:
        fail(f"{manifest_path}: unsupported schema {data.get('schema')!r}")
    if not isinstance(data.get("artifacts"), list) or not data["artifacts"]:
        fail(f"{manifest_path}: 'artifacts' must be a non-empty list")
    names = set()
    for entry in data["artifacts"]:
        for key in ("name", "env", "file", "source"):
            if key not in entry:
                fail(f"{manifest_path}: entry {entry.get('name')!r} lacks {key!r}")
        if entry["name"] in names:
            fail(f"{manifest_path}: duplicate entry {entry['name']!r}")
        names.add(entry["name"])
    return data


def cache_dir(data: dict, override: str | None) -> Path:
    value = override or os.environ.get(data["cache"]["variable"]) or data["cache"]["default"]
    return Path(os.path.expanduser(os.path.expandvars(value)))


def entry_path(data: dict, entry: dict, cache: Path) -> Path:
    source = entry["source"]
    if source.get("kind") == "committed":
        return ROOT / source["path"]
    return cache / entry["file"]


def digest(path: Path) -> tuple[int, str]:
    size = 0
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            size += len(block)
            sha.update(block)
    return size, sha.hexdigest()


def volumes_for(entry_path_: Path) -> list[Path]:
    """The entry file plus its `.part-NNNNN` continuations, entry first, parts by name.

    A `.ninfer` artifact may be split across volumes; the entry file names the parts and they must
    sit beside it with the same names. An artifact is only usable as a complete set.
    """
    if not entry_path_.is_file():
        return []
    parts = sorted(
        path
        for path in entry_path_.parent.iterdir()
        if path.name.startswith(entry_path_.name + ".part-")
    )
    return [entry_path_] + parts


def digest_set(paths: list[Path]) -> tuple[int, str]:
    """Total bytes and one sha256 over the concatenated volumes in order."""
    size = 0
    sha = hashlib.sha256()
    for path in paths:
        with path.open("rb") as handle:
            while block := handle.read(CHUNK):
                size += len(block)
                sha.update(block)
    return size, sha.hexdigest()


def place(source: Path, destination: Path, link: bool) -> None:
    """Hard-link or copy one volume into the cache.

    `link` is only set when link_plan() allowed it, so a cross-filesystem source copies: a symlink
    fallback would put the cache entry on the source's filesystem, which is the thing the loader
    refuses when that filesystem is a network one.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if link:
        try:
            os.link(source, destination)
            return
        except OSError:
            # A filesystem can refuse hard links even when link_plan() allowed one (some do not
            # support them at all). A copy is always usable, so degrade to it rather than to a link
            # that would put the entry back on the source's filesystem.
            print(f"fetch_test_artifacts: {destination} refused a hard link; copying instead",
                  file=sys.stderr)
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists() or partial.is_symlink():
        partial.unlink()
    shutil.copyfile(source, partial)
    partial.replace(destination)


def describe(entry: dict, path: Path) -> str:
    pinned = entry.get("sha256") is not None
    volumes = volumes_for(path)
    if not volumes:
        return "absent"
    expected = len(entry.get("volumes") or [path])
    if len(volumes) != expected:
        return f"MISSING VOLUMES ({len(volumes)} of {expected} present)"
    for volume in volumes:
        refused = refused_filesystem(volume)
        if refused is not None:
            return (f"UNUSABLE filesystem: {volume.name} is on {refused}, "
                    "which the loader refuses")
    if not pinned:
        return "present (unpinned)"
    size, sha = digest_set(volumes)
    if entry.get("bytes") is not None and size != entry["bytes"]:
        return f"MISMATCH bytes {size} != {entry['bytes']}"
    if sha != entry["sha256"]:
        return f"MISMATCH sha256 {sha[:12]} != {entry['sha256'][:12]}"
    return "present, verified"


def command_list(data: dict, cache: Path) -> int:
    print(f"manifest {MANIFEST}")
    print(f"cache    {cache}{'' if cache.exists() else ' (absent)'}")
    print()
    width = max(len(e["name"]) for e in data["artifacts"])
    for entry in data["artifacts"]:
        path = entry_path(data, entry, cache)
        print(f"  {entry['name']:<{width}}  {entry['env']:<38}  {describe(entry, path)}")
    return 0


def command_check(data: dict, cache: Path) -> int:
    bad = 0
    for entry in data["artifacts"]:
        path = entry_path(data, entry, cache)
        state = describe(entry, path)
        print(f"{entry['name']}: {state}")
        if state.startswith(FAILURE_PREFIXES):
            bad += 1
    if bad:
        print(f"{bad} entr{'y' if bad == 1 else 'ies'} failed verification", file=sys.stderr)
    return 1 if bad else 0


def fetch_file(entry: dict, cache: Path, link: bool) -> Path:
    source = entry["source"]
    raw = os.path.expanduser(os.path.expandvars(source["path"]))
    origin = Path(raw) if os.path.isabs(raw) else ROOT / raw
    volumes = volumes_for(origin)
    if not volumes:
        fail(f"{entry['name']}: source {origin} does not exist")
    cache.mkdir(parents=True, exist_ok=True)
    if link:
        link, reason = link_plan(origin, cache)
        print(f"{entry['name']}: {'linking' if link else 'copying'} ({reason})", flush=True)
    destination = cache / entry["file"]
    for index, volume in enumerate(volumes):
        target = destination if index == 0 else destination.with_name(
            destination.name + volume.name[len(origin.name):]
        )
        place(volume, target, link)
    return destination


def fetch_https(entry: dict, cache: Path) -> Path:
    destination = cache / entry["file"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    with urllib.request.urlopen(entry["source"]["url"], timeout=60) as response, partial.open(
        "wb"
    ) as out:
        shutil.copyfileobj(response, out, CHUNK)
    partial.replace(destination)
    return destination


def fetch_hf(entry: dict, cache: Path) -> Path:
    destination = cache / entry["file"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    hf = shutil.which("hf")
    if hf is None:
        fail(f"{entry['name']}: the 'hf' CLI is required for hf:// sources")
    source = entry["source"]
    staging = cache / (entry["name"] + ".hf")
    staging.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [hf, "download", source["repo"], source["file"], "--local-dir", str(staging)],
        check=True,
    )
    (staging / source["file"]).replace(destination)
    shutil.rmtree(staging, ignore_errors=True)
    return destination


def command_fetch(data: dict, cache: Path, names: list[str], link: bool) -> int:
    by_name = {e["name"]: e for e in data["artifacts"]}
    if "all" in names:
        names = [e["name"] for e in data["artifacts"]]
    unknown = [n for n in names if n not in by_name]
    if unknown:
        fail(f"unknown artifact(s): {', '.join(unknown)}")
    status = 0
    for name in names:
        entry = by_name[name]
        kind = entry["source"]["kind"]
        if kind == "committed":
            print(f"{name}: committed at {entry_path(data, entry, cache)}, nothing to fetch")
            continue
        print(f"{name}: fetching ({kind})", flush=True)
        if kind == "file":
            path = fetch_file(entry, cache, link)
        elif kind == "https":
            path = fetch_https(entry, cache)
        elif kind == "hf":
            path = fetch_hf(entry, cache)
        else:
            fail(f"{name}: unknown source kind {kind!r}")
        state = describe(entry, path)
        print(f"{name}: {state} -> {path}")
        if state.startswith(FAILURE_PREFIXES):
            status = 1
    return status


def command_pin(data: dict, cache: Path, name: str, origin: str, manifest_path: Path,
                link: bool) -> int:
    by_name = {e["name"]: e for e in data["artifacts"]}
    if name not in by_name:
        fail(f"unknown artifact {name!r}")
    entry = by_name[name]
    source = Path(os.path.expanduser(origin))
    volumes = volumes_for(source)
    if not volumes:
        fail(f"--from must name the artifact's entry file: {source}")
    cache.mkdir(parents=True, exist_ok=True)
    if link:
        link, reason = link_plan(source, cache)
        print(f"{name}: {'linking' if link else 'copying'} ({reason})", flush=True)
    destination = cache / entry["file"]
    for index, volume in enumerate(volumes):
        target = destination if index == 0 else destination.with_name(
            destination.name + volume.name[len(source.name):]
        )
        if volume.resolve() != target.resolve():
            place(volume, target, link)
    cached = [destination] + [
        destination.with_name(destination.name + volume.name[len(source.name):])
        for volume in volumes[1:]
    ]
    size, sha = digest_set(cached)
    entry["bytes"] = size
    entry["sha256"] = sha
    entry["volumes"] = [
        {"name": volume.name, "bytes": volume.stat().st_size} for volume in volumes
    ]
    entry["source"] = {"kind": "file", "path": str(source)}
    manifest_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"{name}: pinned {len(volumes)} volume(s) bytes={size} sha256={sha}")
    print(f"{name}: cached at {destination}")
    return 0


def command_print_env(data: dict, cache: Path) -> int:
    seen: set[str] = set()
    for entry in data["artifacts"]:
        path = entry_path(data, entry, cache)
        if not path.exists() or entry["env"] in seen:
            continue
        seen.add(entry["env"])
        print(f"export {entry['env']}={path}")
    return 0


# --- self-test -------------------------------------------------------------------------------

TINY = b"provenance-check" * 512


def self_test() -> int:
    """Exercise the link decision and the --check state on real and simulated filesystems.

    Two cases need no simulation: one filesystem (a hard link) and two local filesystems (a copy).
    The refused-filesystem cases classify a directory as refused because no network filesystem is
    mounted in the test environment; the decision, the copy and the reported state are the
    production path, and the real statfs(2) call is still the one that runs for every other case.
    """
    failures: list[str] = []
    skipped: list[str] = []
    for name, case in _SELF_TEST_CASES:
        with tempfile.TemporaryDirectory() as directory:
            try:
                problem = case(Path(directory))
            except _SelfTestSkip as skip:
                skipped.append(f"{name}: {skip}")
                continue
            if problem:
                failures.append(f"{name}: {problem}")
    if failures:
        print(f"self-test failed: {len(failures)} of {len(_SELF_TEST_CASES)} case(s)")
        for failure in failures:
            print(f"  {failure}")
        return 1
    for note in skipped:
        print(f"self-test skipped: {note}")
    print(f"self-test ok: {len(_SELF_TEST_CASES) - len(skipped)} case(s) passed")
    return 0


class _SelfTestSkip(Exception):
    """A case that this host cannot run, such as one needing a second local filesystem."""


def _self_test_source(work: Path) -> tuple[Path, Path]:
    source = work / "source" / "tiny.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(TINY)
    cache = work / "cache"
    cache.mkdir()
    return source, cache


def _self_test_manifest(work: Path, source: Path, sha256: str | None = None) -> Path:
    digest = hashlib.sha256(TINY).hexdigest() if sha256 is None else sha256
    manifest = work / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": 1,
        "cache": {"variable": "NINFER_TEST_ARTIFACT_CACHE",
                  "default": "~/.cache/ninfer/test-artifacts"},
        "fixtures": [],
        "artifacts": [{"name": "tiny", "env": "NINFER_TINY_ARTIFACT", "file": "tiny.bin",
                       "bytes": len(TINY), "sha256": digest,
                       "source": {"kind": "file", "path": str(source)}, "tests": []}],
    }, indent=2), encoding="utf-8")
    return manifest


def _self_test_capture(function: object, *arguments: object) -> tuple[object, str]:
    """Call `function(*arguments)`, capturing both streams; SystemExit becomes its exit code."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            value = function(*arguments)
        except SystemExit as exit_request:
            value = int(exit_request.code or 0)
    return value, out.getvalue() + err.getvalue()


def _self_test_run(manifest: Path, cache: Path, *arguments: str) -> tuple[int, str]:
    """Run one command in-process and return (exit code, stdout and stderr)."""
    code, output = _self_test_capture(
        main, ["--manifest", str(manifest), "--cache", str(cache), *arguments]
    )
    return int(code), output


class _refused:
    """Classify `paths` as a refused filesystem for the duration of the block."""

    def __init__(self, *paths: Path) -> None:
        self._roots = [path.resolve() for path in paths]
        self._original = None

    def __enter__(self) -> None:
        self._original = filesystem_type
        roots = self._roots

        def classify(path: Path) -> int:
            return 0x6969 if Path(path).resolve().is_relative_to(*roots) else self._original(path)

        globals()["filesystem_type"] = classify

    def __exit__(self, *_: object) -> None:
        globals()["filesystem_type"] = self._original


def _case_same_filesystem(work: Path) -> str | None:
    source, cache = _self_test_source(work)
    allowed, reason = link_plan(source, cache)
    if not (allowed and reason == "same local filesystem"):
        return f"link_plan refused a same-filesystem link: {reason}"
    manifest = _self_test_manifest(work, source)
    code, output = _self_test_run(manifest, cache, "--fetch", "tiny", "--link")
    placed = cache / "tiny.bin"
    if code != 0 or "linking (same local filesystem)" not in output:
        return f"--fetch --link did not link: exit {code}, output {output!r}"
    if placed.is_symlink() or os.stat(placed).st_ino != os.stat(source).st_ino:
        return "--fetch --link did not produce a hard link"
    return None


def _case_other_filesystem(work: Path) -> str | None:
    shared = Path("/dev/shm")
    if not shared.is_dir() or not os.access(shared, os.W_OK):
        raise _SelfTestSkip("/dev/shm is not available")
    source, _ = _self_test_source(work)
    cache = Path(tempfile.mkdtemp(dir=shared))
    try:
        if os.stat(source).st_dev == os.stat(cache).st_dev:
            raise _SelfTestSkip("/tmp and /dev/shm share one device")
        allowed, reason = link_plan(source, cache)
        if allowed or "different filesystems" not in reason:
            return f"link_plan allowed a cross-device link: {reason}"
        manifest = _self_test_manifest(work, source)
        code, output = _self_test_run(manifest, cache, "--fetch", "tiny", "--link")
        placed = cache / "tiny.bin"
        if code != 0 or "different filesystems" not in output:
            return f"--fetch --link did not copy: exit {code}, output {output!r}"
        if placed.is_symlink() or os.stat(placed).st_ino == os.stat(source).st_ino:
            return "--fetch --link across devices did not produce a real copy"
        return None
    finally:
        shutil.rmtree(cache, ignore_errors=True)


def _case_refused_source(work: Path) -> str | None:
    source, cache = _self_test_source(work)
    manifest = _self_test_manifest(work, source)
    with _refused(source.parent):
        allowed, reason = link_plan(source, cache)
        if allowed or reason != "the source is on NFS, which the loader refuses":
            return f"link_plan allowed a link from a refused source: {reason}"
        code, output = _self_test_run(manifest, cache, "--fetch", "tiny", "--link")
    if code != 0:
        return f"--fetch from a refused source failed: exit {code}"
    if "copying (the source is on NFS, which the loader refuses)" not in output:
        return f"--fetch did not say it copied: output {output!r}"
    placed = cache / "tiny.bin"
    if placed.is_symlink() or os.stat(placed).st_ino == os.stat(source).st_ino:
        return "--fetch from a refused source did not produce a real copy"
    if refused_filesystem(placed) is not None:
        return f"the copy landed on a refused filesystem: {refused_filesystem(placed)}"
    return None


def _case_refused_cache(work: Path) -> str | None:
    source, cache = _self_test_source(work)
    with _refused(cache):
        verdict, output = _self_test_capture(link_plan, source, cache)
    if verdict != 2:
        return f"link_plan accepted a refused cache: {verdict!r}, {output!r}"
    if "point the cache at local storage" not in output:
        return f"the refusal did not say what to do: {output!r}"
    return None


def _case_check_unusable(work: Path) -> str | None:
    source, cache = _self_test_source(work)
    manifest = _self_test_manifest(work, source)
    code, _ = _self_test_run(manifest, cache, "--fetch", "tiny")
    if code != 0:
        return f"populating the cache failed: exit {code}"
    entry = cache / "tiny.bin"
    with _refused(cache):
        entry_state = describe(load(manifest)["artifacts"][0], entry)
        if not entry_state.startswith("UNUSABLE filesystem"):
            return f"--check state was {entry_state!r}, not UNUSABLE"
        code, output = _self_test_run(manifest, cache, "--check")
    if code != 1:
        return f"--check accepted an unusable entry: exit {code}"
    if "UNUSABLE filesystem" not in output:
        return f"--check did not report the unusable entry: output {output!r}"
    return None


def _case_check_symlink_to_refused(work: Path) -> str | None:
    """The cache the old tool could leave behind: a link whose target is on a refused filesystem."""
    source, cache = _self_test_source(work)
    manifest = _self_test_manifest(work, source)
    entry = cache / "tiny.bin"
    entry.symlink_to(source)
    with _refused(source.parent):
        entry_state = describe(load(manifest)["artifacts"][0], entry)
        code, output = _self_test_run(manifest, cache, "--check")
    if not entry_state.startswith("UNUSABLE filesystem"):
        return f"--check state for a link to a refused filesystem was {entry_state!r}"
    if code != 1 or "UNUSABLE filesystem" not in output:
        return f"--check accepted a link to a refused filesystem: exit {code}, output {output!r}"
    return None


def _case_check_content(work: Path) -> str | None:
    source, cache = _self_test_source(work)
    manifest = _self_test_manifest(work, source)
    code, _ = _self_test_run(manifest, cache, "--fetch", "tiny")
    if code != 0:
        return f"populating the cache failed: exit {code}"
    (cache / "tiny.bin").write_bytes(b"x" * len(TINY))
    code, output = _self_test_run(manifest, cache, "--check")
    if code != 1 or "MISMATCH sha256" not in output:
        return f"--check accepted wrong content of the right length: exit {code}, output {output!r}"
    return None


_SELF_TEST_CASES = (
    ("same filesystem hard-links", _case_same_filesystem),
    ("another local filesystem copies", _case_other_filesystem),
    ("a refused source copies", _case_refused_source),
    ("a refused cache is refused", _case_refused_cache),
    ("--check reports an unusable entry", _case_check_unusable),
    ("--check rejects a link to a refused filesystem", _case_check_symlink_to_refused),
    ("--check verifies content", _case_check_content),
)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--cache", default=None, help="cache directory (default: the manifest's)")
    parser.add_argument("--check", action="store_true", help="verify every cached entry")
    parser.add_argument("--fetch", nargs="+", metavar="NAME", default=None, help="artifacts to fetch")
    parser.add_argument("--all", action="store_true", help="fetch every entry")
    parser.add_argument("--link", action="store_true",
                        help="hard-link local sources when the source and the cache are on one "
                             "accepted local filesystem; otherwise copy")
    parser.add_argument("--pin", metavar="NAME", default=None, help="record bytes+sha256 for an artifact")
    parser.add_argument("--from", dest="origin", metavar="PATH", default=None, help="with --pin: the artifact")
    parser.add_argument("--print-env", action="store_true", help="print shell exports for cached entries")
    parser.add_argument("--self-test", action="store_true",
                        help="exercise the --link decision and --check on temporary trees")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    manifest_path = Path(args.manifest)
    data = load(manifest_path)
    cache = cache_dir(data, args.cache)

    if args.pin:
        if not args.origin:
            fail("--pin needs --from PATH")
        return command_pin(data, cache, args.pin, args.origin, manifest_path, args.link)
    if args.fetch or args.all:
        names = args.fetch or ["all"]
        return command_fetch(data, cache, names, args.link)
    if args.check:
        return command_check(data, cache)
    if args.print_env:
        return command_print_env(data, cache)
    return command_list(data, cache)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
