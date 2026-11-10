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

The cache directory is --cache, else NINFER_TEST_ARTIFACT_CACHE, else the manifest default
(~/.cache/ninfer/test-artifacts). A test whose entry is absent from the cache skips with exit 77;
it does not fail.

Source kinds in the manifest:
    committed  a file already in the repository; the cache is not used, the path is checked in
    file       a local path, copied (or hard-linked with --link) into the cache
    https      a URL, downloaded
    hf         a Hugging Face repository and file, fetched with the `hf` CLI if it is installed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "fixtures" / "artifacts" / "manifest.json"
CHUNK = 1 << 22


def fail(message: str) -> "NoReturn":  # noqa: F821 - typing only
    print(f"fetch_test_artifacts: {message}", file=sys.stderr)
    raise SystemExit(2)


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
    """Copy, hard-link or symlink one volume into the cache.

    A cross-filesystem hard link fails, so --link falls back to a symlink: the cache then costs no
    space and the artifact stays where the maintainer keeps it.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if link:
        try:
            os.link(source, destination)
            return
        except OSError:
            destination.symlink_to(source.resolve())
            return
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
        if state.startswith("MISMATCH"):
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
        print(f"{name}: {describe(entry, path)} -> {path}")
        if describe(entry, path).startswith("MISMATCH"):
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


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--cache", default=None, help="cache directory (default: the manifest's)")
    parser.add_argument("--check", action="store_true", help="verify every cached entry")
    parser.add_argument("--fetch", nargs="+", metavar="NAME", default=None, help="artifacts to fetch")
    parser.add_argument("--all", action="store_true", help="fetch every entry")
    parser.add_argument("--link", action="store_true",
                        help="link local sources instead of copying (hard link, else symlink)")
    parser.add_argument("--pin", metavar="NAME", default=None, help="record bytes+sha256 for an artifact")
    parser.add_argument("--from", dest="origin", metavar="PATH", default=None, help="with --pin: the artifact")
    parser.add_argument("--print-env", action="store_true", help="print shell exports for cached entries")
    args = parser.parse_args(argv)

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
