# Test artifacts: cache, manifest, and skipping

The real-artifact tests load a model of 17–70 GiB. Those artifacts are local prerequisites, never in
git. This page defines where they live, how a test finds one, and what happens when one is missing.

## The rule

A real-artifact test **skips with exit 77 when its artifact is absent**. It never fails because a
machine lacks a file. A test fails only when an artifact it was given is broken.

That rule has two halves, and both are required:

1. the test returns 77 when its environment variable is unset (it always has);
2. the test's registration carries `SKIP_RETURN_CODE 77`, so CTest reads the 77 as a skip.

`tests/cmake/TestArtifacts.cmake` applies the second half from the manifest to every test the
manifest names, and sets the variable only when the artifact is in the cache. A test therefore
needs no per-test plumbing: name it in the manifest once.

## The manifest

`tests/fixtures/artifacts/manifest.json` names every artifact a test can consume:

| Field | Meaning |
|---|---|
| `name` | the manifest entry's stable id |
| `env` | the environment variable the test reads |
| `file` | the file name in the cache |
| `bytes`, `sha256` | the pinned size and digest; both `null` until pinned |
| `source` | `committed`, `file`, `https` or `hf` |
| `tests` | the CTest names that consume it |
| `default` | the entry a bare `NINFER_TEST_ARTIFACT` resolves to |

The manifest also carries a `fixtures` array for the small committed fixtures
(`tests/fixtures/artifacts/tiny.ninfer`, `tests/fixtures/corpus/tokens.ids`,
`tests/fixtures/prompts/corpus.json`). Those are always present, and their digests are checked on
every host by `ninfer_test_artifacts_manifest_test`.

## The cache

```
tools/fetch_test_artifacts.py                  # what the manifest names and what is cached
tools/fetch_test_artifacts.py --check          # verify every cached entry against bytes + sha256
tools/fetch_test_artifacts.py --fetch NAME     # copy/download one entry into the cache
tools/fetch_test_artifacts.py --all            # every entry
tools/fetch_test_artifacts.py --pin NAME --from /path/to/artifact
                                               # copy it in and record bytes + sha256 in the manifest
tools/fetch_test_artifacts.py --print-env      # export lines for every cached entry
```

The cache directory is `--cache`, else `NINFER_TEST_ARTIFACT_CACHE`, else the manifest default
(`~/.cache/ninfer/test-artifacts`). Nothing writes it except this tool.

`--pin` is the one command that edits the manifest: it records the size and digest of an artifact
the maintainer already has. Pin every entry once on a machine that holds the artifact; after that
`--check` verifies the cache on any machine.

## Resolution order for one variable

1. an explicit environment value at **configure** time, if it names an existing file;
2. the manifest entry's file in the cache;
3. neither → the variable stays unset and the tests skip.

An explicit value at **run** time does not override CTest's environment property; set it at
configure time, or point `NINFER_TEST_ARTIFACT_CACHE` at a directory and re-run configure.

## Populating a machine

```bash
tools/fetch_test_artifacts.py --list                    # see what is missing
tools/fetch_test_artifacts.py --fetch qwen3_8_27b_groupwise --link
cmake --preset <target>-dev -DCMAKE_BUILD_TYPE=Release -DNINFER_TARGET=rtx5090 -DBUILD_TESTING=ON
ctest --test-dir build/<preset> -L real
```

`--link` hard-links a local source instead of copying it, which matters for a 70 GiB artifact.

## What a fixture cannot do

`tests/fixtures/artifacts/tiny.ninfer` is a one-tensor v3 artifact: it exercises framing, the reader,
and codecs. It is not a model and cannot stand in for one. The tests named in the manifest need the
real weights; the tiny fixture is for the artifact layer, and the fixed prompt and token corpora are
for tests that need deterministic input rather than a deterministic model.
