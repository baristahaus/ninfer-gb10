# Test-artifact inventory

The real-artifact tests load a model from a cache outside the repository. That is deliberate: a
17–118 GiB artifact is a local prerequisite, never a checked-in fixture. The mechanism — the cache,
the manifest, the resolution order, the skip rule — is described in
[the maintainer page](maintainer/test-artifacts.md). This page is the **inventory**: which artifact
each test needs, where it is resolved from, how large it is, where it comes from, and what happens
when it is absent.

## Running the suite on a host without a GPU

The head node has no GPU, so `libcuda.so.1` is not installed and every test binary that links against
it fails to start:

```
./tests/ninfer_sampling_defaults_test: error while loading shared libraries: libcuda.so.1: cannot open
shared object file: No such file or directory
```

The build provides a stub in the build directory's `cuda-stub` folder, and the link step finds it,
but the runtime loader does not. Put it on the loader path:

```sh
LD_LIBRARY_PATH=$PWD/<build-dir>/cuda-stub ctest -j 4 --test-dir <build-dir>
```

Without this, roughly a hundred tests fail in 0.00 s each and the report looks like a broken build
rather than a missing driver. The failure is environmental, not a defect: the same suite is 235/235
with the stub on the path.

**A green test run is not a run where every test ran.** On a host without the cache, the
real-artifact tests skip with exit 77 and CTest reports them as skipped; the configure step and
`tools/check_test_artifacts.py` both count how many. The count is the thing to read before believing
a green run.

## Artifacts

| Artifact | Environment variable | Cache file | Size | Produced by | Consuming tests | Absent |
|---|---|---|---|---|---|---|
| `exl3_codec_ground_truth` | `NINFER_EXL3_FIXTURE` | `exllamav3-1.6.1.json` | 0.0 GiB | committed: `tests/fixtures/exl3/exllamav3-1.6.1.json` | 1 | skip (77) |
| `qwen3_8_27b_v3_plain` | `NINFER_TEST_ARTIFACT` | `qwen3_8_27b_v3-plain.ninfer` | 20.0 GiB | `/nfs/ninfer-gb10/qwen3_8_27b_v3_plain/qwen3_8_27b_v3-plain.ninfer` (the **fetch source**, not a runtime path — see [storage requirements](maintainer/test-storage.md)) | 16 | skip (77) |
| `qwen3_8_flash_next_125b_a6b_fp8_mtp` | `NINFER_QWEN38_FLASH_NEXT_WEIGHTS` | `qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer` | 118.2 GiB | `/nfs/ninfer-gb10/qwen3_8_flash_next_125b_a6b_fp8_mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer` (the **fetch source**, not a runtime path — see [storage requirements](maintainer/test-storage.md)) | 4 | skip (77) |
| `qwen3_8_flash_next_125b_a6b_dflash2` | `NINFER_QWEN38_FLASH_NEXT_DFLASH2_WEIGHTS` | `qwen3_8_flash_next_125b_a6b_dflash2.ninfer` | not pinned | not pinned | 1 | skip (77) |
| `qwen3_6_35b_a3b` | `NINFER_TEST_ARTIFACT` | `qwen3_6_35b_a3b.ninfer` | not pinned | not pinned | 3 | skip (77) |
| `qwen3_8_27b_dflash2` | `NINFER_TEST_ARTIFACT` | `qwen3_8_27b_dflash2.ninfer` | not pinned | not pinned | 2 | skip (77) |
| `qwen4_exp_banks` | `NINFER_QWEN4_EXP_TEST_ARTIFACT` | `qwen4_exp_banks.ninfer` | not pinned | not pinned | 2 | skip (77) |

Seven artifacts, 23 distinct consuming tests. Four are not pinned yet (`bytes` and `sha256` are
null in the manifest): the inventory knows they exist and which tests need them, but not their
digest, so `fetch_test_artifacts.py --check` cannot verify them until someone pins them on a host
that holds one.

## How a pinned digest is computed

For a single-file artifact the digest is the file's `sha256`. **For a sharded container it is one
`sha256` over the volumes concatenated in order — the entry file first, then the `.part-NNNNN`
continuations sorted by name — and `bytes` is the total across all volumes.** It is not the digest of
the entry file alone, so a hand check that runs `sha256sum` on the first file will report a mismatch
on a correct copy. Verify with `tools/fetch_test_artifacts.py --check`, which applies the same rule
(`volumes()` then one hash over the concatenation).

Both real artifacts are pinned, and both pins were verified against the local copies in November 2026:

| artifact | volumes | bytes | sha256 |
|---|---|---:|---|
| `qwen3_8_27b_v3_plain` | 1 | 21,492,938,224 | `231ae09a158ab5fc5aade8dc3eaf4e6a977e6d5faf6bbf1fc97cd574be7c5f4d` |
| `qwen3_8_flash_next_125b_a6b_fp8_mtp` | 4 | 126,921,610,496 | `3be54b07772d1876925755df84f81bacad3f6dbf1123c2110bb64ab004b833b2` |

## What "produced by" means

- `committed:` — the file is in the repository and always present. Its digest is checked on every
  host by `ninfer_test_artifacts_manifest_test`.
- an absolute path — the maintainer's source of truth for that artifact. The cache is populated
  from it with `tools/fetch_test_artifacts.py --fetch NAME --link` (a hard link for the 20 GiB and
  118 GiB entries), then verified with `--check`.
- `not pinned` — no source path and no digest are recorded yet. The tests still skip cleanly; what is
  missing is the pin, not the test.

## Which tests skip

Each row's consuming tests are named in `tests/fixtures/artifacts/manifest.json` under `tests`. The
configure step prints, per absent artifact, `N test(s) will skip`, and a summary line:

```
-- Test artifacts: 1 resolved, 6 absent, 22 test(s) will skip (cache ~/.cache/ninfer/test-artifacts)
```

The count is **distinct tests**, not the sum of the rows: `ninfer_qwen3_5_moe_real_test` and the
DFlash tests consume more than one artifact, so they are counted once. `tools/check_test_artifacts.py`
reports the same number from the same manifest and the same resolution rule, and fails if the table
above and the manifest disagree.

## What is not here

- **The bytes.** This page does not download or regenerate anything. Populating a machine is
  `tools/fetch_test_artifacts.py`; the [maintainer page](maintainer/test-artifacts.md) has the
  commands.
- **The converter recipes.** The path column is where the maintainer's copy lives, not how it was
  built. The recipe is the artifact's own `provenance.recipe`, and the conversion guide
  ([weight conversion](weight-conversion.md)) covers how one is produced.
- **Artifacts that no test consumes.** A manifest entry with no registered consumer is reported by
  the check as a manifest problem, not listed here.
