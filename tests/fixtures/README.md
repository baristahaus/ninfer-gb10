# tests/fixtures

Small, deterministic, checked-in inputs. Nothing here is a model; the model artifacts live in a
cache and are described by [`artifacts/manifest.json`](artifacts/manifest.json)
(see [test-artifacts.md](../../docs/maintainer/test-artifacts.md)).

| Path | What | Digest |
|---|---|---|
| `artifacts/tiny.ninfer` | a one-tensor v3 artifact (BF16 8×16) for framing and reader tests | `def829d3…` |
| `artifacts/make_tiny_artifact.py` | rebuilds `tiny.ninfer` byte-for-byte | — |
| `corpus/tokens.ids` | the first 2,048 ids of the committed benchmark corpus | `fdefedde…` |
| `corpus/manifest.json` | provenance of that slice (source corpus, digest, range) | — |
| `prompts/corpus.json` | eight fixed, model-agnostic prompts with budgets and coarse expectations | `cc383848…` |
| `exl3/exllamav3-1.6.1.json` | EXL3 codec ground truth, generated once and committed | `41c0557c…` |
| `exl3/generate_fixture.py` | regenerates the EXL3 ground truth (needs a GPU) | — |
| `serve/qwen3_6_thinking_preservation.json` | a recorded serving transcript for the preservation test | — |

## Rules

- **Deterministic.** A fixture that changes between regenerations breaks every digest that names
  it. `make_tiny_artifact.py` fixes the writer's artifact identity for that reason; the manifest
  records the resulting digest.
- **Small.** The whole directory is a few hundred kilobytes. A fixture that needs a real model goes
  in the artifact manifest and the cache, not here.
- **Digested.** Every file listed in `artifacts/manifest.json`'s `fixtures` array is verified by
  `ninfer_test_artifacts_manifest_test` on every host. Add a fixture there when you add a file that
  other tests depend on.
- **Not a benchmark.** `bench/fixtures/` owns measurement inputs. `corpus/tokens.ids` is a slice of
  the benchmark corpus only because it is real, in-distribution text; it is not a measurement
  workload.

## Regenerating

```bash
python3 tests/fixtures/artifacts/make_tiny_artifact.py     # tiny.ninfer, digest must not change
python3 tools/check_test_artifacts.py .                    # verify every pinned digest
```
