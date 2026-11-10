# tests/fixtures

Small, deterministic, checked-in inputs. Nothing here is a model; the model artifacts live in a
cache and are described by [`artifacts/manifest.json`](artifacts/manifest.json)
(see [test-artifacts.md](../../docs/maintainer/test-artifacts.md)).

| Path | What | Digest |
|---|---|---|
| `corpus/tokens.ids` | the first 2,048 ids of the committed benchmark corpus | `fdefedde…` |
| `corpus/manifest.json` | provenance of that slice (source corpus, digest, range) | — |
| `prompts/corpus.json` | eight fixed, model-agnostic prompts with budgets and coarse expectations | `cc383848…` |
| `exl3/exllamav3-1.6.1.json` | EXL3 codec ground truth, generated once and committed | `41c0557c…` |
| `exl3/generate_fixture.py` | regenerates the EXL3 ground truth (needs a GPU) | — |
| `serve/qwen3_6_thinking_preservation.json` | a recorded serving transcript for the preservation test | — |

## Rules

- **Deterministic.** A fixture that changes between regenerations breaks every digest that names
  it. The artifact reader tests build their v3 fixture in a temporary directory from
  `tests/artifact/fixture.h` instead of checking one in: a committed copy would be a second
  representation of the same bytes, and the reader's own writer is the authority for them.
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
python3 tools/check_test_artifacts.py .                    # verify every pinned digest
```
