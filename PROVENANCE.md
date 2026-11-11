# Provenance

NInfer is licensed under the Apache License 2.0 ([LICENSE](LICENSE)). This file is the single
ledger of where the code came from. It consolidates four records that already exist and stay
authoritative for their own scope:

- [HISTORY.md](HISTORY.md) — the lineage, and the per-pull-request provenance of every
  cherry-picked change with its source commit and original author.
- `third_party/<component>/LICENSE` — the vendored dependency licences.
- [`src/ops/linear/exl3/tensorfold/NOTICE`](src/ops/linear/exl3/tensorfold/NOTICE) — the vendored
  TensorFold kernels and their ExLlamaV3 derivative.
- `tools/chat_templates/LICENSE` — the Qwen chat templates.

Nothing here replaces those records. This file exists so that a reader, a licence scanner, or a
downstream distributor can see every component at once.

## 1. Base and lineage

| stage | project | licence | note |
|---|---|---|---|
| origin | [Neroued/ninfer](https://github.com/Neroued/ninfer) | Apache-2.0 | the base engine |
| Flash-Next peer line | [lkarlslund/ninfer](https://github.com/lkarlslund/ninfer) (archived as `ninfer6000`) | Apache-2.0 | Qwen3.8 Flash-Next 125B-A6B support: the local conversion path and the v3 frontend work; this fork descends from its tip `2aa87467` |
| successor peer | [quench6k](https://github.com/lkarlslund/quench6k) | — | a separate Rust/CUDA engine. Exchange is at the design and evidence level only; **no code crosses** |
| this fork | `baristahaus/ninfer-gb10` | Apache-2.0 | the GB10 (`sm_121a`) port and the 5090 lane |

## 2. Merged upstream revisions

| revision | what |
|---|---|
| `e31bc99b` | Neroued/ninfer master: unified Q4–Q8/FP8/NVFP4 templates with A16 MMA routes, BF16 template expansion, two-stage GDN, KDA, `NINFER_CUDA_SYNC` |
| `81c8ce09` | the constrained-decoding source for the vendored XGrammar CPU core |
| `d44ab584` | the merge base of this fork and the two-GPU forks |

## 3. Cherry-picked changes (code taken)

Every cherry-pick keeps its original author and records its source commit (`git cherry-pick -x`).
Full per-PR detail is in [HISTORY.md](HISTORY.md) §"Pull requests merged since the port".

| change | source project | source commit | original author |
|---|---|---|---|
| Layout state preserved after overflow | giveen/ninfer-ext | `1d4a6162` | Duncan Betts |
| Converter reads shard headers without a safetensors index (+ test) | giveen/ninfer-ext | `a59c3b4a`, `7605e257` | giveen |
| `ignore_eos` on chat completions (+ docs/test) | giveen/ninfer-ext | `b003327c`, `1b471efb` | Thireus |
| OpenAI requests publish stable shared prefixes | giveen/ninfer-ext | `da0f6cd2` | giveen |
| Quoted reasoning closes and later tool-call markers | giveen/ninfer-ext | `71304581` | Fedor Suchkov |
| Duplicate tool-call parameter keeps its last value | giveen/ninfer-ext | `d3a44d21` (combined per `932c549a`) | adubkov |
| Engine worker recovers from OOM | Doelfke/ninfer-yarn, carried by gzenz/ninfer | `3f3272d6` | David Oelfke, ported by Ian Ranson |
| Token log-probabilities | this fork (pull request #24) | — | mtdphn (Daphne) |

## 4. Vendored source

| component | origin | licence | scope |
|---|---|---|---|
| TensorFold EXL3 kernels | [ashhart/TensorFold](https://github.com/ashhart/TensorFold) | Apache-2.0 from 0.6.0; MIT for code written before 0.6.0 | `src/ops/linear/exl3/tensorfold/` — two kernel files, adapted to this artifact ABI. See that directory's [NOTICE](src/ops/linear/exl3/tensorfold/NOTICE) |
| ExLlamaV3 trellis decoder | ExLlamaV3 | MIT, Copyright (c) 2025 Turboderp | derivative inside `exl3_decode.cuh`, credited in the file header |
| Qwen chat templates | the Qwen model releases | see `tools/chat_templates/LICENSE` | `tools/chat_templates/*.jinja` |
| capy-flash pinned-host mailbox transport | [capy-flash](https://github.com/ValerioDolci/capy-flash) (formerly ninfer-tp2) | Apache-2.0 | `src/ops/kernel/peer_exchange.cuh`, `src/ops/kernel/peer_exchange_lane.cuh`, `src/ops/common/peer_mailbox.cu`, `include/ninfer/ops/peer_mailbox.h` ported unchanged except for this tree's `ExecutionContext` interface; `include/ninfer/ops/allreduce.h` and `src/ops/common/allreduce.cu` are a reduced adaptation of capy's all-reduce seam, keeping the mailbox transport and dropping the staged path (see the header for the measurement behind that) |
| capy-flash durable disk KV tier | [capy-flash](https://github.com/ValerioDolci/capy-flash) | Apache-2.0 | `src/core/disk_kv_cache.{h,cpp}` and `tests/test_disk_kv_cache.cpp` ported from capy's `src/core/disk_kv_cache.{h,cpp}` (format 2, v0.9.0) and its test; the on-disk format, eviction order, reference counting and atomicity rules are unchanged. This tree adds `Writer::submit(MemorySpan)`, drops the writer's stderr progress line, and adds the restart durability and `MemorySpan` test cases. Not wired into the engine yet |
| capy-flash serving telemetry and prefill-miss diagnostic | [capy-flash](https://github.com/ValerioDolci/capy-flash) | Apache-2.0 | `src/serve/gpu_telemetry.{h,cpp}` and `src/serve/prefill_miss_dump.{h,cpp}`, ported from capy's same-named files at capy HEAD `3d4b4b76`; the NVML subset, the sampling rules, the prefill-miss qualification rules and the dump format are capy's. This tree takes the device identities from `ServerLogEnvironment` and keeps the telemetry path free of CUDA |
| capy-flash turn anchors | [capy-flash](https://github.com/ValerioDolci/capy-flash) | Apache-2.0 | `src/serve/turn_anchors.{h,cpp}`, ported from capy `src/serve/turn_anchors.{h,cpp}` v0.9.0; the anchor placement and the marker vocabulary are capy's. `include/ninfer/types.h` carries only the `--turn-anchors` option default and its comment — no capy code is in that file |
| capy-flash reasoning-close retry | [capy-flash](https://github.com/ValerioDolci/capy-flash) | Apache-2.0 | `src/serve/reasoning_retry.{h,cpp}` and `tests/test_reasoning_retry.cpp`, ported from capy's same-named files (release commit `c2d9e4ba`, capy-flash v0.7.7; capy HEAD `3d4b4b76`). The planner is unchanged in behaviour; the surrounding types are adapted to this tree, and the caller-stop case uses this Engine's `FinishReason::StopString` instead of capy's stop-token-id test |
| capy-flash YaRN rope math | [capy-flash](https://github.com/ValerioDolci/capy-flash) | Apache-2.0 | `src/ops/kernel/yarn.h`, adapted from capy `src/ops/kernel/yarn.h` @ `ccaf64cd` (2026-11-07), which follows HF transformers' `_compute_yarn_parameters`. This tree moves the frequency computation to the device side; capy keeps it on the host behind a `__constant__` table |

## 5. Vendored dependencies

Each directory carries its own licence file.

| dependency | licence | copyright |
|---|---|---|
| `third_party/cpp-httplib` | MIT | yhirose |
| `third_party/llama-jinja` | MIT | The ggml authors |
| `third_party/nlohmann` | MIT | Niels Lohmann |
| `third_party/spdlog` | MIT | Gabi Melman and spdlog contributors |
| `third_party/tomlplusplus` | MIT | Mark Gillard |
| `third_party/utf8proc` | MIT | the utf8proc authors |
| `third_party/xgrammar` | Apache-2.0 | the XGrammar authors |

## 6. Techniques taken without code

Recorded because a provenance ledger that lists only code understates the debt and overstates the
originality. No source code from these projects is present in the tree.

| source | what was taken |
|---|---|
| [HawkBearPig/dgpp](https://github.com/HawkBearPig/dgpp) (Stephen Douglas Hawkins) | design facts: FP8 dense weights, lossless 12-bit BF16, L2 weight prefetch; source read for the C4 gap |
| [igorls/ninfer](https://github.com/igorls/ninfer) | shapes and schedule pattern for the dense-FP8 profile (`3ea24b4`, `3db1caee`); the integrated-device startup-sizing idea (`57a277c9`) |
| cometkim | quantization arithmetic for the `nvfp4_maxabs` recipe |
| vcruz305 | the GB10 runtime recipe |
| ExLlamaV3 source | the EXL3 evaluation and the dense-FP8 decision |
| lkarlslund | the FP8 projection experiment |
| [Niko1221/Strata](https://github.com/Niko1221/Strata) | tiered-expert design direction (cited in the fork strategy) |

## 7. What this fork gives back

Correctness fixes that matter to `sm_120a` are offered upstream as goodwill pull requests.
`share/lk-upstream/` holds five such series against `lkarlslund/ninfer6000`, and
`share/lk-gb10/` holds the six-patch GB10 port. The GB10-specific work does not land upstream.

## 8. Policy for new code

1. Code taken from another project keeps its original author on the commit
   (`git cherry-pick -x`) and gets a row in §3 or §4 here.
2. A new vendored directory carries its own `LICENSE` and, where the upstream provides one, its
   `NOTICE`.
3. A file that is a derivative of another project's file carries an `SPDX-License-Identifier` and
   an `SPDX-FileCopyrightText` line, and names the upstream file in a comment.
4. A technique taken without code gets a row in §6. Say "no code taken" when that is what
   happened — it is a stronger statement than silence.
5. Do not add a component whose licence is not compatible with Apache-2.0.

## 9. Known gaps

- SPDX identifiers are present in `tools/chat_templates/*.jinja`, in the two TensorFold kernel
  files (`exl3_decode.cuh`, `exl3_linear.cu`), and, since 2026-11-10, in the capy-flash files
  named in §4. The earlier note here said TensorFold carried none; it does, and the note was
  wrong.
- `tools/check_provenance.py` enforces the ledger row for a header's port claim, not the SPDX
  lines themselves: a ported file with a row but no `SPDX-License-Identifier` would pass. The
  lines are present today; keeping them present is a review responsibility, not a guard.
- The provenance columns in [HISTORY.md](HISTORY.md) describe the tree as it stood on 2026-10-28.
  Changes after that date are recorded in the git history and in the maintainer documents, not in
  HISTORY.md.
