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

- SPDX identifiers are present only in `tools/chat_templates/*.jinja`. The two TensorFold kernel
  files identify their origin in comments and in the directory `NOTICE`, but carry no SPDX line,
  so automated licence scanners cannot classify them. Adding those two lines is the remaining
  work in this area.
- The provenance columns in [HISTORY.md](HISTORY.md) describe the tree as it stood on 2026-10-28.
  Changes after that date are recorded in the git history and in the maintainer documents, not in
  HISTORY.md.
