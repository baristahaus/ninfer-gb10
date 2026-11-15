# NInfer documentation

Start with the [project README](../README.md) to build NInfer, download a published artifact, and
run the CLI or HTTP server.

## User guides

| Document | Purpose |
|---|---|
| [CLI](cli.md) | text, chat-history, image/video input, output streams, sampling, MTP, and common runtime options |
| [HTTP serving](serving.md) | OpenAI Responses/Chat Completions, Anthropic Messages, state, streaming, token counting, authentication, and tool calls |
| [Performance](performance.md) | RTX 5090 / RTX PRO 6000 / GB10 measurement coverage, per-model serving results, methodology, and publication rules |
| [Weight conversion](weight-conversion.md) | official recipes, custom formats and sources, conversion methods, optional components and artifact output |
| [Cross-engine parity](performance/cross-engine-parity.md) | one artifact on two engines on one box: the measured decode and TTFT parity, with its conditions and limits |
| [Perplexity](perplexity.md) | fixed-corpus and custom-text causal perplexity, comparison rules, progress, and reports |
| [CLI examples](../examples/cli/README.md) | committed text, multimodal, thinking, long-decode, and long-context inputs |
| [Operations (GB10 fleet)](operations.md) | running the engine on GB10 nodes: node model, build, serve, monitoring, failure response, upgrades |
| [Startup configuration](startup-configuration.md) | configuration files, environment, and the startup option contract |

The executable `--help` output is the exact source for command-line option spelling and defaults.

## Model artifacts

| Model | Weights | Download | Versioned model card source |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | [model card](../model-cards/Qwen3.6-27B-NInfer/README.md) |
| Qwen3.6-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) |
| Qwen3.8-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | [model card](../model-cards/Qwen3.8-27B-NInfer/README.md) |
| Qwen3.8-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) |
| Qwen3.6-35B-A3B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | [model card](../model-cards/Qwen3.6-35B-A3B-NInfer/README.md) |

Flash-Next 125B-A6B NVFP4 is available through [local conversion or v2 upgrade](maintainer/qwen3.8-flash-next-125b-a6b-artifact.md); see its [model contract](maintainer/qwen3.8-flash-next-125b-a6b-model.md) and [performance results](performance/qwen3.8-flash-next-125b-a6b.md).

## Repository-local guides

- [Benchmarks](../bench/README.md)
- [Tests](../tests/README.md)
- [Tools](../tools/README.md)
- [Two-GPU lane](lane.md)
- [Test-artifact inventory](test-artifacts.md)
- [Storage requirements for validation](maintainer/test-storage.md) — local storage only; NFS is distribution and archive
- [Capability evaluation](../eval/README.md)

## Project records

| Document | Purpose |
|---|---|
| [Provenance](../PROVENANCE.md) | the single ledger of where the code came from: lineage, merged revisions, cherry-picks, vendored source and dependencies |
| [History](../HISTORY.md) | the README of this fork as it stood on 2026-10-28, and the per-pull-request provenance of every cherry-picked change |

## Maintainer references

The active references under [`maintainer/`](maintainer/) record current architecture, model,
artifact, and maintenance contracts. These files are not additional user workflows or installed
API documentation.

[Engine architecture](maintainer/engine-architecture.md) is the single top-level reference. The
other references own narrower contracts:

| Document | Responsibility |
|---|---|
| [Engine architecture](maintainer/engine-architecture.md) | model/config/weight ownership, loading-to-execution flow, requests, scheduling, transactions and graphs |
| [Build system](maintainer/build-system.md) | CMake targets, explicit source ownership, CUDA compilation boundaries, presets and developer configuration |
| [Artifact container](maintainer/artifact-container.md) | v3 directory, objects, logical bindings, Uses, resources and file framing/sharding |
| [EXL3 trellis contract](maintainer/exl3-trellis-contract.md) | the EXL3 trellis codec contract (SP-7): the exllamav3-compatible format, the codec and SparseMoe oracles, the container extension, variable-extent expert tiering, the converter routes, kernel contracts, acceptance, and the delegation boundary |
| [Numeric formats](maintainer/tensor-formats.md) | represented values, codes/scales, conversion arithmetic and numerical interpretation |
| [Storage layouts](maintainer/storage-layouts.md) | packing, plane offsets, padding, encoded sizes and view addressing |
| [Qwen3.5 model](maintainer/qwen3_5-model.md) | Dense/MoE mathematics, instance config, logical parameters, MTP, Vision and state semantics |
| [DFlash and DFlash2](maintainer/dflash.md) | conditioning, masked draft computation, proposal distributions and backend state |
| [Constrained decoding](maintainer/constrained-decoding.md) | GBNF/JSON/choice/regex, tool policies, masks, speculative transactions and observations |
| [Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md) | candidate selection, retention, materialization and Device/Host checkpoint policy |
| [Paged KV context store](maintainer/paged-kv-cache.md) | typed pools, pages, replicas, address spaces, reservations and consumer views |
| [Expert weight cache](maintainer/expert-weight-cache.md) | resident and spilled expert banks, admission, and the cache's own accounting |
| [ReplaySSM GDN](maintainer/replayssm-gdn.md) | raw transition records and faithful commitment of the verified state prefix |
| [Op development](maintainer/op-development.md) | semantic boundaries, source ownership, numerical qualification and performance evidence |
| [Engineering lessons](maintainer/engineering-lessons.md) | transferable method rules learned in this tree: dead-code tests, testing at the failing value, ownership seams, full builds, inference sources, fixture blindness, contract sentences, and measuring in the deployment's state |
| [Operational logging](maintainer/logging.md) | log ownership, presentation, severity and data policy |
| [Linear benchmark](maintainer/linear-tuning.md#6-pure-linear-timing) | pure Linear measurement, metrics and suites |
| [Linear tuning and reports](maintainer/linear-tuning.md) | tuning ranges, priority points, dispatch tradeoffs and final performance report format |
| [Test inventory](maintainer/test-inventory.md) | the test tree by area, and which artifacts each test consumes |

The Qwen 4 experimental family and the GLM5-Next design tree have their own references. The Qwen 4
documents describe a family that is not part of the current product contract.

| Document | Responsibility |
|---|---|
| [Qwen 4 experimental model](maintainer/qwen4_exp-model.md) | the experimental family's configuration and mathematics |
| [Qwen 4 experimental execution](maintainer/qwen4_exp-execution.md) | its planning and execution route |
| [Qwen 4 parallel design](qwen4_exp-parallel-design.md) | the parallel plan for that family |
| [Qwen 4 distributed roadmap](qwen4_exp-distributed-roadmap.md) | the distributed roadmap for that family |
| [Distributed execution design](distributed-execution-design.md) | the multi-rank execution design |
| [GLM5-Next model](maintainer/glm5next-model.md) | the GLM5-Next model contract |
| [GLM5-Next quantization](maintainer/glm5next-quantization.md) | its quantization routes |
| [GLM5-Next design tree](designs/glm5next/README.md) | the design index: MHC, parallelism, the op checklist, ReplaySSM-KDA and kpool sparse MLA |

### Dated plans in force

A dated plan is not an authority. Each plan below states its own status, and the plan is removed
when its work is executed or abandoned.

| Plan | Status as the document states it |
|---|---|
| [Fork strategy, 2026-10-12](maintainer/2026-10-12-fork-strategy.md) | ratified direction; the ratified strategy for the fork's work packages |
| [Native quantization routes, 2026-10-15](maintainer/2026-10-15-native-quant-routes-plan.md) | C, A1-A3, B1-B2 and E2a-E2c executed; A4, B3 and E2d remain |
| [Agent-class cache tiering, 2026-10-24](maintainer/2026-10-24-agent-cache-tiering-design.md) | design, pre-implementation |
| [EXL3 follow-up and lane split, 2026-10-29](maintainer/2026-10-29-exl3-followup-plan.md) | follow-up work split between the 5090 lane and the GB10 lane |
| [Spill acceptance hang, 2026-10-29](maintainer/2026-10-29-spill-acceptance-hang.md) | investigation dossier; the remaining hang is the open question |
| [EXL3 SC 3.85bpw campaign, 2026-11-02](maintainer/2026-11-02-exl3-sc385-campaign-plan.md) | active plan for the GB10 lane |
| [SP-6 / SP-4-R / SP-5 workplan, 2026-11-04](maintainer/2026-11-04-sp645-workplan.md) | dated plan for active work; handoff package for the GB10 agent |
| [SP-4-R restore design, 2026-11-04](maintainer/2026-11-04-sp4r-restore-design.md) | executed; the design record for the tiered-restore lane |

### Records and drafts

These are records, not authorities. They are linked here so that nothing in the tree is
unreachable.

- [5090 test battery guidance](maintainer/5090-test-guidance.md) — the two `sm_120a` engine bugs
  the extended battery found, both closed, and where their repros live
- [Upstream pull-request drafts](../share/lk-upstream/README.md) — drafts prepared for upstream
- [Upstream transfer analysis](../share/ninfer-gb10-transfer-analysis.md)
- [lk-gb10 record](../share/lk-gb10/README.md)
- [Torch parity report](../tools/gb10/torch_parity/REPORT.md)
- [Test fixtures](../tests/fixtures/README.md), [EXL3 fixtures](../tests/fixtures/exl3/README.md),
  [two-GPU acceptance fixtures](../tests/fixtures/tp2_accept/README.md)

Bench reports, one per campaign:

- [GB10 acceptance, 2026-10-18](../profiles/bench/gb10/acceptance-2026-10-18/summary.md)
- [GB10 file-page probe](../profiles/bench/gb10/file_page_probe/summary.md)
- [GB10 MTP economics, 2026-10-18](../profiles/bench/gb10/mtp-economics-2026-10-18/summary.md)
- [GB10 probe](../profiles/bench/gb10/probe/summary.md)
- [GB10 proposal-head rows](../profiles/bench/gb10/proposal-head-rows/summary.md)
- [GB10 SP-6 serving session](../profiles/bench/gb10/sp6-serving-session/summary.md)
- [GB10 step 2, Flash-Next v3 fork](../profiles/bench/gb10/step2-qwen3_8_flash_next_v3_fork/summary.md)
- [GB10 step 3](../profiles/bench/gb10/step3/summary.md)
- [GB10 step 7a, QAD fix](../profiles/bench/gb10/step7a-qadfix/summary.md)
- [GB10 step 7a, Flash-Next v3 fork](../profiles/bench/gb10/step7a-qwen3_8_flash_next_v3_fork/summary.md)

Model cards contain official artifact facts and source provenance. The
[conversion guide](weight-conversion.md) is the entry point for making an artifact. Exact config
fields, parameter expansion and native supported domains are maintained by the code linked from
these references.
