# NInfer

> Selected checkpoints. Maximum single-GPU inference performance.

## GB10 (sm_121a) — this fork

**Lineage:** NInfer (Neroued) → lkarlslund/ninfer (Flash-Next support) → this fork (GB10 port),
plus selected fixes cherry-picked from giveen/ninfer-ext (see below).

- **Neroued** — the original NInfer engine and the five official upstream artifacts in the
  table below ([huggingface.co/neroued](https://huggingface.co/neroued)).
- **lkarlslund** — [lkarlslund/ninfer](https://github.com/lkarlslund/ninfer) adds
  Qwen3.8 Flash-Next 125B-A6B (MoE, thinking + MTP) support: the local conversion path
  (last row of the table) and the v3 frontend work. This fork is based on the tip of
  that repository (commit `2aa87467`).
- **This fork** — the GB10 (sm_121a) port of the full stack. Upstream targets
  `sm_120a` parts (RTX 5090, RTX PRO 6000); GB10 is a different Blackwell part
  (`sm_121a`, 48 SMs vs the 170-SM class the kernel sizing was tuned for). Without the
  changes below the build gate rejects GB10 and conversion short-reads large tensors;
  the launch-sizing changes are performance-only.

### What this fork changes

1. **GB10 SM format (kernel sizing).** The architecture gate now accepts `sm_121a`
   alongside `sm_120a` (CMake + runtime capability check). Every launch constant
   derived from the 5090's 170 SMs (170/510/680/1020/5440) is replaced by a runtime
   `device_sm_count()` query (`src/ops/common/device_info.{h,cu}`; cached, 170 fallback).
   The persistent-grid kernels stride their work list by `gridDim.x`, so grid sizing is
   performance-only — any grid size is a correct result. These values are scaled from
   the reference part, not yet measured on GB10. `tools/convert` also reads safetensors
   tensors in chunks: Linux caps a single `read`/`pread` just under 2 GiB, so an
   unbounded `pread` short-reads.
2. **JSON and tool calls.** Serve accepts `response_format: text | json_object |
   json_schema` and `tool_choice: auto | none | required | allowed_tools`. Output is
   prompt-guided (no constrained decoding — schema conformance is not guaranteed); a
   leading system block carries the JSON/tool instructions and `json_output::extract`
   returns the first well-formed JSON object or array after dropping whitespace, closed
   think blocks, fences and prose. Streamed JSON arrives as one cleaned content chunk.
   See `docs/serving.md`.
3. **Lowered CUDA requirements.** The whole stack — clean build, op conformance
   (ctest), serve, probes — is validated on the GB10 stock **CUDA 13.0.88** toolkit;
   upstream's validated toolkit is CUDA 13.1. CMake imposes no CUDA version floor.

### Pull requests merged since the port

Each cherry-picked commit keeps its original author and records its source commit
(`git cherry-pick -x`).

| PR | Change | Provenance |
|---|---|---|
| [#1](https://github.com/baristahaus/ninfer-gb10/pull/1) | Streamed JSON `response_format` output sent as one cleaned chunk; parse-first `json_output::extract`; port-note corrections; GB10 plan | This fork (Claude Code session) |
| [#2](https://github.com/baristahaus/ninfer-gb10/pull/2) | `tools/gb10/` scripts for plan steps 0–3 and one pasteable report; `tools/bench/hardware/gb10.json` | This fork (Claude Code session) |
| [#3](https://github.com/baristahaus/ninfer-gb10/pull/3) | Fixes taken from [giveen/ninfer-ext](https://github.com/giveen/ninfer-ext), listed below, and the tool-call fixes ported to the Flash-Next frontend | Cherry-picks as listed; Flash-Next port by this fork |
| [#4](https://github.com/baristahaus/ninfer-gb10/pull/4) | Findings from other NInfer forks folded into the GB10 plan, with source commits | This fork (survey of `Neroued/ninfer` forks) |
| [#5](https://github.com/baristahaus/ninfer-gb10/pull/5) | This provenance record | This fork (Claude Code session) |
| [#7](https://github.com/baristahaus/ninfer-gb10/pull/7) | `tools/gb10/memory_probe.cu` and `probe_memory.sh`: standalone unified-memory probe | This fork (Claude Code session) |
| [#8](https://github.com/baristahaus/ninfer-gb10/pull/8) | Probe weight-sample compression check; `gb10.json` at the measured 246 GB/s with the device's reported name; GB10 plan updated from the first probe run and from [HawkBearPig/dgpp](https://github.com/HawkBearPig/dgpp) (FP8 dense weights, lossless 12-bit BF16, L2 weight prefetch) | This fork (Claude Code session); techniques from DGPP, no code taken |
| [#9](https://github.com/baristahaus/ninfer-gb10/pull/9) | Step-script and `ninfer_bench` trace-define fixes found on the first GB10 campaign (`--spec` flags, served model id, `--profile-measured`, multi-volume artifact size, `NINFER_PERFORMANCE_TRACE` on the bench target); GB10 plan updated with the campaign's baseline, attribution and PLE residency results | Fixes by twoFour (the GB10 box), cherry-picked from `twoFour/gb10-campaign` (`fad5ebdd`, `3df43884`); plan by this fork (Claude Code session) |
| [#10](https://github.com/baristahaus/ninfer-gb10/pull/10) | File-backed page probe (device-side PLE gather gate) and round-boundary sync/graph-launch probe; GB10 plan step 6 re-ordered from the between-round gap attribution | File-page probe and gap attribution by twoFour (the GB10 box, `twoFour/gb10-campaign` `34bcaab7`); sync probe and plan by this fork (Claude Code session) |
| [#11](https://github.com/baristahaus/ninfer-gb10/pull/11) | Yield synchronize on integrated devices (GB10), blocking kept on discrete GPUs: decode +1.6–3.3%; GB10 plan updated with the sync probe and A/B results | Change and measurements by twoFour (the GB10 box, `twoFour/gb10-campaign` `c00795fd`); plan by this fork (Claude Code session) |
| [#12](https://github.com/baristahaus/ninfer-gb10/pull/12) | NVTX sub-ranges on the Flash-Next MTP host path; post-yield trace showing the round boundary is no longer host-bound; GB10 plan step 6 narrowed to GPU work | Instrumentation and trace by twoFour (the GB10 box, `gb10/step6-item2-host-path-nvtx` `0bbc803d`); plan follow-up by this fork (Claude Code session) |
| [#13](https://github.com/baristahaus/ninfer-gb10/pull/13) | Plan step 7a: dense-FP8 Flash-Next profile. Re-encoder from the existing artifact, FP8 A16 routes for the six 2560-wide geometries, fused FP8 HyperConnection and QSA decode forms, FP8 MTP rescoring, binder support, oracle tests | This fork (Claude Code session). Shapes and schedule pattern informed by [igorls/ninfer](https://github.com/igorls/ninfer), lkarlslund's FP8 projection experiment and DGPP; no code taken |
| [#14](https://github.com/baristahaus/ninfer-gb10/pull/14) | GB10 plan: runtime settings for Flash-Next on unified memory (KV dtype, MTP depth, Host tiers, core pinning) and the EXL3 evaluation with its decision | This fork (Claude Code session), from ExLlamaV3's source and vcruz305's GB10 recipe; no code taken |
| [#15](https://github.com/baristahaus/ninfer-gb10/pull/15) | Runnable step 7 quality gate (64K fixed-token scores plus a position-binned drift comparison, `compare_token_drift.py`); FP8 small-T gap and corrected post-7a BF16 bytes recorded; no-speculation decode graphs re-enabled by filling valid columns on the device (`fill_i32`) instead of from freed host memory | Diagnosis and review pointers by mtdphn; changes by this fork (Claude Code session) |
| [#16](https://github.com/baristahaus/ninfer-gb10/pull/16) | Step 7a baseline campaign (`tools/gb10/step7a_*`: token scores, MTP acceptance, TEB, X925 pinning, page-cache drops) and per-recipe real-engine goldens, merged from `twoFour/gb10-step7a-campaign`; 64K drift phase and the BF16 baseline recorded in the plan | Campaign, goldens and measurements by twoFour (the GB10 box, `8054dc10`); drift phase and plan by this fork (Claude Code session) |
| [#17](https://github.com/baristahaus/ninfer-gb10/pull/17) | Step 7a gate results (FP8 +0.29% PPL at 64K, flat drift, acceptance and TEB unchanged) and the no-speculation graph fix verified (+2.5–2.7 ms/token); contract 7.4 worker failure policy with a test-only fault-injection seam; shared FP8 weight helpers, Qwen3.8 labels, English sections in the contract docs; agent config kept machine-local; post-OOM admission backoff restored | Gate runs, verification, failure policy and cleanup by twoFour (the GB10 box, `twoFour/gb10-step7a-gate-results` `49f46eb7`), prompted by mtdphn's review; backoff fix by this fork (Claude Code session) |

Commits taken in #3 (source commit in giveen/ninfer-ext, then original author):

| Change | Source | Author |
|---|---|---|
| Layout state preserved after overflow | `1d4a6162` | Duncan Betts |
| Converter reads shard headers without a safetensors index (+ test) | `a59c3b4a`, `7605e257` | giveen |
| `ignore_eos` on chat completions (+ docs/test) | `b003327c`, `1b471efb` | Thireus |
| OpenAI requests publish stable shared prefixes | `da0f6cd2` | giveen |
| Quoted reasoning closes and later tool-call markers | `71304581` | Fedor Suchkov |
| Duplicate tool-call parameter keeps its last value (combined per ext `932c549a`) | `d3a44d21` | adubkov |
| Engine worker recovers from OOM | `93165378` | Ian Ranson, porting David Oelfke's `3f3272d6` (Doelfke/ninfer-yarn, carried by gzenz/ninfer) |

### Validated on GB10

Clean build 728/728 (nvcc V13.0.88, `-DCMAKE_CUDA_ARCHITECTURES=121a`); op conformance
36 pass / 4 skip / 0 fail; unit suites green; 11/11 live smoke; probe suite all 200, no
early EOS. Serves the Qwen3.8 Flash-Next 125B-A6B v3 artifact (fp8 KV cache, MTP
draft-tokens 2, max concurrency 2).

Open GB10 work — baseline measurement, decode attribution, `tool_choice: required`
enforcement, and NVFP4 tuning — is tracked step by step in
[`docs/maintainer/plan-2026-09-gb10.md`](docs/maintainer/plan-2026-09-gb10.md).

NInfer is a from-scratch C++/CUDA inference engine for Qwen3.5 Dense/MoE and Qwen3.8 Flash-Next
on one Blackwell GPU. The 27B/35B models target RTX 5090; Flash-Next targets RTX PRO 6000. It runs text, image, and video prompts through a local CLI or
OpenAI-/Anthropic-compatible HTTP APIs. The runtime is deliberately specialized: one GPU, one
resident model, and a startup-fixed capacity of one to eight active requests.

Five official upstream artifacts and the local Flash-Next conversion are supported. The quick-start commands use Qwen3.8-27B NVFP4.

| Model | Weights | Artifact | Download and model card |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |
| Qwen3.8 Flash-Next 125B-A6B | `nvfp4` | `qwen3_8_flash_next_125b_a6b_nvfp4.ninfer` | [local conversion and v2 upgrade](docs/maintainer/qwen3.8-flash-next-125b-a6b-artifact.md) |

Each v3 `.ninfer` artifact carries model configuration, encoded weights, logical bindings and
frontend resources. Runtime execution uses those facts with the implemented model and Op
capabilities. You can also [convert your own weights](docs/weight-conversion.md), reuse an official
recipe or choose another supported mixture of formats.

The current engine requires v3 artifacts. Existing official v2 downloads can be
[upgraded locally](docs/weight-conversion.md#upgrade-an-existing-v2-artifact) without downloading
the weights again.

## Quick start

NInfer requires 64-bit Linux, an RTX 5090 or RTX PRO 6000 Blackwell, a CUDA toolkit supporting `sm_120a`,
CMake 3.28 or newer, a C++20 host compiler, Ninja, `pkg-config`, FFmpeg development libraries
(`libavformat`, `libavcodec`, `libavutil`, and `libswscale`), and `libcurl >= 7.85`.
CUDA 13.1 is the validated development toolkit; CMake does not impose a CUDA version floor.
The build rejects CUDA architectures other than `sm_120a`.

Build the product binaries:

```bash
git clone https://github.com/Neroued/ninfer.git
cd ninfer

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Tests and benchmarks are excluded from the default build. `cmake --preset release` configures
the same product build; `cmake --preset dev` also enables tests and benchmarks and finds a
Python 3 interpreter. Both presets use `build/` and explicitly reset the build options.
Machine-specific compiler and Python paths belong in the ignored `CMakeUserPresets.json`.
See [build organization and configuration](docs/maintainer/build-system.md) for details.

There is no install target or packaged binary distribution; run NInfer from its source build tree.
Python tools run independently of CMake; the standalone HBM probe has its own
[build command](tools/README.md#standalone-hbm-probe).

Download the artifact used by this example with the Hugging Face CLI:

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

Start a long-running text/agent server with two active-request lanes and explicit Device/Host
checkpoint capacity:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

Each request has a 240,000-token logical ceiling. A shared 240,000-token Device KV pool serves
admitted requests; two requests run concurrently when their combined reservations fit. The cache
tiers provide two Device checkpoint slots, eight pinned Host State slots, and 8 GiB of pinned Host
KV beyond the two active StateImages.

Send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

Run a one-shot CLI request with a 32,768-token allocation:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode, then give a concise conclusion." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Answer content is written to stdout. Human-readable startup/runtime diagnostics and the CLI-owned
reasoning, timing, throughput, memory, and speculative-decoding report are written to stderr;
reasoning and the result report remain unprefixed product output. On a terminal, weight
materialization uses one transient progress line followed by a compact Engine-ready summary.
Redirected stderr receives persistent readable progress without terminal control sequences. Use
`--log-level debug` for complete startup detail. Option and local input errors remain direct command
diagnostics. Use `--messages FILE` and `--vision` for structured image/video input; see the
[CLI guide](docs/cli.md) and [committed examples](examples/cli/).

## Resource-aware long-context reuse

A reusable prefix checkpoint contains KV and the complete continuation state for its exact prompt
frontier. A Device-resident checkpoint resumes directly. Under pressure, the planner weighs Device
retention, pinned Host State/KV, and eviction by immediate restore work and later reuse cost. Active
requests retain their completion reservations.

See [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
for the algorithm and [Serve TTFT benchmark](tools/bench/ttft/) for public-HTTP coverage of hot
reuse, Host resume, eviction, shared prefixes, scheduling boundaries, and multimodal load.

## Performance

Published measurements use an RTX 5090. The [performance index](docs/performance.md) links to
per-model run records and the [measurement rules](docs/performance/methodology.md). The tables
below are excerpts from those detailed results.

### Concurrent MTP3 decode

Saturated decode used INT8 group-64 KV, CUDA Graphs, MTP3, and one 8,192-token generation per active
request. Throughput uses aggregate committed decode tokens from complete intervals whose actual
decode batch equaled the configured concurrency. Acceptance covers the complete request wave;
these rates are steady decode (tok/s).

| Model profile | C=1 tok/s / accept | C=2 tok/s / accept | C=4 tok/s / accept | C=8 tok/s / accept | C8 / C1 |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% | 2.88× |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% | 5.67× |
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#decode-saturation) `groupwise-int` | 642.5 / 68.6% | 907.2 / 66.3% | 1,213.5 / 69.6% | 1,380.7 / 68.0% | 2.15× |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#decode-saturation) `nvfp4` | 143.8 / 48.9% | 267.6 / 48.1% | 461.1 / 45.8% | 766.6 / 46.0% | 5.33× |

### Single-request serving

The serial serving corpus used INT8 group-64 KV, CUDA Graphs, a 1,024-token prefill chunk, and five
fixed seeds after warm-up. The table keeps one short-prefill, one extreme-prefill, and one
structured-output MTP3 point for each published profile; the full context and scenario matrices are
linked from each model below.

| Model profile | 7,680-token prefill | 260,096-token prefill | Structured MTP3 decode |
|---|---:|---:|---:|
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) `groupwise-int` | 17,705.4 tok/s | 5,247.0 tok/s | 779.6 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `groupwise-int` | 3,218.1 tok/s | 1,614.8 tok/s | 193.0 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `nvfp4` | 11,191.5 tok/s | 2,510.6 tok/s | 252.2 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `groupwise-int` | 3,274.7 tok/s | 1,609.7 tok/s | 224.4 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `nvfp4` | 8,340.4 tok/s | 2,203.1 tok/s | 219.8 tok/s |

## Evaluation

Capability scores were measured through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

The Qwen3.6 rows used temperature 0.6 and presence penalty 1.0; the Qwen3.8 rows used temperature
1.0 and presence penalty 0.0. Multimodal evaluation used `--vision` and an 81,920-token context
limit. Text evaluation used 262,144 tokens except Qwen3.8-27B NVFP4, which used 252,928 tokens to
fit the RTX 5090 after weights. Each score is one sample per problem; model cards contain the
correct/total counts and evaluation notes.

## Startup notes

GPU residency is fixed at process startup. `--spec` selects speculative decoding residency, and
`--vision` independently selects Vision residency. Qwen3.6-35B-A3B DFlash can be combined with
Vision; it accelerates generated-text decode after multimodal prefill, not Vision encode itself.

## Docker

Build the runtime image on a host with the NVIDIA Container Toolkit:

```bash
docker build --tag ninfer:local .
```

Mount the downloaded model and run the same example server profile:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

## Capabilities and limits

The official artifacts provide the following capabilities, with optional components enabled at startup:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill, exact-batch CUDA Graph decode, and startup-bounded batched decode;
- MTP speculative decoding with draft windows from one to five;
- BF16, INT8, FP8, NVFP4, and K8V4 KV storage;
- offline causal-perplexity scoring;
- private and shared exact-prefix reuse with Device/Host State and KV retention;
- model-aware sampling defaults and explicit sampler overrides;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming,
  tools, local response state, token counting, and usage accounting.

The 35B-A3B target additionally supports DFlash with draft windows from one to fifteen for Text and
image/video Vision prompts. Qwen3.8-27B artifacts with the DFlash2 companion weights support
`--spec dflash2 --draft-tokens 7` for the same Text/Vision Engine path, with draft counts 1..15
and either full or optimized proposal heads.

The product boundary remains intentionally small:

- one RTX 5090 and one resident model per Engine;
- a startup-fixed capacity of one to eight active requests with bounded FIFO ingress;
- no request preemption, priority/QoS, active-request swapping, weight offload, multi-GPU, or
  distributed serving;
- one shared startup-fixed KV pool across active requests and retained prefixes;
- model architectures and format/shape combinations use explicitly implemented native paths;
- parsed tool calls are returned to the client; NInfer does not execute tools;
- the in-tree C++ headers are not distributed as an installed SDK.

`--max-context` is each sequence's logical limit. `--kv-capacity` sizes the shared Main Text KV pool
used by active requests and retained prefixes; `auto` resolves the largest legal capacity at
startup from the memory remaining after weights while keeping 1 GiB of sizing headroom. Explicit
capacities remain fixed for the process lifetime.

## Documentation

- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [Perplexity evaluation](docs/perplexity.md)
- [Weight conversion and custom recipes](docs/weight-conversion.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Serve TTFT benchmark](tools/bench/ttft/)
- [CLI examples](examples/cli/)
- [Contributing](CONTRIBUTING.md)

Run the relevant `--help` for the exact current option contract.

## Support

NInfer is a personal project that I develop out of interest. If you find it useful and would like
to support its continued development, you can [support the project on Ko-fi](https://ko-fi.com/neroued).

Support is entirely voluntary. It is not a purchase or investment and does not come with financial
returns, promised services or features, or a role in project decisions. The project's direction,
priorities, technical choices, and release schedule remain independently determined by the
maintainer.

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.
