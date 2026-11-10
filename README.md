# NInfer GB10

A from-scratch C++/CUDA inference engine for the Qwen3.5/3.6/3.8 family and the Qwen3.8
Flash-Next 125B-A6B model, built for the NVIDIA GB10 desktop superchip and the RTX 5090. One
GPU, one resident model, maximum single-device serving performance.

Status: development fork, no tagged release yet. Run NInfer from a source build. The product line is
`master`. The engine is validated on the GB10 (`sm_121a`, aarch64) and on the RTX 5090
(`sm_120a`, x86_64).

This repository is a fork of [Neroued/ninfer](https://github.com/Neroued/ninfer). The earlier
README content, including the upstream quick start and the RTX 5090 measurement tables, is
preserved in [HISTORY.md](HISTORY.md). The record of where the code came from is in
[PROVENANCE.md](PROVENANCE.md).

## What runs today

The engine serves the Qwen3.8 Flash-Next 125B-A6B model on the GB10 as a local HTTP server:

- text generation with thinking and non-thinking prompts.
- image and video input (Vision).
- MTP and DFlash2 speculative decoding (the model drafts several tokens per decode step).
  Per-position acceptance is visible on `GET /metrics`.
- exact prefix reuse across requests (the server reuses work from earlier prompts).
- OpenAI Chat Completions, OpenAI Responses Core, and Anthropic Messages APIs, with streaming,
  tools, token counting, and usage accounting.
- a command line interface and a benchmark tool over the same engine.

The model ships as a `.ninfer` artifact. The current production artifact for this machine is the
`fp8mtp` recipe: NVFP4 routed experts, FP8 dense projections, an MTP layer, and a 51 GB file-backed
PLE table. It loads about 90 GiB of weights and KV into the 121.6 GiB unified memory pool and
leaves room for two full-length requests.

## Quick start

Requirements: 64-bit Linux, CMake 3.28 or newer, Ninja, a C++20 compiler, FFmpeg development
libraries, and `pkg-config`. For the GB10, install CUDA 13.4. The validated toolkit is CUDA
13.4.92.

`NINFER_TARGET` selects the hardware lane at configure time. It sets the CUDA architecture and
checks the host processor. The default is `rtx5090` (`sm_120a`, x86_64). Use `dgx_spark` for the
GB10 (`sm_121a`, aarch64).

Build on the GB10:

```bash
git clone https://github.com/baristahaus/ninfer-gb10.git
cd ninfer-gb10

cmake --preset dgx-spark-release
cmake --build build/dgx-spark-release -j
```

Build on the RTX 5090:

```bash
git clone https://github.com/baristahaus/ninfer-gb10.git
cd ninfer-gb10

cmake --preset rtx5090-release
cmake --build build/rtx5090-release -j
```

The checked-in presets are:

- `<target>-release`: the product binaries.
- `<target>-dev`: the product binaries, the tests and the benchmarks.
- `<target>-ops`: Core, Artifact and Ops development only.

Configure the tests with `cmake --preset rtx5090-dev` and run them with
`ctest --preset rtx5090-dev`. The [build-system reference](docs/maintainer/build-system.md)
covers the targets and the CUDA compilation boundaries.

Download a Flash-Next artifact with the Hugging Face CLI, then start the server:

```bash
./build/dgx-spark-release/apps/ninfer-serve \
  models/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer \
  --host 0.0.0.0 --port 8000 \
  --max-context 262144 \
  --kv-capacity auto \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --proposal-head full \
  --vision \
  --preserve-thinking
```

Send a request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-flash-next-125b-a6b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

`--kv-capacity auto` sizes the KV pool from the memory left after weights. `--spec mtp` selects
speculative decoding at startup. `--vision` loads Vision resources. Read `--help` for the exact
option contract. The [CLI guide](docs/cli.md) and [serving guide](docs/serving.md) cover the
routes in detail.

## Capabilities

| Capability | What it does | Notes |
|---|---|---|
| Text generation | Thinking and non-thinking prompt modes | The artifact supplies the model defaults. |
| Vision | Image and video input | `--vision` loads the Vision resources at startup. Vision is off by default. |
| MTP speculative decoding | The model drafts 1 to 5 tokens per decode step | `--spec mtp --draft-tokens N`. Per-position acceptance is on `/metrics`. |
| DFlash2 speculative decoding | A second draft head, 1 to 15 draft tokens | `--spec dflash2 --draft-tokens N`. |
| Prefix reuse | Exact reuse of an earlier prompt across requests | Device and Host retention of the checkpoint state and KV. |
| KV storage | BF16, INT8, FP8, NVFP4 and K8V4 KV cache | `--kv-dtype`. |
| Decode and prefill | Chunked prefill and exact-batch CUDA Graph decode | `--prefill-chunk` and `--no-cuda-graph`. |
| HTTP APIs | OpenAI Chat Completions, OpenAI Responses Core and Anthropic Messages | Streaming, tools, token counting and usage accounting. |
| Authentication | `--api-key` requires a bearer token or an `x-api-key` header | `/health` and the CORS preflight stay open. |
| Telemetry | Prometheus text on `GET /metrics` | Request, cache, pool and speculative-decoding counters. |
| Constrained decoding | GBNF, JSON, choice and regex masks, and the Qwen tool-call contract | Qwen3.5 family only. The Flash-Next family refuses explicit constraints. |
| CLI and tools | `ninfer`, `ninfer-serve`, `ninfer-perplexity` and `ninfer_bench` | All of them use the public Engine. |
| EXL3/Trellis artifacts | A quantization route on the RTX 5090 lane | The dense route's performance is the open SP-7 work item. |

## Measured performance

All numbers come from this machine and are recorded in the
[GB10 record](docs/performance/qwen3.8-flash-next-125b-a6b.md#gb10-sm_121a). Raw reports live
under `profiles/bench/gb10/`. Rates are decode tokens per second.

Serving decode against DGPP (HawkBearPig/dgpp), same protocol, same artifact format:

| concurrency | NInfer K=1 | NInfer K=3 | DGPP depth 1 |
|---|---:|---:|---:|
| 1 | 48.4 | 59.9 | 47.3 |
| 2 | 72.9 | 91.5 | 68.2 |
| 4 | 107.1 | 121.4 | 91.3 |

At the same draft depth the engine leads DGPP by 2.4% at one request and 17.2% at four. Deeper
drafts add more: K=3 is 32.9% ahead of DGPP at four requests. Acceptance is 2.9 tokens per round
at K=3.

Single-stream, no speculation: 31.5-32.6 tok/s. Prefill: about 2,500 tok/s from 8K to 64K prompts.

### Speculative drafting

Two draft lanes run on the same artifact. Acceptance is greedy, measured with the CLI
(`--temperature 0 --spec <lane> --draft-tokens 3 --no-cuda-graph`):

| lane | acceptance | per block position | notes |
|---|---:|---|---|
| MTP | 89.6% short / 50.2% long | 18/18, 18/12, 18/9 attempted/accepted (live scrape) | 43/48 and 307/611 drafted tokens; 45-48 tok/s decode |
| DFlash2 | 45% (27/60) | 14 / 8 / 5 | draft quality is the limiter, not the round machinery |

The DFlash2 lane is a second, block-diffusion draft head (five sliding-attention layers over the
target's tap layers 5/15/25/35/45, a candidate path selector, and two-tap dynamic
convolutions). Its draft forward and selector picks are verified against a reference replay
of the companion model on the same context. The acceptance trail follows two fixed defects.
First, the draft's final RMSNorm ran the 5120-row geometry on the 2560-row draft (8.7% to
25.9%). Second, `column_topk` wrote its candidates column-major against its rank-fastest
contract (25.9% to 45%). The remaining gap to MTP is draft quality. The decode rate has not
been re-measured after these fixes. The lane record lives in the beads issue history (`ninfer-gb10-cd3`,
`ninfer-gb10-j91`).

`/metrics` exposes `ninfer_spec_decode_{rounds,draft_tokens,accepted_tokens}_total` with a
`speculative_backend` label, the `spec_decode_acceptance_ratio` gauge, and per-position
attempted/accepted counters, so draft quality is trackable per position over time.

Long prompts (October 2026 cycle, draft-tokens 3):

| prompt tokens | TTFT | prefill tok/s | decode tok/s | facts found |
|---:|---:|---:|---:|---:|
| 131,624 | 63.4 s | 2,076 | 48.9 | 3 of 4 |
| 261,266 | 140.7 s | 1,857 | 46.1 | 3 of 4 |

The full 262,144-token model window serves end to end. The engine answered questions about
261,000-token incident bundles correctly, with no stream stalls.

Quality gates on the FP8 artifact, same harness as the development record:

- perplexity: 3.9987 on the 64K fixed corpus, within 0.1% of the recorded FP8 reference.
- tool use: 87-90 on the hard-mode tool benchmark (92 scenarios), inside the recorded band.
- vision: the 64x64 red-image probe answers "red" end to end.
- streaming tool calls publish complete calls with normal OpenAI semantics.

## Where the work lives

### Branches

| branch | position | contents |
|---|---|---|
| `master` | `9b6d1f17` | the product line: the Flash-Next 125B serving lane on GB10, the merged work line (EXL3/Trellis, tiering, telemetry and constrained decoding), and the recent serving and security review |
| `5090-test` | `3d2de369` | the tiering lane: the model-load tier drive and the SP-4 adaptive-cache runtime that consume the merged tiering machinery |
| `feat/tp2-program` | `33e79d81` | the two-GPU lane: `ExecutionContext`, the sharding vocabulary, the Flash-Next placement table, shard binding ranges and the binder shard transform |
| `dev` | `3ea7de40` | the `/metrics` work line |
| `archive/*` tags | October 2026 | older work-line and verification branches, archived |

### The two hardware lanes

The GB10 (`sm_121a`) is an integrated Grace Blackwell part. The CPU (20 Arm cores) and the GPU
(48 Blackwell SMs) share one 121.6 GiB LPDDR5X pool. There is no separate GPU memory. The GPU
reads the pool at 232-247 GB/s measured. The CUDA architecture is `sm_121a`. Upstream targets
`sm_120a`. The validated CUDA toolkit is 13.4.92.

One pool changes the rules. A "Host tier" copy costs the same bandwidth as a Device copy. The
engine accounts for this: startup sizing reads the pool directly, and the cache treats a Host
demotion as a bandwidth-neutral parking space, not a slow fallback.

The RTX 5090 (`sm_120a`) is the second build lane. It carries the tiering work: models larger
than the card run across GPU, RAM, and disk. The EXL3/Trellis quantization lever, serving
telemetry, and constrained decoding for the Qwen3.5 family landed on `master` in the October
2026 integration. The 27B-class models serve fully hot on that lane and are the regression
baseline. The tiering lane's SP-7 packages merged into `master` in October 2026. These packages are
the artifact tiering machinery (binder, placement, materializer) and the EXL3 dense and
expert-bank routes. The EXL3 27B artifact now serves end to end on that lane. A streaming
tool-call smoke decodes at roughly 10 tok/s single-stream, against roughly 83 tok/s for the
groupwise route at the same request shape. The dense EXL3 kernel route's performance remains
the open SP-7 work item. What stays on the `5090-test` line is the model-load tier
drive and the SP-4 adaptive-cache runtime that consume the merged machinery.

## What the fork owns

Upstream does not carry these. This fork is their home:

- the Qwen3.8 Flash-Next family runtime: frontend, Vision, MTP, state, CUDA Graphs.
- the context-cache machinery: pressure planning, materialization, host restore.
- the GB10 measurement record.
- serving features this workload needs: tools, streaming, CORS, shutdown drain.

## Boundaries

The product is deliberately small:

- one GPU and one resident model per engine.
- one to eight active requests, fixed at startup, no request preemption.
- no multi-GPU serving and no distributed execution.
- tools are returned to the client, never executed by the engine.
- the C++ headers are not an installed SDK.

## Documentation

- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance index](docs/performance.md)
- [GB10 performance record](docs/performance/qwen3.8-flash-next-125b-a6b.md#gb10-sm_121a)
- [Flash-Next model reference](docs/maintainer/qwen3.8-flash-next-125b-a6b-model.md)
- [Fork strategy](docs/maintainer/2026-10-12-fork-strategy.md)
- [Provenance](PROVENANCE.md)
- [History](HISTORY.md)

## Related engines

- [Neroued/ninfer](https://github.com/Neroued/ninfer) is the upstream this fork descends from.
  The relationship is bounded: selected ports in, 5090-relevant fixes out, nothing waits on it.
- [quench6k](https://github.com/lkarlslund/quench6k) serves the same model in Rust/CUDA on the
  RTX PRO 6000. The exchange with it is design- and evidence-level, because the codebases do
  not share code.

## License

NInfer is licensed under the [Apache License 2.0](LICENSE). This fork inherits it. The provenance
ledger is [PROVENANCE.md](PROVENANCE.md).

The Flash-Next artifacts are converted from their published source checkpoints by the conversion
path that [lkarlslund/ninfer](https://github.com/lkarlslund/ninfer) introduced. The upstream model
repositories are distributed under Apache-2.0. Vendored dependencies keep their own license files
under `third_party/`.
