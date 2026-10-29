# NInfer GB10

A from-scratch C++/CUDA inference engine for Qwen3.8 Flash-Next 125B-A6B, built for the
NVIDIA GB10 desktop superchip. One GPU, one resident model, maximum single-device serving
performance.

This repository is a fork of [Neroued/ninfer](https://github.com/Neroued/ninfer). The earlier
README content, including the upstream quick start and the RTX 5090 measurement tables, is
preserved in [HISTORY.md](HISTORY.md).

## What runs today

The engine serves the Qwen3.8 Flash-Next 125B-A6B model on the GB10 as a local HTTP server:

- text generation with thinking and non-thinking prompts;
- image and video input (Vision);
- MTP speculative decoding (the model drafts several tokens per decode step);
- exact prefix reuse across requests (the server reuses work from earlier prompts);
- OpenAI Chat Completions, OpenAI Responses Core, and Anthropic Messages APIs, with streaming,
  tools, token counting, and usage accounting;
- a command line interface and a benchmark tool over the same engine.

The model ships as a `.ninfer` artifact. The current production artifact for this machine is the
`fp8mtp` recipe: NVFP4 routed experts, FP8 dense projections, an MTP layer, and a 51 GB file-backed
PLE table. It loads about 90 GiB of weights and KV into the 121.6 GiB unified memory pool and
leaves room for two full-length requests.

## The machine

GB10 is an integrated Grace Blackwell part. The CPU (20 Arm cores) and the GPU (48 Blackwell SMs)
share one 121.6 GiB LPDDR5X pool. There is no separate GPU memory. The GPU reads the pool at
232-247 GB/s measured. The build target is `sm_121a`. Upstream targets `sm_120a`. The validated
CUDA toolkit is 13.4.92.

One pool changes the rules. A "Host tier" copy costs the same bandwidth as a Device copy. The
engine accounts for this: startup sizing reads the pool directly, and the cache treats a Host
demotion as a bandwidth-neutral parking space, not a slow fallback.

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

Long prompts (October 2026 cycle, draft-tokens 3):

| prompt tokens | TTFT | prefill tok/s | decode tok/s | facts found |
|---:|---:|---:|---:|---:|
| 131,624 | 63.4 s | 2,076 | 48.9 | 3 of 4 |
| 261,266 | 140.7 s | 1,857 | 46.1 | 3 of 4 |

The full 262,144-token model window serves end to end. The engine answered questions about
261,000-token incident bundles correctly, with no stream stalls.

Quality gates on the FP8 artifact, same harness as the development record:

- perplexity: 3.9987 on the 64K fixed corpus, within 0.1% of the recorded FP8 reference;
- tool use: 87-90 on the hard-mode tool benchmark (92 scenarios), inside the recorded band;
- vision: the 64x64 red-image probe answers "red" end to end;
- streaming tool calls publish complete calls with normal OpenAI semantics.

## What the fork owns

Upstream does not carry these. This fork is their home:

- the Qwen3.8 Flash-Next family runtime: frontend, Vision, MTP, state, CUDA Graphs;
- the context-cache machinery: pressure planning, materialization, host restore;
- the GB10 measurement record;
- serving features this workload needs: tools, streaming, CORS, shutdown drain.

The second build lane is the RTX 5090 (`sm_120a`). It carries the tiering work: models larger
than the card run across GPU, RAM, and disk. The EXL3/Trellis quantization lever, serving
telemetry, and constrained decoding for the Qwen3.5 family landed on `master` in the October
2026 integration. The 27B-class models serve fully hot on that lane and are the regression
baseline. The tiering lane's SP-7 packages — the artifact tiering machinery (binder,
placement, materializer) and the EXL3 dense and expert-bank routes — merged into `master` in
October 2026, and the EXL3 27B artifact now serves end to end on that lane: a streaming
tool-call smoke decodes at roughly 10 tok/s single-stream against roughly 83 tok/s for the
groupwise route at the same request shape, so the dense EXL3 kernel route's performance
remains the open SP-7 work item. What stays on the `5090-test` line is the model-load tier
drive and the SP-4 adaptive-cache runtime that consume the merged machinery. Older work-line
and verification branches were archived as `archive/*` tags in October 2026.

## Build and run

Requirements: 64-bit Linux, CMake 3.28 or newer, Ninja, a C++20 compiler, FFmpeg development
libraries, and `pkg-config`. For the GB10, install CUDA 13.4. The build accepts `sm_121a` and
`sm_120a`.

```bash
git clone https://github.com/baristahaus/ninfer-gb10.git
cd ninfer-gb10

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Download a Flash-Next artifact with the Hugging Face CLI, then start the server:

```bash
./build/apps/ninfer-serve models/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer \
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

## Boundaries

The product is deliberately small:

- one GPU and one resident model per engine;
- one to eight active requests, fixed at startup, no request preemption;
- no multi-GPU serving and no distributed execution;
- tools are returned to the client, never executed by the engine;
- the C++ headers are not an installed SDK.

## Related engines

- [Neroued/ninfer](https://github.com/Neroued/ninfer) is the upstream this fork descends from.
  The relationship is bounded: selected ports in, 5090-relevant fixes out, nothing waits on it.
- [quench6k](https://github.com/lkarlslund/quench6k) serves the same model in Rust/CUDA on the
  RTX PRO 6000. The exchange with it is design- and evidence-level, because the codebases do
  not share code.

## Documentation

- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance index](docs/performance.md)
- [Flash-Next model reference](docs/maintainer/qwen3.8-flash-next-125b-a6b-model.md)
- [Fork strategy](docs/maintainer/2026-10-12-fork-strategy.md)
- [History](HISTORY.md)

## License

NInfer is licensed under the [Apache License 2.0](LICENSE). This fork inherits it.

The Flash-Next artifacts are converted from their published source checkpoints by the conversion
path that [lkarlslund/ninfer](https://github.com/lkarlslund/ninfer) introduced. The upstream model
repositories are distributed under Apache-2.0. Vendored dependencies keep their own license files
under `third_party/`.
