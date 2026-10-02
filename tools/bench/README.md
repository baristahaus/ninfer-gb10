# tools/bench

Maintainer orchestration for the public `ninfer_bench` throughput tool, serving corpus/concurrency
runners, and the external Serve TTFT client. Correctness is owned by the affected suites under
[`tests/`](../../tests/README.md).

## External Serve TTFT

[`ttft/README.md`](ttft/README.md) defines the black-box latency benchmark. The measurement runner
uses frozen text/media requests and public streaming protocols without calling Engine. A separate
controller manages the fixed Qwen3.8-27B NVFP4/FP8 Serve profiles and fresh-process isolation.

```bash
python3 tools/bench/run_serve_ttft_campaign.py --campaign resource --samples 5
```

The controller chooses the profile, starts and stops Serve for every sample, runs the external
client, stages the NVFP4 artifact once in `/dev/shm`, stores raw/progress/Serve artifacts below
`profiles/bench/ttft/`, records structured per-request Serve diagnostics, and writes Markdown,
JSON, and CSV summaries. The case catalog, exact profiles, TTFT boundary, and fixture qualification
are documented in the dedicated README.

## Flash-Next paired serving and numerical diagnostics

`run_flash_next_serving.py` measures fixed 8K/64K single requests, simultaneous pairs, a 64K
prefill admitted during an 8K decode, and optional eight-turn paired conversations. Each case
has one complete unmeasured warmup followed by five measured waves; the output limit defaults
to 512 tokens and natural stops remain enabled. The server must support two active requests. Configure NInfer cold runs with
`--no-prefix-reuse`; vLLM uses a fresh `cache_salt` per request. Run `--cases turns` separately
with prefix reuse enabled. Use NInfer `--preserve-thinking` and vLLM
`--reasoning-parser qwen3`; add vLLM `--enable-prompt-tokens-details` to retain per-request cache
counts. Conversation histories preserve reasoning separately from content,
and each conversation receives a distinct root and a stable salt.

The `--contexts` JSON maps token lengths to decoded text. Produce it from the committed
Flash-Next fixture using the matching local tokenizer (the tokenizer environment needs
`transformers`; the measurement scripts themselves use Python 3.11's standard library):

```python
import json
from pathlib import Path
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/path/to/Flash-Next", local_files_only=True)
ids = list(map(int, Path("bench/fixtures/qwen3_8_flash_next_context.ids").read_text().split()))
Path("/tmp/flash-next-contexts.json").write_text(json.dumps({
    str(n): tokenizer.decode(ids[:n]) for n in (1980, 2049, 8192, 65536)
}))
```

```bash
python3 tools/bench/run_flash_next_serving.py --url http://127.0.0.1:18087 \
  --contexts /tmp/flash-next-contexts.json --output /tmp/baseline.jsonl
python3 tools/bench/compare_flash_next_serving.py /tmp/baseline.jsonl /tmp/candidate.jsonl \
  --output /tmp/comparison.json
python3 tools/bench/check_flash_next_isolation.py --url http://127.0.0.1:18087 \
  --contexts /tmp/flash-next-contexts.json --output /tmp/isolation.json
```

The comparator checks matched prompt/output lengths for cold workloads and output lengths for
rolling conversations. Unequal output lengths invalidate a completion-time comparison; use
actual token counts and cache/latency observations to describe such conversation runs. It
bootstraps paired wave completion-time ratios; conversation turns
remain grouped by repetition. Its timing gate requires a 2% gain with a positive 95% interval
on a concurrent workload, no single-request regression above 2%, and no mean TTFT/ITL regression
above 5%. Confidence intervals describe these repeated fixtures, not variation across arbitrary
prompts. Stream gaps are delivery gaps, not individual device-token latencies. The isolation
probe checks public-label recall across different request lengths and reversed admission order;
this behavioral probe complements numerical/state tests and is not a mathematical oracle.

`compare_flash_next_scores.py` compares `ninfer-perplexity --token-scores` output with vLLM
causal prompt scores using identical token IDs. See [perplexity](../../docs/perplexity.md) and
[Flash-Next numerical diagnostics](../../docs/maintainer/qwen3.8-flash-next-125b-a6b-model.md#numerical-diagnostics).

`compare_token_drift.py` compares two `ninfer-perplexity --token-scores` runs of the same corpus
and protocol, typically two weight formats, by offset within each window: per-bin mean NLL
difference and its slope, so error that accumulates through recurrent state is visible apart from
the overall difference.

## Corpus baker

`ninfer_bench` benchmarks prefill at an exact length by slicing the first `P` token ids of a
committed corpus, so the corpus must be real, in-distribution text (not random tokens) and at
least as long as the largest prefill you want to run. `make_bench_corpus.py` bakes that corpus
offline with a local Hugging Face Qwen3.6 tokenizer.

Outputs (committed):

```text
bench/fixtures/bench_corpus.ids            whitespace-separated decimal token ids (exactly --tokens)
bench/fixtures/bench_corpus.manifest.json  tokenizer id, token count, and source description
```

Content sources:

- Built-in curated multi-domain prose (Chinese / English / code / math) — the default. It is
  encoded WITHOUT the chat template or special tokens, then tiled (paragraphs rotated each cycle)
  and truncated to exactly `--tokens`. Repetition only fills length; because prefill/decode
  throughput is token-count / bandwidth bound, it does not bias the numbers.
- `--source-text <file>` (repeatable) — tokenize your own long meaningful text instead, e.g. a
  downloaded public-domain book or a concatenated document set, for genuinely diverse very long
  content. The committed default is `~64k` tokens; raise `--tokens` and/or pass `--source-text`
  for more.

The binary slices `[0:P]`; the manifest is provenance only.

## Requirements

Install the tokenizer dependencies into the active Python environment:

```bash
pip install -r tools/bench/requirements.txt
```

The tokenizer is loaded locally only; the tool never downloads from the network. Pass
`--tokenizer-path` or set `NINFER_TOKENIZER_PATH`.

## Regenerate / check

```bash
# Regenerate the committed corpus from the built-in bank (default 65536 tokens).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --tokens 65536

# Bake from your own downloaded/assembled text instead (kept local; not committed).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --tokens 131072 --source-text /path/to/book.txt

# Check that the committed .ids and its descriptive manifest agree; no tokenizer or source needed.
python3 tools/bench/make_bench_corpus.py --check
```

`--tokens` is the exact committed corpus size and the ceiling on prefill length; increase it (and
optionally use `--source-text`) to benchmark longer prefills, memory permitting.

## NInfer performance matrix

`run_ninfer_bench_matrix.py` runs the layered public-Engine `ninfer_bench` matrix against the native
`.ninfer` artifact and stores its local reports under `profiles/bench/`. Its defaults are:

```text
artifact: out/qwen3_6_27b.ninfer
binary:   build/bench/ninfer_bench
corpus:   bench/fixtures/bench_corpus.ids
```

The matrix treats MTP `k=3` with the optimized proposal head as the primary path, keeps `k=0` and
`k=5` as controls, and sweeps `k=0..5` on representative context-decode cases. Decode-bearing cases
cover CUDA Graph and eager execution; prefill-only cases vary prompt length and prefill chunk.

```bash
# Configure the benchmark targets once; they are off in the default public build.
cmake -S . -B build -DNINFER_BUILD_BENCHMARKS=ON

# Inspect commands without running the model.
python3 tools/bench/run_ninfer_bench_matrix.py --preset core --dry-run

# Main run. Builds build/bench/ninfer_bench first, then writes JSON and summary.csv.
python3 tools/bench/run_ninfer_bench_matrix.py --preset core

# Longer run that adds 32k/64k prompt and context-decode points.
python3 tools/bench/run_ninfer_bench_matrix.py --preset full

# Run only the MTP draft-window sweep.
python3 tools/bench/run_ninfer_bench_matrix.py --preset full --suite mtp_sweep
```

Default outputs:

```text
profiles/bench/ninfer-<preset>-<timestamp>/
  commands.sh
  manifest.json
  json/<suite>/<case>.json
  logs/<suite>.<case>.stderr.txt
  summary.csv
  summary.json
```

Use `--resume` to skip completed JSON reports in an existing `--output-dir`, and `--preset smoke`
for a minimal script/runner check. `--no-build` uses the binary supplied by `--bench` without
building it.

Each raw report must be `ninfer_bench_report` schema v15. The flattened summary and schema-v4 matrix
manifest carry native facts from the report: architecture, public name, actual formats, prefill signature, artifact,
load/read/upload/staging values, Engine memory arenas including the non-additive Vision layout
inside the unified workspace and CUDA Graph allowance, per-test planned logical and
allocator-observed workspace peaks, KV capacity and
payload, configured proposal head and graph mode, phase timings and throughput, and speculative
rounds/drafts/acceptance/fallbacks. The matrix manifest is descriptive and records the commands and
selected local inputs; it does not make repository state part of report validity.

## Serving corpus benchmark

[Published coverage and model results](../../docs/performance.md) identify the recorded runs.
The [serving methodology](../../docs/performance/methodology.md) owns workload definitions,
metric boundaries, aggregation, comparison rules, and publication format. This section describes
runner usage and output files.

`run_serve_corpus.py` accepts explicit `--artifact LABEL=PATH` entries. Labels identify report groups;
the selected artifact supplies the architecture, public name and weight bindings.
Omitting `--mode` selects MTP0 and MTP3; repeat `--mode` to select a subset. Use `dflash7` for
Qwen3.6-35B-A3B DFlash K=7 and `dflash2_7` for Qwen3.8-27B DFlash2 K=7, with companion weights
in the selected artifact. `--sampling greedy` selects exact argmax; the default is stochastic.
Run commands with a selected Python 3.11 interpreter, as in the model-page reproduction entries.

The serial runner writes `run.jsonl`, `summary.csv`, `summary.md`, and per-server logs under
`server/`. JSONL contains the completed requests and responses; CSV/Markdown contain fixture and
category summaries. The output directory is supplied explicitly with `--output`.

Its schema-v7 result and flattened summaries retain the actual `prefill_signature`, request Host
exposure, and decode Host/Device-wait time per round received from the schema-v21 serving records.
Request exposure is a latency distribution value and is never summed across concurrent requests;
worker aggregation uses the serving `throughput.host_work` interval deltas. The stochastic route pins its complete
temperature/top-p/top-k/min-p/presence/frequency profile explicitly, so model-default changes do
not alter the measurement method.

## Concurrent serving benchmark

`run_serve_concurrency.py` selects `--suite decode-saturation` or `--suite corpus-makespan`.
Their distinct time boundaries and workload dispatch are defined in the
[serving methodology](../../docs/performance/methodology.md#workloads-and-measurement-boundaries).
Repeat `--concurrency` to select C points; each point starts a fresh server. The point report
records the actual Engine configuration, automatic KV capacity, shuffle seed where applicable,
dispatch method, and per-request positions.

Schema-v3 outputs include `points/*.json`, `server/*.jsonl`, and combined `summary.json`, `summary.csv`, and
`summary.md`. Corpus runs also write complete responses in `corpus/<point>/results.jsonl` and
per-request phase summaries in that directory; older campaigns may have only point reports and
server logs. Historical model pages identify the report directory associated with each table.

```bash
python3 tools/bench/run_serve_concurrency.py \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 \
  --decode-tokens 8192 \
  --output profiles/bench/concurrent-decode

python3 tools/bench/run_serve_concurrency.py \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite corpus-makespan \
  --concurrency 1 --concurrency 2 \
  --output profiles/bench/concurrent-corpus
```

Use `--kv-capacity auto` when the fixed corpus needs more shared KV than the default 262,144-token
pool. A point is intentionally not resumable: combining fragments from separate server processes
would not preserve either a steady interval or one continuous makespan.

## Flash-Next GPU work and roofline estimates

`flash_next_performance.py` joins compile-time-enabled work annotations to Nsight Systems GPU
activities for `qwen3.8-flash-next-125b-a6b/nvfp4`. It emits JSON and Markdown with per-stage
GPU work time, phase share, launch geometry, useful bytes/FLOPs, estimated time, and estimate
efficiency. PLE hash/gather host time and unaccounted GPU activities are retained separately.
This is an attribution and tuning tool, not a measurement of distance from an achievable optimum.

Configure the profiling build explicitly. `NINFER_PERFORMANCE_TRACE` defaults to `OFF`; disabled
builds do not evaluate annotation arguments. Enabled builds add host NVTX ranges, without device
reads, CUDA events, synchronization, or changes to inference arithmetic/state.

```bash
cmake -S . -B build -DNINFER_BUILD_BENCHMARKS=ON -DNINFER_PERFORMANCE_TRACE=ON
cmake --build build -j --target ninfer_bench
mkdir -p profiles/bench/flash_next_performance

# Keep graph construction in the trace: it provides ownership for subsequent replayed nodes.
# The report selects only the measured repetition, excluding load, capture and warmup.
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --export=sqlite --output=profiles/bench/flash_next_performance/mtp0 \
  ./build/bench/ninfer_bench \
    --weights /absolute/path/to/qwen3_8_flash_next_125b_a6b_nvfp4.ninfer \
    --corpus bench/fixtures/qwen3_8_flash_next_context.ids \
    -pg 8192,32 --max-ctx 16384 --prefill-chunk 8192 --kv-dtype bf16 \
    --mtp-draft-tokens 0 --warmup 1 -r 1 -o json \
    --output-file profiles/bench/flash_next_performance/mtp0-bench.json

python3.11 tools/bench/flash_next_performance.py \
  profiles/bench/flash_next_performance/mtp0.sqlite \
  --hardware tools/bench/hardware/rtx_pro_6000_blackwell.json \
  --benchmark profiles/bench/flash_next_performance/mtp0-bench.json \
  --output profiles/bench/flash_next_performance/mtp0-report.json
```

Run on an otherwise idle GPU with the explicitly selected artifact. This example requests 33
outputs (32 decode outputs). For MTP3 use `--mtp-draft-tokens 3 --lm-head-draft` and distinct output
paths. The input benchmark JSON must be from the same capture and contain one Flash-Next test.
`--sample N` selects a zero-based measured repetition when the capture contains several. The
report carries the complete supplied benchmark metadata, including configuration, memory,
committed output throughput and speculative counts. Profiled throughput is diagnostic: use
fresh paired **unprofiled** runs to claim an inference speed improvement.

A serving capture has no measured repetition. `--serve-rounds decode.mtp_round` (in place of
`--benchmark`) attributes the GPU work launched inside those per-round host ranges and adds a
per-round table that also lists unattributed kernels by name. `--batch N` keeps only rounds whose
NVTX payload (the number of active rows) is N, and `--trim F` drops a fraction of the rounds at
each end. The serve process must run under `nsys profile` from launch with a
`NINFER_PERFORMANCE_TRACE=ON` build, so that graph construction is in the trace.
`tools/gb10/round_attribution.sh` runs the complete C4 capture.

The full capture command above is deliberate. Narrowing collection to `cudaProfilerApi` without
also collecting graph-construction metadata loses replay ownership. Newer Nsight versions offer
`node:nvtx-precapture`, but the qualified command collects construction directly. Export existing
captures with `nsys export --type sqlite --output trace.sqlite trace.nsys-rep`. Graph-only traces
without node activities cannot produce this report. The importer follows creation/clone ancestry,
including instantiation clones, and associates each invocation with its actual replay launch.
It counts work once per Op invocation, not once per constituent kernel. Missing ancestry remains
unattributed; partially captured graph Ops receive no efficiency estimate. Host NVTX durations
are never substituted for GPU activity durations. The SQLite join follows NVIDIA's
[Nsight Systems analysis schema](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html).

### Interpreting the numbers

`gpu_work_ms` sums GPU kernel/copy/set durations; `gpu_busy_ms` measures their interval union
(within each invocation for stage rows). Parallel GPU activities therefore contribute separately
to work but only once to busy time. Phase shares use summed GPU work, so they are additive.
Measured wall time also includes host execution and gaps. Attribution coverage is the share of
GPU work mapped to an annotated Op, **not** the share of model mathematics covered by a cost model.
Zero batch/context metadata means that the Op interface does not provide that field.

Each Op emits its work estimate at the real call site. The formulas live in
[`src/ops/flash_next_work.h`](../../src/ops/flash_next_work.h); the generic NVTX emission mechanism
lives in [`src/core/performance.h`](../../src/core/performance.h). Runtime regions distinguish target
prefill/verify and predictor work. Op ownership includes calls from the MTP predictor, not just
main-model scheduling. The initial model coverage is:

| Stage | Counted work | Explicitly omitted or uncertain |
|---|---|---|
| MoE | Router/shared projections and top-10 routed GEMMs; NVFP4 codes, K16 scales and divisors, or BF16 weights; public I/O | Unique experts range from 10 to `min(512,10*T)`; routing/quantization/reduction scratch and elementwise compute omitted |
| QSA | Q/gate, K/V, output, index-key and non-reused index-query projections; public I/O | Scoring, selection, attention, codec and KV traffic need device-resident causal/valid/selected data; context is an envelope, not a substitute for it |
| GDN | Projections, one FP32 state read per request, state write for updates or replay record publication; public I/O; dominant sequential recurrence compute for update/record | Chunked prefill recurrence compute, convolution, norm/control traffic and nonlinear work omitted |
| HyperConnection | Mix/injection projections and public stream I/O; repeat/add/combine I/O | Norms, gates and private intermediate traffic omitted |
| PLE | Key/value projections and public I/O; host hashing/gather timed separately | Convolution/state traffic, nonlinear work and overlap-dependent host/transfer latency omitted |

T counts columns in the **execution envelope**, including any padded columns. It is not the number
of committed outputs, and speculative acceptance does not rescale the work of an already executed
verification. MoE traffic bounds are the all-shared versus maximally distinct expert scenarios,
not a uniform-routing assumption or measured expert counts. Useful bytes assume cold weights read
once per Op and public inputs/outputs transferred once; they are not measured DRAM bytes. Cache
reuse can invalidate the cold-traffic assumption, while scratch and repeated loads increase actual
traffic. These are implementation-independent useful-work estimates, not a reproduction of a
kernel's staging or tile padding.

For each invocation the report calculates the maximum of byte/rate and the three precision-specific
FLOP/rate terms. It retains the MoE traffic interval. Rates use decimal GB/s and TFLOP/s; an FMA is
two operations. Stage estimated times sum per invocation, preserving differences in geometry.
Estimate efficiency is estimated time divided by measured invocation GPU busy time. Values above
100% are retained and flagged, since cache reuse, envelope work or inappropriate hardware rates
can make the estimate exceed observed time. There is no whole-model “percent optimal” field:
incomplete math coverage, host dependencies and overlap do not support that claim.

The supplied theoretical hardware profile uses the RTX PRO 6000 **Workstation Edition** dense,
FP32-accumulating rates from NVIDIA's
[architecture whitepaper, Appendix A](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf).
It does not use the advertised structured-sparse AI rate. To use measured attainable rates, supply
another JSON file with the same fields, set `kind` to `sustained`, and record the measurement's
hardware, workload/toolchain and conditions in `reference`. Do not label a specification-derived
rate as sustained or use a different card's bandwidth probe default.

### Focused verification

```bash
python3.11 -m unittest tests/test_flash_next_performance.py
cmake -S . -B build -DBUILD_TESTING=ON -DNINFER_PERFORMANCE_TRACE=ON
cmake --build build -j --target ninfer_performance_trace_fixture
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --export=sqlite --output=profiles/bench/flash_next_performance/fixture \
  ./build/tests/ninfer_performance_trace_fixture
```

The small fixture checks actual HyperConnection repeat outputs after eager execution and two
graph launches, one cloned. It needs no model artifact. Import its SQLite with `read_report()`
from the Python module and verify three `hyper.repeat` calls, each with T=8 and 204800 useful bytes,
100% GPU attribution, and no partial calls. It is a trace/accounting test, not an inference
benchmark; do not attach an unrelated benchmark JSON just to use the product report CLI.
CPU tests cover clone relationship records, repeated replay, asynchronous eager launches, process
isolation, missing metadata, incomplete measurement, overlap and estimate accounting.

Qualified with CUDA 13.3, Nsight Systems 2026.3.1 and Python 3.11 on RTX PRO 6000 Blackwell.
The real artifact passed the Text/Vision/MTP/prefix integration test. Full-model captures at
8,192 prompt tokens plus 32 decode outputs attributed 96.8% of MTP0 and 97.2% of MTP3 GPU work,
with no partial Op instances. Remaining work includes output heads, copies and publication helpers.
See the [unprofiled tuning baseline](../../docs/performance.md#flash-next-tuning-baseline) for
throughput measurements; no inference speedup is claimed by the instrumentation itself.
