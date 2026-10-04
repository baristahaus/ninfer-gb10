# Qwen3.8 Flash-Next 125B-A6B performance

## V3 migration qualification — 2026-09-23

Matched loopback serving measurements on RTX PRO 6000 Blackwell compare the saved v2 binary with
the v3 migration built using CUDA 13.4.59. Both use the same represented NVFP4 weights, BF16 KV,
MTP3, the 147456-row optimized proposal, CUDA Graphs, context limit 73728, shared KV capacity
147456, prefill chunk 8192, two active lanes, greedy sampling and disabled prefix reuse.
Each case has one warmup and three measured waves, with 512 output tokens per request.

| Workload | V2 mean completion | V3 mean completion | V3 time change |
|---|---:|---:|---:|
| One 8K request | 3.497 s | 3.527 s | +0.88% |
| Two 8K requests | 5.214 s | 5.266 s | +0.99% |
| One 64K request | 9.537 s | 9.668 s | +1.37% |
| Two 64K requests | 17.275 s | 17.561 s | +1.65% |
| Mixed 8K/64K requests | 11.560 s | 11.743 s | +1.58% |

Prompt and output counts match, and generated text is identical for all measured requests.
Mean TTFT changes range from +1.4% to +2.0%; mean inter-token latency changes from +0.5% to +1.3%.
The existing 2% single-request and 5% latency regression guards pass. This migration does not
pass the separate optimization gate requiring a 2% concurrent speedup, and is not advertised as
a performance improvement. The comparison uses the saved pre-migration binary rather than a v2
rebuild with the new compiler; code, toolchain and run-order effects are not separated.

Reproduce the serving matrix with `tools/bench/run_flash_next_serving.py --repetitions 3` and
compare with `tools/bench/compare_flash_next_serving.py`. Local results are under
`profiles/bench/flash_next_v3/`. Full artifact reading, real Text/Vision/MTP/prefix/C2 integration,
independent HyperConnection/PLE/QSA/GDN/MoE numerical oracles, and exact conversion-word tests
qualify the migration. The source converter's 296472 checkpoint tensor descriptors were validated
against the local source; the complete artifact used here was produced by the v2-to-v3 upgrader.


## v2 comparison with vLLM

### Paired serving qualification, September 2026

The September 22 campaign uses one RTX PRO 6000 Blackwell 96 GB at 450 W, driver 610.43.03,
CUDA 13.3, sm_120a, and the same represented RadixArk NVFP4 text weights in both engines.
The NInfer artifact's additional Q4/Q5 inventory concerns Vision and the optional proposal head;
it is not a different quantization of the main text model. The isolated vLLM checkout is
`c42f5285ec9851dacfa370e163e7837e87fd7b6d` with its matching CUDA wheel, Torch 2.13.0+cu130,
FlashInfer CUTLASS NVFP4 MoE, Marlin linear projections, and CPU-offloaded PLE.

Both servers permit two active requests, use CUDA Graphs, a 73,728-token request limit, 8,192-token
prefill chunks/budget, BF16 KV unless stated otherwise, greedy generation, and a 512-token output limit.
NInfer reserves 147,456 KV tokens. Single-request measurements use that same two-request server
configuration. Cold-prefix workloads have one complete unmeasured warm wave per case followed by
five measured waves; file-backed PLE is warm. NInfer disables prefix reuse, while vLLM isolates
requests with distinct cache salts. The 8K/64K fixture produces 8,277/65,620 prompt tokens after
chat formatting. The mixed case admits a 64K request after the 8K request begins streaming.
All cold requests reach 512 output tokens. Eight-turn paired conversations are separate
cache-enabled runs and preserve reasoning history. Their NInfer cache has four Device state
slots, eight Host state slots, and 4 GiB of Host KV; vLLM uses its default prefix-cache policy.
Natural stops remain enabled: NInfer's HTTP API has no minimum-output control. Conversation output lengths can differ, so their completion
times are descriptive and excluded from the fixed-output speedup gate.

The acceptance rule is at least 2% lower paired completion time with a paired-bootstrap 95%
interval above zero, no single-request slowdown above 2%, and no mean TTFT/ITL regression above
5%. The intervals describe repeated fixed fixtures, not a population of arbitrary prompts.
The [paired serving tools](../../tools/bench/README.md#flash-next-paired-serving-and-numerical-diagnostics)
retain completion time, TTFT, mean ITL, stream gaps, actual token counts, and response content.
Earlier runs that inadvertently reused prefixes, lacked a complete warm wave, or flattened
reasoning history are excluded from qualification comparisons.

Mean end-to-end completion time in seconds (five measured waves, lower is better):

| Workload | NInfer MTP0 | NInfer MTP3 full | NInfer MTP3 optimized | vLLM MTP0 | vLLM MTP3 |
|---|---:|---:|---:|---:|---:|
| 8K single | 6.123 | 3.859 | 3.521 | 5.773 | 3.967 |
| 8K pair | 7.825 | 5.718 | 5.304 | 7.496 | 4.963 |
| 64K single | 12.397 | 10.524 | 9.915 | 10.510 | 8.567 |
| 64K pair | 20.288 | 18.844 | 18.202 | 16.949 | 14.674 |
| Mixed 8K/64K | 14.193 | 12.471 | 12.041 | 12.405 | 10.040 |

The existing `--spec mtp --draft-tokens 3 --lm-head-draft` profile passes the five cold-workload
acceptance gates against the full proposal head. Completion time falls 7.24% for 8K pairs
(95% interval 6.98–7.50%), 3.41% for 64K pairs (2.90–3.87%), and 3.44% for mixed requests
(3.41–3.49%). Single requests improve 8.75% at 8K and 5.78% at 64K; mean TTFT and ITL also
improve. This is qualification of an existing option, not a new kernel speedup. The approximate
head only proposes tokens; the unchanged full target head verifies them. The real Engine test
covers both full and optimized heads, including prefix rollback and concurrent lane reuse.

In five eight-turn paired conversations, mean follow-up TTFT is 155.8 ms without speculation,
182.1 ms with the full MTP3 proposal head, and 181.7 ms with the optimized head. Mean per-request
ITL is 12.20/8.01/7.24 ms respectively. Actual response lengths span 172–512, 263–512, and
235–512 tokens, so these conversation observations do not qualify a completion-time speedup.
The corrected histories reuse generated prefixes (typically about 99.6% cached on follow-ups),
and the optimized head does not show a cache-response latency regression in this run.

The current vLLM conversation runs average 260.3 ms follow-up TTFT without speculation and
513.3 ms with MTP3; response lengths span 187–512 and 94–512 tokens respectively. Its selected
attention block sizes are 1,568/1,600 tokens to align attention and recurrent cache pages. MTP3
reports a mean 7,474 cached tokens per follow-up, whereas NInfer's private continuations typically
leave only about 35–39 prompt tokens to compute. This cache-policy distinction matters when
interpreting latency; the rolling histories also differ with each engine's generated responses.
The MTP0 reference did not expose per-response cache counts, but server counters confirm hits
in conversations and zero hits in the cold matrix. MTP3 enables `--enable-prompt-tokens-details`.

Sequential candidate decisions:

| Candidate | Evidence | Decision |
|---|---|---|
| [QSA output-gate fusion](https://github.com/vllm-project/vllm/pull/55309) preserving the BF16 attention boundary | Independent QSA oracle and real Engine test pass; 8K pair slows 2.6%, 64K pair improves only 0.06%; 64K single slows 2.4% | Reverted |
| [True query-union QSA](https://github.com/vllm-project/vllm/pull/55430), tiles of 2/4/8 queries | Independent per-query oracle passes for overlapping/disjoint selections, holes and inactive rows. Refined two-query route passes real Engine and isolation tests, but full qualification gives only 1.5% faster 64K pairs and 8.7% slower 8K singles | Reverted |
| [Small-token GDN projection](https://github.com/vllm-project/vllm/pull/57318) | Existing native fused projection/control takes 2.1–2.8 µs at 2/4/8 tokens; the tested library projection alone takes about 9.1–9.2 µs | Keep existing native route |
| Sorted/deduplicated PLE reads | Exact gathered bytes match. At 64K, cold `pread` improves about 1.28 to 1.25 s, but warm gather grows from 9.3 to 70 ms; sorted mmap also loses warm | Keep existing mmap gather |
| Existing FP8 KV profile | One warmed paired screen: 8K 5.845 s versus BF16 5.718 s; 64K 18.997 s versus 18.843 s. Saves about 1.82 GiB at the same 147,456-token capacity | Keep BF16 for speed; FP8 remains a capacity option |

The union prototype builds a compact union with per-query membership bits, stages each union K/V
once, and maintains separate softmax state. Its final one-warp-per-query layout removes the first
prototype's occupancy penalty. A one-wave screen suggested about 9% lower 64K paired time; the
full workload did not reproduce that gain. Its improved NLL on the small scoring sample does not
rescue the failed serving acceptance gate. No rejected kernel, environment selector, or extra
union workspace remains in the runtime.

Numerical qualification includes the independent HyperConnection, PLE, QSA, GDN, replay, and MoE
Op tests, plus real Text/Vision/MTP/prefix rollback and two-lane Engine checks. Public-label probes
cover prompts below and above the sparse-selection boundary, 8K/64K mixed lengths, and reversed
admission order. Matching single/concurrent MTP captures have exactly equal committed GDN tensors
at the tested frontier. Ordinary and MTP state arithmetic is not bit-identical: one matched
frontier has a 2.42% relative RMS recurrent-state difference. This is diagnostic evidence, not
proof of an incorrect state transition.

The fixed-history scoring comparison uses 8,192 WikiText targets with identical input token IDs
and BF16 KV. NInfer NLL/PPL is 0.637672/1.892070 versus vLLM 0.597365/1.8173. Mean absolute
log-probability disagreement is 0.2486 nats; 7.80% of positions differ by more than one nat.
The engines therefore cannot be called numerically equivalent, and neither engine is a
mathematical oracle. No confirmed model/state bug was established by this campaign. Retained
fixed-token score export and opt-in target-logit/committed-GDN diagnostics support further
localization; they do not establish a general quality ranking.

### Earlier single-request throughput profile

The Flash-Next campaign measures one request on an NVIDIA RTX PRO 6000 Blackwell Workstation
Edition (96 GiB), CUDA compile/runtime and driver 13.3, BF16 KV, an 8,192-token NInfer prefill
chunk, CUDA Graph decode, greedy selection, and 512 generated tokens. The maximum prompt is
261,632 tokens, so prompt plus output reaches the model's complete 262,144-token context. vLLM
`0.1.dev20740+g55f69ea17` and NInfer use the same `RadixArk/Qwen3.8-Flash-Next-NVFP4` represented
weights. vLLM used one persistent warm process for MTP0 and one for MTP3; every prompt length had
one unmeasured warm request followed by five measured requests. NInfer used warmup=1 and one
measured request per point. Prefix reuse was disabled for these uncached PP/TG comparisons.

The ratio columns are NInfer divided by vLLM. Thus `TG ratio >= 1.20` is the acceptance gate, while
PP ratios near one show comparable prompt processing. NInfer's measured PP range is 0.86–1.10x
vLLM, and every measured TG point clears 1.20x.

| Prompt | MTP0 PP | vLLM PP | PP ratio | MTP0 TG | vLLM TG | TG ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 11,235 | 10,222 | 1.10x | 102.9 | 79.0 | 1.30x |
| 8,192 | 12,080 | 13,716 | 0.88x | 100.0 | 79.0 | 1.27x |
| 32,768 | 11,412 | 12,767 | 0.89x | 97.0 | 79.0 | 1.23x |
| 65,536 | 10,900 | 12,109 | 0.90x | 96.7 | 79.0 | 1.23x |
| 131,072 | 10,086 | 11,217 | 0.90x | 94.8 | 79.0 | 1.20x |
| 196,608 | 9,423 | 10,413 | 0.91x | 95.5 | 79.0 | 1.21x |
| 261,632 | 9,299 | 9,846 | 0.94x | 95.7 | 79.0 | 1.21x |

| Prompt | MTP3 PP | vLLM PP | PP ratio | MTP3 TG | vLLM TG | TG ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 10,435 | 10,260 | 1.02x | 178.2 | 112.3 | 1.59x |
| 8,192 | 11,349 | 13,246 | 0.86x | 274.4 | 198.5 | 1.38x |
| 32,768 | 10,805 | 11,985 | 0.90x | 253.6 | 196.1 | 1.29x |
| 65,536 | 10,316 | 11,293 | 0.91x | 256.1 | 194.5 | 1.32x |
| 131,072 | 9,536 | 10,376 | 0.92x | 243.9 | 191.6 | 1.27x |
| 196,608 | 8,797 | 9,809 | 0.90x | 232.2 | 190.1 | 1.22x |
| 261,632 | 8,814 | 9,371 | 0.94x | 240.9 | 191.4 | 1.26x |

### Flash-Next QSA selected-work experiments

On 2026-09-10, branch `perf/flash-next-qsa-selected-work` screened two ideas from recent
vLLM QSA work: skip wholly padded selection tiles, and share selected K/V loads between
neighboring prefill queries. NInfer already uses selected paged attention and incremental
compressed index keys; the full-context-mask and full-history-pooling improvements in llama.cpp
are not missing mechanisms here.

The shared-load prototype pairs two BF16 prefill queries within a request, matches overlapping
rows inside each 16-position tile, and reuses their staged K/V while preserving each query's
selection order, mask, and separate softmax. It applies at even request widths above 16; other
routes retain the original kernel. This is a conservative shared-load experiment, not a port of
vLLM's SM121 tile-union kernel. Both candidates preserve the public selected-index representation,
including interior padding and MTP reuse. Neither adds persistent state or workspace allocation.

Qualification uses the complete QSA Op's represented-input FP64 attention oracle. Added cases
cover 17/18-token prefill, identical/overlapping/disjoint neighboring selections, nonuniform K/V,
complete padded tiles, a final selected entry after the holes, an inactive final query, output
guards, and unchanged reused indices. The shared-load candidate also passes the real
Text/Vision/MTP/prefix integration test.

Measurements use RTX PRO 6000 Blackwell, sm_120a, CUDA 13.3, 450 W, the existing hybrid-Q4
Flash-Next NVFP4 artifact, BF16 KV, context capacity 73728, chunk 8192, MTP3 with the optimized
proposal head, and `bench/fixtures/qwen3_8_flash_next_context.ids`. Each process uses
`-pg '1024,128;8192,128;65536,128'`, one warmup and three measured repetitions. ComfyUI was stopped;
a process guard aborts the benchmark if an unexpected GPU compute process appears. Independent
binaries run baseline–pruning–shared–shared–pruning–baseline. Local reports and telemetry are in
`profiles/bench/flash_next_qsa_selected_work/`.

Results and decisions:

| Candidate | Observed result | Decision |
|---|---|---|
| Skip wholly padded tiles | Initial PP results vary with warming; refined version has no consistent PP gain and reduces 8K committed TG about 2.3–2.8% | Rejected |
| Pair neighboring queries and share K/V staging | 8K/64K PP about 10–12% lower than adjacent pruning-only controls | Rejected |
| Compact paired statistics, two resident CTAs, pruning disabled | 8K/64K PP about 7–9% lower than surrounding original-kernel controls; TG essentially unchanged | Rejected |

The refined pruning variant replaces the existing synchronization with a tile-validity reduction
instead of adding another barrier. Its 8K MTP acceptance still changes from 100% to 98.96%;
committed throughput is the acceptance metric, rather than removed attention work alone.
The final paired variant removes duplicated four-lane statistics and uses a two-block launch
bound: compiled register use is 121 per thread, static shared memory 1344 bytes, dynamic shared
memory 49280 bytes, and no local-memory spills. This fixes the initial occupancy limit but does
not recover the matching/staging cost.

For the final paired screen, median PP (tok/s), first/reverse comparison:

| Route | 8K PP | 64K PP |
|---|---:|---:|
| Original kernel controls | 10799 / 10825 | 9826 / 9840 |
| Compact shared-load candidate | 10063 / 9865 | 9110 / 8929 |

The compact candidate passes the strengthened QSA oracle and real Text/Vision/MTP/prefix test.
The original kernel is restored and the strengthened oracle also passes against it. No runtime
optimization from this screen is retained. The strengthened tests and this result record remain;
reports for the two refinements are in the `refined/` and `compact/` subdirectories of the campaign
directory. These negative results concern the tested CUDA implementations; they do not measure
vLLM's different SM121 tile-union implementation.

### Flash-Next tuning baseline

The 2026-09-05 tuning baseline uses the original Flash-Next implementation with performance
annotations disabled, one request, BF16 KV, CUDA Graph decode, greedy selection, no prefix reuse,
a 73,728-token capacity and an 8,192-token prefill chunk. Hardware is RTX PRO 6000 Blackwell
Workstation Edition with a 450 W power limit, CUDA compile/runtime/driver 13.3. Each point has one
warmup and five measured repetitions with warm file-backed PLE pages. The explicitly selected
artifact is `out/candidate-hybrid-q4/qwen3_8_flash_next_125b_a6b_nvfp4.ninfer`, the registered
NVFP4 identity with the 147,456-row Q4 optimized proposal head. The similarly named top-level
artifact has an older proposal-head inventory and is not the artifact used here.

Commands use `ninfer_bench --corpus bench/fixtures/qwen3_8_flash_next_context.ids
-pg '8192,512;65536,512' --max-ctx 73728 --prefill-chunk 8192 --kv-dtype bf16
--warmup 1 -r 5`, with `--mtp-draft-tokens 0` or `--mtp-draft-tokens 3 --lm-head-draft`.
Each request therefore has 513 total outputs, including 512 decode outputs. This is a fresh
NInfer tuning baseline, not an exact reproduction of the preceding vLLM comparison.

Median tokens/s, with the five-run min–max range in parentheses:

| Mode | Prompt | PP | Committed TG |
|---|---:|---:|---:|
| MTP0 | 8,192 | 12,226 (12,169–12,281) | 95.8 (95.8–95.9) |
| MTP0 | 65,536 | 10,957 (10,829–11,092) | 91.6 (91.5–91.6) |
| MTP3 | 8,192 | 11,234 (11,208–11,313) | 260.5 (260.3–260.5) |
| MTP3 | 65,536 | 10,147 (10,050–10,261) | 247.6 (247.5–247.9) |

Full-model diagnostic traces at 8,192 prompt tokens and 32 decode outputs use Nsight Systems
2026.3.1 with compile-time annotations enabled. The offline report attributes 96.8% of MTP0 and
97.2% of MTP3 GPU work, with no partial Op instances. In MTP0, prefill GPU work is 33.4% GDN,
21.9% MoE, 21.7% QSA and 20.8% HyperConnection. With MTP3, decode GPU work is 34.2% MoE,
23.1% GDN, 16.8% HyperConnection and 15.5% QSA. These trace shares support attribution only;
the throughput baseline above comes from unprofiled execution. Local JSON reports are under
`profiles/bench/flash_next_performance/baseline/`; the
[capture/report workflow](../../tools/bench/README.md#flash-next-gpu-work-and-roofline-estimates)
defines the cost-model limitations. The real Text/Vision/MTP/prefix integration test passed.

### Flash-Next KV profile tradeoff

Flash-Next exposes BF16 as the default throughput profile and row-scaled FP8 E4M3 as an explicit
capacity profile through `--kv-dtype`. A paired NInfer run on the same RTX PRO 6000, artifact,
262,144-token capacity, 8,192-token chunk, CUDA Graph configuration, seven prompt lengths, 512
generated tokens, warmup=1, and repetitions=3 measured the following FP8 changes relative to BF16:

| Mode | Prefill throughput change | Decode throughput change | Runtime reservation saved |
|---|---:|---:|---:|
| MTP0 | −17.3% at 1K; −11.9% at 8K; −5.0% at 261K | −0.4% to −1.2% | 3,196,059,648 bytes |
| MTP3, optimized proposal head | −15.0% at 1K; −3.0% at 8K; −5.5% at 261K | −25.8% at 1K; −6.1% at 8K; −0.3% at 261K | 3,462,462,976 bytes |

The decode penalty is workload-dependent because FP8 can change the generated continuation and
therefore MTP acceptance. The fixed `ninfer-ppl-1m-v1 --quick` corpus at context/stride 4096/2048
scored 261,167 tokens: BF16 PPL was 3.537131 and FP8 PPL was 3.536871. This is a −0.007% aggregate
difference, so the measured choice is capacity versus speed rather than a detectable aggregate
quality loss on that corpus.

Independent of KV format, Flash-Next QSA workspace planning now reserves its hierarchical
candidate buffers only for the `tokens <= 16` route that allocates them. At prefill chunk 8,192 and
maximum context 262,144 this removes 2,147,483,648 bytes from the startup workspace reservation
without changing execution.

NInfer startup is 23.5–25.2 seconds for the measured profiles. About 21.6–23.1 seconds is artifact
read/upload; the remaining time pins Host state/KV and prepares CUDA Graphs. The base MTP0
residency is 77,843,526,912 bytes, while MTP3 with the optimized proposal head is 83,258,958,592
bytes. The 51.2 GB PLE n-gram table stays file-backed: the target reads selected rows through its
read-only mapping and the operating-system page cache owns physical Host residency.

A separate persistent-server prefix check used a 9,522-token continuation sharing 9,503 tokens
with the prior request. Computed prefill fell from 9,512 tokens in 898.6 ms to 19 tokens in 57.0 ms;
end-to-end latency fell from 1.071 seconds to 0.269 seconds. This cache-hit result is intentionally
separate from the uncached PP table above.

The single-request corpus requests were submitted serially to a persistent `ninfer-serve` process
over the loopback OpenAI-compatible HTTP endpoint. Each reported corpus fixture used five fixed
seeds. Values are arithmetic mean ± sample standard deviation, and server warm-up completes before
the measured requests. The concurrent campaign has its own sustained-wave method below.

## GB10 (`sm_121a`)

The GB10 fork (`baristahaus/ninfer-gb10`) builds the engine for `sm_121a`. This section is the
record of its measurements, the changes they led to and the approaches it rejected. Raw reports
live under `profiles/bench/gb10/`.

### Platform and configuration

- **Hardware:** NVIDIA GB10 (Grace Blackwell) with 48 SMs and 24.0 MiB L2. It has 121.6 GiB of
  unified LPDDR5X; the specification is 273 GB/s and the measured GPU streaming read is
  232–247 GB/s, 246 in `tools/bench/hardware/gb10.json`. The CPU has 20 Arm cores (10 Cortex-X925
  and 10 A725).
- **Toolchain:** CUDA 13.0.88, driver 580.173.
- **Artifact:** recipe `qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp-v3` (`fp8mtp`, 119 GB):
  - NVFP4 routed experts;
  - FP8 E4M3 row-scaled dense projections: GDN qkv/z/out, QSA query-gate/output, HyperConnection
    down and up, shared expert, and `lm_head`;
  - an MTP layer in the main layers' formats;
  - the file-backed FP8 PLE table, 51 GB.
- **Serving settings:**
  - `--kv-dtype fp8`: decode speed is identical to BF16 KV; FP8 KV halves the KV footprint and
    gives up 7–9% prefill.
  - `--spec mtp --draft-tokens 3 --lm-head-draft`: 1–3 is the package's limit, and K=3 is the
    best or within 1% of the best on every measured class.
  - `--host-kv-mib 0`: a Host tier on unified memory is a copy in the same pool. TEB score and
    prefix hits are identical with and without it.
- **Pinning:** pinning the server to the X925 cores measured neutral; campaigns pin for
  repeatability.
- **Memory:**
  - On integrated devices, startup sizing uses `MemAvailable` less the pinned Host KV and a 6 GiB
    reserve. `cudaMemGetInfo` counts reclaimable page cache as used.
  - Load takes about 90 GiB with the default Host tier.
  - Up to eight active requests start and serve.
- **Synchronization:** `NINFER_CUDA_SYNC` unset selects `yield` on integrated devices. Blocking
  synchronization cost about 1.4 ms per wake-up on GB10.

### Results against DGPP, October 3

Protocol: DGPP's `scripts/serve_load.py` with five prompt classes (prose, code, json, math, chat),
256 output tokens, greedy, thinking off, three repetitions, and caches dropped before each server
start. DGPP is `HawkBearPig/dgpp` at `dd58d6d3`: single-Spark template, NVFP4 experts, FP8 dense,
MTP depth 1, rerun in the same campaign. NInfer is tree `05ed1f2a` with `fp8mtp`. Rates are wall
tok/s including prefill, averaged over all classes (`profiles/bench/gb10/parity-2026-10-03/`).

| C | NInfer K=1 | NInfer K=2 | NInfer K=3 | DGPP |
|---|---:|---:|---:|---:|
| 1 | 48.42 | 56.90 | 59.89 | 47.29 |
| 2 | 72.94 | 82.37 | 91.53 | 68.24 |
| 4 | 107.07 | 116.68 | 121.39 | 91.33 |

- **Matched draft depth** (NInfer K=1 against DGPP's depth 1): +2.4% at C1, +6.9% at C2 and +17.2%
  at C4.
  - At C1 K=1 only code trails DGPP, by 1.0%.
  - At C4 NInfer leads every class by 14.6–20.0%.
- **Deeper drafts:** at C4, K=3 is +32.9%. Whether DGPP gains as much at depth 2–3 is untested.
- **Acceptance:** 1.803 / 2.425 / 2.917 tokens per round at K=1/2/3.
- **Decode-only rates, C1/C2/C4:**

  | K | C1 | C2 | C4 |
  |---|---:|---:|---:|
  | 1 | 50.5 | 39.5 | 30.2 |
  | 2 | 59.6 | 45.2 | 34.1 |
  | 3 | 63.6 | 50.9 | 36.0 |

- **Per class at C4:**
  - code, json and math are fastest at K=3 (130.8 / 143.0 / 130.8);
  - prose and chat peak at K1–K2 (99.5 / 105.0), and K=3 is within 1% of that.
- **Earlier scoreboards on the same protocol:**

  | campaign | NInfer K=1 vs DGPP at C1 / C2 / C4 |
  |---|---|
  | 2026-09-30, before the QSA and MTP-round work | −10 / −9 / −10% |
  | after the PLE page-in | at parity at C4 |
  | after K5, 2026-10-02 | −3.2 / −0.3 / +5.1% |

Single-stream bench rows (`ninfer_bench`, 8K/64K prompts, 512 outputs, `fp8mtp`):

| | value |
|---|---|
| decode without speculation | 32.6 / 31.5 tok/s |
| prefill | about 2.5k tok/s from 8K to 64K (1.7k at 1K, peaks at 8K) |
| MTP rows on the fixture corpus | 63.2 (K=2), 72.1 (K=3) |

The fixture corpus tiles one 3,442-token text, so drafts accept about 99% and its MTP rows are an
upper bound. Speed claims use the served rates above or the natural-text corpus.

### Measurement rules

- **Greedy C4 decode is not run-to-run deterministic**, with or without MTP; C1 is bitwise.
  - Several ops choose kernels or reduction trees by the call's token count: BF16 linear at T=1,
    T=2..16 and above. So does the QSA split count, which depends on the visible-key envelope.
  - Admission timing decides which rows share a round.
  - Each output stays within its op's oracle criterion, but a row's bits depend on its
    batch-mates. Batch invariance is not part of the product contract.
  - C4 speed claims therefore use GPU work or device wait per round, or kernel medians; tok/s from
    four-request legs is acceptance noise. C1 tok/s A/B comparisons remain valid.
- **Profilers:**
  - nsys with `--cuda-graph-trace=node` adds a fixed cost of about 1–3 µs to every launch and
    inflates graph-launch host time. Compare kernel medians and shares, not absolute sums, and
    treat a launch-count saving as unmeasured until it is.
  - ncu cannot profile a model-size process on GB10: its replay cannot allocate in the shared
    pool. A 70 GiB repro exhausted the pool and rebooted the node. Never point ncu at a process
    without free memory of the model size plus about 30 GiB.
  - GB10's ncu DRAM counters do not reflect streaming kernels. Bandwidth questions are answered by
    timing against a plain read of the same bytes (`flash_next_moe_bench --probe`,
    `flash_next_hc_bench`, `flash_next_fp8_small_t_bench`).
- **Page cache:** a run that reads more than the page cache holds leaves it full. Drop it before a
  benchmark that loads the model.

### Where a C4 K=1 round goes

Round attribution, nsys, tree `3b257c91` (`profiles/bench/gb10/latency-batch/`). There are 65 ms
of GPU work per round. Floors come from the attribution's byte model at 246 GB/s.

| stage | ms per round | status |
|---|---:|---|
| `moe.nvfp4` routed experts (96 launches) | 33.1 | at the read floor |
| `moe.nvfp4` other: shared expert, router, routing, quantize, reduce | 3.2 | latency-bound pieces of 0.1–1.3 ms |
| `gdn.record` | 13.5 | see below |
| `hyper.combine_mix` | 4.6 | projections at 74–83% of a plain read; the norm is now 3.6 µs per launch |
| `qsa.select` | 4.4 | FP8 projections near the floor; split attention 1.0 ms |
| unscoped: FP8 `lm_head` | 2.6 | 636 MB, at the floor |
| unscoped: Q4 draft head | 0.9 | |

- **Routed experts:** a trace-build counter (`NINFER_MOE_ROUTE_STATS`) measured 60.07 distinct
  experts per 8-row call, out of 80 assignments. That is 166 MB in 689 µs: 241 GB/s, 98% of the
  measured read, equal to the microbench.
- **`gdn.record`:**
  - the projections run at 96–97% of a plain read (medians);
  - the recurrent fold and record read and write each row's 3.15 MB of FP32 state per layer, a
    103 µs floor that the kernel already beats in 78.5 µs, presumably from L2 hits;
  - conv and gating take 0.54 ms.

What remains is small latency work, about 0.1–0.2 ms per round per item. Fewer bytes per token
(more accepted tokens per round, smaller formats) is the remaining large lever.

### Long prompts: operations workload, October 4

Method: `tools/gb10/long_context.sh` (tree `6cd02900`, `profiles/bench/gb10/long-context/`).
- **Prompts:** synthetic incident bundles with one planted root-cause chain.
- **Tasks:** a scripts task and a root-cause triage task, 1,536 output tokens, greedy, thinking
  off, two reps, each with a prefix-reusing follow-up turn.
- **Arms:** NInfer at its defaults (`--prefill-chunk 1024`, two lanes, KV pool equal to
  `--max-context`, 73,728 tokens), and DGPP `dd58d6d3`.

| | 15K | 30K | 60K |
|---|---:|---:|---:|
| NInfer TTFT, s (prefill tok/s) | 9.8 (1,590) | 18.8 (1,585) | 38.9 (1,550) |
| DGPP TTFT, s (prefill tok/s) | 7.9 (1,960) | 15.6 (1,915) | 32.3 (1,867) |
| NInfer decode tok/s, K=1 / K=3 | 48.7 / 60.5 | 48.8 / 62.3 | 48.8 / 61.0 |
| DGPP decode tok/s | 46.6 | 46.8 | 46.4 |

- **Prefill:** at these defaults NInfer prefills about 20% slower than DGPP, flat from 15K to
  60K, and below the 2.5k tok/s of the `ninfer_bench` rows. Those rows used
  `--prefill-chunk 8192` on a page-cached tiled corpus.
- **Decode:** does not depend on prompt length. K=3 leads DGPP by 25–35% at every size.
- **Acceptance:** 1.82 tokens per round at K=1 and 2.92 at K=3, the same as on short prompts.
- **Follow-up turns** start in 0.2–0.8 s. 22 of 24 recomputed only 43–55 prompt tokens. Two
  recomputed the previous answer as well (1,595 tokens, 1.3 s); that partial miss is not yet
  explained.
- **Planted facts:** every arm's answers named 3–4 of the 4.
- **Interference** (a 60K prompt arriving while a 15K request decodes):
  - DGPP stalled the running request for 32.9 s in total, with gaps up to 2.4 s.
  - NInfer showed no stall, but only because the 60K prompt was not admitted: two lanes share the
    73,728-token default pool, and the running request's reservation leaves no room for a 60K
    prompt (`no_feasible_plan`). So it waited 22–28 s for the running request to finish. With a
    pool that fits both, the two would share the GPU; that case is not yet measured.
  - At the defaults, a newcomer waits for the running request whatever its length.

**Prefill chunk sweep** (`profiles/bench/gb10/long-context-chunks/`; K=3, one rep,
`--kv-capacity auto`, which gives a 147,456-token pool so both requests fit):

| `--prefill-chunk` | 15K TTFT s (tok/s) | 60K TTFT s (tok/s) | interference: newcomer TTFT s | running request stalled s | largest gap s |
|---:|---:|---:|---:|---:|---:|
| 1024 (default) | 9.76 (1,592) | 39.04 (1,549) | 41.8 | 41.6 | 0.75 |
| 2048 | 8.18 (1,900) | 32.59 (1,856) | 34.0 | 33.8 | 1.19 |
| 4096 | 7.20 (2,159) | 28.34 (2,134) | 29.1 | 28.9 | 2.03 |
| 8192 | 6.85 (2,271) | 27.32 (2,214) | 27.7 | 27.5 | 3.81 |

- **The 1024-token default is the whole prefill gap to DGPP.** At 2048 NInfer is at parity, at
  4096 +10–14% and at 8192 +16–19%. Decode is unchanged.
- **Interference:** with room in the pool, the newcomer is admitted at once (about 120 ms queue)
  and its prefill takes the GPU.
  - The running request gets one decode round per prefill chunk. Its stall equals the newcomer's
    whole prefill at every chunk size, and its largest gap equals one chunk's prefill time.
  - Prefill is compute-bound, about 0.46–0.66 s per 1,024 tokens against a 23 ms weight read, so
    every decode round taken during it comes out of the newcomer's prefill time.
  - How to divide the GPU between a long prefill and a running request is a scheduling policy the
    Engine does not have yet.

### Changes and their measured effect

Each change kept outputs bitwise unless the row says otherwise. Gates: op oracle tests, the
real-artifact Engine test against per-recipe goldens, and for numerical changes the perplexity and
drift gate below.

| change | effect on GB10 |
|---|---|
| `yield` synchronize on integrated devices | decode +1.6–3.3% |
| capture-safe valid columns re-enable no-speculation graphs | MTP-off +2.5–2.7 ms per token |
| FP8 dense projections (7a, then HyperConnection up and shared expert) | MTP-off 20.4 → 32.2 tok/s, prefill 1.5k → 2.5k; numerical change, gate below |
| T=2..4 FP8 on Tensor Cores, not SIMT, so verify and draft share one compute profile | acceptance 63.4 → 66.4% |
| MTP layer in the main layers' formats (7d, `fp8mtp`) | tok/s up at every K, +5% at K=3 single stream; acceptance −0.15 / −0.22 / −0.59 points at K=1/2/3 |
| dense QSA when every visible key fits the 2051-key budget | C4 +5.2% |
| admission re-arm when a blocking StateImage fork settles | removes a multi-second admission hold on concurrent replays |
| PLE history read from the fork source during prefill | correctness: concurrency-dependent output after a context capture |
| deferred fold (K1), device-resident frame (K2), in-graph PLE stage (K3) | C4 host time 8.9 → 0.05 ms per round |
| fused GDN fold inside verify (K5) | C4 +2.1%; one 108 MiB state pass per row and round removed |
| overlapped PLE page-ins | C4 long generations +18%, 135 → 0.1 major faults per round |
| expert-grouped NVFP4 MoE decode from 8 rows | C4 +3.1% |
| MoE decode schedules from the bandwidth probe (BN64 BK512 S2 gate/up, BN64 BK128 S4 down) | routed experts 87 → 97% of the read, C4 +4.7%, C1 +1.7%; at T ≤ 4 the per-layer fixed cost rose 12–32 µs |
| PLE replay record with one weight pass per 8 columns | `ple.record` 2.31 → 0.33 ms per round |
| QSA split decode staging each KV tile once in 16-byte pieces | split kernel 2.89 → 1.01 ms per round at C4 K=1; −6.4% GPU work at K=3 |
| HyperConnection grouped RMSNorm with loads issued up front | 14.0 → 3.6 µs per launch, −0.89 ms per C4 round |
| QSA key, value and index-query projections in one launch | QSA BF16 0.70 → 0.59 ms per round |

### Quality gate for numerical changes

- **Perplexity and drift:** `ninfer-perplexity` over the 1,044,557-token corpus at 4K and 64K
  windows, plus `tools/bench/compare_token_drift.py` on the 64K token scores. Drift is the slope of
  the NLL difference over offsets.

  | artifact | 4K PPL | 64K PPL | drift slope |
  |---|---:|---:|---|
  | BF16-projection reference | 3.9731 | 3.7540 | — |
  | FP8 dense projections | 3.9999 | 3.7527 | −0.0007 nats per 10K tokens |
  | `fp8proj` (also HyperConnection up and shared expert) | 3.9982 | 3.7581 | −0.0008 nats per 10K tokens |

  `fp8mtp` has the same target weights as `fp8proj`, and its 4K PPL is identical. Both drift
  slopes are flat: there is no GDN-state accumulation.
- **Same-artifact run-to-run floor:** ΔPPL 1.3e-4.
- **Draft acceptance** on the 16-stream natural corpus (`tools/gb10/k_sweep.sh`). Greedy and fixed,
  it reproduces exactly across servers.
- **TEB hard mode, seed 42:** a behavioural smoke check, not a score gate. `fp8proj` scored
  87/88/91/91 against BF16's 87/87/87.

### Evaluated and not adopted

| approach | result |
|---|---|
| EXL3 3-bit trellis experts | about 10% fewer bytes than NVFP4 experts, but no FP4 tensor-core prefill, new GEMV and MoE kernels, and a lossy re-quantization source |
| lossless 12-bit BF16 (DGPP `bf12`) | under 1% of a step's bytes are still BF16 in `fp8mtp` |
| generic memory compression (`cuMemCreate`) | 1.00–1.01× on every real weight class; it pays only on zero-filled data |
| two MTP rounds in flight (K4) | about 2% slower; after K3 the serial loop exposes 0.05 ms of host time per round, so there was nothing to hide. Removed |
| grouped cold prefill | admission stagger is at most about 6% of a C4 wave on the parity protocol and does not arise under steady arrival |
| FP8 sliced-K row tiles, K-warp pairs split across CTAs | no gain on any shape; split pairs were the slowest variant everywhere |
| three-stage FP8 tiles for long-K shapes | hc.down +4 points of the read floor, but the attribution showed the MoE stage 0.7 ms per round slower on that tree, unexplained, against a gain of at most 0.06 ms. Reverted |
| MoE route kernel load hoisting | median unchanged at 8.67 µs per launch. Reverted |
| lower-bit KV (K4V2 and similar) | decode is flat from 1K to 128K (32.6 / 31.1 / 32.2 / 31.1 tok/s at K=0); only capacity would change |
| ncu-driven MoE tuning | replaced by the timing probe above |

### Open items

- **DGPP at draft depth 2–3:** untested, so the depth-1 comparison is the like-for-like claim.
- **Batch-invariant decode:** a product decision with a throughput cost.
- **Long-prompt follow-ups:**
  - the default prefill chunk: 4096 gains 38% at 60K with a 2 s worst gap, against 0.75 s at 1024;
  - a policy for sharing the GPU between a long prefill and a running decode;
  - the two follow-up turns that recomputed the previous answer.
- **MTP at 32K and 128K:** K=3 measured 37.7 tok/s at 32K and 60.8 at 128K against 56–57 at
  1K–8K. Each row continues different natural text, so do not cite 32K/128K MTP rates until that
  is separated from a context effect.
- **Not started, with estimates:**
  - BF16 GDN recurrent state (needs the drift gate): about 1.5–2% of decode;
  - NVFP4 for the remaining dense projections;
  - an int6 PLE table (memory, not speed);
  - constrained decoding for `tool_choice: required`: the measured format failure rate is 0/180
    without it; one semantic miss.

### Reproduce

`tools/gb10/`, each script with `ART=` set to the artifact:

| script | purpose |
|---|---|
| `step0_build_test.sh` | build and tests |
| `step2_baseline.sh` | bench rows, telemetry, effective bandwidth |
| `step7a_baseline.sh` | perplexity, acceptance, TEB |
| `k_sweep.sh` | acceptance and served rate per K |
| `block_i.sh` with `PHASES=I9` | the DGPP scoreboard |
| `round_attribution.sh` | per-stage GPU work at a fixed concurrency; `ROUTE_STATS=1` adds the distinct-expert tally |
| `kernel_shapes.py` | per-launch-shape medians from a trace |
| `moe_microbench.sh` | MoE bandwidth probe |
| `long_context.sh` | 15K–60K operations workload: TTFT, prefill, decode, follow-up reuse, prefill interference |

Reports land under `profiles/bench/gb10/`.
