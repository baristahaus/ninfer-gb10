# Qwen3.8 Flash-Next 125B-A6B performance

## Swift 1.5 cold PLE prefill — 2026-10-01

The Swift NVFP4 variant retains a 102.4 GB BF16 PLE table. On the local Btrfs volume with
zstd compression, serial cold gathering caused a fresh 5,869-token HTTP prompt to take
17.242 seconds at 340 tokens/s, with 72,101 major page faults and 7,334 MiB of disk reads.
The gathered values occupy only about 29 MiB; compressed filesystem extents amplified
random row reads, and serial gathering exposed the individual page-fault latency.

The package now selects random-access mappings, and each Program owns sixteen prefill
gather workers. The converter disables Btrfs compression on PLE-bearing output shards;
the existing Swift table ranges were uncompressed in place without changing artifact bytes.
The same prompt after this fix took 1.180 seconds at 4,975 tokens/s, with 90,142 major page
faults and 352 MiB of disk reads. Cold prefill time improved by 14.6x. A warm run with a
different leading prompt and zero prefix-cache hits took 0.714 seconds for 5,873 tokens,
or 8,220 tokens/s. These are prefill measurements, not a decode-throughput comparison.

Measurements use RTX PRO 6000 Blackwell, CUDA 13.4.92, the Swift 1.5 NVFP4 v3 artifact,
BF16 KV, MTP3, CUDA Graphs, two active lanes, 1,024-token prefill chunks, greedy sampling,
thinking disabled and one generated token. The probe selects 5,000 words using Python
`random.Random(784123)` from the ordered unique lowercase `[a-z]{4,14}` words in
`docs/maintainer/engine-architecture.md`, joins them with spaces and appends `\nReply OK.`.
The fixed cold run evicts only PLE file-cache ranges before service startup. The warm run
prepends `Second warm test. ` so it repeats prefill rather than restoring a cached prefix.

An isolated cold gather of the first 4,096 corpus token IDs took 4.486 seconds with the
former mapping and serial loop, 4.062 seconds with random mapping alone, 1.021 seconds
with eight workers, and 0.478 seconds with sixteen workers on compressed storage. All
gathered bytes match exactly. The synthetic gather test additionally protects token/head
ordering across uneven worker partitions for both FP8 and BF16 tables.

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

### Follow-up diagnosis — 2026-09-26

Repeating the same HTTP matrix with the unchanged production binary gives the following
descriptive comparison against the stored v2 measurements, not a fresh controlled v2/v3 A/B:

| Workload | Current mean completion | Change from stored v2 |
|---|---:|---:|
| One 8K request | 3.532 s | +1.02% |
| Two 8K requests | 5.436 s | +4.26% |
| One 64K request | 9.525 s | -0.12% |
| Two 64K requests | 17.262 s | -0.07% |
| Mixed 8K/64K requests | 11.471 s | -0.77% |

The first paired 8K wave took 5.733 s; the other two took 5.288 s each. A separate short-workload
repeat after compilation finished averaged 3.598 s single and 5.396 s paired, respectively
+2.90% and +3.49% against stored v2. The first matrix overlapped the tracing build's compilation.
The single-request outputs and token counts match the old fixture exactly. The earlier 64K
prefill slowdown does not reproduce: current single-request TTFT is 6.351 s versus 6.453 s for v2
and 6.568 s immediately after migration. Timing variation within the same current binary is
comparable to the originally reported migration difference.

A separate Release tracing build with CUDA 13.4.92 and Nsight Systems 2026.3.1 measured the
public Engine on 65536 corpus tokens plus 32 generated tokens, BF16 KV, MTP3, optimized proposal,
8192-token chunks, one warmup and one measured repetition. Its 6784 ms measured interval contains
6724 ms of GPU busy time (99.1%); prefill GPU work is 6599 ms. Warm PLE gathering takes 23.3 ms
and hashing 3.9 ms. CPU scopes may overlap GPU work. These costs cannot explain the earlier
115 ms single-64K TTFT increase by themselves; current prefill is predominantly GPU execution.
Profiled throughput is not a speed comparison with the HTTP runs or the old build.

The v3 segmented PLE reader adds a lookup per row, but this trace does not establish its
incremental cost against v2. GPU operating conditions, compiler effects and a small kernel
regression remain unseparated: the original runs have no clock telemetry, and the saved v2
binary/artifact were removed during requested cleanup. A uniform 2% migration regression and
a PLE/scheduling explanation are therefore not established. Local measurements and the selected
trace report are under `profiles/bench/flash_next_v3_diagnosis/`.

The follow-up also repairs benchmark tracing: `NINFER_PERFORMANCE_TRACE` must be defined on
`ninfer_bench` itself so its measured-region marker is emitted alongside the engine's work scopes.
The corrected capture contains one measured marker and is accepted by `flash_next_performance.py`.


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
