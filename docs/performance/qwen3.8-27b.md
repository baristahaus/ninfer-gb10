# Qwen3.8-27B serving performance

[Performance index](../performance.md) · [Measurement and publication rules](methodology.md)

On this page: [run records](#scope-and-run-records), [context profile](#no-speculation-context-profile),
[single-request decode](#single-request-speculative-decode), [corpus makespan](#corpus-makespan),
[decode saturation](#decode-saturation), [completion outcomes](#completion-outcomes),
[reproduction](#reproduction-and-reports).

## Scope and run records

Measured on September 28–29, 2026 (Asia/Shanghai), using one RTX 5090, NVIDIA driver 617.14,
CUDA 13.4 compile/runtime/driver API, and NInfer revision
`7f6aafedb5f20200def820cfe51ab81c09c20eeb`. The serving binary was fixed for the entire campaign.
All runs use FP8 E4M3 row-256 KV, a 1,024-token prefill chunk, CUDA Graphs, disabled prefix reuse,
and the [common stochastic sampling profile](methodology.md#common-serving-profile).
MTP3 and DFlash2 K=7 use the optimized proposal head. Startup and warmup are outside measurement.

The artifacts are `out/qwen3_8_27b.ninfer` (`groupwise-int`) and
`out/qwen3_8_27b_nvfp4.ninfer` (`nvfp4`), including their selected speculative companion weights.
All 20 configurations and 820 formal requests completed without request, CUDA, or allocation errors.
The tables contain this campaign's measurements only.

| Run | Weights ID | Measurement | Context ceiling | C | Actual KV capacity (tokens, in C order) |
|---|---|---|---|---|---|
| G0 | `groupwise-int` | MTP0 | 262,144 | 1 | 262,144 |
| G3 | `groupwise-int` | MTP3 corpus | 131,072 | 1, 2, 4, 8 | 131,072, 262,144, 347,200, 318,592 |
| GD | `groupwise-int` | DFlash2 K=7 corpus | 131,072 | 1 | 131,072 |
| GS | `groupwise-int` | MTP3 saturation | 16,384 | 1, 2, 4, 8 | 16,384; 32,768; 65,536; 131,072 |
| N0 | `nvfp4` | MTP0 | 262,144 | 1 | 262,144 |
| N3 | `nvfp4` | MTP3 corpus | 131,072 | 1, 2, 4, 8 | 131,072, 262,144, 253,632, 225,024 |
| ND | `nvfp4` | DFlash2 K=7 corpus | 131,072 | 1 | 131,072 |
| NS | `nvfp4` | MTP3 saturation | 16,384 | 1, 2, 4, 8 | 16,384; 32,768; 65,536; 131,072 |

MTP0 reserves its KV capacity explicitly; corpus and saturation points use `--kv-capacity auto`.
The workspace arena is 243.3 MiB in every configuration. Resident weight arenas depend on the selected backend:

| Backend | groupwise-int weights (GiB) | nvfp4 weights (GiB) |
|---|---|---|
| No speculation | 15.920 | 18.976 |
| MTP3 | 16.672 | 19.729 |
| DFlash2 K=7 | 18.326 | 21.383 |

### Dense EXL3 route (October 2026)

Measured on October 28, 2026 (America/New_York), on the same RTX 5090, with the master
generation's serving binary and the `out/qwen3_8_27b_exl3.ninfer` dense artifact
(payload digest `eaa9d921a32e…388974d`). This campaign runs the fused TensorFold decode
kernel (`src/ops/linear/exl3/tensorfold/`, one rotated-activation kernel plus one fused
tensor-core matmul with in-kernel split-K per projection, replacing the two-pass SIMT decode
measured by the first version of this section). Runs use FP8 E4M3 row-256 KV, a 1,024-token
prefill chunk, CUDA Graphs, disabled prefix reuse, and the stochastic sampling profile. Twenty
formal requests completed without request, CUDA, or allocation errors. These rows come from
this campaign only and are not comparable to the September groupwise and NVFP4 rows beyond the
shared method.

#### MTP0 context-length profile

| Prompt tokens | Samples | Prefill phase (tok/s) | Server TTFT (ms) | Decode phase (tok/s) |
|---|---|---|---|---|
| 7,680 | 5 | 1,288.1 ± 2.0 | 5,964.8 ± 9.7 | 60.4 ± 0.2 |
| 64,512 | 5 | 1,281.8 ± 1.6 | 50,345.6 ± 62.6 | 56.4 ± 0.0 |
| 130,048 | 5 | 1,213.4 ± 0.2 | 107,209.7 ± 20.9 | 52.4 ± 0.1 |
| 260,096 | 5 | 1,095.4 ± 0.1 | 237,506.1 ± 22.3 | 46.1 ± 0.0 |

The fused route decodes at 60.4 down to 46.1 tok/s from 8K to 256K — 6.3× to 5.0× the
two-pass route's 9.6–9.2 tok/s on the same fixtures. The prefill phase is unchanged within
run-to-run variation (its composed route is untouched). The decode rate declines with
context because the growing KV stream crosses over the fixed projection cost; the
attribution below quantifies the split.

#### Decode saturation

Measured with the `--ignore-eos` server knob (a benchmark knob that suppresses
end-of-sequence stop tokens on the instance, so every stream decodes to the full
8,192-token budget and the suite's completion contract holds at every C):

| C | Steady (s) | Steady decode (tok/s) | Mean decode batch | Wave completion |
|---|---|---|---|---|
| 1 | 135.0 | 60.30 | 1.0 | output limit, 8,141 tokens/stream |
| 2 | 138.0 | 117.93 | 2.0 | output limit, 8,137 tokens/stream |
| 4 | 144.0 | 225.39 | 4.0 | output limit, 8,114 tokens/stream |
| 8 | 156.0 | 418.82 | 8.0 | output limit, 8,167 tokens/stream |

Batched decode scales near-linearly: 1.96× at C=2, 3.74× at C=4, 6.95× at C=8,
with the per-row rate at C=8 at 52.4 tok/s (87% of the solo rate). Every stream in
every wave reaches the output limit; the batched EXL3 decode scaling that the
two-pass campaign could not measure is now recorded. Without the knob, the same
waves are refused by the completion contract: under the stochastic profile at
least one stream samples a stop token inside the 8,192-token window, the surviving
stream then runs solo, and the wave measures no batched rate.

MTP3 and DFlash2 modes are not part of this route's record: the GDN replay-record path
refuses speculative decoding over EXL3 weights at request time, so the speculative suites
cannot run.

#### Where the decode time goes

Two measured levels, no kernel probe in between. The projection level is an operator-level
microbench of the dispatched unit — one rotated-activation kernel, one fused decode matmul
with in-kernel split-K, one output transpose — timed at every registered decode shape with
the artifact's real geometry (all 256 EXL3 objects are k2=4, codebook mcg; the per-round
dispatch census below sums to exactly 512 calls, eight per layer across 64 layers):

| Shape (T=1, k2=4) | Calls per round | Measured (µs/call) | Effective weight BW (GB/s) |
|---|---|---|---|
| 1024×5120 | 32 | 10.2 | 130 |
| 2048×5120 | 96 | 10.2 | 259 |
| 6144×5120 | 128 | 18.5 | 426 |
| 5120×6144 | 64 | 15.8 | 498 |
| 17408×5120 | 128 | 36.8 | 606 |
| 5120×17408 | 64 | 31.1 | 719 |

Weighted by the census, the fused projections cost 11.39 ms per round. This is an eager
upper bound — each call carries roughly 3–6 µs of launch/dispatch overhead that the
serving path's graph replay removes — so the graph-replayed share is somewhat lower. The
end-to-end level is the battery: a 16.6 ms mean round at 8K (60.4 tok/s), i.e. a ~5.2 ms
residual for everything else (attention compute, KV reads, LM head, per-round schedule).
The fp8 KV read floor gives the context dependence: 0.125 MiB per token (64 layers ×
1,024 × K+V) means ~1.0 GB at 8K (0.6 ms), ~8.5 GB at 64K (4.8 ms), ~17.1 GB at 128K
(9.6 ms), and ~34.2 GB at 256K (19.1 ms at the 1.79 TB/s peak). The measured rounds
(16.6 / 17.7 / 19.1 / 21.7 ms) fit that shape: through 128K the fixed projection cost is
the critical path and the KV stream grows underneath it; by 256K the KV floor alone
(19.1 ms) exceeds the projection cost, the round becomes KV-bound, and the projections
overlap under the KV stream — which is also why the round grows by only 5.1 ms over a
32× larger context. The 16.5 ms steady round at 8K therefore splits into roughly 11.4 ms
of fused projections (the upper bound including launch overhead), under 1 ms of KV and
attention floor, and ~4 ms of fixed per-round work (LM head, norms, GDN recurrence,
sampling, schedule).

Two ceilings follow. At short contexts the route is projection-bound: the effective
weight bandwidth of the small slices (130–260 GB/s against the 719 GB/s of the largest)
says the per-call launch floor, not the kernel, is what costs on the 1,024- and 2,048-row
slices — 128 of the 512 calls are the 17408×5120 pair at 606 GB/s, so batching or fusing
the small slices is the candidate there. At 256K the route is KV-bandwidth-bound and
projection work is irrelevant; that lane is the tiering machinery's, not the kernel's.

## No-speculation context profile

G0/N0 use the [single-request method](methodology.md#single-request-phases): five fixed seeds per fixture, thinking disabled, and a 128-token output budget.

### groupwise-int

| Prompt tokens | Samples | Prefill phase (tok/s) | Server TTFT (ms) | Decode phase (tok/s) |
|---|---|---|---|---|
| 7,680 | 5 | 3,331.9 ± 7.6 | 2,308.5 ± 5.2 | 84.1 ± 0.3 |
| 64,512 | 5 | 2,964.4 ± 3.6 | 21,798.1 ± 28.1 | 76.8 ± 0.1 |
| 130,048 | 5 | 2,623.8 ± 0.6 | 49,643.7 ± 10.3 | 69.7 ± 0.2 |
| 260,096 | 5 | 2,139.4 ± 0.4 | 121,723.1 ± 19.7 | 59.2 ± 0.1 |

### nvfp4

| Prompt tokens | Samples | Prefill phase (tok/s) | Server TTFT (ms) | Decode phase (tok/s) |
|---|---|---|---|---|
| 7,680 | 5 | 12,819.1 ± 16.8 | 602.7 ± 1.1 | 74.1 ± 0.3 |
| 64,512 | 5 | 8,658.2 ± 44.8 | 7,487.4 ± 37.7 | 68.2 ± 0.2 |
| 130,048 | 5 | 6,198.5 ± 19.6 | 21,055.7 ± 67.2 | 62.5 ± 0.3 |
| 260,096 | 5 | 4,016.4 ± 10.4 | 64,910.0 ± 171.8 | 53.4 ± 0.6 |

## Single-request speculative decode

These phase statistics come directly from the C=1 corpus points, G3/N3 and GD/ND.
Each reasoning fixture has five samples and a 65,536-token output budget; each other category
pools three fixtures × five seeds with a 4,096-token budget. Values are arithmetic mean ± sample
standard deviation of per-request rates and ratios. Category deviation also includes differences
between fixtures. Actual output lengths vary; [completion outcomes](#completion-outcomes) apply
to all the means below. In particular, all C=1 AIME 15 samples reach the output limit.

### groupwise-int

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode phase (tok/s) | MTP3 acceptance | MTP3 tokens/round |
|---|---|---|---|---|---|
| `long_decode_aime26_01` | 5 | 1,440.8 ± 290.3 | 196.1 ± 4.7 | 77.9% ± 2.6% | 3.34 ± 0.08 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 149.2 ± 1.3 | 54.9% ± 0.7% | 2.65 ± 0.02 |
| `long_decode_aime26_30` | 5 | 49,543.0 ± 9,464.5 | 155.1 ± 3.2 | 57.3% ± 2.0% | 2.72 ± 0.06 |

#### MTP3 cross-scenario decode

| Category | Samples | Decode phase (tok/s) | MTP3 acceptance | MTP3 tokens/round |
|---|---|---|---|---|
| Code | 15 | 193.5 ± 5.9 | 76.4% ± 3.3% | 3.29 ± 0.10 |
| Story | 15 | 123.4 ± 10.3 | 36.6% ± 5.9% | 2.10 ± 0.18 |
| Translation | 15 | 190.3 ± 10.6 | 74.6% ± 6.1% | 3.24 ± 0.18 |
| Structured | 15 | 214.7 ± 13.7 | 88.4% ± 7.8% | 3.65 ± 0.23 |

#### DFlash2 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode phase (tok/s) | DFlash2 acceptance | DFlash2 tokens/round |
|---|---|---|---|---|---|
| `long_decode_aime26_01` | 5 | 1,572.2 ± 585.1 | 305.7 ± 17.7 | 65.9% ± 4.7% | 5.61 ± 0.33 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 177.3 ± 4.8 | 33.9% ± 1.3% | 3.37 ± 0.09 |
| `long_decode_aime26_30` | 5 | 41,378.4 ± 7,618.2 | 189.0 ± 8.2 | 36.4% ± 2.3% | 3.55 ± 0.16 |

#### DFlash2 cross-scenario decode

| Category | Samples | Decode phase (tok/s) | DFlash2 acceptance | DFlash2 tokens/round |
|---|---|---|---|---|
| Code | 15 | 258.6 ± 24.6 | 53.6% ± 6.5% | 4.75 ± 0.45 |
| Story | 15 | 115.0 ± 28.7 | 15.9% ± 7.5% | 2.11 ± 0.53 |
| Translation | 15 | 241.6 ± 49.1 | 49.1% ± 12.9% | 4.44 ± 0.91 |
| Structured | 15 | 348.9 ± 51.4 | 77.3% ± 13.6% | 6.41 ± 0.95 |

### nvfp4

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode phase (tok/s) | MTP3 acceptance | MTP3 tokens/round |
|---|---|---|---|---|---|
| `long_decode_aime26_01` | 5 | 1,559.4 ± 727.2 | 207.4 ± 3.6 | 77.2% ± 1.8% | 3.32 ± 0.05 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 161.7 ± 3.8 | 57.2% ± 2.1% | 2.72 ± 0.06 |
| `long_decode_aime26_30` | 5 | 37,978.0 ± 7,474.4 | 170.0 ± 2.3 | 59.9% ± 1.5% | 2.80 ± 0.05 |

#### MTP3 cross-scenario decode

| Category | Samples | Decode phase (tok/s) | MTP3 acceptance | MTP3 tokens/round |
|---|---|---|---|---|
| Code | 15 | 205.3 ± 9.5 | 76.8% ± 5.1% | 3.30 ± 0.15 |
| Story | 15 | 132.6 ± 13.1 | 37.6% ± 7.1% | 2.13 ± 0.21 |
| Translation | 15 | 202.9 ± 12.3 | 75.2% ± 6.6% | 3.26 ± 0.20 |
| Structured | 15 | 231.7 ± 10.6 | 90.6% ± 5.6% | 3.72 ± 0.17 |

#### DFlash2 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode phase (tok/s) | DFlash2 acceptance | DFlash2 tokens/round |
|---|---|---|---|---|---|
| `long_decode_aime26_01` | 5 | 1,921.6 ± 947.3 | 323.6 ± 28.8 | 62.8% ± 6.9% | 5.40 ± 0.48 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 200.4 ± 13.9 | 35.3% ± 3.4% | 3.47 ± 0.24 |
| `long_decode_aime26_30` | 5 | 38,535.8 ± 3,520.6 | 214.9 ± 3.4 | 38.1% ± 0.8% | 3.66 ± 0.05 |

#### DFlash2 cross-scenario decode

| Category | Samples | Decode phase (tok/s) | DFlash2 acceptance | DFlash2 tokens/round |
|---|---|---|---|---|
| Code | 15 | 285.7 ± 25.2 | 53.9% ± 6.0% | 4.77 ± 0.42 |
| Story | 15 | 129.2 ± 33.8 | 16.5% ± 8.1% | 2.15 ± 0.56 |
| Translation | 15 | 264.0 ± 64.1 | 48.6% ± 15.3% | 4.40 ± 1.07 |
| Structured | 15 | 377.3 ± 57.7 | 75.7% ± 13.8% | 6.30 ± 0.96 |

## Constrained decode

Grammar-constrained generation (SP-2: JSON-object masking through the MTP round), measured
2026-10-13 on the RTX 5090 (CUDA 13.4.92) with the v3 NVFP4 27B artifact, greedy sampling,
fp8 KV, prompt 2048 + 512 generated tokens, three measured repetitions after one warmup. The
pre-port binary (`5090-test@3b885469`) and the SP-2 binary ran identical commands:

```bash
./build/bench/ninfer_bench --weights <27b-v3-nvfp4>.ninfer --corpus bench/fixtures/bench_corpus.ids \
  -pg 2048,512 -r 3 --kv-dtype fp8 --spec mtp --draft-tokens 3 --lm-head-draft [--json-object]
```

| Configuration | decode out t/s | MTP acceptance | Spec rounds |
|---|---:|---:|---:|
| Unconstrained, pre-port binary | 228.89 ± 0.22 | 92.38% | 408 |
| Unconstrained, SP-2 binary | 228.86 ± 0.03 | 92.38% | 408 |
| Unconstrained, no speculation | 75.62 ± 0.02 | — | — |
| JSON-object, no speculation | 75.50 ± 0.02 | — | — |
| JSON-object, MTP3 | 120.93 ± 0.07 | 33.16% | 771 |

Readings:

- The port's structural costs measure zero on this workload: unconstrained MTP3 decode is
  unchanged (228.86 vs 228.89 tok/s; acceptance and round counts bit-identical between the two
  binaries), and grammar masking without speculation costs 0.16% (75.50 vs 75.62 tok/s).
- The constrained cost is the speculation interaction. Under the JSON-object grammar the MTP
  drafts are legal under the mask but diverge from the target's grammar-concentrated
distribution, so acceptance falls to 33.16% and the round count rises from 408 to 771 for the
  same 512 tokens: constrained MTP3 decode runs at 53% of the unconstrained rate. Requests that
  need constrained output at full throughput can disable speculation for that request class.

## Constrained decode

Grammar-constrained generation (SP-2: JSON-object masking through the MTP round), measured
2026-10-13 on the RTX 5090 (CUDA 13.4.92) with the v3 NVFP4 27B artifact, greedy sampling,
fp8 KV, prompt 2048 + 512 generated tokens, three measured repetitions after one warmup. The
pre-port binary (`5090-test@3b885469`) and the SP-2 binary ran identical commands:

```bash
./build/bench/ninfer_bench --weights <27b-v3-nvfp4>.ninfer --corpus bench/fixtures/bench_corpus.ids \
  -pg 2048,512 -r 3 --kv-dtype fp8 --spec mtp --draft-tokens 3 --lm-head-draft [--json-object]
```

| Configuration | decode out t/s | MTP acceptance | Spec rounds |
|---|---:|---:|---:|
| Unconstrained, pre-port binary | 228.89 ± 0.22 | 92.38% | 408 |
| Unconstrained, SP-2 binary | 228.86 ± 0.03 | 92.38% | 408 |
| Unconstrained, no speculation | 75.62 ± 0.02 | — | — |
| JSON-object, no speculation | 75.50 ± 0.02 | — | — |
| JSON-object, MTP3 | 120.93 ± 0.07 | 33.16% | 771 |

Readings:

- The port's structural costs measure zero on this workload: unconstrained MTP3 decode is
  unchanged (228.86 vs 228.89 tok/s; acceptance and round counts bit-identical between the two
  binaries), and grammar masking without speculation costs 0.16% (75.50 vs 75.62 tok/s).
- The constrained cost is the speculation interaction. Under the JSON-object grammar the MTP
  drafts are legal under the mask but diverge from the target's grammar-concentrated
distribution, so acceptance falls to 33.16% and the round count rises from 408 to 771 for the
  same 512 tokens: constrained MTP3 decode runs at 53% of the unconstrained rate. Requests that
  need constrained output at full throughput can disable speculation for that request class.

## Corpus makespan

Each point is one uninterrupted run of the same 75-request corpus, using shuffle seed `20260811`
and ordered HTTP sends. Makespan includes prefill, decode, admission waits, transitions, and drain.
Corpus rates divide token totals by full makespan; acceptance is the ratio of summed accepted and
drafted tokens. Average batch covers the entire run. Stochastic continuations differ between points.

### MTP3 groupwise-int

| C | Requests | Computed prefill tokens | Decode tokens | Makespan (s) | Requests/s | Corpus prefill (tok/s) | Corpus decode (tok/s) | Avg batch | MTP acceptance |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 75 | 15,460 | 768,038 | 4,918.67 | 0.0152 | 3.1 | 156.1 | 1.00 | 58.0% |
| 2 | 75 | 15,460 | 728,617 | 2,676.67 | 0.0280 | 5.8 | 272.2 | 1.92 | 58.4% |
| 4 | 75 | 15,460 | 779,134 | 2,023.84 | 0.0371 | 7.6 | 385.0 | 3.47 | 59.0% |
| 8 | 75 | 15,460 | 755,705 | 1,719.23 | 0.0436 | 9.0 | 439.6 | 4.31 | 60.0% |

### MTP3 nvfp4

| C | Requests | Computed prefill tokens | Decode tokens | Makespan (s) | Requests/s | Corpus prefill (tok/s) | Corpus decode (tok/s) | Avg batch | MTP acceptance |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 75 | 15,460 | 693,701 | 4,115.22 | 0.0182 | 3.8 | 168.6 | 1.00 | 59.6% |
| 2 | 75 | 15,460 | 709,989 | 2,394.23 | 0.0313 | 6.5 | 296.5 | 1.90 | 58.9% |
| 4 | 75 | 15,460 | 722,202 | 1,640.17 | 0.0457 | 9.4 | 440.3 | 3.13 | 58.5% |
| 8 | 75 | 15,460 | 708,589 | 1,443.37 | 0.0520 | 10.7 | 490.9 | 3.52 | 59.5% |

At C=8, the full-corpus average batch is 4.31 for groupwise-int and 3.52 for NVFP4.
Their shared KV capacities are 318,592 and 225,024 tokens; the sampled maximum waiting counts
are four and five requests. NVFP4 C=4 also records one waiting request. These are measured admission
limits, and the full makespans retain their effect. No spill, owner degradation/eviction, or
search-exhaustion event was recorded.

### DFlash2 K=7

| Weights ID | Requests | Completion tokens | Decode tokens | Makespan (s) | Requests/s | Corpus decode (tok/s) | DFlash2 acceptance |
|---|---|---|---|---|---|---|---|
| `groupwise-int` | 75 | 725,080 | 725,005 | 3,896.89 | 0.0192 | 186.0 | 35.8% |
| `nvfp4` | 75 | 697,637 | 697,562 | 3,354.86 | 0.0224 | 207.9 | 36.6% |

Each DFlash2 point computes 15,460 prefill tokens, has average decode batch 1.00, and uses 131,072 KV tokens.

## Decode saturation

GS/NS use the [sustained-wave method](methodology.md#decode-saturation): one wave at each C,
`long_decode_aime26_15`, a 335-token prompt, thinking enabled, and an 8,192-token output budget
per request. All 30 requests reach that budget. Steady throughput selects complete full-batch
intervals; acceptance covers the entire wave, and wave makespan includes ramp-up and drain.

### groupwise-int

| C | Steady (s) | Avg batch | Steady decode (tok/s) | MTP acceptance (wave) | Wave makespan (s) |
|---|---|---|---|---|---|
| 1 | 58.00 | 1.00 | 136.5 | 44.4% | 59.89 |
| 2 | 63.00 | 2.00 | 253.3 | 45.2% | 65.25 |
| 4 | 80.00 | 4.00 | 398.1 | 46.1% | 83.32 |
| 8 | 108.00 | 8.00 | 582.4 | 46.4% | 115.39 |

### nvfp4

| C | Steady (s) | Avg batch | Steady decode (tok/s) | MTP acceptance (wave) | Wave makespan (s) |
|---|---|---|---|---|---|
| 1 | 54.00 | 1.00 | 147.7 | 46.2% | 55.34 |
| 2 | 54.00 | 2.00 | 291.0 | 48.7% | 56.52 |
| 4 | 60.00 | 4.00 | 522.2 | 45.8% | 63.11 |
| 8 | 69.00 | 8.00 | 922.4 | 46.1% | 72.00 |

## Completion outcomes

Token accounting and termination reasons were checked for every formal request. All 40 MTP0
requests terminate at a stop token. The table below gives stop-token / output-limit counts for
the C=1 speculative corpus; every sample remains in the published statistics.

| Workload | Samples per profile | Groupwise MTP3 | NVFP4 MTP3 | Groupwise DFlash2 | NVFP4 DFlash2 |
|---|---|---|---|---|---|
| `long_decode_aime26_01` | 5 | 5 / 0 | 5 / 0 | 5 / 0 | 5 / 0 |
| `long_decode_aime26_15` | 5 | 0 / 5 | 0 / 5 | 0 / 5 | 0 / 5 |
| `long_decode_aime26_30` | 5 | 5 / 0 | 5 / 0 | 5 / 0 | 5 / 0 |
| Code | 15 | 0 / 15 | 5 / 10 | 0 / 15 | 5 / 10 |
| Story | 15 | 10 / 5 | 10 / 5 | 9 / 6 | 11 / 4 |
| Translation | 15 | 15 / 0 | 15 / 0 | 15 / 0 | 15 / 0 |
| Structured | 15 | 1 / 14 | 0 / 15 | 3 / 12 | 1 / 14 |
| **Total** | **75** | 36 / 39 | 40 / 35 | 37 / 38 | 42 / 33 |

Across MTP3 C=1/2/4/8, groupwise-int stop/limit counts are 36/39, 35/40, 36/39, and 37/38;
NVFP4 counts are 40/35, 41/34, 40/35, and 42/33. Output-limit samples measure generation through
the configured budget and do not establish completed reasoning or task success.

Screening the complete C=1 responses for obvious large verbatim repetitions found no candidates.
Concurrent C>1 runs retain token and termination records, but no complete-response repetition audit
was performed. Answer accuracy, prompt compliance, and perplexity were not measured in this campaign.

## Reproduction and reports

Reports are local under `profiles/bench/qwen3_8_27b_fp8kv_20260928/`.
`campaign.json` records the build, artifact identities, commands, and execution status.
`run.jsonl`, `points/*.json`, `server/*.jsonl`, and the existing summaries retain the measured evidence;
C=1 corpus responses and phase summaries are in `corpus/<point>/`.

| Runs | Report directories relative to campaign root |
|---|---|
| G0 / N0 | `groupwise-int/mtp0/`, `nvfp4/mtp0/` |
| G3 / N3 | `groupwise-int/mtp3-corpus/c{1,2,4,8}/`, `nvfp4/mtp3-corpus/c{1,2,4,8}/` |
| GD / ND | `groupwise-int/dflash2-corpus/`, `nvfp4/dflash2-corpus/` |
| GS / NS | `groupwise-int/mtp3-saturation/c{1,2,4,8}/`, `nvfp4/mtp3-saturation/c{1,2,4,8}/` |

Build `ninfer-serve`, then run from the repository root with Python 3.11 and a fresh output directory:

```bash
export NINFER_BENCH_PYTHON=/home/neroued/miniconda3/envs/py311/bin/python
export NINFER_PERF_OUTPUT=profiles/bench/qwen3_8_27b_fp8kv_rerun

for perf_profile in nvfp4 groupwise-int; do
  case "$perf_profile" in
    nvfp4) perf_artifact=out/qwen3_8_27b_nvfp4.ninfer ;;
    groupwise-int) perf_artifact=out/qwen3_8_27b.ninfer ;;
  esac
  perf_common=(--serve build/apps/ninfer-serve
    --artifact "qwen3_8_27b=$perf_artifact"
    --kv-dtype fp8 --sampling stochastic --port 18080)

  "$NINFER_BENCH_PYTHON" tools/bench/run_serve_corpus.py "${perf_common[@]}" \
    --mode mtp0 --output "$NINFER_PERF_OUTPUT/$perf_profile/mtp0"

  for perf_c in 1 2 4 8; do
    "$NINFER_BENCH_PYTHON" tools/bench/run_serve_concurrency.py "${perf_common[@]}" \
      --mode mtp3 --suite decode-saturation --concurrency "$perf_c" \
      --decode-tokens 8192 --max-context 16384 --kv-capacity auto --prefill-chunk 1024 \
      --output "$NINFER_PERF_OUTPUT/$perf_profile/mtp3-saturation/c$perf_c"
  done

  for perf_c in 1 2 4 8; do
    "$NINFER_BENCH_PYTHON" tools/bench/run_serve_concurrency.py "${perf_common[@]}" \
      --mode mtp3 --suite corpus-makespan --concurrency "$perf_c" \
      --max-context 131072 --kv-capacity auto --prefill-chunk 1024 \
      --output "$NINFER_PERF_OUTPUT/$perf_profile/mtp3-corpus/c$perf_c"
  done

  "$NINFER_BENCH_PYTHON" tools/bench/run_serve_concurrency.py "${perf_common[@]}" \
    --mode dflash2_7 --suite corpus-makespan --concurrency 1 \
    --max-context 131072 --kv-capacity auto --prefill-chunk 1024 \
    --output "$NINFER_PERF_OUTPUT/$perf_profile/dflash2-corpus"
done
```
