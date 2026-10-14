# Fork strategy: tiered serving, one core, and the Qwen 4 runway (2026-10-12)

Status: **ratified direction, dated plan.** This document records the product and architecture
strategy agreed with the owner on 2026-10-12. It authorizes no implementation by itself; every
work package below requires its own task-level authorization, and the contract changes it names
(the capability reframing, multi-GPU, multi-node) revise `AGENTS.md` at execution time, not now.
It supersedes no active reference. When the strategy is executed or abandoned, this plan is
removed or folded into the active authorities per the document lifecycle.

---

## 1. Where we stand (the evidence base)

- **The fork is independent.** 404 fork commits vs 60 upstream since the merge-base; whole
  subsystems exist only on one side (fork: `pressure.cpp`/`materialization.cpp`, the host-restore
  and shared-prefix machinery; upstream: constrained decoding, tuple schemas). Upstream accepts
  only RTX 5090 implementation PRs, so the fork's direction never lands there. Staying in step is
  pure merge cost. The relationship continues as: bounded ports of upstream features, 5090-relevant
  correctness fixes offered upstream as goodwill, and nothing else.
- **The baseline is validated.** The Qwen 3.8 27B serves fully hot on the RTX 5090 (vision + MTP3
  + preserve-thinking, fp8 KV, 240k context, concurrency 6): 142-test battery green, both 5090
  bugs closed, serving smokes green, production dogfooding live, hardware canaries deterministic
  after the memory fix (DDR5-3600 JEDEC, 900 GiB discriminator traffic, zero flips).
- **The fork outperforms upstream on its own axes.** The 2026-10-12 A/B (`tool-eval-bench --seed
  42 --hardmode --trials 2`, identical artifact/flags/GPU): fork 88 vs upstream 90 on quality —
  a net of 14 bidirectional marginal flips from the two builds' different reuse paths (8 favoring
  upstream, 6 the fork; no systematic category gap; upstream's own trial-to-trial variance was
  ±1 score point) — while the fork was **14% faster end-to-end** (735 s vs 858 s of scenario
  time; its shared-prefix reuse eliminated 149 re-prefills) and reproduced **exactly** across
  trials (stddev 0.0 on all 97 scenarios) where upstream wobbled ±0.7.
- **The engine owns a validated hot/cold spine for runtime state.** Device state slots + host
  state slots + host KV spill + the pressure planner + demote/restore transactions: the
  frontier-exact reuse, the C+H capacity model, and the Host-only restore path all passed the
  battery and the A/B this week. This machinery is the foundation the weight tiering below
  extends; it is the fork's hardest-won semantic asset and stays sovereign (section 12).

## 2. The product decision: two hardware lanes, two capability profiles

**Lane A — GB10 (121 GiB unified memory): native, mostly in memory.**
The 125B Flash-Next NVFP4 artifact is 125.5 GiB on disk; its resident set is the ~78 GiB of
weights plus KV/state, because the 320M×160 PLE embedding (51.2 GB FP8) is already file-mapped
and demand-paged — the operating-system cache owns its residency, and prompt preparation
gathers 16 rows per token. That is the disk tier working in production today. GB10 lane work is
therefore validation and tuning, not new architecture: serve the 125B natively on GB10, measure,
and treat the PLE pattern (file-backed, demand-paged, gathered reads) as the precedent for any
future cold weight class.

**Lane B — RTX 5090-class (32 GB discrete): GPU + RAM + disk tiering for the MoE class.**
The 125B's resident weights (~78 GiB) are 2.4× the 5090's VRAM. The lever is MoE sparsity: 512
routed experts per layer, **10 selected per token** (~2% of each layer's bank), plus a
sigmoid-gated shared expert that stays hot. Tiering the routed banks (hot set on device, warm
set in pinned host RAM, cold set in the artifact file) brings the 125B-class onto 5090-class
hardware; Strata proved the approach at 62–94 tok/s for this exact model family on a 12 GB
gaming card.

**Two capability profiles, three model classes.**

| Class | Profile | Members |
|---|---|---|
| 27B dense | **fully hot** (the baseline) | Qwen 3.5 / 3.6 / 3.8 27B only |
| MoE | **tiered** (aligned to the layered approach) | Qwen 3.6 35B-A3B (fits today), Qwen 3.8 Flash-Next 125B-A6B, future MoEs |
| Qwen 4.0 / 4.x | whichever profile its shapes require | not yet specified; no speculative code (section 13) |

The profiles are engine capabilities, not per-family code: a model selects the fully-hot or the
tiered profile by its class, and the profile machinery is shared across families.

## 3. The hard constraint: the 27B baseline is untouchable

No tiering work may degrade the 27B's in-memory capability on the 5090: weights, KV, state
slots, concurrency, CUDA Graph decode, or prefix reuse. The 27B is the validation baseline for
every package below. Concretely: the tiering machinery must be **dormant unless the loaded model
class requires it** (a fully-hot model allocates no expert cache, streams nothing, and keeps
its today's memory plan unchanged), and the 27B battery + serving smokes must pass unchanged at
every step of the convergence and tiering work. Where a code path is shared between profiles,
the 27B's behavior is the reference the tests pin.

## 4. The tiering architecture

Built on the existing spine, extended from runtime state to weights. The MoE **expert slice** is
the tiering unit (the routed banks dominate the 125B's weight footprint; per-layer banks are
already separate tensors in the v3 container).

**Placement model.**
- *Hot:* device-resident expert slices in an adaptive cache sized by free VRAM after the
  baseline reservation (for the 125B on 5090: dense weights + KV/state first, experts fill the
  rest). Seed the hot set from a ranked profile, then adapt at runtime from the router
  statistics the engine already observes.
- *Warm:* the full routed banks in pinned host RAM (the 125B banks fit 128–256 GB workstations).
- *Cold:* the artifact file itself, read through the OS file cache on demand — the PLE-table
  pattern generalized.

**Execution paths (decided empirically, in this order).**
1. *5090 phase 1 — GPU-only execution with PCIe streaming:* expert slices stream (with
   layer-ahead prefetch, Strata-style) from the warm set onto the device, are computed by the
   existing NVFP4 MoE kernels, and evict. No new kernels; the existing GPU ops are reused
   unchanged.
2. *GB10 — demand paging:* with unified memory the warm set is already GPU-accessible; residency
   is the hardware/OS's business (again the PLE precedent). No streaming code on GB10.
3. *CPU execution of cold experts (phase 2):* Strata's trick — the CPU computes cold experts
   in-place on pinned host RAM in parallel with the GPU's hot work. On GB10 this is natural
   (the Grace CPU shares the memory; no copies). On 5090 it requires new CPU kernels for the
   NVFP4 expert path and is only justified if PCIe streaming measures insufficient.

**KV and state.** Runtime state is already tiered (host state slots — validated this week). The
KV extension is **tail streaming**: for long contexts, the most-read KV pages stay on device and
the rest restores/streams from host on demand — an extension of the existing host-KV-spill
machinery from batch-restore-at-reuse to demand reads.

**Ranked profiles.** The converter already owns a frequency-corpus ranking discipline (the 35B
artifact carries `tools/freq_corpus` ranking provenance). The same discipline produces per-
`(layer, expert)` activation profiles at conversion time; the artifact carries them as component
metadata, and the runtime hot-set cache seeds from them and adapts.

**Container impact.** The v3 container already binds tensors as separate objects with logical
bindings; expert-slice placement metadata (tier hints, ranked profiles) is a bounded extension
of the existing binding/resource model, not a new container version.

## 5. The capability reframing: one core, family math modules

The two model families currently own parallel machinery (`AGENTS.md`: the Flash-Next family owns
its own "planning/Program/Text/Vision/MTP algorithms, state transactions, workspace composition,
and CUDA Graph machinery", explicitly not reusing the Qwen3.5 runtime). Much of the core is
already shared (`src/runtime` engine/scheduler/transactions, `src/core` containers, `src/serve`
protocol/transport). The duplication lives in the families' program/schedule/graph/frontend
layers. The reframing:

- **Shared core (one implementation, family-agnostic):** serving/HTTP, admission/FIFO
  scheduling, KV/state containers and their tiering transactions, context cache and pressure,
  CUDA Graph machinery, telemetry, artifact/container/binder, MoE expert machinery, quant formats.
- **Family modules (per family):** the mathematical Ops, the state shapes, the frontend/output
  semantics, the artifact recipes, and the schedule *policies* (parameterized over the shared
  core instead of owning a private core).

**Qwen 4 readiness is the acceptance test:** when a Qwen 4 family arrives, adding it means a
math module + an artifact recipe + its tests — zero changes to the shared core. Until then, no
speculative Qwen-4 code, placeholders, or abstraction for its hypothetical shapes.

The convergence is also the fork's single largest "simplify and add lightness" move (section 10)
and proceeds in bounded steps, each validated against the 27B baseline.

## 6. Telemetry: EngineMetrics compatibility, vLLM-class data sharing

EngineMetrics (0rand, `../EngineMetrics`) polls `/metrics` in Prometheus text format with
engine-prefixed counters plus a `/health` JSON, and has backends for vLLM, TensorFold, mlx-serve
and oMLX — none for us. Two deliverables:

1. **The fork's server exposure.** The serving layer already computes per-request TTFT, decode
   and prefill rates, MTP drafted/accepted, and prefix-reuse statistics. Aggregate them into
   Prometheus counters on `/metrics` (`ninfer:prompt_tokens_total`,
   `ninfer:generation_tokens_total`, `ninfer:mtp_drafted_total`, `ninfer:mtp_accepted_total`,
   `ninfer:request_latency_seconds_{sum,count}`, `ninfer:time_to_first_token_seconds_{sum,count}`)
   and enrich `/health` with the cache/pool counters EngineMetrics renders (pool tokens/free,
   kept prompts, cached tokens, context length). Add the tiering-specific counters the adaptive
   hot set needs to be tunable: hot-set size and hit rate, tier traffic (slices and bytes per
   tier), and CPU/GPU expert execution split.
2. **A `ninfer` backend for EngineMetrics** — a small, bounded module in 0rand's tool (a
   zero-dependency TUI; a backend is ~100 lines against the existing `fetch_metrics`/
   `fetch_json` helpers). Offered upstream to EngineMetrics, not forked.

This is deliberately sequenced first among the work packages: the tiering work is not tunable
without the observability, and the metrics are immediately useful for today's serving.

**As built (2026-10-12, SP-1 executed).** All telemetry lives on `GET /metrics` in Prometheus
text with `ninfer:`-prefixed counters and gauges (the vLLM pattern); `/health` remains the
minimal liveness probe it always was, so existing consumers are untouched. The serving layer
owns the request-observed counters (requests, prompt/computed-prefill/generation/reasoning/
cached tokens, MTP drafted/accepted/rounds, latency/TTFT/prefill/decode sum-count pairs); the
Engine owns the gauges (batch occupancy, KV pool occupancy, `kept_prompts` — a new
RuntimeStats gauge counting catalogued private continuations plus shared-prefix owners —
context length, state-slot occupancy, host-KV spill). The EngineMetrics `ninfer` backend
(`../EngineMetrics`: `engines/ninfer.py`, registry, auto-detection, render test) renders the
live dashboard from `/metrics` alone and is validated against a live fork server; the diff is
ready to offer upstream to 0rand.

## 7. EXL3 / Trellis (the quantization lever)

TensorFold proves EXL3 works as **one shared trellis module for every family** — any codebook
(3inst, mcg, mul1), any width 1–8, mixed within a checkpoint and inside one MoE layer — on
GB10-class hardware. Its impact profile is exactly the user's observation: better-quantized
models on fewer GB10s, traded against concurrency.

The fork's EXL3 pass: a v3-container format + `src/ops` kernels + a converter recipe, targeting
the MoE routed banks first (where the bit-for-bit quality lever is largest and where it
multiplies with tiering: smaller hot sets, more experts per GB). Numerical discipline per the
standing rules: one independent exact/FP32 oracle for the trellis decode and the MoE path, the
production kernels checked directly against it, tolerance-based comparison per the format's
precision profile. This package is independent of the tiering spine and sequences after it —
EXL3's value roughly doubles once the hot-set size is the binding constraint.

## 8. Multi-GPU and multi-node (phase-gated logical evolution)

- **Phase 1 (this document's core):** one GPU + RAM + disk. No contract change.
- **Phase 2 — secondary GPUs as expert caches:** disjoint ownership (each expert slice has one
  owner: GPU0, a secondary GPU, or the CPU pool; results return via pinned buffers — Strata's
  proven pattern). Requires real multi-GPU hardware in quantity and an explicit `AGENTS.md`
  contract revision.
- **Phase 3 — multi-node:** only with demonstrated demand; the largest contract change of all;
  not designed here beyond this sentence.

Each phase gates on the previous one being validated, and none is speculative.

## 9. Simplify and add lightness (the standing discipline)

- The family convergence (section 5) is the main slimming: delete the duplicated
  program/schedule/graph machinery; **net code deletion is a first-class outcome and is measured**
  at each convergence step.
- Test scope: validate the baseline (27B), the tiered target (125B), the shared core, and the
  external contracts (serving schemas, artifact framing). Delete tests of deleted paths. No
  coverage for coverage's sake; no tests for private shape.
- Every package below states what it deletes or de-duplicates alongside what it adds. A package
  that only adds is disfavored; a package that adds a capability and deletes duplication is
  favored.

## 10. Sovereign work packages

Each package is bounded, independently executable, and specified by a contract plus acceptance
criteria — which is also what makes each one delegable (section 12). Dependencies are stated.

| ID | Package | Depends on | Acceptance |
|---|---|---|---|
| SP-1 | EngineMetrics exposure: `/metrics` + the EngineMetrics `ninfer` backend | — | **EXECUTED 2026-10-12**: the EngineMetrics dashboard renders a live fork server (MTP acceptance, cache hit, pool occupancy, kept prompts all reconciling with the serve-contract smoke) |
| SP-2 | Upstream ports: constrained decoding (choice/regex) + tuple schemas/bounded numbers into serving | — | Tool-calling schema tests pass; the ported features engage on the tool-calling workload (measurable on the SP-1 counters) |
| SP-3 | KV tail streaming (host-KV-spill extension from batch-restore to demand reads; per-lane shape measured 2026-10-12: GB10 reads host-tier pages in place at fabric bandwidth — no staging — while the 5090 sits behind PCIe at 45 GB/s pinned read / 47 GB/s H2D against 1683 GB/s VRAM, so its tail is staged and hidden behind the batch's compute) | SP-1 (tunability) | Long-context serving on the 27B with KV beyond device capacity; the 27B baseline battery unchanged |
| SP-4 | Expert-tiering spine: container placement metadata, ranked profiles, the adaptive hot-set cache, tier transactions on the pressure/materialization spine. Phase 2a executed 2026-10-13 as the positional prefix split (byte-exact hot/warm mini-banks, generation bit-identical, 17.9% device-byte savings; the positional 80/20 split measured −80.5% decode). Phase 2b's ranked proof executed 2026-10-13: the MoE routing stages accumulate always-armed per-expert activation counters (zero measured cost — decode 775.01 vs 774.61 tok/s baseline, acceptance bit-identical), the measured skew shows the top-204/256 experts per layer take 96.91% of selections while the positional prefix took 78.70%, and the profile-ranked permuted artifact (banks and router rows reordered identically; generation bit-identical to the original — acceptance, rounds, and the permuted activation counts all exact) recovers decode from 151.14 to 590.12 tok/s (3.9×) at the same 17.27 GB device residency; the residual gap to full-hot (decode 76.2%, prefill 4828 vs 19097 t/s from long-tail warm re-reads per prefill tile) is the adaptive runtime cache and SP-5 prefetch's target. Phase 2b's scattered split executed 2026-10-14: per-expert residency tables (one 256-entry u32 table per MoE layer; bit 31 marks the warm tier, the low bits the expert's compacted rank) now drive all four decode and four prefill MoE sites, and the binder/materializer/loader consume arbitrary hot/warm lists directly — per-expert compacted mini-banks, tiers normalized to expert-id order, cold folded into the pinned-host warm tier — so no artifact permutation is needed; the scattered artifact (per-layer profile-ranked hot sets on the natural expert numbering) validates bit-identically (acceptance and rounds exact) with decode 590.47 tok/s — parity with the permuted ranked artifact's 590.12, making the permutation optional — and prefill 5338 vs 4822 t/s (+10.7% over the permutation), while the 27B serves unchanged at 82.58 tok/s with the machinery dormant; two assembly defects were found and fixed on the way: identity-rank hot experts received zero table entries (every hot expert read expert 0's weights, and the pre-table validation had been vacuous because the kernel path never ran), and the hot/warm partition keyed on the first view's parent, inverting both banks for layers whose expert 0 is warm. The adaptive hot-set cache executed 2026-10-14 (SP-4 final): every tiered bank's device mini-bank gained a cache tail contiguous with its hot slots — a promoted warm expert takes compacted rank `hot_experts + slot`, addressed by the same pointer and row math as any hot expert, so the kernels needed zero changes — sized by a device budget (free bytes at plan time less a 6 GiB startup reserve for state slots, workspaces, graphs, and KV headroom; an explicit KV capacity that no longer fits fails loudly at resolution), with one uniform slot count across each layer's paired banks; startup promotion copies each plane's warm blocks into the tail as one contiguous block per bank and flips the residency entries, so the tiered model serves device-resident experts wherever the tail covers its warm tier. Measured on all three 35B tiered artifacts (bit-identical generation throughout — acceptance and rounds exact): the split recovers decode 150.60 to 771.26 tok/s and prefill 3682 to 19072 t/s, the ranked 590.43 to 775.26 and 4822 to 19291, the scattered 590.47 to 771.89 and 5338 to 19065 — each at or above 99.6% of the full-hot records (decode 775.01, prefill 19097) at the tiered 17.27 GB planned residency plus the promoted tail, while the 27B stays dormant and unchanged (82.68 vs 82.58 tok/s) and the loading/moe contract tests pass. The runtime adaptation executed 2026-10-14 (SP-4's "adapt at runtime from the router statistics"): the always-armed per-expert activation counters now drive promotion/eviction at the prefill chunk boundary (every chunk) and the decode round boundary (every eighth round) for tail-limited tiers, with hysteresis — a newcomer must clear a minimum recent count and double the incumbent it evicts, at most two swaps per layer per pass — and stream-ordered copies so the flipped entries and promoted blocks land between the boundary's kernels. Measured on the 35B stand-in with the forced tail limit (the 125B's natural tail-limited scenario): at 26/52 slots per bank the warm-tier hit rate reaches 59.08% (the activity-ranked set displacing the id-ordered ~50%), decode 693.50 tok/s (+3.4%) and prefill 9698 t/s (+13.1% over the id-ordered promotion); at 13/52 slots the hit rate is 32.18% with decode 643.00 and prefill 6981; generation is bit-identical at every tail size. The full-tail paths (hit rate 100%, decode 774.81, prefill 19125.90 on the scattered artifact) and the 27B (82.62 tok/s, dormant) are unchanged, and the loading/moe contract tests pass. Remaining refinement: the precise baseline-first budget for automatic KV mode (today a 6 GiB startup reserve bounds the tail's competition with the resolved pool, and an explicit capacity that no longer fits fails loudly); the cold tier stays declarative (cold experts fold into the pinned-host warm tier) until SP-5's streaming generalizes it to the file-backed pattern | SP-1 | A tiered MoE serves with a hot/warm/cold split and a measured hot-set hit rate; the 27B serves with the machinery dormant and unchanged |
| SP-5 | 5090 PCIe expert streaming with layer-ahead prefetch (GPU-only execution; prefetch depth n+2 — two layers ahead, community-validated at +8% chat / +4.2% code on Flash-Next SSD streaming and suggested by upstream's author — and the cache must pin in-flight prefetches or eviction discards them before their layer consumes them) | SP-4 | The 125B (or a stand-in MoE) serves on a 32 GB card; throughput and hit rate measured on SP-1 counters |
| SP-6 | GB10 native 125B bring-up (validation lane; demand paging, no streaming code) | SP-4 (profiles only) | The 125B serves on GB10 natively; the resident-set and PLE demand-paging behavior measured |
| SP-7 | EXL3: container format + trellis kernels + converter recipe, MoE banks first | SP-4 (sequencing) | An EXL3-quantized MoE artifact converts, loads, and matches its oracle within tolerance; quality-per-GB measured against NVFP4 |
| SP-8 | Family convergence: one shared core; families shrink to math + recipes + policies | SP-1 (observability of no-regression) | The 27B baseline battery + smokes green at each step; net lines deleted reported per step; Qwen-4-readiness review (adding a family touches no core code) |
| SP-9 | Multi-GPU expert caches (phase 2) | SP-4, SP-5 + hardware demand + explicit contract revision | A two-GPU box serves a tiered MoE with disjoint expert ownership |

Sequencing guidance: SP-1 and SP-2 are immediately parallel-safe and small. SP-4 is the spine;
SP-3 can proceed alongside it. SP-5/SP-6 follow SP-4. SP-7 sequences after SP-4 for maximal
value. SP-8 proceeds in bounded steps continuously, validated by the baseline at every step.
SP-9 waits.

## 11. Delegation: Opus, upstream, and sovereign work

The owner's framing: the Opus subscription is a fixed $20/month that loses money on our usage —
extract maximum value by routing the right work to it. The strategy's package design exists
precisely to make that routing safe: each package carries a contract and acceptance criteria, so
an Opus session can execute against the spec and the fork reviews and validates.

**Route to Opus (implementation volume against our contracts):**
- SP-2, the upstream constrained-decoding port — mechanical, bounded, upstream code exists as
  the reference.
- SP-1b, the EngineMetrics `ninfer` backend — a small, well-specified Python module.
- SP-7's reference implementation and oracle: the trellis decode math and its independent
  naive oracle, written against our semantic contract, then our kernel validated against it.
- Test fixtures and documentation volume with stated contracts.

**Borrow from upstream:** SP-2's source code; any base-code fixes; nothing else.

**Offer upstream:** 5090-relevant correctness fixes (the capture-frontier chunk-split fix is
written, validated, and in a machinery upstream still has).

**Stays sovereign, never delegated:** the pressure/materialization/host-restore transactions and
the reuse semantics (the fork's hardest-won correctness), the tiering transaction design, the
acceptance criteria and product decisions, the baseline validation, and anything that changes
what "correct" means.

## 12. Guardrails

- The 27B baseline capability is not negotiable (section 3).
- No speculative Qwen-4 code; readiness is architectural, not implemented (section 5).
- Multi-GPU and multi-node happen only per the phase gates and with explicit contract revisions
  (section 8).
- This document authorizes no implementation; each package requires task-level authorization and
  updates the affected active authorities (`AGENTS.md`, the maintainer references) as part of its
  execution, not after.
- When the strategy is fully enabled or abandoned, this plan is removed or folded into the
  active references per the document lifecycle.

## 13. The local engine ecosystem (context, not commitments)

- **Project Maya** (`../project-maya`): a Strata fork serving GLM-5.3-Flash (321B total,
  18B active, 1M context) on one consumer GPU or up to 16, tiering experts across GPU/RAM/NVMe
  with self-tuning to the machine. Its own quants hold 97.7–99.2% of the FP8 model's zero-shot
  accuracy. The team runs GLM-5.3 heavily on it; GLM serving is Maya's lane and stays there —
  the fork remains Qwen-focused. Maya is the strongest external validation of this strategy's
  tiered-MoE direction, and a working reference for the multi-GPU phase (disjoint expert
  placement across cards).
- **The Maya-vs-Flash-Next comparison** (2026-10, community video with owner-reported numbers):
  on a 32 GB card + 96 GB DDR5, Maya serves GLM-5.3-Flash at 30+ tok/s (one RTX 5090 owner);
  on 24 GB cards, Flash-Next on Strata stays ahead (64–95 tok/s vs Maya's 11–17). The
  decisive datum for the quant strategy: at ~110 GB of memory, **a 125B model at 4-bit beats a
  321B model at 2-bit** — quality-per-GB wins over parameter count. This is the strongest
  external argument for the fork's EXL3 package (SP-7): better quantization of the 125B beats
  reaching for a larger model at worse quant. The same comparison notes MTP is disabled on a
  single-card Maya setup — a real speculative-vs-tiering interaction cost that the fork's SP-4/5
  design must account for (the MTP draft layer reads hot dense weights and must stay viable
  alongside tiered experts). RAM economics gate tiering (96 GB DDR5 ≈ $1,899): the strategy
  deliberately targets the RAM users already have.

## 14. Provenance of the numbers

- A/B: 2026-10-12, RTX 5090, identical nvfp4 v3 27B artifact, identical server flags and bench
  command (`tool-eval-bench --seed 42 --hardmode --trials 2`); upstream `81c8ce09` vs fork
  `5090-test@638b27cf`; full logs and JSONs in `~/runs/tool-eval-ab/`.
- 125B artifact: `docs/maintainer/qwen3.8-flash-next-125b-a6b-artifact.md` (125.5 GiB, five
  files, 51.2 GB file-mapped PLE table, 512/10 experts, NVFP4 routed banks).
- SP-5 prefetch depth: r/LocalLLaMA 2026-10-13, "../Based on advice from LocalLLama
  implemented a n+2 lookahead prediction for ssd streaming Qwen Flash Next - and it worked!
  _ r_LocalLLaMA.pdf" (github.com/skeggsguy/Flash-next-ssd) — an n+2 expert lookahead on
  Flash-Next SSD streaming measured +8% chat / +4.2% code on a Mac Mini M5 64 GB; the
  eviction-pinning interaction (a correctly prefetched expert can be evicted before its layer
  consumes it) is the thread's second finding and constrains SP-4 phase 2b's cache and SP-5's
  staging.
- RTX 5090 memory tier: `tools/bench/hardware/rtx_5090.json` — 1683 GB/s VRAM streaming read, 45 GB/s
  pinned-host read in place, 47 GB/s pinned H2D payload; measured 2026-10-12 by
  `tools/gb10/memory_probe.cu` (0.20 GiB buffers, CUDA 13.0.88, driver 13040, production server resident).
- Strata: `../Strata` `docs/HOW_IT_WORKS.md` (expert cache ~700 experts/GB, CPU-parallel cold
  experts, layer-ahead prefetch, KV tail streaming, disjoint secondary-GPU caches).
- TensorFold: `../TensorFold` `docs/recipes/cuda.md` (shared EXL3 trellis module, any codebook,
  width 1–8, mixed per layer).
- EngineMetrics: `../EngineMetrics` `engines/` (`/metrics` Prometheus + `/health` JSON; the
  TensorFold backend's counter set as the shape reference).
