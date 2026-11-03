# Campaign plan: EXL3 SC 3.85bpw 27B on the 5090 with MTP and DFlash2, plus our own re-validated recipe

Status: active plan for the GB10 lane. Execution hardware: the 5090 box (serving,
batteries, short-window development) and the GB10 (sensitivity measurement and, if
windows require, the re-quantization compute). The pooled server on 172.30.30.16:8000
(NVFP4 MTP3, shared consumers) is never interrupted during development; every GPU job
below that needs the 5090 lists its window.

## 1. Objective

1. **Serve `JRamirez-UAB/Qwen3.8-27B-EXL3-3.85bpw-24GB` on this engine (ninfer-gb10, 5090)
   with both speculative routes — MTP3 and DFlash2 K=7 — and measure which works best**
   per axis (single-request latency, pool throughput at C≥2, long-context 256K).
2. **Re-validate his recipe by building our own quantization with his pipeline**
   (the exllamav3 SC pipeline: self-trace → rfn probe → sensitivity → greedy allocation →
   convert → held-out KLD), calibrated on **our deployment domain (sysadmin/agent)**, and
   head-to-head our artifact against his on the same public trace.
3. **Land reusable pipeline assets before Qwen 4 27B (or equivalent) drops**: the
   replay-over-EXL3 route, the SC recipe machinery wired through our converter, and the
   external-artifact import route. When the next 27B-class model arrives, the loop is
   trace + rfn + measure + optimize + convert ≈ 2–3 h of compute plus one day of glue.

The 27B class is the fully-hot baseline and the validation baseline for strategy work;
this campaign must not regress the NVFP4 MTP3 pool posture (the pool decision is a
separate user call after P3 — this plan does not switch the pool).

## 2. Verified preconditions (this repo, current tree)

Facts established by code inspection on 2026-11-02; each removes a class of risk from
the plan:

- **Width and codebook coverage: zero kernel work.** The dense EXL3 dispatch admits
  `k2 ∈ [2, 16]` per object (1.0–8.0 bpw, derived from the payload extent), and the
  TensorFold decode launcher registers every even `k2` (2…16) for both `mul1` and `mcg`
  plus odd `k2` (3…13) for `mul1`. His allocation uses 2/3/4/5-bit `mul1` linears
  (`k2` = 4/6/8/10): every width is registered.
- **MUL1 constant: bit-identical.** The contract-pinned constant `0x83DCD12D` equals the
  `mul1_multiplier` (2212286765) in his published `quantization_config.json`. The decoder
  hard-codes the constant; no per-object multiplier plumbing is needed and no
  version-skew exists between his v1.5.1 build and the pinned exllamav3 v1.6.1.
- **The speculative wall is one op.** `text.cpp:233` throws only in the GDN
  projection/conv section. The recurrence record (`gated_delta_net_replay_record`,
  `text.cpp:276`) is already route-agnostic — it consumes the q/k/v/g/β tensors that the
  decomposed EXL3 path already produces. The replay fold machinery
  (`src/ops/linear_attention/gated_delta_net/replay/launch.h`) already registers the 27B
  geometry (`Layers48Heads48`: 48 layers, 16 qk heads, 48 value heads, 10240 conv
  channels). The single missing piece is a conv-state record output for the standalone
  `causal_conv1d_silu_snapshot` (the non-EXL3 path gets it fused in
  `gdn_projection_record`).
- **Converter recipe engine exists.** `tools/convert/recipe.py` is an ordered
  fnmatch-wildcard per-parameter assignment engine (per-expert widths in the 35B bank
  were built on it); per-layer EXL3 bitrates are first-class.
- **Our quantizer is an exllamav3 reimplementation with oracles**
  (`tools/convert/sources/exllamav3.py`), so the same recipe can be quantized by both
  toolchains and compared.

## 3. Inputs and references

| Input | Identity | Role |
|---|---|---|
| His artifact | `JRamirez-UAB/Qwen3.8-27B-EXL3-3.85bpw-24GB` (main = 3.85bpw, 13.62 GiB; `3.70bpw` branch = 13.19 GiB) | import target (P0) |
| His recipe | per-tensor table parsed from his `quantization_config.json`: GDN qkv/z 12×3+36×4, out 42×4+6×5, gate/up 1×2+14×3+31×4+2×5, down 7×3+35×4+6×5 (GDN); full-attn q/k/v 2×3+10×4+4×5, o 9×4+7×5, MLP 3–5; a/b fp16; embed int8 row; head 5-bit; `mul1`; cal 196×2048 | the allocation to re-validate (P2) |
| His pipeline | exllamav3 `sc_trace.py` / `sc_rfn_probe.py` / `sc_measure.py` / `sc_optimize.py` / `convert.py -rcp` / `eval/qbench.py` | the method to run (P2) |
| His public eval trace | `turboderp/Qwen3.8-27B-exl3` → `qbench_prompts_gen.json` (24 prompts) | shared KLD ground truth for head-to-head |
| Our accepted 2.00bpw | `out/qwen3_8_27b_exl3.ninfer` (7.92 GiB, PPL 5.18 vs 4.55 FP8 control) | rfn probe anchor (P2), decode baseline |
| NVFP4 27B | pool artifact, MTP3 150–207 tok/s C=1, 18.98/19.73 GiB | serving baseline, reference model for trace sampling |
| Quality anchors | his: KLD 0.0067 (public trace), PPL 1.3430 vs 1.3332 BF16 public; MTP 3.03 tok/step, per-position acceptance 0.86/0.66/0.52; needle 8/8 at 197,991 on a 3090 | gate calibration (§6) |

## 4. Phases

### P0 — Import his artifact into the engine (5090 dev + one short window)

**W0.1 Converter route `qwen3_8_27b_exl3_sc385`.** Import his safetensors into a
`.ninfer` v3 artifact: per-object trellis/suh/svh with the per-layer `k2`, `mul1`
markers, int8 row-split embeddings (same family as our q8_g32), bf16 norms, fp16 a/b
projections, and his 5-bit EXL3 `lm_head` (248320×5120, `k2=10`). Emit the conversion
report with per-object widths (the report format already carries per-object format;
add the `k2`/codebook fields for EXL3 objects).

**W0.2 EXL3 lm_head decode route.** The contract's "remaining dense consumers (the
embedding and lm_head)" step: route the head matmul (T=1: `x[5120]·W[5120,248320]`)
through the EXL3 dense dispatch. Verify kernel generality at `n=248320` (grid over n
tiles); if the fused route cannot take the shape, fall back to converter-side
re-encoding of the head to q8_g32 (+0.37 GB, zero engine work) and record the decision.

**W0.3 MTP companion.** His `mtp_bits=4` module enters the artifact as the speculative
companion (the arena currently carries the NVFP4 companion; the MTP linears must
dispatch through the EXL3 dense route). Verify with a draft-token smoke.

**W0.4 DFlash2 companion.** The DFlash2 draft model is independent of the main model's
weight quantization: re-extract it from the existing NVFP4 27B artifact (or the BF16
source) and re-encode with the existing companion route. No MTP/DFlash2 *schedule*
changes — both schedules operate on `ReplayRecords` and state slots, which are
route-agnostic.

**Acceptance P0.**
1. Load + `--spec mtp --draft-tokens 3` smoke and `--spec dflash2 --draft-tokens 7`
   smoke at MTP0 semantics (speculative disabled) with TEB `--short` (≥14/15).
2. Oracle: the `a16` argv-mode real-tensor check (follow-up plan M3, long pending)
   against **his** tensors — relative L2 ≤ 2/256 on K∈{4,6,8,10} × `mul1` at the 27B
   decode shapes. This also closes M3 with the strongest possible fixtures.
3. Non-spec decode numerics at 8K within the composition-oracle tolerance of §3.2
   against the FP64 reference.

**Window:** one short 5090 window (smoke + oracle, < 2 h). Development otherwise.

### P1 — Speculative decoding over EXL3 (the critical path)

**W1.1 Conv-record op.** `causal_conv1d_silu_snapshot_record` (or a record-output
variant of the existing op): identical numerics to the snapshot path plus a
`records.conv` plane per valid column. Oracle: the recorded conv state must equal an
independent conv-state evaluation at the valid columns (and match the fused
`gdn_projection_record` conv record where a fused-path layer exists for comparison).

**W1.2 Replay wiring.** In `text.cpp`'s GDN section: replace the throw with the
decomposed replay path — EXL3 `gdn_projection` → `causal_conv1d_silu_snapshot_record`
→ the existing route-agnostic `gated_delta_net_replay_record` → the existing
fold/commit machinery. Both `ReplayRecords` consumers (MTP3 and DFlash2 K=7) then run
unmodified over the EXL3 main model.

**W1.3 Speculative-invariant test.** A permanent test: seeded draft tokens with forced
rejections (deterministic, small fixture) — the committed GDN state and logits after
replay must match the non-speculative sequence to the oracle tolerance. This is the
regression guard for the record/fold numerics; it runs in the CPU/GPU test suite, not
the pool.

**Acceptance P1.**
1. W1.3 green for both MTP3 and DFlash2 over EXL3 (the 27B fold geometry is already
   registered).
2. No regression on the non-spec EXL3 2.00bpw route (the existing suite + a TEB pass).
3. Per-position acceptance and tokens/step recorded for both modes (gate anchors:
   his 3.03 tok/step, 0.86/0.66/0.52 — measured on a different card and engine, so
   treat as reference, not gate).

**P1 measured (2026-11-02, RTX 5090, `qwen3_8_27b_exl3_sc385_df2.ninfer`, TEB --short
15/15 quality 100 both modes, `--kv-dtype k8v4 --prefill-chunk 1024
--max-concurrency 2`):**

| Mode | Rounds | Drafted | Accepted | Acceptance | Tokens/step | Per-position acceptance |
|---|---|---|---|---|---|---|
| MTP3 | 1319 | 3956 | 3368 | 85% | 3.55 | not recorded (single run) |
| DFlash2 K=7 | 880 | 6155 | 3861 | 63% | 5.39 | 91% / 79% / 69% / … of rounds |

Tokens/step = generation_tokens / (rounds + fallback steps); DFlash2's
`spec_decode_accepted_tokens_per_pos_total` series render on /metrics (the MTP run
predates that counter's landing). Both modes beat his anchors.

**P3 decode-saturation progress (same artifact, the published driver
`run_serve_concurrency.py`, `long_decode_aime26_15`, 8192-token budgets; points under
`profiles/bench/p3-exl3sc/points/`):**

Steady-decode tok/s (the published definition; the NVFP4 MTP3 reference rows are
147.7/291.0/522.2/922.4):

| Mode | C=1 | C=2 | C=4 | C=8 |
|---|---|---|---|---|
| MTP0 steady | 59.5 | 116.3 | 220.5 | 407.1 |
| MTP3 steady | 116.2 | 213.2 | 381.5 | blocked: T>16 |
| DFlash2 K=7 steady | 147.9 | 264.8 | blocked: T>16 | blocked: T>16 |

MTP0 is the first MTP0 saturation record for this route. MTP3 C=1/C=2 steady reaches
79/73 percent of the NVFP4 MTP3 rows at half the weight footprint. MTP3 acceptance at
this fixture is 47-50 percent (the AIME domain, below TEB's agentic 85 percent);
DFlash2 collapses to ~30 percent here. Makespans: MTP0 138.3/142.1/151.1/165.9 s;
MTP3 71.0/78.0 s at C=1/C=2.

**Root cause of the C>=4 collapse, found and fixed:** the record rounds project at
T = width x batch; past the register route's T<=8 domain the EXL3 linear composed
its route and dequantized a 176 MB prime per layer per round. The register domain
now covers T<=16 (the kernel's validated bound; T=32/64 produce non-finite columns
on the token tiling and stay composed until launch_linear's token tiling extends).
Re-measured with the fix: MTP3 C=4 381.5 tok/s steady (was 18.1; makespan 88.9 s),
DFlash2 C=2 264.8 tok/s (was 11.8; makespan 64.4 s). The MTP3 C=8 (T=32) and
DFlash2 C>=4 (T>=32) rows await the kernel's token-tiling extension.

**Window:** development iterations are short (single-request speculative rounds); one
medium window for the acceptance battery.

### P2 — Our own quantization with his pipeline, sysadmin/agent domain (GB10 primary)

The pipeline runs as published (his tools, pip exllamav3), not a port — "his pipeline"
is the point. The outputs feed P0's import route (W0.1) unchanged.

**W2.1 Self-trace (5090, pool model, no window needed — samples through the pool API
when the owner agrees, else a dedicated short instance).** Sample the NVFP4 27B
(thinking mode on, `preserve_thinking`, the deployment profile) over sysadmin/agent
scenarios: TEB scenario pack (opencode-style requests, tool calls, real command
execution transcripts) plus long agent sessions — our domain, English. Pack 250×2048
rows (`sc_trace.py` format). His trace was 196×2048 of Spanish sysadmin; domain match
is rationale #2 of their pipeline and the explicit reason for doing this.

**W2.2 rfn probe (5090, short window).** Dequant our accepted 2.00bpw artifact
per-tensor, record `rfn_t = ‖W_q − W‖/‖W‖` (`sc_rfn_probe.py` against the checkpoint).
K=2 anchors extrapolate across K per the amplitude-halving law — our artifact is the
anchor.

**W2.3 Sensitivity measurement (GB10, ~1–2 h of compute).** `sc_measure.py`:
LDLQ-shaped noise injection per tensor into the **unquantized** BF16 27B (54 GB —
fits the 121 GiB unified pool natively; their doc: "fits comfortably in 96 GB"),
suffix-forward from cached boundary states, KLD vs clean reference, two noise levels.
Fallback if the GB10 test setup lags: the 5090 tiered lane (experimental for 54 GB
dense; flagged, not planned).

**W2.4 Greedy allocation → recipe diff (CPU).** `sc_optimize.py` at two targets —
**3.85 bpw** (his size point, head-to-head) and **2.5 bpw** (a smaller point inside our
current 7.92-GiB posture). Then the re-validation itself: diff our per-tensor
allocation against his published table per tensor class (qkv/z/out/gate/up/down/q/k/v/o,
per layer-class) — agreement rate and per-class divergence, with the divergence
explained by the domain (English vs Spanish sysadmin) and the reference model
(our NVFP4-sampled trace vs his). This is the "what was used per layer, and does our
independent measurement agree" record.

**W2.5 Our quant (GB10 or 5090 window, ~1–2 h).** His `convert.py -rcp recipe.yaml
-cd cal_trace.safetensors` (his quantizer, primary) → standard-format EXL3 → import via
W0.1 → `out/qwen3_8_27b_exl3_sc385_ours.ninfer`. **Cross-check:** the same recipe
quantized by our converter (the exllamav3 reimplementation) on the full artifact;
dequant parity between the two toolchains against the FP64 oracle at a per-tensor
sample (this proves the reimplementation on `mul1` and K∈{4,6,8,10} end-to-end, which
M3's a16 check proves only in one direction).

**Acceptance P2.**
1. KLD on his public trace (`qbench_prompts_gen.json`): our 3.85 artifact vs his —
   gate: **ours ≤ 1.10 × his 0.0067** (domain shift budget; same architecture, same
   target bpw).
2. Full-corpus PPL ≤ 4.70 (his PPL delta vs BF16 is +0.7% on the public trace; our
   corpus control is 4.55, so ≤ 4.70 is the honest bound; his 2-bit-class would be
   ~5.2, ours at 2-bit is 5.18).
3. TEB `--short` ≥ 14/15 with quality ≥ 95 (NVFP4 baseline: 15/15 at 100).
4. Toolchain parity: max per-tensor dequant divergence (his quantizer vs ours, same
   recipe) within 2/256 relative L2.

### P3 — Performance battery and the decision (5090, pool window, ~1 day)

Configuration matrix (all on the 5090, same fixtures as `docs/performance/qwen3.8-27b.md`):

| Route | Weights | Roles |
|---|---|---|
| EXL3 SC 3.85 (his) × {MTP0, MTP3, DFlash2 K=7} | 13.62 GiB | new |
| EXL3 SC 3.85 (ours) × {winner mode} | ≈13.6 GiB | recipe winner |
| NVFP4 × {MTP3, DFlash2 K=7} | 18.98/21.38 GiB | baselines (existing rows) |
| EXL3 2.00 uniform (ours) × {MTP0} | 7.92 GiB | decode-rate reference |

Measurements: decode C=1/2/4/8 with the `--ignore-eos` knob (8K), context profile
8K/64K/130K/260K (MTP0), prefill/TTFT, per-position acceptance, tokens/step, KV pool
sizing at 3.85 (13.62 GiB weights → ~17 GiB free → ~700K-token pool at k8v4: 4×171K or
8×85K concurrent long-RoPE contexts — record the achieved pool), TEB responsiveness
median turn, 256K needle 8/8.

**Decision rules (recorded, not discretionary):**
- **Pool mode winner:** argmax aggregate throughput at C=4 and C=8 subject to the P2
  quality gates; tie-break on per-request median latency. A speculative mode whose
  tokens/step lands below 2.5 after P1 is dropped from the pool candidates and the
  recorded reason says so.
- **Pool switch:** user decision after P3 — the plan does not pre-declare it. If the
  switch happens, the pool restarts on EXL3 SC 3.85 + winner mode with the existing
  flag set (plus the spec-mode flags); counters restart at zero (pooled consumers
  delta-sample across the gap, per the documented restart convention).
- **Recipe winner:** our allocation vs his on the P2 KLD head-to-head; adopt the
  better, keep both reports. The per-class overlap from W2.4 is published with it.

**Acceptance P3.** Battery tables land in `docs/performance/qwen3.8-27b.md` (new EXL3
SC rows, same method as the October EXL3 section); the fork-strategy SP-7 note gains
the campaign closure record (gates recorded on both sides per the ratified cadence).

### 4a. W0.1 binding design (resolved 2026-11-02, payload verified)

Download verified against SHA256SUMS at
`/nfs/hub/models--JRamirez-UAB--Qwen3.8-27B-EXL3-3.85bpw-24GB/snapshots/dd840e08…/`
(14.62 GB, 2 shards, 3081 tensors, 573 EXL3 objects: trellis last dim = 16·bpw —
289×4-bit, 77×3-bit, 41×5-bit incl. the head, 2×2-bit, 164×6-bit vision).

The controlling constraint: his artifact carries a per-tensor input transform (`suh`)
on every linear, while this engine's fused groups (gate_up, qkvz) assume one shared
suh. Verbatim import therefore keeps his objects separate and binds them through the
engine's existing pair/rows parameter forms:

| Group | Import | Engine work |
|---|---|---|
| Attention | 4 objects/layer (query/gate split from his `q_proj` by 256-row-per-head bands — tile-aligned, 16-row granularity holds); per-role rows form is already what Qwen3.5 uses | none |
| GDN | `(in_proj_qkv, in_proj_z)` pair — the op signature's pair form exists | the decomposed EXL3 route must recognize the pair (currently single-only) — folds into W1.2 |
| Dense MLP | `(gate, up)` pair + `down` | one new `DenseParameters` variant arm + a two-dispatch branch in `dense_mlp`; workspace sizing follows the MoE separate-bank pattern |
| Head/embed/norms | verbatim (5-bit EXL3 head, int8 row embed) | W0.2 head-route check only |
| Vision / MTP | text-only first artifact; his vision tower carries 247 biases (our linear op takes none) and his MTP companion is a 7-tensor fc head (~20 MB) | deferred; both are explicit P0 verification items, not assumptions |

Text model has zero biases (config-confirmed), so the bias question is vision-only.

## 5. Hardware and window map

| Job | Machine | Duration | Pool impact |
|---|---|---|---|
| W0.1/W0.4 converter work, W1.x development | 5090 dev tree / GB10 | — | none (no device-allocating runs during dev builds) |
| P0 smoke + oracle | 5090 | < 2 h | one short window |
| W1.3 + P1 acceptance | 5090 | ~2 h | one medium window |
| W2.1 trace sampling | 5090 (through the pool API, owner-approved) or dedicated instance | ~1 h | none if through the pool |
| W2.2 rfn probe | 5090 | ~1 h | one short window |
| W2.3 sensitivity | **GB10** | ~1–2 h | none |
| W2.5 quant × 2 | GB10 (primary) | ~2–4 h | none |
| P3 battery | 5090 | ~1 day | one long window (pool down for the battery) |

Total 5090 window: ~1.5–2 days, negotiable per phase. Single-GPU rule holds throughout:
never a second device-allocating run while the pool holds the card.

## 6. Quality gates (consolidated)

| Gate | Criterion | Anchor |
|---|---|---|
| KLD (public trace) | ours ≤ 1.10 × 0.0067 | his 3.85 build, his public trace |
| PPL (full corpus) | ≤ 4.70 | FP8 control 4.55; our 2-bit 5.18 |
| TEB short | ≥ 14/15, quality ≥ 95 | NVFP4 15/15 @ 100 |
| Needle 256K | 8/8 at ≥ 197,991 | his 3090 run |
| Speculative invariants | W1.3 replay/reject equivalence | oracle tolerance §3.2 |
| Toolchain parity | ≤ 2/256 relative L2 per tensor | FP64 oracle |
| Decode shapes | a16 real-tensor (his tensors) ≤ 2/256 | contract §3.2 |

## 7. Risks

- **R1 — EXL3 head at n=248320.** The fused route may not be shape-generic at the head
  geometry. Mitigation: P0 smoke first; fallback is converter-side q8_g32 head
  (zero engine work). Cost if fallback: +0.37 GB, negligible KLD delta expected.
- **R2 — conv-record numerics.** The record must be the conv state exactly at the valid
  columns (fold replay rewinds from it). Mitigation: the W1.1 oracle and W1.3 before
  any battery.
- **R3 — DFlash2 companion domain shift.** The companion was trained against the
  BF16-class main; acceptance over the EXL3-3.85 main may sit below the NVFP4
  baseline. This is measured, not a blocker; the P3 decision rule handles it.
- **R4 — his allocation is domain-tuned** (Spanish sysadmin, his trace). The W2.4 diff
  decides how much to trust the import over the re-quant; if the per-class divergence
  is large, ours wins the recipe slot and his artifact is still the import target for
  the P3 baseline (both measured).
- **R5 — window availability.** Every long 5090 job has a GB10 fallback except P3
  (the battery defines the 5090 claim). If windows slip, P0→P1 ship first (the
  speculative route is the durable asset), P2/P3 follow.
- **R6 — GB10 test setup maturity.** SP-6 bring-up is "test setup comes later"; if the
  GB10 is not serving-ready for W2.3, the 5090 tiered lane is the flagged fallback
  (experimental at 54 GB dense) or W2.3 waits.

## 8. Out of scope

- EXL3 prefill tensor-core route (prefill stays the SIMT composed route; the world's
  prefill numbers confirm the 10× gap is a format property, not a tuning target here).
- YaQA output scales and n-gram PLE quantization (125B lane; only if a P2 gate fails
  and the attribution points at output-channel scales).
- Pool switchover (user decision after P3), GB10 125B bring-up (SP-6), Qwen 4 work
  (the plan's deliverable is that the pipeline is ready when it drops).

## 9. Effort and timing

P0: 2–3 days development + one short window. P1: 2–4 days development + one medium
window (one op, one wiring, one invariant test — the code structure makes it smaller
than "implement speculative over EXL3" suggests). P2: ~2 days glue + ~4–6 h of GB10
compute. P3: ~1 day window + analysis. Total: 2–3 weeks of mixed-lane effort, 1.5–2
days of 5090 window. Durable assets for the Qwen 4 27B arrival: the replay-over-EXL3
pattern (per-family, transfers to the next hybrid), the SC machinery wired through our
converter (model-agnostic), and the import route (format-generic).

## 10. Sequence and dependencies

W0.1 → {W0.2, W0.3, W0.4} → P0 acceptance → W1.1 → W1.2 → W1.3 → P1 acceptance → P3
(needs P1). W2.1 → W2.2 ∥ W2.3 → W2.4 → W2.5 → P2 acceptance. P3 consumes P1 and P2
winners. P0 acceptance (oracle on his tensors) is the earliest hard gate; if the
import fails the oracle, stop and re-check format compatibility before spending P1.
