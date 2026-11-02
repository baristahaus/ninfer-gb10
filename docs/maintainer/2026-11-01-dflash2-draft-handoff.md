# Flash-Next DFlash2 draft: handoff

Status of the DFlash2 draft lane as of 2026-11-01 23:58Z. Everything below was measured on
the GB10 workstation against the 128 GB Flash-Next FP8/MTP artifact with the DFlash2 draft
component appended. Read this before touching the draft path.

## 1. What works

- `ninfer-serve --spec dflash2 --draft-tokens N` starts, warms up (one full DFlash2 round),
  and serves coherent text. Verified: `17*23` answered `391` with `finish=stop`.
- The draft round executes end to end: context append, masked-block propose, target verify,
  greedy accept, PLE staging for the target's verify columns.
- Acceptance is nonzero, so the draft carries real signal: 12/138 (8.7%) on a 59-token reply,
  19/501 (3.8%) on a 187-token reply.

## 2. What is broken

Acceptance is far below the reference DFlash2 lane. `docs/performance/qwen3.8-27b.md:190-198`
records 33.9-65.9% acceptance and 3.4-5.6 tokens/round for the 27B DFlash2 route; this lane
produces 2-9%.

**Measured against the MTP lane on the same artifact** (2026-11-02, greedy,
`--no-cuda-graph`, `--max-context 4096`, one request; short = "17*23" 64-token reply, long =
photosynthesis 512-token reply):

| Configuration | Short acceptance | Long acceptance | Long decode |
|---|---|---|---|
| `--spec mtp --draft-tokens 3` | 43/48 (**89.6%**) | 307/611 (**50.2%**) | **45-48 tok/s** |
| `--spec dflash2 --draft-tokens 3` | 10/152 (6.6%) | 38/1413 (2.7%) | ~20.4 tok/s |
| `--spec dflash2 --draft-tokens 1` | 8/53 (15.1%) | 8/502 (1.6%) | ~22.4 tok/s |

MTP is ~2.2x faster end to end and needs no special flags (note `--lm-head-draft` no longer
exists; the MTP draft head is automatic). The DFlash2 lane currently performs at the plain
decode baseline — its verify batches cost more than the near-zero acceptance returns, and
dt3 is *slower* than dt1. The MTP lane also works unchanged on the DFlash2-appended
artifact, so the appended component does not disturb it.

The draft's *structure* is sound: its top-16 candidates usually contain the token the target
picks, but that token sits at ranks 3-6 instead of rank 0. That "right token, wrong rank"
signature is a systematic distortion, not a broken assembly.

## 3. Prime suspect: the draft's RoPE base — CONFIRMED AND FIXED 2026-11-02, with a twist

The suspicion was right, but the fix has a trap that cost the 2026-11-02 session most of a
day: **the whole DFlash2 lane reads `Variant::DFlashConfig` — the v1 `DFlashConfig` — not
`DFlash2Config`** (`variant.h:22` aliases `DFlashConfig = detail::DFlashConfig`; both
`propose_dflash2_batch_impl` and `append_context_impl` do `using Config = typename
V::DFlashConfig`). Editing `DFlash2Config::rope_theta` alone was completely inert: the
propose's `rmsnorm_rope` call and the append's `ops::rope` call both kept reading the v1
constant. An earlier serve A/B that showed bit-identical acceptance before/after the
`DFlash2Config` edit was the tell (that A/B also ran a stale `ninfer-serve`; rebuild both
binaries).

What is in place now:

1. `ops::rmsnorm_rope` takes a `float theta` on both overloads (contract updated; the
   theta=1e7 checkpoints keep a bit-exact constant-table fast path inside
   `rope_d128_sincos`). The qwen3_5 draft passes its runtime `config.rope_theta`.
2. `DFlashConfig::rope_theta` (the constant the lane actually reads) and
   `DFlash2Config::rope_theta` are both `1.0e4F`, documented from the companion
   `config.json`. `load_plan.cpp` validates the artifact's
   `dflash_config`-adjacent `rope_parameters.rope_theta` against it, so a mismatched
   artifact refuses to bind.
3. Verified by a full host oracle of the context-K chain (tap features ->
   `feature_projection` -> rmsnorm -> `context_key` -> per-head rmsnorm -> rope): the
   lane-cache K at the appended position matches theta=1e4 to BF16 rounding (max diff
   0.016) and rejects 1e7 (5.28).

Re-measured acceptance after the fix: essentially unchanged (dt1: 8/53 15.1% short,
8/502 1.6% long). The theta inconsistency was real but not the dominant loss. Do not copy
the `qwen3_5` reference's assertion that the draft's theta must be 1e7: that checkpoint is
a different model. The companion config is the authority for this one.

## 4. Ruled out, with evidence

| Item | Evidence |
|---|---|
| Head logits layout | The vocabulary producers store token-major (`LinearBf16Output::at`, and `ops::sample` reads `logits[col * physical_rows + v]`). `column_topk` was fixed to match. |
| NaN in the draft forward | Root cause was a swiglu row extent, fixed — see §5. |
| Selector | `NINFER_DFLASH2_BYPASS=1` zeroes the selector's projected hidden so the edge score collapses to the raw head score and the op picks the per-column head argmax. Acceptance stayed 0, so the loss is upstream of the selector. |
| Context append / cache contents | Lane-cache K-plane scan after prefill shows exactly 13 positions x 128 dims nonzero for head 0, zeros beyond — correct ring layout. |
| Prefill context values | Full-buffer scan: 0 NaN, 0 zero, sane magnitudes. |
| Dynamic-conv prepare | Host oracle on real weights: residual `mean|x|=0.0040`, prepared `mean=1.52 max=62.0`, base kernel `mean=0.505`, `finish_delta` per group in `[-8,+10]`; consistent with `(base + projected_delta) * rmsnorm(x)`. |
| Slot choice | Append and propose both use the round's destination slots, matching the v1 DFlash path and the reference. |
| `output` / `mlp_down` binding | `native_weight` carries the artifact's Q8 row-split format; the `bf16()` helper name is only misleading. |
| Feature tap | Three variants measured: completed-layer `block_input` 3.8%, final-mix `hyperconnection_mix` 2.3%, post-mixer `block_output` 1.5%. Completed-layer is kept. |

The tap is still unproven (the reference definition of the Flash-Next per-layer hidden has no
counterpart in this repo), but it is now second to the RoPE base: a wrong theta explains the
rank demotion on its own, and it is a one-parameter change.

### Ruled out 2026-11-02, with evidence (all reads stream-synchronized)

| Item | Evidence |
|---|---|
| `column_topk` fidelity | Host check on the dumped round: `scores[rank]` equals the logit at `column*V + id` for every rank, descending, and rank 0 is the column's global argmax. The candidates are the head's true top-16. |
| Greedy accept | Synced pre-accept dump of `drafts`/`verify_ids`/`extents` matches the egress `accepted`/`licensed_counts` in every round under the `row_targets[a] == row_drafts[a]` kernel rule. |
| Verify column geometry | Ingress anchors/frontiers/extents, block ids/positions (`[anchor@E, mask@E+1..]`), append positions/counts, and lane ids all dumped per round and all current; commit-stream arithmetic reconciles token-for-token with the generated text. |
| Dynamic-conv prepare (exact) | Host oracle on the full token-major buffers: max abs err 0.14/0.018/0.021/0.021 per column (BF16 rounding); `finish_delta` matches the projection to 0.12. Base indexing is `base[h]`/`base[H+h]` (tap-major flat). |
| Embedding + token-major convention | The dumped block residual bit-matches `embed_tokens` rows; the repo-wide activation convention is token-major content inside `[D, T]`-declared tensors (pinned by the `ops::linear` and embedding tests). Earlier "scrambled embedding" and "conv prepare mismatch" readings were dump-stride artifacts. |
| Tap feature layout | Cross-token block correlation on the dumped `compact_features`: diagonal-dominant (same-layer 0.41-0.59, cross-layer 0.06-0.25) — the 5-layer concat order and per-token layout are correct. |
| Target verify columns | The target's per-column verify logits differ and correctly condition on rejected draft tokens (col1's argmax shifts legally with the draft's proposal); the earlier "degenerate columns" reading was an extent-0 round plus stale-buffer reads. |
| Tap layer index | A/B 2026-11-02 with the sensitive metric (the target's top-1 rank inside the draft's same-position distribution + top-2k overlap, per-round synced dumps): `layer-1` tagging mean rank 1805 / overlap 0.163; `layer` tagging mean rank 3556 / overlap 0.166. No difference — **reverted to the original `layer - 1` semantics**; the tap index is not the defect. |
| Selector value | Post-theta-fix BYPASS (pure head argmax) is 0/28 while the selector-enabled dt1 run is 15.1% — the selector *helps*: it promotes the true token from deeper head ranks via the edge score. The defect is upstream, in the head distribution itself. |
| RoPE theta (post-fix) | Both the propose Q/K and the context keys now rope at 1e4 (oracle-verified); acceptance unchanged, so the remaining demotion has another cause. |

**New signature to chase**: the draft's distribution is *warm but wrong* — its top-2000
overlap with the target's same-position distribution is only ~0.16 (measured on synchronized
per-round dumps, stable across both tap indexings), and the true token's rank swings from 2
to 13888 across rounds. That is not a constant offset; the draft's effective context or
forward differs per round. Earlier "true token at rank 3-6 in the top-16" readings came from
the *unsynchronized* candidates/drafts dumps and should not be relied on quantitatively; the
overlap number supersedes them. The unsynchronized-dump trap (§6) bit twice more this
session — the `dump` lambda and the drafts dump now synchronize too; any new probe must.

**Token-identity check (2026-11-02, decisive)**: `--print-token-ids` on the compiler prompt,
16 tokens, all three specs — `none`, `mtp dt3`, and `dflash2 dt1` produce **bit-identical**
token streams. The DFlash2 round machinery (propose, verify, accept, commit, state) is
*correct*; the low acceptance is purely the draft's proposal quality. Over a 300-token
generation the specs diverge at benign numeric near-ties, but DFlash2 flips ~3.5x earlier
than MTP (visible-text divergence at ~18 tokens vs ~65; TEB deltas 90 vs 97 agree) — the
DFlash2 verify round's numeric/state handling (GDN replay/fold, PLE staging) flips argmaxes
more often than the MTP round's and is the second campaign target after the tap semantics.
The earlier "round-0 chimera / warmup leak" reading was wrong: the round-0 ledger is the
real prompt plus the model's own first generated token ('The' — the thinking opens "The user
is asking…"), and `--print-token-ids` confirmed all specs agree from token one.

## 5. Fixes already landed (all required for startup)

| Area | Change |
|---|---|
| Planner | Removed stale v1-style Q8 `linear_add` scratch; mirrored the reference DFlash2 planning (conv branch buffers, prepare/add capacities, logits projection, selector scratch) |
| Ops registered | BF16 swiglu for the draft's packed gate/up — see the row-extent note below; BF16 linear `(1024,2560)`, `(256,2560)`, `(2560,12800)`; BF16 split `(6144,2560) -> 4096/1024/1024`; `rmsnorm_pack_tail` at 2560 rows; new `column_topk` op |
| Binding | Conv base kernels and selector codebooks need reversed axes (`{2560,2,2}`, `{256,248320}`): the ops index with the hidden/rank axis fastest, matching `parameter_tensor`'s convention |
| Prefill | Feature tap added to the Flash-Next layer runner; BF16 context projections split out of `linear_pair` (which is Q8-only) |
| Round | Masked-draft-aware guards (backend KV entitlement, DFlash state check, decode dispatch, context append, backend frontier); proposal buffers sliced on the batch axis; DFlash PLE staging via a new ingress `ple_history` plus the stage mailbox |
| Diagnostics | BF16 linear route tables now report `(n,k,t)` on failure instead of an opaque message |

**Row-extent trap worth remembering**: the BF16 swiglu kernel takes the *gate-plane* row
extent, not the packed weight's row count. For the draft's `[15360, 2560]` packed gate/up the
shape constant is `kRows = 7680`; passing 15360 makes the kernel read rows 15360..30720 out of
bounds and every downstream value becomes NaN. This was the NaN that hid the real problem for
most of the session.

## 6. Debug facilities (all env-gated, safe to leave in)

- `NINFER_DFLASH2_DUMP=1` — stream-synchronized dumps and full-buffer NaN scans of the draft
  residual per layer, the MLP intermediate, the normed hidden, the head logits (token-major),
  the prefill context, the context cache K plane, per-round target argmax, per-round accepted
  count, and the layer-0 conv oracle inputs written to `/tmp/oracle_*.bin`.
- `NINFER_DFLASH2_BYPASS=1` — zeroes the selector's projected hidden so the selector picks the
  per-column head argmax; used to separate head quality from selector quality.

Probes synchronize the compute stream before every read. The compute stream is
`cudaStreamNonBlocking`, so an unsynchronized `cudaMemcpy` D2H races the kernels and produces
misleading dumps — this cost a full false-positive root cause earlier in the session.

## 7. Reproduction

```bash
# serve
build/apps/ninfer-serve out/dflash2/<artifact>.ninfer --host 127.0.0.1 --port 8047 \
  --max-context 4096 --max-concurrency 1 --kv-dtype fp8 --spec dflash2 --draft-tokens 3 \
  --no-cuda-graph

# single-step baseline (cleanest comparison point)
build/apps/ninfer-serve ... --spec dflash2 --draft-tokens 1

# CLI with diagnostics. The output-length flag is --max-tokens (shared config schema,
# generation.max_tokens); this binary rejects docs/cli.md's --max-new as an unknown argument.
NINFER_DFLASH2_DUMP=1 build/apps/ninfer out/dflash2/<artifact>.ninfer \
  --prompt "The capital of France is" --max-tokens 12 --temperature 0 \
  --spec dflash2 --draft-tokens 3 --no-cuda-graph
```

Acceptance appears in the serve log as `dflash2 accepted A/B (P%)` per request and in the CLI
summary as `accepted by pos`.

## 8. Open items, in order

1. ~~Draft RoPE theta.~~ **Done 2026-11-02** — see §3 for the `Variant::DFlashConfig` trap.
2. ~~Unary scores vs the packed top-k key.~~ **Top-k verified exact** (§4).
3. ~~Feature tap semantics.~~ **Resolved 2026-11-02 from the reference itself.** The checkpoint's
   capture patch (`patches/qwen38-dflash2-capture.patch`, incoai/Qwen3.8-Flash-Next-DFlash2)
   defines it: *"DFlash2 was trained on completed, zero-based layer outputs, averaged over the
   four hyper-connection streams"* — `hidden_states.unflatten(-1, (hc_count, hidden_size))
   .mean(dim=-2)` at layers [5,15,25,35,45]. Our capture took the next block's hyper-*mixed*
   `block_input`; it now emits `mean(hyper)` tagged `layer - 1` (timing was already right).
   Measured: draft-vs-target top-2k overlap **0.163 → 0.204**, mean true-token rank 1805 →
   1235; greedy acceptance 9.3% (France) / 5.5% (compiler) at dt1. The upstream lineage's own
   spec (`lkarlslund/ninfer6000/docs/maintainer/dflash.md`) otherwise matches this
   implementation clause by clause (context K/V, conv prepare/finish, masked-block
   fill-in-the-blank, selector walk, verify/commit).
4. ~~Full draft-forward oracle.~~ **Done 2026-11-02** — layer-0 host replay from the dumped
   artifact weights: q (post-norm+rope) max diff 0.020, k 0.019, v 0.247 (BF16 rounding on
   values ~10-18); the conv prepare had already matched to 0.014-0.14. The forward chain is
   numerically sound; no defect localized.
5. ~~The one-page over-map crash.~~ **Fixed and verified 2026-11-02**: `materialize_to_tokens`
   treats `target <= mapped` as a no-op. The TEB dflash2-dt3 battery now completes with **0
   invariants, 0 x 503, and scores 97.0** — equal to MTP's 97.0 and only 3 points below plain
   (100). The earlier 0.0 was 100% the crash. dt3 also beats dt1 (90) because dt1 runs ~3x more
   verify rounds per token.
6. ~~The "3.5x flip rate" vs MTP.~~ **Resolved by analysis**: the per-*round* flip rate is
   comparable; the per-*token* difference is the tokens-per-round ratio (dt3 = 97 = MTP). No
   numeric defect; dt3 is the correct operating point. Residual per-token flips are benign
   near-tie argmax flips inherent to batched verify.
7. **Batch > 1**: verified working (batch 2, both requests correct, batch 1.70). Acceptance is
   lower in batch (1.0-1.1%) than single (~9%) — a batch-path draft-quality item for the next
   campaign, not a correctness bug.
8. ~~CUDA Graph mode.~~ **Done 2026-11-02**: `Variant::dflash_graph_profiles` now returns the
   profile set (the `mtp_graph_profiles` analog: `profiles(capacity)` plus the verify-envelope
   topology adjustment — dense QSA where the envelope fits, classes 3/4 across the 65K/131K
   boundaries; the draft schedule is startup-fixed so no extra classes). Graph mode starts
   (ready in 1.2 s), serves correctly, and is quality-equivalent to eager: TEB
   `--short --trials 3` = **97.0 / Pass@3 93.3%**, the same as eager dt3 and MTP. Decode
   22.4-24.2 tok/s (eager ~20.4); acceptance 5.1-11.7% (eager 2.7-6.9%). The earlier
   empty-family throw and its unwind segfault are gone; the failure path's heap corruption
   (`ServeOptions::~ServeOptions`) is no longer reachable from this configuration.
9. **Draft quality remains the headline gap**: acceptance ~5-9% (dt1) / ~2.7-6.9% (dt3) vs the
   reference's 3.26 acceptance length (~81% per position). Further verification (2026-11-02):
   the tap-index A/B with the corrected mean value is null (0.204 vs 0.200 overlap — adjacent
   block means correlate, so the slot assignment barely matters; `layer-1` kept per the doc);
   BYPASS (pure head argmax) is still 0/28, so the draft's top-1 never matches the target's
   argmax; and the **finish** (`linear_dynamic_grouped_conv_add`, the last unverified stage)
   now matches its contract — Q8_G32 row-split weight decoded, mean abs err 0.101, large
   errors only on ~±400-magnitude elements at 0.3-3% relative. Every computable stage of the
   draft forward is oracle- or contract-verified. **Final verification 2026-11-02**:
   weight provenance is bit-exact against `~/models/dflash2-fused/model.safetensors` for
   feature_projection, layers/0 query_key_value, the context_key/value slices
   ([4096:5120]/[5120:6144]), mlp gate_up, the selector hidden_projection and the
   predecessor_codebook — no conversion/binding error; the tap slice order is confirmed
   absolutely (monotone correlation with the target's own final hidden,
   [+0.044, +0.048, +0.060, +0.086, +0.207], argmax at layer 45); the finish's 58 large
   absolute errors are all within 1 BF16 ulp of their magnitude (kernel correct); and the
   draft's dist is only weakly closer to the previous position's target dist (0.249) than
   the same position's (0.196) — a weak lean, not a clean off-by-one. The residual gap
   requires the source model's `hidden_states` (absent; a download-scale prerequisite) or a
   reference-runtime comparison (SGLang DFlash2), not any further local code change.

## 9. Coverage limits of everything above

Measurements are single-request, `--no-cuda-graph`, one artifact, greedy sampling unless a
row says otherwise (batch 2 was exercised once; CUDA Graph mode is unsupported). The
draft-forward gap is now closed for layer 0 (the host replay in §8 item 4) and covered by
the upstream spec (`lkarlslund/ninfer6000/docs/maintainer/dflash.md`) clause by clause for
the rest; the remaining blind spot is the *numeric fidelity of the tap at the capture
boundary*, which no oracle on this machine can settle without the source model's
`hidden_states`. Per-round synchronized dumps live in `/tmp/oracle_*.bin`
(`oracle_prop_logits_N.bin`, `oracle_target_logits_N.bin`, `oracle_feats_N.bin`,
`oracle_cachek.bin`, `l0_*.bin`); the conv-prepare, context-K, and layer-0 replays are the
reproducible checks.
