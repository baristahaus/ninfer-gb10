# Effort report: the EXL3/Trellis quantization package (SP-7), closed 2026-10-24

Status: closed. This document is the closing record of the SP-7 EXL3/Trellis effort: what was
built, what was measured, what was decided, and the logical next steps parked at close. The
semantic contract remains `exl3-trellis-contract.md`; the campaign economics remain in the
closed `2026-10-21-q38fn-exl3-campaign-plan.md`; the accepted artifact provenance lives in this
report and the conversion reports beside the artifacts.

## 1. What the effort set out to do

Take the fork's MoE and dense weights from fixed FP8/groupwise recipes to EXL3 trellis coding
(per-expert bitrates, trellis-constrained Viterbi quantization, uncalibrated — no Hessians), on
three fronts: the 27B dense route end to end, the 35B MoE bank route end to end, and the 125B
Flash-Next banks campaign (planned, not executed). The purpose was quality-per-byte: the same
serving quality at half the resident bytes, as the tiering posture lever on both lanes.

## 2. What landed (all committed and pushed on `5090-test`)

- **The 27B dense EXL3 route: accepted.** `out/qwen3_8_27b_exl3.ninfer`, 7.92 GiB, full-corpus
  perplexity 5.18 vs the FP8 control's 4.55 at 26.5 GiB. Fused decode kernels, converter route,
  oracles, container layout (`exl3_dense_v1`), and the contract all landed earlier in the
  package; this effort's campaigns produced the accepted artifact.
- **The 35B MoE bank route: accepted.** `out/qwen3_6_35b_a3b_exl3_banks.ninfer`, 14.0 GiB at
  k2=6 uncalibrated banks (dense projections keep FP8), full-corpus overall 4.613 vs the FP8
  control's 4.552 — within 1.4% at 53% of the bytes. Per corpus: better than the control on
  chinese_reference (5.281 vs 5.480) and english_reference (5.931 vs 6.080), tied on
  english_long_form (7.808 vs 7.807), worse on ninfer_code (1.842 vs 1.640, the one clear
  regression). Single run, no repeats; same corpus, windows, context/stride and fp8 KV on both
  sides. Early-window trend readings (~5.9) proved to be cold-start artifacts — partial windows
  are not a go/no-go instrument.
- **The engine reconciliation** (`e7960dc5`): the tier-rank stride made layout-aware (the packed
  native route's 2·intermediate stride vs the EXL3 banks' one row block per expert), the FFN
  workspace sizing for the EXL3 decode plan, and the removal of the artificial single-token
  wrapper restriction (the decode loop already covers arbitrary token counts, so prefill runs
  as numerically identical per-token decode).
- **The shape-readiness sweep for the 125B shapes**: completed and verified by twoFour
  (commit `e9e35a67`, branch `q38fn-bank-shapes`, their box) — decode kernels templated on
  `SparseMoeShape` with exactly two instantiations (35B and 125B geometries), a 512-expert/top-10
  selector, workspace/plan arithmetic, wrapper admission, and the Flash-Next family binding
  branch. Plus one latent bug found and fixed on the way (`dot_fp32_rows` group count). Verified
  on their box (175/175 targets, both suites); not yet reviewed in-tree.
- **Campaign economics, measured**: ~2800 s per 35B layer streaming (separate gate/up linears),
  ~2230 s packed (executed then reverted — the packed layout cannot feed the separate-bank
  kernel route, contract §2.2); the 5090 box streams at ~1.07 s per million routed parameters;
  GB10 streaming measured at 18.5k params/s (~50× slower) — the campaign, if ever revived, runs
  on the 5090 box.
- **The repo strip-down** (`bf3e0551` and follow-ups): raw profiling traces untracked (repo
  974→74 MB), ~270 GB of dead experiment artifacts removed locally, GitHub cleaned across the
  default branch and stale branches/tags.

## 3. Decisions taken

- **The 125B will not be re-encoded.** It serves directly from the existing FP8 MTP artifact
  (70.5 GiB on the ~121 GiB GB10 pool). A faithful load keeps the original authors' quality
  guarantee; a lossy repack transfers the quality burden to the fork, with a ~5-day campaign and
  a quality gate the effort would have to pay for. The banks route stays the documented denser-
  posture option, now with an accepted 35B anchor.
- **The 35B acceptance closed the dataset**: every planned acceptance run on the perplexity-1m
  corpus is complete; nothing is half-scored.
- **The packed-gate_up detour was reverted** — the correct layout is separate gate/up banks
  (contract §2.2); the detour's reusable residue is the measured packed rate and the reuse
  pattern for finished projections across campaigns.

## 4. Logical next steps, parked at close (in the order they would matter)

1. **GB10 125B direct-serving bring-up (SP-6)** — now the actual product path: the FP8 MTP
   artifact serving on the unified-memory lane. Its test setup comes later per the strategy.
2. **The agent-cache tiering package** — design complete and committed
   (`2026-10-24-agent-cache-tiering-design.md`): session-scoped retention weighting plus the
   state spill tier (Device/Host/Spill, host-layout reuse, page-in-never-recompute invariant).
   This is the strongest next implementation: it makes long-horizon agent sessions survive idle
   gaps at O(tail) resumption cost, which the cache-economics frame identifies as the dominant
   serving lever.
3. **Merge `5090-test` into `master`** — the fork's default branch is one README commit and 95
   work commits behind; the effort's whole record is only on the work branch.
4. **Review and land twoFour's `q38fn-bank-shapes` sweep in-tree** — verified on their box, not
   reviewed here; it is standing preparation for any banks revival and touches kernel surfaces
   the fork owns.
5. **The sanctioned upstream port** — the length-bounded JSON string mask optimization
   (upstream `0a8530d4`): a 387-line rewrite across the diverged vendored xgrammar; portable
   only when its test suite can run.
6. **The stale-branch work** — the `--cors` serve feature and the Flash-Next NVFP4 MoE oracle
   tests on stripped branches; real work, unmerged, needing review.
7. **If a denser 125B posture is ever actually needed**: the banks route is accepted and the
   shape sweep exists; the prerequisites are a batched EXL3 MoE prefill kernel (a hard
   requirement under agent-class traffic — the looped decode disqualifies it, strategy §4),
   the mixed-rate map design (per-expert rates are natively expressible; the sensitivity anchor
   is one more 35B acceptance run), and a campaign on the 5090 box.

## 5. Provenance summary

- Artifacts: `out/qwen3_8_27b_exl3.ninfer` (7.92 GiB, accepted), `out/qwen3_6_35b_a3b_exl3_banks.ninfer`
  (14.0 GiB, accepted), `out/qwen3_6_35b_a3b.ninfer` (22.8 GiB FP8 control); conversion reports
  beside each. Campaign layer files under `/home/barista/calib/exl3_banks_35b_k6` (separate-bank
  layout, complete) and `exl3_banks_packed` (packed, reverted).
- Runs: `profiles/perplexity/exl3_banks_k6_full/report.json` and
  `profiles/perplexity/control_fp8_full/report.json` (local-only per the strip-down policy);
  logs under `/home/barista/calib/`.
- Key commits: `f2ca42b6`/`b0fc2af3` (packed detour and revert), `e7960dc5` (engine
  reconciliation), `bf3e0551` (strip-down), `db01aa86` (contract path fixes), `e724e351`
  (125B campaign closed), `5a678d76` (cache-tiering design), `8914f212` (llama.cpp expert-cache
  ingestion), `52dea3d5` (this acceptance recorded).
- Hardware: RTX 5090 (32 GiB VRAM + 123 GiB host), CUDA 13.4.92, `sm_120a`; GB10 streaming rate
  measured by twoFour on the unified-memory lane.
