# GB10: lkarlslund/ninfer master (8f574ee4 + share/lk-gb10 patches) vs this fork

Run list from `share/lk-gb10/README.md`. Executed on the GB10 workstation, 2026-10-10.

## Setup

- lk tree: `lkarlslund/ninfer` at `8f574ee4` (2026-10-07) + the four `share/lk-gb10` patches
  (`git am`): sm_121a gate, MemAvailable startup sizing, SM-count persistent grids, MTP RoPE
  element-count gate.
- Build: `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121a`,
  `cmake --build build -j`. Prerequisite installed on the box: `libssl-dev`
  (lk's artifact layer requires OpenSSL; this fork does not).
- lk binary: `lk-ninfer/build/apps/ninfer-serve`. Our arm: this repo's
  `build/apps/ninfer-serve` at head `91a1405e`.

Incident (2026-10-10 14:51-14:56Z): the first build attempt ran concurrently with
an unknown second build of the same tree (background job `bg_2`, 472-step graph,
its own command line) in the same `build/` directory. The two raced and the first
`apps/ninfer` link failed on a CUDA device-link registration
(`__fatbinwrap_...small_t_cu_...` undefined). Fallback: discarded the raced tree
and did a clean single-process rebuild (verified before any GPU work).

Incident (2026-10-10 15:28-15:36Z): the first parity arm (lk, K=3, v3_fork entry, max-concurrency 4)
failed at startup: `view element count mismatch: requested 36, available 12` during CUDA graph
preparation. The K-scaling (K=3: 36/12, K=2: 27/9, K=1: 18/6, i.e. requested 9(K+1) vs available
3(K+1)) located the view. lk's `mtp_forward_flash_next` decides the MTP RoPE layout with a shape
test on `rope_positions.ne[1]`, which misreads a `{width,batch}` text-position tensor as three-axis
MRoPE whenever the decode batch is 3; the batch-3 MTP decode graph capture then views it as
`{width,batch,3}` (three times the element count) and throws. Fallback: backport the fork's
element-count fix (`11f9e7a6`) into the lk tree as the fourth `share/lk-gb10` patch (commit
`601c7369` in the lk clone), rebuild, rerun the arm. The buggy path throws rather than computes, so
the fix changes no numerics on a previously working path. The fork's own serve under the identical
flags is the positive control.

## ctest (run-list item 1)

`ctest --output-on-failure` on the patched tree, env:
`NINFER_QWEN38_FLASH_NEXT_WEIGHTS=<v3_fork entry>`,
`NINFER_TEST_ARTIFACT=<27B v3-plain artifact>`.
**129/135 passed, 1 skipped, 5 failed** (814 s, 2026-10-10 15:04-15:18Z;
log `/tmp/lk-ctest.log`).

All Flash-Next tests pass, including the real-model ones:
`ninfer_qwen3_8_flash_next_real_test` (144 s) and
`ninfer_qwen3_8_flash_next_graph_real_test` (83 s) load and run the 125B
model from the v3_fork entry.

| # | test | result | triage |
|---|---|---|---|
| 36 | qwen3_5_loading_real | skipped | needs an explicit `--artifact` argument; not a failure |
| 43 | qwen3_5_prefix_real | failed | registered template token-count goldens (16/18) vs the 27B artifact's registered template; host-side golden drift, Qwen3.5 family |
| 46 | qwen3_5_dflash2_real | failed | "missing component dflash2" — the 27B v3-plain artifact has no DFlash2 companion component |
| 47 | qwen3_5_moe_real | failed | requires the 35B MoE artifact, which is not on this machine |
| 48 | qwen3_5_dflash_real | failed | "missing component dflash" — no DFlash companion in the 27B artifact |
| 123 | hyperconnection | failed | `std::bad_alloc`: arena exhaustion (`end > cap_`) in `DeviceArena::alloc` from `finish_mix`; lk's rewritten test/op pair only (the fork's older test passes on this box). The 1,024 B slack in `hyperconnection_mix_workspace_capacity_bytes` leaves a sub-KB margin on the general (non-fused) FP8 route. The real 125B tests prove the model-level path is unaffected. Upstream fix candidate: size the capacity from the actual allocations |
## Artifact gate (run-list item 1)

Probe: lk's own load-plan test
(`./build/tests/ninfer_qwen3_8_flash_next_load_plan_test`, env
`NINFER_QWEN38_FLASH_NEXT_WEIGHTS`; pure directory/binding check, no device
allocation; exit 77 skips when unset).

- **fp8mtp entry: REFUSED.**
  `model.language_model.layers.0.mlp.shared_expert.down_proj.weight:
  representation does not match its mathematical value type`. lk's load plan
  requires the shared-expert down projection to be exact BF16; the fp8mtp entry
  stores it row-scaled FP8 (this fork's load plan allows either).
- **v3_fork entry: ACCEPTED.**
  `~/models/qwen3_8_flash_next_v3_fork/qwen3_8_flash_next_125b_a6b_nvfp4.ninfer`
  (entry + 4 parts, 126 GB): `text_logical_parameters=125743702913`
  (124B..126B check), device 77.8 GB, MTP 83.1 GB (NVFP4 drafter banks), full
  83.3 GB, PLE table file-backed 51.2 GB (FP8, 320001536 x 160).

**Decision: all four arms serve the v3_fork NVFP4 entry.** The run list's
prescribed separate conversion (re-encode the 44 `shared_expert.down_proj`
tensors as BF16, a full ~127 GB re-conversion) is not needed: an artifact both
binders accept already exists. Both trees load it; the parity question is
engine vs engine on one artifact, not artifact vs artifact.

## Parity protocol (run list item 2)

DGPP's `scripts/serve_load.py` (HawkBearPig/dgpp checkout, `bench_stream` alongside):
five classes (prose, code, json, math, chat), C1/C2/C4, 256 output tokens, greedy,
thinking off, 3 repetitions per class-concurrency (45 phases, 105 requests per arm),
caches dropped before each server start. Metric: wall tok/s including prefill,
averaged over all classes.

Serve flags, all arms (run-list prescription, same as the October 3 rows):
`--max-context 73728 --max-concurrency 4 --kv-dtype fp8 --preserve-thinking --lm-head-draft
--host-kv-mib 0 --prefill-chunk 4096` plus the arm's spec:

| arm | spec flags |
|---|---|
| lk K=3 | `--spec mtp --draft-tokens 3` |
| lk adaptive | `--spec mtp` (lk default 3..7 drafts) |
| lk K=1 | `--spec mtp --draft-tokens 1` |
| ours K=3 | `--spec mtp --draft-tokens 3` (this repo at head, same day/box) |

Notes:
- The run list prescribes `--host-kv-mib 0 --prefill-chunk 4096` for the October 3 rows;
  the committed October 3 driver (`block_i.sh serve_args`, tree `05ed1f2a`) does not carry
  those two flags. Read the new scoreboard primarily as lk-vs-ours (identical flags, same
  day); cross-campaign comparison with the committed October 3 rows is approximate.
- `--max-context 73728` and no `--vision` keep the harness comparable with the October 3
  scoreboard (short prompts, text-only classes). The canonical production profile
  (262144 + vision) is exercised by the test cycle, not by this benchmark.
- October 3 reference (this box, our tree `05ed1f2a`, `profiles/bench/gb10/parity-2026-10-03/`):
  ours K1/K2/K3 at C1/C2/C4 = 48.42/72.94/107.07, 56.90/82.37/116.68, 59.89/91.53/121.39;
  DGPP 47.29/68.24/91.33; acceptance 1.803/2.425/2.917.

### Results (2026-10-10 15:39-16:14Z; all four arms rc=0, 45/45 phases)

All arms: v3_fork NVFP4 entry, identical serve flags, same day and box. lk tree
`8f574ee4` + the four `share/lk-gb10` patches (including the rope fix); ours at
head `91a1405e` (binary built Oct 5).

Wall tok/s, prefill included, mean over the five classes (per-class tables in
`results/*.txt`):

| arm | C1 | C2 | C4 |
|---|---|---|---|
| lk K=1 | 29.4 | 47.1 | 67.0 |
| lk K=3 | 37.6 | 56.3 | 67.9 |
| lk adaptive 3..7 | 39.2 | 58.0 | 69.0 |
| ours K=3 | 40.9 | 63.0 | 85.7 |

Ours K=3 vs. lk K=3 (same spec, same artifact): +3.3 / +6.7 / +17.7 tok/s at
C1/C2/C4 (+8.8% / +11.8% / +26.1%); vs. lk's best arm (adaptive): +4.4% /
+8.5% / +24.2%. The gap is C4-driven: on code/json/math at C4 ours holds
94.5/99.7/94.5 where lk holds 71.9/83.2/74.5; at C1 the arms are within
1.7 tok/s of each other. Draft depth on the lk side: K=1 to adaptive lifts
C1 29.4 to 39.2 but leaves C4 at 67.0 to 69.0 (C4 rounds are
bandwidth/MoE-bound, not acceptance-bound).

Artifact context: the October 3 rows above ran the fp8mtp entry, which lk's
load plan refuses (artifact gate, above), so no lk arm could serve it. Ours
on the v3_fork entry here (40.9/63.0/85.7) versus ours on fp8mtp on October
3 (59.89/91.53/121.39): the NVFP4 entry carries roughly twice the projection
bytes and costs ~30% at C4 on this engine. The engine-vs-engine question is
the table above on one artifact.

## k_sweep (run list item 3)

`tools/gb10/k_sweep.sh` protocol (16 natural-corpus streams, greedy, 1024 tokens, C1),
`KS="1 3"`, both arms on the v3_fork NVFP4 entry (the original plan's ours-on-fp8mtp arm is
not possible: lk's load plan refuses fp8mtp, and a mixed-artifact acceptance comparison would
confound drafter with quantization). Two passes under one shared output name `ksweep-parity`:
first with `SERVE_BIN` pointed at the lk build (arms `lk`), then at this repo's build (arms
`ours`).

Results (2026-10-10 16:17-16:54Z; outputs under
`profiles/bench/gb10/step7a-ksweep-parity-*`):

| run | acceptance | drafted | accepted | tok/s (sum over 16 streams) |
|---|---:|---:|---:|---:|
| lk k1 | 0.7545 | 9302 | 7018 | 27.8 |
| lk k3 | 0.5437 | 18643 | 10137 | 33.1 |
| ours k1 | 0.7564 | 8838 | 6685 | 29.2 |
| ours k3 | 0.5799 | 17790 | 10317 | 36.4 |

K=1 acceptance is nearly identical (0.7564 vs. 0.7545) on the shared drafter
weights. At K=3 ours holds 0.5799 vs. 0.5437: lk drafts more tokens per step
and loses acceptance more steeply with depth. The C1 parity-protocol gap
(39-41 vs. 29-39 tok/s by arm) is consistent with these acceptance numbers.

## Batch invariance (run list item 4)

lk build, C4: the same four requests replayed twice (fresh server per round, caches
dropped between rounds); outputs compared for exact equality. lk's `2fa4c756` claims a
row's result does not depend on its batch. This box: our branch is not invariant at C4
(known).

Results (2026-10-10 16:51-17:16Z):

- Run-list check (two fresh servers, caches dropped, K=3, thinking off):
  **FAIL** — prose/code/json matched exactly between rounds; math diverged
  (844 vs. 833 chars, splitting from the first sentence).
- Attribution, same build, same flags, v3_fork entry:
  - C1, one server, sequential same-prompt pairs: **deterministic**
    (4/4 exact matches).
  - C4, one server, the same 4-request batch back-to-back: **non-invariant**
    (prose and code diverged; json and math matched).
  - C4, two fresh servers with dropped caches: **non-invariant** (math).

C4 greedy replay is not reproducible on the lk master build (which includes
`2fa4c756`), and the divergence is C4-specific: C1 is clean. Whether the
cause is batch composition, KV page state, or execution timing is an
lk-side question; the operational fact is that the batch-invariance claim
does not hold on this box, matching this branch's known C4
non-invariance.
