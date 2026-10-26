# 5090-test: engine bugs from the sm_120a test battery (closed)

This branch carries the repro programs and guidance for the two engine bugs surfaced by the
RTX 5090 (`sm_120a`) 27B extended test battery. Both are closed: bug #7 is fixed in the engine
(`301d28f9`), and bug #8 was resolved as not an engine defect (the scenario probed the wrong
reuse frontier). The repros remain as regression guards.

**Status of the battery that found them:** clean `sm_120a` build (zero compile bumps, CUDA
13.4 / V13.4.92), 141/142 CTest pass (the 1 failure is the hardware canary below), real-model
GPU tests and serving smokes pass on both 27B artifacts except the two open bugs. The
environment was restored to its pre-battery state.

## The two battery bugs — both resolved

| Bug | Beads | Status | Resolution |
|---|---|---|---|
| #7 — vision + reuse flag divergence | `ninfer-gb10-hdt` | **FIXED** (engine) | capture-frontier chunk splits are now a structural prompt property (`301d28f9`); validated on both artifacts |
| #8 — host-restore state materialization | `ninfer-gb10-dn9` | **RESOLVED** (not an engine defect) | the Engine's Host State restore is correct; the scenario probed the wrong reuse frontier — see the bug #8 section |

Both need the 27B artifacts (groupwise and nvfp4) and a GPU. The full battery re-run (below)
is green on both artifacts.

## Bug #7 — vision + reuse flag divergence (`ninfer-gb10-hdt`)

**Symptom.** A fresh engine, plain decode (no MTP), a vision prompt, temperature 0, the first
request in the engine (`reused_prompt_tokens == 0`, fresh vision encode) computes a different
prompt state when `allow_prefix_reuse` is true vs false:

```
allow_prefix_reuse=false -> 27775 383 279 2099 3766   ("Based on the image provided")
allow_prefix_reuse=true  -> 760 2099 18126 264 5215   ("The image displays a pattern")
```

The flag alone changes the computation. No cache hit, no MTP, no prior request. Text-only
prompts are unaffected. Both artifacts compute the perturbation, but only the groupwise (Q4/Q5)
margins flip the greedy token; the nvfp4 margins are large enough that the token is unchanged.

**Root cause (confirmed on the 5090).** `allow_prefix_reuse=true` builds capture groups whose
frontiers split the prefill chunk (`split_frontier = capture_frontier` in `prefill.cpp`). The
vision prompt carries a rewrite-checkpoint capture at token 75 (the user-message boundary), so
the reuse-enabled prefill ran two chunks [0,75) + [75,82) while the flag-disabled prefill ran one
chunk [0,82). The chunked prefill's floating-point arithmetic is chunk-decomposition-dependent
(two-stage GDN association and chunk-boundary continuation), so the split changes the computed
prompt state at ULP level; the first generated token's logits differ and prompts with tight
top-1/top-2 margins flip the greedy token. The capture offer/transaction itself is
computation-neutral (verified: split without offer still diverges, no split with the flag on
matches the flag-off chain exactly). The text-only control takes the same message-boundary split
(frontier 8 of 15) and stays identical only because its margin is wide - the perturbation is
prompt-independent, the *flip* is margin-dependent.

**Fix (landed).** Chunk splits at semantic capture frontiers are now a structural property of the
prompt: `RequestBasePlanImpl::split_frontiers` collects the identity's rewrite-checkpoint frontier
and every prepared cache opportunity whenever the identity is reusable and the context cache is
enabled, independent of `allow_prefix_reuse`. The staged prefill walks `split_frontiers` next to
the capture groups and splits the chunk at the earliest pending frontier. The flag gates only the
checkpoint/publication (capture groups and offers remain flag-gated); the chunk decomposition -
and with it the computed state - no longer depends on it. Both flag values now produce the same
chain on both artifacts; the flag-off chain on tight-margin prompts changes to the split-chunked
chain (both routes were always valid chunkings; they are now consistent).

Validated on the RTX 5090: the repro exits 0 on both artifacts (vision and text controls
identical), and the prefix real test's vision scenario - the check that originally caught the
bug - passes on the groupwise artifact. This is an upstream-inherited bug (merge base
`e31bc99b`); the fix should be upstreamed.

**Provenance note.** The exact token sequences above are from the 5090 battery on the groupwise
artifact. The repro program prints the sequences for whatever artifact it is run against; do not
assume they are identical to the ones above on a different artifact or engine configuration.

## Bug #8 — host-restore state materialization (`ninfer-gb10-dn9`)

**Symptom.** After a host demotion (the pressure pass verifiably increased the `d2h` counts), the
restored request reuses 316 tokens and uploads the KV pages from Host, but uploads **zero state
images** — `state_h2d_count` stays flat:

```
path=1 reused=316 outputs=2 state=0 degraded=1
```

Groupwise (Full head) and nvfp4 (optimized head) fail identically (corrected attribution — see
below).

**Corrected attribution (2026-10-12, verified on the 5090).** The original "groupwise passes"
claim was wrong: it came from the focused `pressure-resume` scenario, which maps to
`exercise_pressure_partial_spill_and_resume` - a different function from the host-restore check.
The artifact-aware repro runs the identical scenario on the groupwise artifact and fails the
same way (`state_d2h=1` after pressure, `state_h2d=0` after restore). Bug #8 is not
optimized-head-specific; the `host_state_restore` gate does not fire for either head.

**Code path (the gate).** `src/models/qwen3_5/program/transactions/materialization.cpp:657`:

```cpp
if (residency == StateReplicaResidency::HostOnly) {
    host_state_restore = state;
    ...
}
```

and the actual upload at `materialization.cpp:848`:

```cpp
if (host_state_restore) {
    ...
    state_store->begin_host_to_device(*host_state_restore, device.transfer_stream);
    ...
}
```

The KV pages are restored via `prepare_kv_restores` (`materialization.cpp:831-847`) and reach
`main_kv_h2d` / `backend_kv_h2d`. But `host_state_restore` is never set, so
`begin_host_to_device` is never called and `state_h2d` stays flat. `host_state_restore` is only
set when the state image's residency is exactly `HostOnly`.

**Resolution (2026-10-12, verified on the 5090, both artifacts).** Not an Engine defect. The
Engine's Host-to-Device State restore path is correct; the failing check probed the wrong reuse
frontier. Three facts, established with temporary state-store traces (handle indices, physical
slots, residency — since removed):

1. The pressure planner demotes the **typed rewrite checkpoint first**, by design:
   `pressure.cpp` — "A typed rewrite is the less destructive state relief while the endpoint
   remains usable." With `device_state_slots=1` (H) and `max_concurrency=1` (C), Device State
   capacity is `C + H = 2` physical slots
   (`resource-scheduling-and-context-cache.md` §5.1). The retained continuation held two
   Device-resident checkpoint images (endpoint on the active slot, rewrite on the H slot); the
   pressure demoted the rewrite (freeing the H slot for the pressure request's own state) and
   the endpoint kept riding the active-concurrency slot.
2. The original scenario's restore follow-up reproduced the assistant turn, so the identity
   match extended past the turn boundary to the **full cached endpoint** (316 tokens): a
   `PrivateEndpoint` reuse (path=1) binding the still-Device-resident endpoint zero-copy. Zero
   State uploads is the correct outcome for that shape — the endpoint never left the Device.
   (Reuse matching is frontier-exact: the follow-up probes its own structural frontiers against
   the catalog's published checkpoint frontiers.)
3. The Host State restore fires when a follow-up binds the demoted rewrite: a **regeneration
   follow-up** (only the user turn replaced) matches at the turn boundary (305 tokens), binds
   the `HostOnly` rewrite checkpoint, and the Engine uploads the State image plus the demoted
   KV pages from Host:
   `path=2 reused=305 state_h2d=1 main_kv_h2h=3 backend_kv_h2h=2`.

**Fix.** No Engine change. The repro and `exercise_host_restore` (`test_engine_prefix_real.cpp`)
were corrected to the regeneration shape plus a negative control:

- the pressure request (unchanged) demotes the rewrite State image and the KV pages to Host;
- the regeneration restore must reuse the turn boundary (`PrivateTurnClosure`,
  `reused < prompt_tokens`) and upload the State image and KV pages from Host (`state_h2d`,
  `main_kv_h2d`, `backend_kv_h2d` all increase);
- a repeat of the same regeneration prompt reuses the now-Device-resident committed
  checkpoints and must upload nothing further (all h2d counters flat) — the restore policy
  uploads a `HostOnly` source exactly once, never redundantly.

Both checks pass on the groupwise (Full head) and nvfp4 (optimized head) artifacts (exit 0).

**Provenance note.** The exact token sequences above are from the 5090 battery on the groupwise
artifact. The repro program prints the sequences for whatever artifact it is run against; do not
assume they are identical to the ones above on a different artifact or engine configuration.

## The hardware finding (not a code bug — machine fix)

The 5090 machine has a marginal host-bound memory path. Five collected corruption samples, all
bit 5 of byte 0 (XOR 0x20): four in `gdn_replay_fold_test`, one in a standalone pure-CUDA
discriminator with no fork code. Evidence: 600 GiB of host-only `memcmp` traffic → 0 flips;
300 GiB of GPU→host downloads → 1 stored bit-5 flip; every fold-test mismatch vanished on
re-download (GPU data correct; kernels clean per memcheck/initcheck/synccheck). Flip rate ≈ 1 per
100 GB of GPU→host traffic.

**Consequence for this work:** the two flaky tests (`gdn_replay_fold`, `kv_cache_append`) are
canaries for this hardware issue. Until the machine is fixed, treat any bit-exact test failure on
the 5090 as "re-download and re-check" before blaming code. If a repro run on the 5090 shows a
single-bit flip that vanishes on re-run, it is the hardware, not the engine.

**Machine fix (owner: the 5090 operator):** disable EXPO (run at JEDEC timings), run memtest86+,
re-run the discriminator. This is a machine fix, not a code fix. The bit-exact tests are correct
as written.

## Building and running the repros

Both repros are registered in `tests/models/qwen3_5/tests.cmake` as
`ninfer_qwen3_5_repro_vision_reuse_divergence` and
`ninfer_qwen3_5_repro_host_restore_state`. They are skipped under plain CTest (no artifact) and
run individually:

```bash
cmake --build build -j                      # build everything, including the two repros
NINFER_TEST_ARTIFACT=/path/to/27b.ninfer ./build/tests/ninfer_qwen3_5_repro_vision_reuse_divergence
NINFER_TEST_ARTIFACT=/path/to/27b.ninfer ./build/tests/ninfer_qwen3_5_repro_host_restore_state
```

- Bug #7 repro (FIXED): exit 0 = identical token sequences on the vision and text controls.
  The vision divergence and the text-only control both print their chains; a regression exits 1.
- Bug #8 repro (RESOLVED): exit 0 = the host-restore validation passed (the demoted State
  image came back from Host on the turn-boundary reuse, and the negative control uploaded
  nothing further); exit 1 = a check failed. Runs on both artifacts (the proposal head is
  selected from the artifact).

Both print the full stat / token lines on failure so the next session can pick up the evidence
without re-deriving it.

## Battery re-run results (2026-10-12, post-fix, DDR5-3600 JEDEC)

- Full CTest battery with the groupwise artifact: **exit 0** (136 passed, 14 skipped — the
  skips are the artifact-argument tests, component-absent tests, and Flash-Next tests). The
  hardware canaries `gdn_replay_fold` and `kv_cache_append` both pass deterministically on the
  fixed memory setup.
- nvfp4 artifact: the full prefix real test (all scenarios), score, loading, vision-workspace,
  and both repros pass.
- 35B-A3B MoE: the `/nfs` v2 artifact (`models--neroued--Qwen3.6-35B-A3B-NInfer`, container
  version 2) was upgraded in place with `tools/upgrade_ninfer_v2_to_v3.py` (weights preserved,
  `qwen3_6.jinja` installed). The MoE real test, the DFlash real test, and the loading test
  (vision + MTP, 22.2 GiB on device) all pass on the upgraded artifact.
- Serving smokes: `serve_thinking_preservation.py` (MTP, groupwise) and `serve_contract.py`
  (resident fork server, vision enabled) both pass.
- Test-suite fixes carried by this branch beyond the two bugs: the vision-workspace, DFlash2,
  MoE, and DFlash real tests are now artifact-aware (the proposal head is selected from the
  artifact, and component-scoped tests skip with exit 77 when the artifact lacks the
  component) instead of aborting or failing on mismatched artifacts. The prefix real test
  remains dense-scoped (its load-summary check requires `Qwen3_5ForCausalLM`); extending it
  to MoE artifacts is a separate task.

## Out of scope for this branch

- The hardware fix (machine-side, owner is the 5090 operator; the 2026-10-12 EXPO change was
  verified clean on the 5090 with the bit-flip discriminator - 900 GiB of traffic, zero flips).
- The Flash-Next real tests (need the 125B-A6B artifact, which does not fit the 32 GB card;
  NInfer has no host-RAM weight offloading). Skipped by design.
