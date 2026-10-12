# 5090-test: open engine bugs from the sm_120a test battery

This branch carries the repro programs and guidance for the two open engine bugs surfaced by the
RTX 5090 (`sm_120a`) 27B extended test battery. The bugs are real correctness issues that need
engine-side work; they were characterized and minimally reproduced on the 5090 but not fixed.

**Status of the battery that found them:** clean `sm_120a` build (zero compile bumps, CUDA
13.4 / V13.4.92), 141/142 CTest pass (the 1 failure is the hardware canary below), real-model
GPU tests and serving smokes pass on both 27B artifacts except the two open bugs. The
environment was restored to its pre-battery state.

## The two open bugs

| Bug | Beads | Status | Symptom | Artifact |
|---|---|---|---|---|
| #7 — vision + reuse flag divergence | `ninfer-gb10-hdt` | **FIXED** (see below) | `allow_prefix_reuse` changes the computation with no cache hit | both artifacts; groupwise flips the greedy token |
| #8 — host-restore state materialization | `ninfer-gb10-dn9` | **OPEN, re-attributed** | `state_h2d` stays flat on restore (KV pages upload, state images do not) | both artifacts, both heads |

Both need the 27B artifacts (groupwise and nvfp4) and a GPU. Bug #8 remains open; the repro
program and the code-path references below are enough to start cold.

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

**Hypothesis (superseded — the optimized-head layout was not the cause).** The state image IS
demoted to Host for both heads (`state_d2h=1` after the pressure pass), so the question is why
the restore gate at `materialization.cpp:657` does not match. Either the residency label after
demotion is not `HostOnly` (a `Both`/`DeviceOnly` label the gate misses), or the restored
request binds the state through a path that skips the `host_state_restore` selection entirely.

**Next steps.**
1. Run the repro on both artifacts: `NINFER_TEST_ARTIFACT=<artifact>
   ./build/tests/ninfer_qwen3_5_repro_host_restore_state`. Both currently fail with
   `state_h2d=0` on restore (the corrected expectation - the original "groupwise passes" note
   was wrong).
2. **Discriminating experiment A** — dump the state image's residency right after the pressure
   pass (before the restore). Add a trace at `materialization.cpp:657` (the `residency` read).
   Is it `HostOnly`, `DeviceOnly`, or `Both` after the verified demotion?
   - `DeviceOnly`/`Both` → the demotion ran but the gate at `materialization.cpp:657` does not
     match the actual residency; the restore path skips the state upload.
   - `HostOnly` → the gate matched but the upload path dropped it. Look at
     `materialization.cpp:848` and `state_store->begin_host_to_device`, and at how the restored
     request selects its state source (`selected_state` / `PrivateSourceMode`).
3. **Discriminating experiment B** — trace which state source the restored request binds: the
   demoted checkpoint's Host replica, or a device-resident fallback. If the request binds a
   device fallback without uploading, the accounting check is right but the binding path skips
   the Host materialization - decide whether the contract is "upload the Host state" or "any
   valid state source", and fix the gate or the check accordingly. Read `ninfer-gb10-10l`
   (own-source checkpoint demotion at the reusing follow-up's admission) first: it is the
   adjacent demote-vs-drop design question and a fix here may constrain it.
4. Fix so the optimized head's state images are demoted to Host and restored on resume. This is
   a host-replica-path bug; fix the fork and upstream it.

**Provenance note.** The `path=1 reused=316 outputs=2 state=0 degraded=1` line is from the 5090
battery on the nvfp4 artifact. `path=1` is `PrefixReusePath::PrivateTurnClosure`. The repro
program prints the full stat line for whatever artifact it is run against.

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
- Bug #8 repro (open): exit 0 = `state_h2d` increased (correct), exit 1 = `state_h2d` flat (bug).
  Runs on both artifacts (the proposal head is selected from the artifact).

Both print the full stat / token lines on failure so the next session can pick up the evidence
without re-deriving it.

## Out of scope for this branch

- Bug #8's fix (in progress; see the bug #8 section for the corrected attribution and the
  discriminating experiments).
- The hardware fix (machine-side, owner is the 5090 operator; the 2026-10-12 EXPO change was
  verified clean on the 5090 with the bit-flip discriminator - 900 GiB of traffic, zero flips).
- The MoE real test (needs a v3 35B-A3B artifact; the local one is a v2 container) and the
  Flash-Next real tests (need the 125B-A6B artifact, which does not fit the 32 GB card; NInfer
  has no host-RAM weight offloading). Both are skipped by design.
