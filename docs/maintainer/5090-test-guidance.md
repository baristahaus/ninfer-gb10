# 5090-test: open engine bugs from the sm_120a test battery

This branch carries the repro programs and guidance for the two open engine bugs surfaced by the
RTX 5090 (`sm_120a`) 27B extended test battery. The bugs are real correctness issues that need
engine-side work; they were characterized and minimally reproduced on the 5090 but not fixed.

**Status of the battery that found them:** clean `sm_120a` build (zero compile bumps, CUDA
13.4 / V13.4.92), 141/142 CTest pass (the 1 failure is the hardware canary below), real-model
GPU tests and serving smokes pass on both 27B artifacts except the two open bugs. The
environment was restored to its pre-battery state.

## The two open bugs

| Bug | Beads | Symptom | Artifact |
|---|---|---|---|
| #7 — vision + reuse flag divergence | `ninfer-gb10-hdt` | `allow_prefix_reuse` changes the computation with no cache hit | groupwise flips the greedy token; nvfp4 below the margin |
| #8 — host-restore state materialization | `ninfer-gb10-dn9` | `state_h2d` stays flat on restore (KV pages upload, state images do not) | nvfp4 + optimized head only; groupwise (Full head) passes |

Both need the 27B artifacts (groupwise and nvfp4) and a GPU. They are open to whoever picks the
beads issue up; the repro programs and the code-path references below are enough to start cold.

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

**Why it is a bug.** `allow_prefix_reuse` is a *participation* flag: it gates whether the
continuation is *published* for later reuse. It must not change the *computation* of the current
request. When there is no reuse, both flag values must produce the identical prompt state.

**Code path (the gate).** `src/models/qwen3_5/program/planning/request_plan.cpp:256-257`:

```cpp
base->summary.publish_continuation =
    options.allow_prefix_reuse && prompt.identity.reusable && context_cache.enabled;
```

When `publish_continuation` is true, the plan additionally:
- assigns `prefix_digests` and sets `prefix_identity_tag` (`request_plan.cpp:321-324`);
- builds the `capture_groups` via the `add_capture` lambda (`request_plan.cpp:326-342`);
- and the prefill copies the flag onto the request (`prefill.cpp:646`), which the commit path
  uses to decide the finish disposition (`transactions/commit.cpp:598`).

So the flag changes the *planning* (page allocation, state layout, capture bookkeeping) before
any reuse decision. That is the perturbation.

**Hypothesis (to confirm, not assumed).** The vision prompt's identity is what makes
`prompt.identity.reusable` change the capture plan's shape: the media tokens put the prompt on a
planning branch that a text-only prompt does not take, so the `publish_continuation` flag flips a
different materialization shape for vision than for text. The materialization path then computes
a numerically perturbed prompt state. The fix direction: make the materialization path
*independent* of `publish_continuation` when there is no reuse — the flag should gate
publication, not the computation.

**Next steps.**
1. Run the repro: `NINFER_TEST_ARTIFACT=<artifact> ./build/tests/ninfer_qwen3_5_repro_vision_reuse_divergence`
   on both artifacts. Confirm: groupwise diverges, nvfp4 identical.
2. Diff the materialization plan (page allocation, state slots, `capture_groups`,
   `prefix_digests`, `prefix_identity_tag`) for `allow_prefix_reuse=true` vs `false` on the
   vision prompt — that pins exactly which field diverges. Add a trace at
   `request_plan.cpp:321` and at the prefill's plan copy (`prefill.cpp:646`).
3. Confirm the same diff is *empty* for a text-only prompt (validates the vision-specific
   hypothesis).
4. Locate the first divergent value in the prompt state (KV, state image, or the vision
   transient). The perturbation is numerical, so a `memcmp`-style diff of the prefill output
   between the two flag values is the discriminating experiment.
5. Fix so that `publish_continuation` does not change the computation when there is no reuse.
   This is an upstream bug (merge base `e31bc99b`); the fork barely touched these files. Fix the
   fork and upstream it.

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

Groupwise (Full head) passes the identical scenario; only the nvfp4 artifact with the optimized
proposal head fails.

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

**Hypothesis (to confirm, not assumed).** The optimized proposal head has a different state-image
layout (slot count, or a split across multiple state images) than the Full head. The pressure /
demotion path handles the Full head's layout but not the optimized head's, so the optimized
head's state image either (a) stays on Device (residency `DeviceOnly` — never demoted), or (b)
gets a residency label the `HostOnly` gate does not match. Either way `host_state_restore` stays
null.

**Next steps.**
1. Run the repro: `NINFER_TEST_ARTIFACT=<nvfp4> ./build/tests/ninfer_qwen3_5_repro_host_restore_state`
   and confirm `state_h2d` stays flat (the failure line). Run on groupwise and confirm it passes.
2. **Discriminating experiment A** — dump the state image's residency right after the pressure
   pass (before the restore). Add a trace at `materialization.cpp:646` (the `residency` read).
   Is it `HostOnly`, `DeviceOnly`, or `Both` for the optimized head?
   - `DeviceOnly` → the demotion path did not demote the optimized head's state image. Look at
     the pressure / demotion path for a layout-specific branch.
   - `Both` → the demotion ran but the residency label does not match the `HostOnly` gate. Look
     at the gate at `materialization.cpp:657`.
   - `HostOnly` → the gate matched but the upload path dropped it. Look at `materialization.cpp:848`
     and `state_store->begin_host_to_device`.
3. **Discriminating experiment B** — compare the demotion's `state_d2h_count` for the optimized
   vs Full head. If the optimized head's `state_d2h` did not increase during the pressure pass,
   the state image never went to Host (confirms hypothesis (a)).
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

- Bug #7 repro: exit 0 = identical token sequences (correct), exit 1 = divergence (bug).
- Bug #8 repro: exit 0 = `state_h2d` increased (correct), exit 1 = `state_h2d` flat (bug).

Both print the full stat / token lines on failure so the next session can pick up the evidence
without re-deriving it.

## Out of scope for this branch

- The fixes themselves (they need the 27B artifacts and a GPU, which the GB10 workstation does
  not have). This branch is the *handoff*: repros + guidance + beads issues.
- The hardware fix (machine-side, owner is the 5090 operator).
- The MoE real test (needs a v3 35B-A3B artifact; the local one is a v2 container) and the
  Flash-Next real tests (need the 125B-A6B artifact). Both are skipped by design.
