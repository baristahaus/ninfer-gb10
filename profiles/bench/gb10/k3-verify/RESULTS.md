# K3 verify - 2026-10-01

Binary: `efa6d2b216aaeb6e7af5a59f8e754de0b331f402d9fba40cc62880b2d3b622e9` (clean build of
`b90fd5a4`, 176 targets). Host: GB10, driver 580.173.02, CUDA 13.0.88. Artifact:
`out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer` (same as K1/K2).

## Verdict: FAIL - the real test crashes on the first MTP round

A K3 implementation bug, not a verifier environment issue. The crash is deterministic
(reproduced 3x: verify.sh, two manual runs).

| Check | Result |
|---|---|
| 1. `test_flash_next_ple_stage`, `test_mtp_round` (+ `gdn_replay_fold`, `flash_next_ple` regressions) | PASS |
| 2. Real test bitwise identical to K2 | FAIL - `WORKER CRASH: Flash-Next PLE gather missed the round's deadline` on the first MTP round (`exercise_mtp_and_prefix`, `ProposalHead::Full`, K=3) |
| 3. C4 K=1 warm serve, decode host ms/round vs K2's 8.93 | not run - the engine dies on its first MTP round |
| 4. `flash_next_ple_wait_staged` a few us per round in the nsys trace | not run - same reason |

## Root cause (established with temporary in-tree diagnostics; removed afterwards)

Mailbox event trace from the failing run (`real-test-diag.log`):

```text
stage thread start served=0
begin_round req=0 ans=0        <- prepare_graphs code-warm round (wrapped in begin/end)
served seq=1 tokens=4          <- the warm round's publish is gathered and acknowledged
end_round req=1 ans=1
check: req=1 ans=1             <- late was 0; the engine continues
begin_round req=3 ans=1        <- first request round: the host already sees TWO unacked publishes
served seq=4 tokens=4
end_round req=4 ans=4
check: late=1 -> throws "missed the round's deadline"
```

Causal chain:

1. `instantiate_graph_family` (program_impl.h:758-771) launches each instantiated graph once as a
   warm launch. For the MTP family that launches the MTP round graph - which contains the PLE
   `publish_ids` kernel - while the `PleGatherStage` is **inactive** (no `begin_round` around the
   `instantiate_graph_family(mtp_graphs, ...)` call at program_impl.h:11633. The code-warm round at
   11549-11557 is wrapped in begin/end/check; the instantiate-time launches are not).
2. The two MTP topologies' warm launches publish `request_sequence` 1->2->3. The host gather thread
   is sleeping (`active_ == false`), so no acknowledgement arrives; each in-graph
   `wait_staged` kernel times out at its 5 s deadline and sets `late = 1`. This also adds
   2 x 5 s of dead time to engine construction (visible in the 35 s run: load ~15-20 s,
   construction ~12 s, one fast round).
3. The first real round itself runs correctly: publish 3->4, the host thread wakes on
   `begin_round`, gathers seq 4, acknowledges `answer_sequence = 4`, and the round's wait passes.
   But the round's `check()` finds the **stale `late = 1` left by step 2** and throws - a false
   attribution of a prepare-phase timeout to the round.

Why the unit test missed it: `test_flash_next_ple_stage` exercises the handshake with a host-side
responder and eager single publish/wait pairs. It never runs the engine's instantiate-time graph
warm launches with the stage inactive, which is the only path that leaves the late flag stale.

## Proposed fix (for K3's owner; ~4 lines in program_impl.h)

Wrap the MTP family's instantiate call in the stage:

```cpp
if (speculative_backend == SpeculativeBackend::Mtp) {
    if (ple_gather_stage) { ple_gather_stage->begin_round(); }
    instantiate_graph_family(mtp_graphs, "MTP", device, prepare_representative);
    if (ple_gather_stage) {
        ple_gather_stage->end_round();
        ple_gather_stage->check();
    }
}
```

With the stage active, the warm launches are acknowledged (each wait passes in microseconds), no
late flag is set, and `check()` is clean. The ordinary and DFlash instantiate calls need nothing:
only the MTP body publishes (`flash_next_ple_publish_ids` is called only from
`mtp_decode_batch_body`). Alternative considered: keep the stage thread permanently active and
drop the begin/end deactivation; that also works but changes the designed poll/sleep semantics.

Note for the fix: after it, re-run checks 2-4 above. The stale-data side effect of the warm
launches (they copy whatever staging held into the PLE embeddings) is harmless for warm runs -
their outputs are not observed and each real round re-publishes and re-copies - so no further
action is needed there.

## Side data: K2 host NVTX breakdown (the optional follow-up ask)

From the k2-verify serve trace (`serve-c4k1.sqlite`, 540 decode rounds; total named-range time per
round):

| Range | ms/round |
|---|---|
| `decode.mtp.submit.ingress` | 7.43 - host-side PLE hash+gather and frame build; the work K3 moves off the critical path |
| `program.submit` | 9.17 (wraps the round submission, includes ingress) |
| `cuda_graph.launch` | 0.88 |
| `engine.commit_output` | 0.08 |

So of the 8.65 ms mean kernel-idle per decode round, the ingress scope accounts for the bulk -
K3's decode-host reduction target, to be confirmed by check 3 after the fix (K2 baseline 8.93
- Diagnostic edits removed; tree clean at `b90fd5a4`; clean real-test binary rebuilt.
- Fix landed as `6d851f69` (`fix(flash-next): activate the PLE gather stage during
  graph instantiation`) on 2026-10-01 and was verified during the K4a gate: the real
  test passes bitwise, and in the K4a C4 K=1 serve trace `wait_staged_kernel` ran 554
  times with a 10.7 us mean and 59.1 ms max - no 5 s timeout, so the late flag is no
  longer left stale.
- Issue `ninfer-gb10-nkw` closed; K4 verification continues under `ninfer-gb10-nb2`.
