# Spill acceptance hang — investigation dossier (2026-10-29)

State: the 27B tiering acceptance suite (`ninfer_qwen3_5_native_transactions_test none`,
artifact `out/qwen3_8_27b_exl3.ninfer`) fails in `state_spill_roundtrip` at the first
spill's `settle()`: the driver's `wait<ContextProgress>` times out after its 2-minute
deadline. Two real defects on the path are already fixed and pushed (862b12cb): the
`finalize_mappings` spill-store wipe, and four authored-blind fixture configuration
errors. The remaining hang is the open question.

## Verified working (each confirmed by live tracing this session)

1. `try_spill_checkpoint` assembles and submits the transaction
   (`src/models/qwen3_5/session/reclaim.cpp` ~560-618): `use_spill_store` set, spill
   actions staged, `submit_context` called.
2. The executor runs the spill: `RankProgramImpl::start`
   (`src/models/qwen3_5/program/lifecycle.cpp` ~93) → `context_actions->submit` → the
   actions execute; `context_ready.record(device.transfer_stream())`.
3. `RankProgramImpl::progress` (`src/models/qwen3_5/program/execution.cpp` ~10) passes
   its gate immediately for every context operation (including the spill), calls
   `observations()`, and emits `ResultReady{ContextResult}` + `DeviceRetired` for the
   spill's ticket. Observed live: gate passes right after each start, tickets 1..6.
4. The session receives and dispatches both events
   (`src/models/qwen3_5/session/execution.cpp` poll → `receive_result` →
   `receive_context_result`; `DeviceRetired` → `retire` → `retire_context`). Observed
   live: both events dispatched for the spill's ticket.
5. `retire_context` (`src/models/qwen3_5/session/context_operation.cpp` ~50) runs
   `complete_spill`, sets `out.complete = out.published = true`, resets
   `context_operation_`, and pushes the `ContextProgress` into the session's `events`.
   Observed live for tickets 1/3/5 (capture + two demotions) and — in one run — for
   ticket 6 (the spill): the driver's pump printed the received event as
   `variant=5` = ContextProgress.

## The contradiction

- In one run the driver's pump received the spill's ContextProgress (variant 5) and
  pushed it into `events_`, yet the waiting `settle()` still hit its deadline.
- In the next run, with only a print added *before* the settle, the pump was never
  entered at all after the settle began (zero pump calls for ~2 minutes).
- The stage marker after `settle()` never prints in any run; the marker before it does.

So the same binary shape behaves nondeterministically at exactly this point. This is
not a missing link in the happy path — every link was observed working at least once —
it is a timing/lifecycle subtlety in or around:

- `EventDriver::wait` / `pump` (`tests/models/qwen3_5/native_fixture.h` ~170-291):
  the check-events-then-pump loop and its 2-minute deadline.
- `SessionImpl::poll` (`src/models/qwen3_5/session/execution.cpp` ~250): the
  `submit_releases()` → events-check → `executor().poll()` loop, and the
  `release_ticket` / `after_releases` interaction with the abort's residency releases
  (the test aborts the writer before demoting, which queues releases; the release
  rides a context frame submitted from inside `poll`).
- `LocalExecutor::poll` (`src/runtime/executor/local_executor.h` ~115): the
  Discarding-flight filter loop and the flight phase machine around
  ResultReady/DeviceRetired for Context frames.
- The interaction of a just-submitted release context frame with the completion of the
  preceding spill context frame (single context frame slot; `pool_.try_reserve`).

## Exact review question

Given the verified chain above, why can the driver's `wait<ContextProgress>` miss a
ContextProgress that the pump demonstrably received (variant 5, pushed to `events_`),
and why can the pump stall entirely for the remainder of the deadline in a near-identical
run? Name the concrete interleaving (release frame vs spill frame vs deadline edge) and
the minimal fix — in the engine if the interleaving is real, in the fixture if the
driver's loop is at fault.

## Files to read (in this order)

1. `tests/models/qwen3_5/native_fixture.h` — wait/pump/settle loop.
2. `src/models/qwen3_5/session/execution.cpp` — poll loop, `submit_releases` interplay.
3. `src/models/qwen3_5/session/sequence.cpp` — `submit_releases` (release frame submit).
4. `src/models/qwen3_5/session/commit.cpp` — `receive_result`/`retire` release_ticket
   branch and `after_releases` re-queue.
5. `src/runtime/executor/local_executor.h` — poll filter loop, flight phase machine.
6. `src/models/qwen3_5/program/execution.cpp` — progress/observations gating.

## Reproduction

```
NINFER_TEST_ARTIFACT=out/qwen3_8_27b_exl3.ninfer \
  ./build/tests/ninfer_qwen3_5_native_transactions_test none
```

Fails after ~3.5 minutes (earlier stages + 2-minute settle deadline). Requires the RTX
5090 lane (Qwen3.5 compute-capability gate); the GB10 workstation cannot execute it.
