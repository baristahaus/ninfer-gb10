# Admission-block counter rerun: default slots, `aff11e8e` (2026-09-30)

Opus's counter test: build master + `aff11e8e`, rerun the slot-flag workload at the
**default** slot count (no `--device-state-slots`; everything else identical, incl.
`--repeat 2`), and report `context_cache.admission_blocked` from the throughput events
during the `running 2 | waiting 2` window.

## Build and unit tests (GB10, sm_121a)

- Build: clean, 377/377 targets, 198 s incremental (22:37–22:40Z).
- ctest: **139/139 passed, 0 failed** (7 skipped, artifact-gated) — includes the updated
  `ninfer_request_log_test` checking the `admission_blocked` per-interval serialization.
  This is the first compile/run of the engine + request-log changes (Opus could not build
  on their side).

## Run

- 23:15–23:16Z, master + `aff11e8e`, default slots (total 8), C4 flags, `--log-level
  debug`, warm + 2 waves. Hold reproduced: wave 2 (4× `private_response_replay`) → 2
  admitted, 2 held (TTFT 9.19 / 9.39 s; queue_wait 9.06 / 9.26 s),
  `running 2 | waiting 2` for ~10 s.

## `context_cache.admission_blocked` (per-interval deltas)

| interval | scheduler | ctx_tx | unsettled_fork | no_free_lane | no_feasible_plan |
|---|---|---:|---:|---:|---:|
| 23:15:56 | running 4 | 0 | 0 | 0 | 0 |
| 23:16:01 | running 4 | 0 | 0 | 0 | 0 |
| 23:16:06 | running 4 | 0 | 0 | 0 | 0 |
| **23:16:11** | **running 2, waiting 2** | 0 | **1** | 0 | 0 |
| **23:16:16** | **running 2, waiting 2** | 0 | 0 | 0 | 0 |
| 23:16:21 | running 2 | 0 | 0 | 0 | 0 |
| 23:16:26 | running 2 | 0 | 0 | 0 | 0 |

**`unsettled_state_fork` is the only gate that fired: 1 block attempt in the first hold
interval, 0 in the second, and every other gate at 0 for the whole run.**

Reading with the agreed tree: the held head was stopped by the unsettled-StateImage-fork
gate (`program.has_unsettled_state_fork()` in `resource_manager.h`). `no_feasible_plan = 0`
rules out the planner-refusal hypothesis for this run; `no_free_lane = 0` says the block
was not the lane count; `context_transaction = 0` says no open context transaction held
it. The single count (rather than one per round) is consistent with the head being
re-inspected only on scheduler state changes, not every round — the "what re-opens the
fork" question is left to the admission-code trace.

## Incident (recorded per the fallback rule)

- 22:44:50Z: the first attempt of this rerun crashed at startup with
  `FATAL ... Flash-Next weights exceed free GPU memory` because it started while the ctest
  suite (launched concurrently) still ran its GPU tests; the job's health loop had no
  liveness check, so it spun to the 1800 s job timeout instead of failing fast.
- Fallback 23:15Z: rerun after ctest completed (139/139 at 22:49:15Z), with a liveness
  check added to the health loop. The invalid attempt left zero-byte artifacts, discarded.
