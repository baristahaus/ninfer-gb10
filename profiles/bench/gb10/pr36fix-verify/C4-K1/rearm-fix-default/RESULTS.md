# Re-arm fix check: admission counter repro at default slots (2026-09-30)

Fix `dc622216` (re-arm admission when a blocking StateImage fork settles) + counters
`645b0a9c`. Same workload as the pre-fix baseline (`C4-K1/slot-flag-default/`, 23:15Z):
default slots (no `--device-state-slots`), C4 flags, `--log-level debug`, warm + 2 waves,
`--repeat 2`.

## Result: the hold is gone

- No `running 2 | waiting 2` window. Wave 2's 2 replays were admitted within the first
  interval after wave 2 arrived (23:35:40: `running 4 | waiting 0`, the two prefills batched
  with the running decodes at batch 3.57, decode 72.4 tok/s).
- `waiting` never left 0.
- `admission_blocked.unsettled_state_fork` fired **2 once** (23:35:40) — the two wave-2
  heads were each inspected once while a fork was still open — then 0 for the rest of the
  run. The gate still detects the open fork; the re-arm clears it within the same interval.

| run | wave-2 window | held TTFTs |
|---|---|---|
| pre-fix (23:15Z, `slot-flag-default`) | `running 2, waiting 2` × ~10 s | 9.06 / 9.26 s |
| fix `dc622216` (23:35Z, this dir) | `running 4, waiting 0` | 101 / 306 / 544 / 752 ms |

Both fix-run expectations met: no hold window, held TTFTs < 1 s.

## Also this round

- ctest (default suite, with fix): 139/139 passed.
- `ninfer_qwen3_8_flash_next_real_test` with the 32 GB artifact (the per-order two-row
  decode check): **passed** (77 s).
- Pre-fix engine (re-arm hunk reverted, new test kept) + real test: **passed** (82 s).
  Per Opus's stated criterion, the new per-order check is therefore **not catching this bug**
  in its current form. Two candidate causes, both need Opus's call:
  1. The fixture's resumed request never forks (the stated unverified assumption) — the fork
     gate is simply not exercised.
  2. The first `submit` drives the resumed request's materialization/first round before
     returning, so the fork settles before the second `submit` enqueues the cold request.
     The cold head is then never inspected inside the open-fork window. [inference —
     consistent with the repro, where the blocked heads were enqueued in the same ingress
     burst as the fork opening]
  Sequential public-API submits apparently cannot place a head inside the open-fork window;
  a fixture that enqueues the cold request in the same boundary as the fork opening would
  be needed, or the serve-level counter repro remains the regression check for this bug.
