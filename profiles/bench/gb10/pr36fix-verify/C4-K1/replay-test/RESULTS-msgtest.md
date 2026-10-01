# Chat-message replay-admission test (495c4995) — GB10 verification, 2026-10-01 (UTC)

## Runs

| # | engine | request length | result |
|---|---|---|---|
| 1 | fixed (`495c4995` incl. `dc622216` re-arm) | 64 outputs (test) | **PASS** (111 s) |
| 2 | pre-fix (`dc622216^` engine_core.h) | 64 outputs (test) | **PASS** (109 s) — expected FAIL |
| 3 | pre-fix | 512 outputs (probe) | **FAIL** — 4th replay held 16.528 s |
| 4 | fixed | 512 outputs (probe) | **FAIL** — 4th replay held 15.718 s |

Runs 1–2 are the two real-test invocations (4 distinct chat-message prompts, shared
system message, 4 state slots, 4 concurrency, MTP draft-3/Full, graphs on, 32 GB
fp8-mtp artifact). The message path reuses: the test's `replays == 4` held in every
run — the raw-token diagnosis from the previous round was correct.

## Finding: the 4th-replay hold is present on BOTH engines; the test cannot catch the fix in any configuration tried

Run 3 (pre-fix, 512 outputs) and run 4 (fixed, 512 outputs) — full logs in `probe-512.log`:

```
warm-0/1/2/3      reused= 0/12/12/12   (shared-prefix reuse on the warm wave)
resub-0           reused=29  total= 16.529 s  queue=  0.005 s
resub-1           reused=29  total= 16.374 s  queue=  0.166 s
resub-2           reused=31  total= 17.481 s  queue=  0.322 s
resub-3           reused=30  total= 27.169 s  queue= 16.528 s   <- held
check: replays=4 longest_queue=16.528 shortest_total=16.374 -> FAIL
```

`resub-3`'s queue wait equals `resub-0`'s total lifetime to the millisecond in BOTH
runs (16.528 s vs 16.529 s pre-fix; 15.718 s vs 15.718 s fixed): the 4th replay is
admitted exactly when the 1st completes. The first three replays are admitted on the
same ~160 ms stagger in both engines (0.005 / 0.166 / 0.322 s pre-fix;
0.005 / 0.167 / 0.323 s fixed) — the `dc622216` re-arm (fork-settlement rescue) does
not change this scenario at all.

This is a different admission situation from the serve repro, where the replays are
submitted while the original requests are still running: there, the pre-fix engine
held the replays and the fixed engine admitted them via the re-arm (counter repro).
Here — 4 replays submitted at once after the warm wave has fully completed, all 4
slots free — the 4th replay waits out the 1st's full lifetime on BOTH engines.
Whether that is a by-design constraint (materialization/admission gating) or a
remaining admission issue is for the engine owner to judge; it is outside the
`dc622216` fix's scope, which targets replays arriving during in-flight work.

## Consequence for the test

The 4-replay burst pattern (warms complete, then 4 replays at once) cannot
distinguish pre-fix from fixed at any request length tried: at 64 outputs both
pass, at 512 outputs both fail on the 4th-replay hold. Per the agreed decision rule
(pre-fix run passed), the C++ test is removed as a regression gate for the
admission hold; the server-level counter repro (replays arriving during in-flight
work) remains the check for the `dc622216` fix.

If a C++-level gate is still wanted, it must reproduce the serve situation: replays
submitted while the original requests are still running (e.g. submit the 4 replays
mid-decode of the warm wave, before any warm request completes). That shape has not
been probed.

The 4th-replay full-lifetime hold on the fixed engine (present in both engines)
should be adjudicated by the engine owner: by-design admission gating, or a
remaining issue worth its own fix.

## Environment

GB10 (sm_121a), CUDA 13.0.88, artifact `qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer`
(32 GB), one GPU job at a time. Probe: throwaway `probe_hold.cpp` (removed after runs).
