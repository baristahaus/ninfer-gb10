# Slot-flag test: `--device-state-slots 8` vs the wave replay admission hold (2026-09-30)

Opus's decision test for the reproducible A/B wave-1 admission hold: rerun the 4-request
C4 workload on one build with `--device-state-slots 8` (default = max-concurrency = 4).
All 4 admit at once → slot count is the cause. Hold persists → trace the admission code.

## Setup

- Build: master `e88b47c0` (post-#36 merge; engine code = the A/B fix tree), 22:31Z.
- Flags: A/C4 measured-pass flags + `--device-state-slots 8` + `--log-level debug`:
  `--max-context 73728 --max-concurrency 4 --kv-dtype fp8 --preserve-thinking --spec mtp
  --draft-tokens 1 --lm-head-draft`. Total device state slots = 4 lanes + 8 = 12
  (default total would be 8).
- Load: `serve_load.py --concurrency 4 --classes prose --max-tokens 256 --repeat 2`
  (one 64-token warm request, then two 4-request waves; wave 2 resends wave 1's 4
  prompts, so all 4 of wave 2 are cached-response replays).

## Result: the hold persists. Verdict branch: the cause is elsewhere.

Wave 1 (4 fresh/root requests): clean — all 4 admitted (TTFT 0.24–1.49 s).
Wave 2 (all 4 = `private_response_replay`, 7-token prefill each): 2 admitted
(queue_wait 0.00 / 0.17 s), 2 held (queue_wait 8.85 / 9.29 s), `running 2 | waiting 2`
for ~10 s — the A/B signature, unchanged by doubling the extra checkpoint capacity.
(In the A/B runs the hold fell on measured wave 1 because the warm-up pass, a separate
serve_load invocation, had already cached the 4 responses; same mechanism, shifted one
wave.)

## What the request log shows during the hold

`context_cache` snapshot at 22:32:03Z (`running 2 | waiting 2`):

- `occupancy.device_state_slots = 12` — **the pool is full (12/12)**. In the A/B runs
  (default, total 8) the same window shows `8/8`. Retention scales with pool size:
  the extra 4 slots just get filled with retained state.
- `state_operations.forks = 4` in the 5 s window (2 replay requests × 2 state forks
  each); wave 1's window shows 8 forks for its 4 requests. The forked checkpoints plus
  the retained originals are what occupy the pool.
- `selections.private_response_replay = 2`, `reused_prompt_tokens = 58`
  (2 × 29-token prefix hits), `pressure.historical_fork_hits = 2`.
- `occupancy.shared_active_references = 2`; `shared_owners_degraded` fires at most once
  (in wave 1's window) and never frees device slots during the hold.
- KV occupancy is flat and small the whole run (26 main / 26 backend pages, 0 host KV)
  — KV is not the resource in question.

Per-request (wave 2, request_done): req6/req7 admitted (queue_wait 0.00/0.17 s),
req8/req9 held (queue_wait 8.85/9.29 s); all four `prefix_reuse_path =
private_response_replay`, `prefix_cache_hit_tokens = 29`.

## Verdict and handoff

Per the agreed branch: **hold persists → not the slot count → trace the admission code.**
The data says admission is starved by retained forked checkpoints, not by the slot
ceiling: the pool saturates at whatever size it is given, and no eviction/degradation
frees device slots while the 2 admitted requests run. Artifacts: this directory
(`server.log` debug, `request.jsonl` with the full `context_cache` snapshots,
`load.txt`); A/B baseline at `../run*/request.jsonl`.
