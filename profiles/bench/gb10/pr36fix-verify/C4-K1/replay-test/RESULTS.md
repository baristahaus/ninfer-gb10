# Replay-admission test verification (6a0629a8) — 2026-09-30/10-01

## Step 1: build + real test with the fix — FAILED (test setup, not the engine)

`6a0629a8` built clean (2.7 s incremental). Real test with the 32 GB artifact, fixed
engine (`dc622216` re-arm + counters):

```
Flash-Next concurrent replays were not admitted together: 0 replays,
longest queue wait 0.561982 s against a shortest request of 2.63528 s
```

The timing side of the check passed on the fixed engine (2 × 0.562 s = 1.12 s < shortest
total 2.64 s). The failure is the reuse count: **0 of 4 replays reused anything.**

With the test as pushed, the pre-fix engine fails the check identically: the re-arm hunk
only re-arms admission — it does not affect whether the engine retains or replays cached
state, so the same `0 replays` is guaranteed. A redundant build + GPU run would not change
the conclusion.

## Diagnosis: the test's prompt path never participates in the context cache

`exercise_replay_admission` builds its prompts with `engine.prepare_tokens(vector)`.
That API is the raw-token measurement path — `engine.h`:

> `// Raw token input is retained for repeatable correctness and performance measurement.`

Raw-token requests produce no structural cache candidates: the engine never retains a
replayable state for them, and never materializes from one. The message-based path
(`engine.prepare(PromptInput)` — what the serve handler uses) does:

| path | warm | resub #1 | resub #2 |
|---|---:|---:|---:|
| `prepare_tokens` (27-tok fixture prompt) | 0 | 0 | 0 |
| `prepare(PromptInput)` (system+user messages) | 0 | **76** | **76** |

76 = the full message-templated prompt restored from retained state.

Ruled out as the cause (all showed 0 reuse): KV capacity (auto/explicit), max_context
(512/73728), MTP config (draft 1/3, head Optimized/Full), device-state slots (4/8), CUDA
graphs (on/off), prompt length (27/54/81), resubmit delay (0/250/1000/3000/6000 ms),
output length (8/64), model stop defaults (on/off). The serve-level repro (chat
completions, same artifact) reuses normally — consistent with the message-path mechanism.

## The test fix

Build the four prompts as chat messages and submit them through `engine.prepare(PromptInput)`:

- 4 distinct system+user message sets (distinct user text), warmed once each, then all 4
  resubmitted at once — the current 4-variant shape, only the prepare path changes;
- keep the relative timing criterion (longest queue wait < half the shortest total);
- the `reused_prompt_tokens != 0` per-request check then measures real reuse.

Probe output for the 4-variant message pattern (validation that the fixed shape works):
see `probe4-messages.log`.

## Environment

GB10 (sm_121a), artifact `qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer` (32 GB),
probe binary linked against the `6a0629a8` engine build, one GPU job at a time.
