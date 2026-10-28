# Branch cleanup and what to validate next (note for the 5090-lane collaborator)

Written 2026-10-28, master at `5335f64e`. This note is a handoff, not an authority. It can be
removed once the 5090 lane has acknowledged it.

## What happened in the cleanup

The shared repository carried 24 local branches and 18 remote branches, most of them work
records from September and October. On 2026-10-28 they were cleaned up:

- 22 local branches and 15 remote branches were deleted.
- Every deleted branch tip is preserved as an `archive/*` tag. The tags are pushed to the
  remote, so nothing is lost. Examples: `archive/verify-pr36fix-2026-09-30` (the ncu and MoE
  verification records, 96 files), `archive/twoFour-gb10-campaign` (the September campaign
  records), `archive/claude-brave-pasteur-hcmpip` (a 1,016-line design overview that was never
  merged).
- What remains as branches: `master`, `5090-test`, `q38fn-bank-shapes`, and the seven
  `upstream/*` provenance mirrors. `5090-test` stays because it is live work, not a record: it
  carries 98 commits of SP-7 strategy and tiering work that is not yet merged into `master`.

If you need any archived content, read it from the tag: `git log archive/<name>` or
`git checkout archive/<name>`.

## What landed on `master` since your last look

The October integration plus the following week of work:

- the Flash-Next streaming tool-call fix. Every OpenAI streaming request that produced tool
  calls failed with HTTP 500 on the merged tree; the fix restores byte-progress publication and
  terminal-call adoption. Tool benchmark went from 44 back to 87 (159/184, error rate 0.0).
- the GB10 test cycle: the full 262,144-token model window serves end to end, and the vision
  probe passes. The vision probe's embedded PNG in `tools/gb10/long_context.py` was malformed
  and is regenerated.
- the demote-then-restore compound fixture (`exercise_forced_own_source_compound`) and a
  determinism fix for the older pressure exercise, which depended on a prefill timing race and
  failed on a warm cache.
- the 125B load-plan test is registered and runs again (`tests/models/qwen3_8_flash_next_125b_a6b/`).
- SP-8-lite batch 1: the Flash-Next SHA-256 digest now re-exports the shared common
  implementation; the duplicate file is gone.
- the README is rewritten as a current-state review; the old README content moved to
  `HISTORY.md`.
- the native-quant plan is reconciled with `master`: A4 and B3 landed through the integration;
  what remains for a servable QAD-native artifact is the E2c binder port, the E2d PLE-consumer
  decision, the E2e oracle, and the serving evidence (3.5 to 6.5 working days by the recorded
  estimate).

One breaking change to know: the `--lm-head-draft` serve flag no longer exists on `master`.
The LM-head draft route is unconditional there. All `tools/gb10/` step scripts were updated;
if you carry scripts outside the repository that pass the flag, remove it there too.

## What to validate on the 5090

Your lane was restructured by the integration. The next `master` build on the 5090 should run,
in this order:

1. Clean build for `sm_120a`: `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release`
   and `cmake --build build -j`. The build accepts `sm_121a` and `sm_120a`; your lane stays
   the default target.
2. The focused test set: `ctest -R 'openai|anthropic|tool_streaming|flash_next'` from the
   build directory, then the 27B-class battery that you normally run. The serving schema tests
   cover the streaming tool contract; the real-artifact tests need the usual environment
   variables and skip otherwise.
3. A serving smoke with tools on the OpenAI route, streaming on: a tool call must stream to a
   normal `finish_reason: tool_calls` chunk with the call emitted as a complete delta. This is
   the path the merge had broken.
4. The tiered 73.7 GiB load profile, if you keep it: the cache and materialization semantics
   were touched by the demote-then-restore work, and the fixture that covers it runs in
   `ninfer_qwen3_8_flash_next_engine_real_test`.

The measurement rules and the record live in
`docs/performance/qwen3.8-flash-next-125b-a6b.md` and `docs/performance/qwen3.8-27b.md`.

## What to push and when

The ratified merge cadence applies: the work frontier merges into `master` at stable points,
with gates green before and after, as an ordinary master-line merge. `5090-test` is now 98
commits ahead of `master` (the SP-7 packages: cache economics, session-scoped retention, the
state spill tier, the 35B banks acceptance record, the SP-7 closure). When your SP-7 work
reaches a stable point, bring the frontier over the same way the October integration did:

- run the gate battery on the 5090 first and record the numbers;
- merge `5090-test` into `master` with a merge commit (no rebase);
- re-run the gate battery on `master` and record both sides.

Do not delete `5090-test` in a future cleanup. It is the tiering lane's working line, not a
record. The `archive/*` tags, by contrast, are records only and are never expected to merge.
