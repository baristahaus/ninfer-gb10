# K1 verification — 2026-10-01

K1 (deferred MTP commit fold, `596a63f9`) vs the item-3 fused baseline
(`8630b6a1`). C4 K=1, fp8 KV, `--spec mtp --draft-tokens 1 --lm-head-draft`,
prose 256 tokens × 3 repeats + 1 warm per request × 4 requests, greedy, ABBA
(A B B A) with the page cache dropped between runs. Artifact:
`out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer`. Runbook:
`abba.sh` (draft A arm), `abba3.sh` (final A arm). Raw logs: `abba*.log`,
`quick.log`, `k1-verify-build.log`, `k1-verify-ctest.log`.

## Correctness — K1-final binary `413ee809` (sha256
`413ee8090218968067aa489e4522a07b8024f842e2a3fe27bf12370cb39a692e`)

| Check | Result |
|---|---|
| `ninfer_gdn_replay_fold_test` | PASS |
| `ninfer_flash_next_ple_test` | PASS |
| ctest (139 tests, `k1-verify-ctest.log`) | 134 passed, 5 skipped (artifact-gated real tests), 0 failed |
| `ninfer_qwen3_8_flash_next_real_test` — bitwise goldens, fp8_mtp (ordinary greedy, MTP verify, prefix reuse, concurrent state, vision) | PASS, no goldens moved |

## A-arm discrepancy (first two campaigns)

The A-arm binaries of `abba.log` and `abba2.log`
(`bin-baseline/ninfer-serve-596a63f9`) were linked with a **pre-commit draft
`recurrent.cu` object** (gated_delta_net), not the committed K1-final source:

- nvcc embeds a source-content hash in its per-TU symbol mangling: the
  `gated_delta_net/recurrent.cu` module is `_6afba1d6_557584` in the A arm vs
  `_f31dc62d_673948` in the final build; the unchanged
  `kimi_delta_attention/recurrent.cu` carries the identical hash
  (`_17934a9b`) in both binaries, so the suffix is a pure content hash.
- The A arm carries K1's `launch_replay_fold_fixed` symbols, so its
  `recurrent.cu` is a K1 draft revision (pre-`596a63f9`), not the pre-K1 file.
- The two binaries differ in 88.3 MB of 227 MB (39 %) and 64 KiB in size, but
  all 1,468 differing host functions are same-size, immediate-only diffs (GOT /
  data-pointer re-layout induced by the one differing object); zero
  size-changed ninfer symbols, so the other 16 K1 files compiled from final
  content in both builds. The size delta is the draft vs final SASS (17 KiB
  `.nv_fatbin`) plus the 64 KiB alignment padding it shifts.
- The A-arm build log (`k1-verify-build.log`, step `[8/194]`) shows
  `recurrent.cu` was compiled at build time, i.e. the tree carried the draft
  content then.

Consequence: the verdicts from `abba.log` and `abba2.log` apply to the draft
GDN kernel. `abba3.log` re-baselines the A arm on the final binary. Tracked in
beads as `ninfer-gb10-q4s`.

## Speed

Decode tok/s and host-exposed ms/round per run (a1 b1 b2 a2), request-log
totals over all requests of the run:

| Campaign | A arm | Decode tok/s | Host exposed ms/round |
|---|---|---|---|
| `abba.log` | draft | 18.9 15.5 18.6 19.3 | 16.7 35.5 17.6 15.1 |
| `abba2.log` | draft | 19.3 17.5 19.4 19.8 | 14.1 24.0 14.8 12.7 |
| `abba3.log` | final `413ee809` | 19.6 18.3 19.6 20.1 | 13.1 20.1 13.5 12.0 |

The loop is serial, so host and GPU never overlap: every host millisecond adds to wall time,
and the removed synchronize plus two launches (well under 1% of a ~70 ms round) were simply
below run-to-run noise. `b1` (first B run) shows a host-side stall on every campaign
(35.5 / 24.0 / 20.1 ms/round) — the same machine-state signature recorded in
the item-3 campaign; `b2` and the A runs agree across campaigns.

**Verdict:** no measurable change, no regression — on both the draft and the
K1-final kernel. The predicted small host-exposed reduction does not rise
above run-to-run noise at C4 K=1.
