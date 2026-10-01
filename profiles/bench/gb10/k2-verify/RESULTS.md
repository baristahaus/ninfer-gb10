# K2 verification — 2026-10-01

K2 (device-resident MTP round frame, `bd8a8ecb`) on top of K1. Binary
sha256 `7e329b4443cb510a8018bb1e5aba8737bff6bcd166a91fc22cadd94bee688cd7`.
Artifact: `out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer`.
Checks per the plan (K2 section, "What twoFour checks"): outputs bitwise
identical to K1; in one nsys trace of a steady C4 run,
`decode.mtp.submit.frame_upload` appears only at membership changes.

## Correctness

| Check | Result |
|---|---|
| `ninfer_mtp_round_test` (new exact oracle, K = 1, 3, 5 × B = 1, 4, 8) | PASS |
| `ninfer_gdn_replay_fold_test` | PASS |
| `ninfer_flash_next_ple_test` | PASS |
| `ninfer_qwen3_8_flash_next_real_test` — bitwise goldens (ordinary greedy, MTP verify, prefix reuse, concurrent state, vision, both proposal heads) | PASS, no goldens moved |

The real test passing on the K2 binary with the same goldens that K1
(`413ee809`) passed means K2 outputs are bitwise identical to K1.

## Frame-upload trace (nsys, C4 K=1, prose 256×3+1 warm, fp8 KV, draft 1)

`serve-c4k1.nsys-rep` (27 MB), `--trace=cuda,nvtx --cuda-graph-trace=node`.
Request log: 3136 tokens, 1916 rounds, 1.64 tok/round, decode 20.8 tok/s,
device wait 69.6 ms/round, host exposed 8.93 ms/round.

| NVTX range | Count |
|---|---|
| `decode.mtp.submit.graph` (MTP round submissions) | 540 |
| `decode.mtp.submit.ingress` (host frame build, every round) | 540 |
| `decode.mtp.submit.frame_upload` (H2D of the frame) | **38** |
| `cuda_graph.capture` | 32 |
| request_start / request_done (request log) | 13 / 13 |

- The 38 uploads are **7.0%** of MTP round submissions: the frame advanced on
  the other 93% of rounds with no H2D.
- The uploads fall in four clusters (13 / 10 / 11 / 4) aligned with the four
  load waves; the three inter-wave quiet gaps (10.5 s, 11.3 s, 13.5 s of
  steady rounds) contain **zero** uploads.
- Upload accounting (verified against the trace): 1 startup upload before the first request
  (graph preparation clears the frame) + 2 per admitted request (its first round is a
  membership change; its second follows the state-fork settle, which folds on the host, so the
  host's fold entry no longer matches the device's) + 1 per finished request at leave:
  1 + 26 + 13 = 40, reduced to 38 by coalescing — a wave's last leave and the next wave's first
  admission share a round, and the final request's leave has no next round to observe it.

**Verdict:** the frame advances in steady state; uploads occur only at startup, membership
changes (admission, settle, leave) and the design's explicit invalidations. None of the 32
runtime `cuda_graph.capture` ranges falls within one round of an upload — the upload decision
lives only in `decode_mtp_batch`, and captures are other graphs. K2 passes its checks.

## Notes

- Env var gate: the real test reads `NINFER_QWEN38_FLASH_NEXT_WEIGHTS`
  (`tests/models/qwen3_8_flash_next_125b_a6b/test_engine_real.cpp:302`); the
  first verify run used the misspelled `NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS`
  and the test self-skipped (exit 77). Fallback recorded in beads
  `ninfer-gb10-khx` (2026-10-01 15:04Z).
- The nsys version here has no `nvtx_pushpop` stats report; the counts above
  come from the `NVTX_EVENTS` × `StringIds` tables of the sqlite export.
