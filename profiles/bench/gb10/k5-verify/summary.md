# K5 run-list verification, d0bc40ff on 2026-10-02, twoFour

Build: `cmake --build build -j` (sm_121a, CUDA 13.0), all run-list targets compiled.

## Step 1: GDN op tests — FAIL (new fused test)

`ninfer_flash_next_gdn_test` (full output):

```
ninfer: persistent grids sized for 48 SMs
Flash-Next GDN pending fold W=4 R=4 B=5: conv records differs from fold-then-record
Flash-Next GDN pending fold W=4 R=4 B=5: key records differs from fold-then-record
Flash-Next GDN pending fold W=4 R=4 B=5: value records differs from fold-then-record
Flash-Next GDN pending fold W=4 R=4 B=5: gate records differs from fold-then-record
FAIL Flash-Next GDN
```

- Failing case W=4 R=4 B=5 is the pending-entries-out-of-order case.
- The test reports only the four record planes; conv/recurrent state and block
  output are bitwise identical in that case. So the fused state math is clean
  and the recorded columns (the input to the next round's fold) are not.

`ninfer_gdn_replay_fold_test`: `OK gdn_replay_fold` — no regression in the old
fold path.

## Step 2: real test — HARD FAIL at graph setup

```
ninfer: persistent grids sized for 48 SMs
Qwen3.8 Flash Next real Engine: MTP CUDA Graph profile 1 (topology class 10):
CUDA Graph executable update failed: cudaErrorGraphExecUpdateFailure
(update result 5)
```

- Fails before any decode: the MTP graph profile 1 (topology class 10) cannot
  be updated against the profile-0 executable after the K5 topology change
  (head fold removed, snapshot memcpy added, fused record kernel).
- No end-to-end bitwise data, no speed data; steps 3-5 blocked.

## Status

Steps 3 (C1/C4 probes), 4 (ABBA tok/s), 5 (attribution) not run: the op test
fails and the engine cannot start the MTP graph. Both defects are in the K5
commit (op record pass; MTP graph update path).

## Re-run on the fix (`cfa71b2a`), 2026-10-02 — all steps pass

The two `d0bc40ff` defects were fixed as recorded in the plan K5 section: the op
test now compares only the valid record columns `[0, valid)` (the unwritten
`[valid, width)` columns are never read by any fold), and the record snapshot is
a kernel node (`copy_record_rows_kernel`) instead of a 2D memcpy, so
`cudaGraphExecUpdate` accepts it across batch profiles. Full run list re-run on
the GB10 workstation at `cfa71b2a`; all acceptance items pass.

### Step 1: GDN op tests — PASS

- `ninfer_flash_next_gdn_test`: `OK Flash-Next GDN` — includes the new bitwise
  comparison of the fused fold+record op against fold-then-record at real shapes
  (conv/recurrent states, block output, all four record planes; pending entries
  out of order; extents 0/1/2/4; W=2 single row).
- `ninfer_gdn_replay_fold_test`: `OK gdn_replay_fold` — old fold path unchanged.

Test-side fix (two lines, recorded): the fused-run comparison pre-filled its
record planes with 0x7f and compared whole planes; the verify writes columns
`[0, valid)` only, and the reference run's unwritten columns still held earlier
contents. Both runs now compare the valid columns.

### Step 2: real test — PASS, bitwise

`qwen3_8_flash_next_real_test` on the fp8 MTP artifact: `OK Qwen3.8 Flash Next
real Engine!` (79.9 s, MTP graph setup succeeds — the previous
`cudaErrorGraphExecUpdateFailure` is gone). Output is bitwise equal to the
`b42ca206` golden.

### Step 3: C1/C4 serve probes — PASS, bitwise

Discard probe (`profiles/bench/gb10/k4-verify/discard_load.py`): four requests
(`maxtok10` length/10, `stopword` stop/14, `maxtok16` length/16, `survivor`
length/256), batch = all four in parallel at C4, solo = sequential C1. Served
with the production config (`--kv-dtype fp8 --preserve-thinking --spec mtp
--draft-tokens 1 --lm-head-draft`, C4). Text = reasoning + content, sha256/16.

| Request (finish) | base batch | base solo | k5 batch | k5 solo |
|---|---|---|---|---|
| maxtok10 (length, 10) | `7aa119c4` | `7aa119c4` | `7aa119c4` | `7aa119c4` |
| stopword (stop, 14) | `f284eda2` | `f284eda2` | `f284eda2` | `f284eda2` |
| maxtok16 (length, 16) | `b62a186c` | `b62a186c` | `b62a186c` | `b62a186c` |
| survivor (length, 256) | `847314b2` | `847314b2` | `847314b2` | `847314b2` |

All 16 outputs byte-identical: k5 == base in both modes, and batch == solo in
both binaries (the C4 batch leaves rows of the cohort while the survivor keeps
decoding; admission is exercised by the real test's concurrent state, which
passes). Raw jsonl: `discard-{base,k5}-{batch,solo}.jsonl`.

### Step 4: ABBA tok/s vs `b42ca206`

Serve per leg (fresh start, `drop_caches` before each leg), load
`concurrency_sweep --max-tokens 256 --prompt-chars 2000 --ignore-eos`.
A = `/tmp/serve-base` (b42ca206, sha `13aeb7f1`), B = `/tmp/serve-k5`
(cfa71b2a, sha `8bcee0f0`).

C4 K=1 (2048 tokens, `--n 4,4`):

| Leg | decode | tok/s | device wait ms/round | host exposed ms/round |
|---|---|---|---|---|
| A1 base | 77.2 s | 26.5 | 65.9 | 0.14 |
| B1 k5 | 76.2 s | 26.9 | 65.3 | 0.13 |
| B2 k5 | 75.0 s | 27.3 | 65.3 | 0.12 |
| A2 base | 77.0 s | 26.6 | 66.6 | 0.16 |
| **means** | | **base 26.55 / k5 27.10 (+2.1%)** | 66.25 → 65.30 | 0.15 → 0.12 |

C1 K=1 (512 tokens, `--n 1,1`):

| Leg | decode | tok/s | device wait ms/round |
|---|---|---|---|
| A1 base | 12.5 s | 40.9 | 39.6 |
| B1 k5 | 12.4 s | 41.2 | 39.3 |
| B2 k5 | 12.5 s | 41.1 | 39.4 |
| A2 base | 12.5 s | 41.0 | 39.5 |
| **means** | | **base 40.95 / k5 41.15 (+0.5%, noise band)** | 39.55 → 39.35 |

Direction and magnitude match the plan estimate (~2.0 ms removed of a ~72 ms
C4 round, a little back to the copy). The C4 k5 legs drift upward B1→B2
(warm-up); the ABBA means absorb it. Note: an earlier C1 ABBA attempt was run
twice concurrently by mistake and discarded (port/GPU collision); the table
above is a single clean re-run.

### Step 5: round attribution (C4 K=1, nsys)

`tools/gb10/round_attribution.sh` on the K5 tree (fresh `build-trace/`), data in
`profiles/bench/gb10/round-attribution/` (pre-K5: `round-attribution-merged/`).

| | pre-K5 (merged run) | K5 |
|---|---|---|
| per-round GPU work (134 / 129 rounds) | 75.195 ms | 73.855 ms (−1.34) |
| `recurrent_fold_kernel` (head fold) | 4.006 ms/round (5.3%) | **gone** |
| snapshot copy | — (2D memcpy that broke graph updates) | `copy_record_rows_kernel` 0.059 ms/round |
| record kernel | `recurrent_record_kernel` 1.914 ms/round | `recurrent_fold_record_kernel` 2.831 ms/round |
| gdn.record stage | 11.643 ms/round | 13.531 ms/round |

Net GDN state work: 4.006 + 1.914 = 5.920 → 2.831 + 0.059 = 2.890 ms/round
(−3.03). The stage table shows +1.89 because the head fold used to sit in the
unattributed bucket; the in-stage mma delta (9.233 → 10.089) is run-to-run
variance between the two captures — its code path is unchanged, and the
unprofiled ABBA confirms the net gain. Profiler request-log total: 27.2 tok/s
under nsys (matches the plain 27.1; timings carry profiler overhead, shares
only).

### Acceptance (plan K5)

1. op test — PASS (both, after the recorded two-line test fix).
2. real test bitwise + C1/C4 serve probes bitwise incl. stop/max_tokens endings
   and admission during decode — PASS.
3. C1/C4 decode ABBA — PASS (C4 +2.1%, C1 +0.5% within noise, no regression).
4. round attribution: head fold kernel gone, record kernel time per round,
   snapshot copy — PASS (table above).
