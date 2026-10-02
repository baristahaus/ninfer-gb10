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
