# M3 verification: PLE multi-column GEMV, a5e4a219 vs 42722673, GB10, 2026-10-02, twoFour

Subject: commit a5e4a219 (`perf(ple)`: PLE multi-column GEMV) vs base 42722673 (the M2 tree).
Protocol, per k5/m2 verify: four test gates, ABBA decode (A1 B1 B2 A2 at C4 then C1; fresh
serve per leg, `drop_caches` per leg, concurrency_sweep --max-tokens 256 --prompt-chars 2000
--ignore-eos), and round attribution (nsys 2025.3.2, C4, MTP draft 1).

Artifact: qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer (70.2 GiB load, FP8 KV, MTP K=1).
Build: tree a5e4a219 (HEAD a80e1066 adds bench outputs only; build unchanged, "no work to do").
Serve A (base): /tmp/m3-a-arm-serve, sha256 a6967c82...c54f096 (42722673, saved pre-rebuild).
Serve B (new): /tmp/m3-bin/ninfer-serve, sha256 fc7872dc...136ace0b (a5e4a219, verified
identical to build/apps/ninfer-serve by size + mtime).

## Gates — all PASS (20:51–20:55Z)

| Gate | Result |
|---|---|
| ninfer_linear_bf16_a16_test (incl. BF16 decode columns) | `OK BF16_A16 Linear` |
| ninfer_flash_next_ple_test | `OK Flash-Next PLE` |
| ninfer_flash_next_ple_stage_test | `OK Flash-Next PLE stage` |
| ninfer_qwen3_8_flash_next_real_test (bitwise vs goldens) | `OK Qwen3.8 Flash Next real Engine` |

Binaries ran from the /tmp/m3-bin disk-recovered copies (see Incidents). The bitwise real
test re-reads the full artifact and the PLE table through the VFS and matched goldens, which
also confirms the machine VFS incident (ninfer-gb10-1kq) did not affect the model data paths.

## ABBA decode (20:57–21:04Z; full re-run after the first run's teardown bug — see Incidents)

C4, 2048 tokens per leg:

| Leg | Binary | Decode tok/s | Tok/round | Device wait ms/round | Queue wait mean |
|---|---|---:|---:|---:|---:|
| A1 | base 42722673 | 27.7 | 1.73 | 62.5 | 1739 ms |
| B1 | new a5e4a219 | 28.9 | 1.77 | 61.1 | 1608 ms |
| B2 | new a5e4a219 | 28.5 | 1.73 | 60.7 | 1605 ms |
| A2 | base 42722673 | 27.3 | 1.70 | 62.2 | 1737 ms |

Pair averages: A = 27.5 tok/s (1.715 tok/round, 62.35 ms/round) vs B = 28.7 tok/s
(1.75 tok/round, 60.9 ms/round) → **B is +1.2 tok/s = +4.4%**; B wins in both interleavings
(B1 > A1 and B2 > A2). Prefill 2.5 s on every leg.

C1, 512 tokens per leg:

| Leg | A1 | B1 | B2 | A2 |
|---|---:|---:|---:|---:|
| Decode tok/s | 41.9 | 42.1 | 42.2 | 41.9 |

Pair averages: A = 41.9 vs B = 42.15 tok/s → +0.6%, within leg-to-leg noise; no regression.
All C1 legs: 1.62 tok/round, device wait ~38.4–38.6 ms/round, queue wait ~2 ms.

## Round attribution (new build, C4 K=1)

- GPU busy 94.17% (9040 / 9132 ms); 133 decode rounds; host wall 68.66 ms/round, GPU work
  67.97 ms/round; 1.72 tok/round; host exposed 1.19 ms/round.
- Stage shares: moe.nvfp4 target.verify 54.0% (nvfp4_w4a4_mma_kernel 91.3% of stage),
  gdn.record 19.9%, qsa.select 9.4%, hyper.combine_mix 8.0%.
- PLE stage (the changed multi-column GEMV): 0.5% of decode GPU work (0.328 ms/round),
  dominated by bf16_gemv_columns_kernel (92.7% of the stage, 0.304 ms/round, 2 launches/round).
  End-to-end headroom from further PLE GEMV gains is bounded by this share.
- Full tables: `profiles/bench/gb10/round-attribution/summary.md` (this run).

## Incidents (20:51–20:58Z) and effect on validity

1. **Kernel VFS path-lookup corruption** on this node (6.17.0-1029-nvidia), onset 20:36:18Z:
   the four test binaries became invisible to the VFS while the on-disk filesystem stayed
   intact (verified via debugfs and raw partition reads). All four were reconstructed by
   dd'ing their physical extents from /dev/nvme0n1p2 into /tmp/m3-bin (exact on-disk sizes,
   ELF verified, sha256 fa25c63b…/822f97af…/847b1026…/741bebd0…). The B serve binary was a
   healthy 20:42Z VFS copy. Node reboot still required (ninfer-gb10-1kq).
2. **Campaign teardown bug (first M3 run, 20:51–20:56Z):** `pkill -x ninfer-serve` did not
   match the renamed A-arm binary (comm "m3-a-arm-serve"), so the leg-1 server survived
   holding 74.6 GiB GPU + port 18087; legs 2–8 died at startup ("weights exceed free GPU
   memory") while the health check answered from the orphan — their Totals are empty and the
   only valid datapoint from that run is abba-c4-1-base (28.2 tok/s, superseded by the
   re-run). Orphan killed 20:57Z (GPU + port verified released); teardown fixed to exact-PID
   kill with a still-alive guard; all eight legs and attribution re-run as recorded above.
   The "load driver reported errors" line in the first run was the sweep hitting the stale
   orphan; the re-run legs had no load-driver errors.

## Decision

Accept a5e4a219: all gates pass including the bitwise real test, +4.4% at C4 and +0.6% at
C1 (both interleavings agree at C4), no regressions; the PLE multi-column GEMV stage is
numerically clean and accounts for 0.5% of decode GPU work.

Raw legs: `abba-c{4,1}-{1..4}-{base,new}/` (serve.log, load.log, load.json, requests.jsonl,
summary.txt).
