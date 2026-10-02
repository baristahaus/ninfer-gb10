# QSA split-decode staging verification: 370a5247 vs a5e4a219, GB10, 2026-10-02, twoFour

B arm (new): 370a5247, sha256 acc402a1da68d3349bb7b3ab0fc0ef94d4d30f0c0ea0538d26def286ee59a120
A arm (base): a5e4a219, sha256 cd80b878e61c2ed6f3cfec6c342904a605bd6b6e3e12c501fc4c0fd54d863014

- Host: Linux 6.17.0-1029-nvidia aarch64; Ubuntu 24.04.4 LTS
- Commit: 2c935f31
- nvcc: release 13.0, V13.0.88
- GPU/driver: NVIDIA GB10, 580.173.02
- Artifact: qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer, 119G (multi-volume total)
- Protocol: ABBA (A1 B1 B2 A2) per config; fresh serve + drop_caches per leg;
  concurrency_sweep --max-tokens 512 --prompt-chars 2000 --ignore-eos (M3 protocol).

## C4 K=1

| Leg | Arm | Decode tok/s | Tok/round | Device wait ms/round | Queue wait mean |
|---|---|---:|---:|---:|---:|
| 1 | base | 32.3 | 1.77 | 54.9 | 4963 ms |
| 2 | new | 33.0 | 1.76 | 53.4 | 4752 ms |
| 3 | new | 33.3 | 1.77 | 53.2 | 4816 ms |
| 4 | base | 33.8 | 1.82 | 54.0 | 4880 ms |

Pair averages: A (base) 33.05 tok/s (legs 32.3, 33.8) vs B (new) 33.15 tok/s (legs 33.0, 33.3) → +0.3%.

## C1 K=1

| Leg | Arm | Decode tok/s | Tok/round | Device wait ms/round | Queue wait mean |
|---|---|---:|---:|---:|---:|
| 1 | base | 38.3 | 1.46 | 38.0 | 1 ms |
| 2 | new | 39.2 | 1.46 | 37.2 | 1 ms |
| 3 | new | 39.1 | 1.46 | 37.3 | 1 ms |
| 4 | base | 38.4 | 1.46 | 37.9 | 1 ms |

Pair averages: A (base) 38.35 tok/s (legs 38.3, 38.4) vs B (new) 39.15 tok/s (legs 39.2, 39.1) → +2.1%.

## C4 K=3

| Leg | Arm | Decode tok/s | Tok/round | Device wait ms/round | Queue wait mean |
|---|---|---:|---:|---:|---:|
| 1 | base | 32.7 | 2.10 | 64.3 | 4644 ms |
| 2 | new | 39.4 | 2.65 | 67.4 | 3674 ms |
| 3 | new | 34.6 | 2.49 | 71.9 | 3756 ms |
| 4 | base | 31.9 | 2.13 | 66.8 | 4919 ms |

Pair averages: A (base) 32.30 tok/s (legs 32.7, 31.9) vs B (new) 37.00 tok/s (legs 39.4, 34.6) → +14.6%.

