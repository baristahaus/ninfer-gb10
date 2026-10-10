# Long-context operations workload

- Host: Linux 6.17.0-1029-nvidia aarch64; Ubuntu 24.04.4 LTS
- Commit: 8bd44941 (with local changes)
- nvcc: release 13.0, V13.0.88
- GPU/driver: NVIDIA GB10, 580.173.02
- Artifact: qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer, 119G (multi-volume total)

Sizes 15000 tokens, 2 reps, max_tokens 1536, greedy, thinking off;
prefill chunks default; decode shares default; KV capacity auto.

## ninfer-k3

| tokens | task | turn | n | TTFT s (median) | prefill tok/s | decode tok/s | max gap s | facts named (of 4) or bugs fixed (of 3, edit) per rep |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 15000 | script | 1 | 2 | 7.11 | 2188 | 60.2 | 0.05 | 3, 4 |
| 15000 | script | 2 | 2 | 0.29 | nan | 60.3 | 0.05 |  |
| 15000 | edit | 1 | 2 | 6.89 | 2182 | 72.5 | 0.06 | 2, 2 |
| 15000 | edit | 2 | 2 | 0.25 | nan | 73.7 | 0.07 |  |
yield: {"yield": {"short_tokens": 2000, "large_tokens": 15000, "short_ttft_s": 1.575, "long_ttft_s": 9.093, "short_first_before_long_first": true}}

## ninfer-k4

| tokens | task | turn | n | TTFT s (median) | prefill tok/s | decode tok/s | max gap s | facts named (of 4) or bugs fixed (of 3, edit) per rep |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 15000 | script | 1 | 2 | 7.13 | 2184 | 62.4 | 0.06 | 3, 4 |
| 15000 | script | 2 | 2 | 0.29 | nan | 58.4 | 0.06 |  |
| 15000 | edit | 1 | 2 | 6.88 | 2185 | 77.8 | 0.07 | 2, 2 |
| 15000 | edit | 2 | 2 | 0.25 | nan | 80.1 | 0.07 |  |
yield: {"yield": {"short_tokens": 2000, "large_tokens": 15000, "short_ttft_s": 1.583, "long_ttft_s": 9.222, "short_first_before_long_first": true}}

## ninfer-k5

| tokens | task | turn | n | TTFT s (median) | prefill tok/s | decode tok/s | max gap s | facts named (of 4) or bugs fixed (of 3, edit) per rep |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 15000 | script | 1 | 2 | 7.15 | 2177 | 61.1 | 0.07 | 3, 4 |
| 15000 | script | 2 | 2 | 0.30 | nan | 57.1 | 0.07 |  |
| 15000 | edit | 1 | 2 | 6.90 | 2179 | 82.7 | 0.08 | 2, 2 |
| 15000 | edit | 2 | 2 | 0.25 | nan | 85.0 | 0.08 |  |
yield: {"yield": {"short_tokens": 2000, "large_tokens": 15000, "short_ttft_s": 1.588, "long_ttft_s": 9.305, "short_first_before_long_first": true}}
