# Long-context operations workload

- Host: Linux 6.17.0-1029-nvidia aarch64; Ubuntu 24.04.4 LTS
- Commit: 6cd02900
- nvcc: release 13.0, V13.0.88
- GPU/driver: NVIDIA GB10, 580.173.02
- Artifact: qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer, 119G (multi-volume total)

Sizes 15000,30000,60000 tokens, 2 reps, max_tokens 1536, greedy, thinking off.

## ninfer-k1

| tokens | task | turn | n | TTFT s (median) | prefill tok/s | decode tok/s | max gap s | facts named |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 15000 | script | 1 | 2 | 9.79 | 1590 | 49.1 | 0.04 | 3/3 of 4 |
| 15000 | script | 2 | 2 | 0.80 | nan | 46.9 | 0.04 |  |
| 15000 | rca | 1 | 2 | 9.83 | 1585 | 48.2 | 0.04 | 4/4 of 4 |
| 15000 | rca | 2 | 2 | 0.23 | nan | 46.8 | 0.04 |  |
| 30000 | script | 1 | 2 | 18.93 | 1579 | 48.7 | 0.04 | 3/3 of 4 |
| 30000 | script | 2 | 2 | 0.29 | nan | 48.6 | 0.04 |  |
| 30000 | rca | 1 | 2 | 18.73 | 1588 | 48.9 | 0.04 | 3/4 of 4 |
| 30000 | rca | 2 | 2 | 0.23 | nan | 46.9 | 0.04 |  |
| 60000 | script | 1 | 2 | 38.88 | 1550 | 48.7 | 0.05 | 3/3 of 4 |
| 60000 | script | 2 | 2 | 0.26 | nan | 47.9 | 0.04 |  |
| 60000 | rca | 1 | 2 | 39.03 | 1546 | 48.8 | 0.05 | 4/4 of 4 |
| 60000 | rca | 2 | 2 | 0.24 | nan | 46.7 | 0.04 |  |
interference: {"interference": {"small_tokens": 15000, "large_tokens": 60000, "arrival_ttft_s": 67.047, "decoder_gap_median_before_s": 0.037, "decoder_gap_max_during_s": 0.041, "decoder_stall_s": 0, "decoder_gap_median_after_s": null, "decoder_finished_before_arrival_prefill": true}}

## ninfer-k3

| tokens | task | turn | n | TTFT s (median) | prefill tok/s | decode tok/s | max gap s | facts named |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 15000 | script | 1 | 2 | 9.78 | 1592 | 62.9 | 0.05 | 3/3 of 4 |
| 15000 | script | 2 | 2 | 0.29 | nan | 57.2 | 0.05 |  |
| 15000 | rca | 1 | 2 | 9.77 | 1594 | 58.1 | 0.05 | 4/4 of 4 |
| 15000 | rca | 2 | 2 | 0.24 | nan | 53.3 | 0.05 |  |
| 30000 | script | 1 | 2 | 18.83 | 1588 | 63.2 | 0.05 | 3/3 of 4 |
| 30000 | script | 2 | 2 | 0.78 | nan | 56.0 | 0.05 |  |
| 30000 | rca | 1 | 2 | 18.67 | 1593 | 61.3 | 0.05 | 4/4 of 4 |
| 30000 | rca | 2 | 2 | 0.24 | nan | 53.6 | 0.05 |  |
| 60000 | script | 1 | 2 | 38.76 | 1555 | 61.4 | 0.06 | 3/3 of 4 |
| 60000 | script | 2 | 2 | 0.30 | nan | 57.5 | 0.05 |  |
| 60000 | rca | 1 | 2 | 38.95 | 1549 | 60.5 | 0.07 | 4/3 of 4 |
| 60000 | rca | 2 | 2 | 0.29 | nan | 54.2 | 0.05 |  |
interference: {"interference": {"small_tokens": 15000, "large_tokens": 60000, "arrival_ttft_s": 60.867, "decoder_gap_median_before_s": 0.048, "decoder_gap_max_during_s": 0.053, "decoder_stall_s": 0, "decoder_gap_median_after_s": null, "decoder_finished_before_arrival_prefill": true}}

## dgpp

| tokens | task | turn | n | TTFT s (median) | prefill tok/s | decode tok/s | max gap s | facts named |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 15000 | script | 1 | 2 | 7.92 | 1965 | 47.5 | 0.06 | 3/3 of 4 |
| 15000 | script | 2 | 2 | 0.36 | nan | 45.4 | 0.05 |  |
| 15000 | rca | 1 | 2 | 7.96 | 1956 | 45.6 | 0.06 | 3/4 of 4 |
| 15000 | rca | 2 | 2 | 1.58 | nan | 44.0 | 0.05 |  |
| 30000 | script | 1 | 2 | 15.65 | 1909 | 46.8 | 0.05 | 4/3 of 4 |
| 30000 | script | 2 | 2 | 0.85 | nan | 46.0 | 0.05 |  |
| 30000 | rca | 1 | 2 | 15.49 | 1919 | 46.7 | 0.05 | 4/4 of 4 |
| 30000 | rca | 2 | 2 | 0.31 | nan | 44.3 | 0.05 |  |
| 60000 | script | 1 | 2 | 32.28 | 1867 | 46.6 | 0.08 | 3/3 of 4 |
| 60000 | script | 2 | 2 | 1.07 | nan | 44.5 | 0.05 |  |
| 60000 | rca | 1 | 2 | 32.31 | 1867 | 46.2 | 0.05 | 4/4 of 4 |
| 60000 | rca | 2 | 2 | 0.32 | nan | 45.1 | 0.05 |  |
interference: {"interference": {"small_tokens": 15000, "large_tokens": 60000, "arrival_ttft_s": 33.086, "decoder_gap_median_before_s": 0.0252, "decoder_gap_max_during_s": 2.393, "decoder_stall_s": 32.942, "decoder_gap_median_after_s": 0.0252, "decoder_finished_before_arrival_prefill": false}}
