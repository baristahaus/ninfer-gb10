# Long-context operations workload

- Host: Linux 6.17.0-1029-nvidia aarch64; Ubuntu 24.04.4 LTS
- Commit: 1872f59f
- nvcc: release 13.0, V13.0.88
- GPU/driver: NVIDIA GB10, 580.173.02
- Artifact: qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer, 119G (multi-volume total)

Sizes 15000,60000 tokens, 1 reps, max_tokens 1536, greedy, thinking off;
prefill chunks 1024 2048 4096 8192; KV capacity auto.

## ninfer-k3-c1024

| tokens | task | turn | n | TTFT s (median) | prefill tok/s | decode tok/s | max gap s | facts named per rep (of 4) |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 15000 | script | 1 | 1 | 9.87 | 1584 | 63.8 | 0.05 | 3 |
| 15000 | rca | 1 | 1 | 9.65 | 1600 | 57.9 | 0.05 | 4 |
| 60000 | script | 1 | 1 | 39.08 | 1548 | 61.6 | 0.05 | 3 |
| 60000 | rca | 1 | 1 | 39.00 | 1550 | 59.8 | 0.05 | 4 |
interference: {"interference": {"small_tokens": 15000, "large_tokens": 60000, "arrival_ttft_s": 41.841, "decoder_gap_median_before_s": 0.0476, "decoder_gap_max_during_s": 0.749, "decoder_stall_s": 41.617, "decoder_gap_median_after_s": 0.0481, "decoder_finished_before_arrival_prefill": false}}

## ninfer-k3-c2048

| tokens | task | turn | n | TTFT s (median) | prefill tok/s | decode tok/s | max gap s | facts named per rep (of 4) |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 15000 | script | 1 | 1 | 8.23 | 1900 | 63.7 | 0.05 | 3 |
| 15000 | rca | 1 | 1 | 8.13 | 1900 | 57.9 | 0.05 | 4 |
| 60000 | script | 1 | 1 | 32.58 | 1856 | 57.6 | 0.05 | 3 |
| 60000 | rca | 1 | 1 | 32.59 | 1855 | 63.4 | 0.05 | 4 |
interference: {"interference": {"small_tokens": 15000, "large_tokens": 60000, "arrival_ttft_s": 34.019, "decoder_gap_median_before_s": 0.0478, "decoder_gap_max_during_s": 1.191, "decoder_stall_s": 33.795, "decoder_gap_median_after_s": 0.048, "decoder_finished_before_arrival_prefill": false}}

## ninfer-k3-c4096

| tokens | task | turn | n | TTFT s (median) | prefill tok/s | decode tok/s | max gap s | facts named per rep (of 4) |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 15000 | script | 1 | 1 | 7.24 | 2160 | 61.7 | 0.05 | 3 |
| 15000 | rca | 1 | 1 | 7.16 | 2158 | 57.5 | 0.05 | 4 |
| 60000 | script | 1 | 1 | 28.33 | 2135 | 61.3 | 0.05 | 3 |
| 60000 | rca | 1 | 1 | 28.34 | 2133 | 61.9 | 0.05 | 4 |
interference: {"interference": {"small_tokens": 15000, "large_tokens": 60000, "arrival_ttft_s": 29.136, "decoder_gap_median_before_s": 0.0481, "decoder_gap_max_during_s": 2.028, "decoder_stall_s": 28.913, "decoder_gap_median_after_s": 0.0484, "decoder_finished_before_arrival_prefill": false}}

## ninfer-k3-c8192

| tokens | task | turn | n | TTFT s (median) | prefill tok/s | decode tok/s | max gap s | facts named per rep (of 4) |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 15000 | script | 1 | 1 | 6.90 | 2265 | 61.7 | 0.05 | 3 |
| 15000 | rca | 1 | 1 | 6.79 | 2276 | 57.5 | 0.07 | 4 |
| 60000 | script | 1 | 1 | 27.30 | 2216 | 61.3 | 0.05 | 3 |
| 60000 | rca | 1 | 1 | 27.34 | 2212 | 61.9 | 0.05 | 4 |
interference: {"interference": {"small_tokens": 15000, "large_tokens": 60000, "arrival_ttft_s": 27.705, "decoder_gap_median_before_s": 0.0481, "decoder_gap_max_during_s": 3.81, "decoder_stall_s": 27.479, "decoder_gap_median_after_s": 0.0483, "decoder_finished_before_arrival_prefill": false}}
