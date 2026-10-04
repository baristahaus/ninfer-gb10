# Long-context operations workload

- Host: Linux 6.17.0-1029-nvidia aarch64; Ubuntu 24.04.4 LTS
- Commit: acf6f5f7
- nvcc: release 13.0, V13.0.88
- GPU/driver: NVIDIA GB10, 580.173.02
- Artifact: qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer, 119G (multi-volume total)

Sizes 15000,60000 tokens, 1 reps, max_tokens 1536, greedy, thinking off;
prefill chunks default; decode shares default; prompt lookup off on; KV capacity auto.

## ninfer-k3-lookup

| tokens | task | turn | n | TTFT s (median) | prefill tok/s | decode tok/s | max gap s | facts named per rep (of 4) |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 15000 | script | 1 | 1 | 7.21 | 2170 | 60.4 | 0.05 | 3 |
| 15000 | rca | 1 | 1 | 7.12 | 2170 | 56.8 | 0.05 | 4 |
| 60000 | script | 1 | 1 | 28.22 | 2144 | 60.7 | 0.05 | 3 |
| 60000 | rca | 1 | 1 | 28.17 | 2146 | 61.1 | 0.05 | 4 |
interference: {"interference": {"small_tokens": 15000, "large_tokens": 60000, "arrival_ttft_s": 42.45, "decoder_gap_median_before_s": 0.0484, "decoder_gap_max_during_s": 2.018, "decoder_stall_s": 28.689, "decoder_gap_median_after_s": 0.0487, "decoder_finished_before_arrival_prefill": false}}
yield: {"yield": {"short_tokens": 2000, "large_tokens": 60000, "short_ttft_s": 1.684, "long_ttft_s": 30.072, "short_first_before_long_first": true}}

## ninfer-k3

| tokens | task | turn | n | TTFT s (median) | prefill tok/s | decode tok/s | max gap s | facts named per rep (of 4) |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 15000 | script | 1 | 1 | 7.19 | 2174 | 61.3 | 0.05 | 3 |
| 15000 | rca | 1 | 1 | 7.11 | 2173 | 57.2 | 0.05 | 4 |
| 60000 | script | 1 | 1 | 28.17 | 2147 | 60.9 | 0.05 | 3 |
| 60000 | rca | 1 | 1 | 28.22 | 2142 | 61.4 | 0.05 | 4 |
interference: {"interference": {"small_tokens": 15000, "large_tokens": 60000, "arrival_ttft_s": 42.547, "decoder_gap_median_before_s": 0.0483, "decoder_gap_max_during_s": 2.022, "decoder_stall_s": 28.769, "decoder_gap_median_after_s": 0.0485, "decoder_finished_before_arrival_prefill": false}}
yield: {"yield": {"short_tokens": 2000, "large_tokens": 60000, "short_ttft_s": 1.691, "long_ttft_s": 30.156, "short_first_before_long_first": true}}
