# Long-context operations workload

- Host: Linux 6.17.0-1029-nvidia aarch64; Ubuntu 24.04.4 LTS
- Commit: 1307107c (with local changes)
- nvcc: release 13.0, V13.0.88
- GPU/driver: NVIDIA GB10, 580.173.02
- Artifact: qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer, 119G (multi-volume total)

Sizes 15000,60000 tokens, 2 reps, max_tokens 1536, greedy, thinking off;
prefill chunks default; decode shares default; prompt lookup off on; KV capacity auto.

## ninfer-k3-lookup

| tokens | task | turn | n | TTFT s (median) | prefill tok/s | decode tok/s | max gap s | facts named (of 4) or bugs fixed (of 3, edit) per rep |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 15000 | edit | 1 | 2 | 6.97 | 2149 | 71.9 | 0.07 | 2, 2 |
| 15000 | edit | 2 | 2 | 0.28 | nan | 69.9 | 0.05 |  |
| 60000 | edit | 1 | 2 | 27.97 | 2142 | 74.5 | 0.07 | 2, 2 |
| 60000 | edit | 2 | 2 | 0.26 | nan | 73.9 | 0.06 |  |
interference: {"interference": {"small_tokens": 15000, "large_tokens": 60000, "arrival_ttft_s": 50.182, "decoder_gap_median_before_s": 0.0486, "decoder_gap_max_during_s": 1.874, "decoder_stall_s": 5.577, "decoder_gap_median_after_s": null, "decoder_finished_before_arrival_prefill": true}}
yield: {"yield": {"short_tokens": 2000, "large_tokens": 60000, "short_ttft_s": 1.617, "long_ttft_s": 30.064, "short_first_before_long_first": true}}
backfill: {"backfill": {"small_tokens": 15000, "short_tokens": 2000, "large_tokens": 60000, "short_ttft_s": 1.53, "held_ttft_s": 51.192, "running_end_after_held_arrival_s": 30.442, "short_first_before_running_end": true}}

## ninfer-k3

| tokens | task | turn | n | TTFT s (median) | prefill tok/s | decode tok/s | max gap s | facts named (of 4) or bugs fixed (of 3, edit) per rep |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 15000 | edit | 1 | 2 | 6.95 | 2154 | 74.8 | 0.07 | 2, 2 |
| 15000 | edit | 2 | 2 | 0.28 | nan | 71.2 | 0.06 |  |
| 60000 | edit | 1 | 2 | 27.91 | 2146 | 74.6 | 0.07 | 2, 2 |
| 60000 | edit | 2 | 2 | 0.26 | nan | 74.6 | 0.05 |  |
interference: {"interference": {"small_tokens": 15000, "large_tokens": 60000, "arrival_ttft_s": 49.826, "decoder_gap_median_before_s": 0.0481, "decoder_gap_max_during_s": 1.869, "decoder_stall_s": 3.699, "decoder_gap_median_after_s": null, "decoder_finished_before_arrival_prefill": true}}
yield: {"yield": {"short_tokens": 2000, "large_tokens": 60000, "short_ttft_s": 1.629, "long_ttft_s": 30.125, "short_first_before_long_first": true}}
backfill: {"backfill": {"small_tokens": 15000, "short_tokens": 2000, "large_tokens": 60000, "short_ttft_s": 1.525, "held_ttft_s": 51.057, "running_end_after_held_arrival_s": 30.272, "short_first_before_running_end": true}}
