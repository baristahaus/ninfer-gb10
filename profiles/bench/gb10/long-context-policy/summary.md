# Long-context operations workload

- Host: Linux 6.17.0-1029-nvidia aarch64; Ubuntu 24.04.4 LTS
- Commit: 7f7af4c1 (with local changes)
- nvcc: release 13.0, V13.0.88
- GPU/driver: NVIDIA GB10, 580.173.02
- Artifact: qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer, 119G (multi-volume total)

Sizes 15000,60000 tokens, 1 reps, max_tokens 1536, greedy, thinking off;
prefill chunks default; KV capacity auto.

## ninfer-k3

| tokens | task | turn | n | TTFT s (median) | prefill tok/s | decode tok/s | max gap s | facts named per rep (of 4) |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 15000 | script | 1 | 1 | 7.21 | 2167 | 61.6 | 0.05 | 3 |
| 15000 | rca | 1 | 1 | 7.13 | 2165 | 57.3 | 0.05 | 4 |
| 60000 | script | 1 | 1 | 28.27 | 2140 | 61.1 | 0.05 | 3 |
| 60000 | rca | 1 | 1 | 28.29 | 2137 | 61.6 | 0.05 | 4 |
interference: {"interference": {"small_tokens": 15000, "large_tokens": 60000, "arrival_ttft_s": 49.018, "decoder_gap_median_before_s": 0.0481, "decoder_gap_max_during_s": 2.025, "decoder_stall_s": 28.81, "decoder_gap_median_after_s": 0.0619, "decoder_finished_before_arrival_prefill": false}}
