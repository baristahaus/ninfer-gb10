# Long-context operations workload

- Host: Linux 6.17.0-1029-nvidia aarch64; Ubuntu 24.04.4 LTS
- Commit: 88e04123 (with local changes)
- nvcc: release 13.0, V13.0.88
- GPU/driver: NVIDIA GB10, 580.173.02
- Artifact: qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer, 119G (multi-volume total)

Sizes 15000 tokens, 2 reps, max_tokens 1536, greedy, thinking off;
prefill chunks default; decode shares default; KV capacity auto.

## ninfer-k3

| tokens | task | turn | n | TTFT s (median) | prefill tok/s | decode tok/s | max gap s | facts named (of 4) or bugs fixed (of 3, edit) per rep |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 15000 | script | 1 | 2 | 7.14 | 2178 | 60.3 | 0.05 | 3, 4 |
| 15000 | script | 2 | 2 | 0.29 | nan | 60.4 | 0.05 |  |
| 15000 | edit | 1 | 2 | 6.91 | 2175 | 72.6 | 0.06 | 2, 2 |
| 15000 | edit | 2 | 2 | 0.25 | nan | 73.8 | 0.06 |  |
