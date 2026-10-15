## Step 2 — unprofiled baseline

- Host: Linux 6.17.0-1029-nvidia aarch64; Ubuntu 24.04.4 LTS
- Commit: 38c4f72e
- nvcc: release 13.0, V13.0.88
- GPU/driver: NVIDIA GB10, 580.173.02
- Artifact: qwen3_8_flash_next_125b_a6b_nvfp4.ninfer, 126G (multi-volume total)
- Benchmark: -pg 8192,512 and 65536,512; --max-ctx 73728 --prefill-chunk 8192 --kv-dtype fp8; 1 warmup + 5 measured, greedy
- Sampled rows: -pg 8192,512 with the model's registered sampling preset (--sampling model --seed 1); 1 warmup + 3 measured
- Power mode: [N/A]

Memory bandwidth probe:
```
nvcc fatal   : Unsupported gpu architecture '121a'
```

| Config | Test | Prefill tok/s | Decode output tok/s | Mean accepted length |
|---|---|---:|---:|---:|
| none, KV fp8-e4m3-row256 | pp8192+tg512 | 1,550.3 ± 1.1 | 21.4 ± 0.0 | — |
| none, KV fp8-e4m3-row256 | pp65536+tg512 | 1,509.8 ± 0.9 | 21.1 ± 0.0 | — |
| mtp K=2, KV fp8-e4m3-row256 | pp8192+tg512 | 1,470.1 ± 0.7 | 47.6 ± 0.0 | 2.99 |
| mtp K=2, KV fp8-e4m3-row256 | pp65536+tg512 | 1,433.5 ± 0.6 | 46.3 ± 0.0 | 2.98 |
| mtp K=3, KV fp8-e4m3-row256 | pp8192+tg512 | 1,468.5 ± 2.0 | 56.1 ± 0.0 | 3.96 |
| mtp K=3, KV fp8-e4m3-row256 | pp65536+tg512 | 1,430.0 ± 0.5 | 54.7 ± 0.0 | 3.96 |
| none, KV fp8-e4m3-row256, model;seed=1 | pp8192+tg512 | 1,542.5 ± 3.7 | 21.5 ± 0.0 | — |
| mtp K=2, KV fp8-e4m3-row256, model;seed=1 | pp8192+tg512 | 1,481.4 ± 1.4 | 47.5 ± 0.0 | 2.99 |

- Device: NVIDIA GB10; CUDA runtime 13.0, driver 13.0
- KV capacity: 73728 tokens (fp8-e4m3-row256); available after startup: 27.2 GiB

GPU telemetry (decode-dominated runs; nvidia-smi every 0.5 s, window = samples between the
first and last with utilization >= 50%, so model loading is excluded):

Effective GB/s = modeled bytes per emitted token (tools/gb10/decode_bytes.py: dense weights,
routed experts at 10/512 per token or their uniform-routing union per MTP round, GDN state,
MTP draft steps; KV/indexer and activations excluded) x decode tok/s, against the probe above.

| Run | Samples | GPU util % mean | Power W mean / p95 / peak | SM clock MHz mean / min | Temp °C max | Decode tok/s | J / output token | Modeled GB / token | Effective GB/s (% of probe) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| idle | 20/20 | 1 | 7.5 / 19.7 / 19.7 | 649 / 208 | 63 | — | — | — | — |
| K=0 | 434/475 | 96 | 31.8 / 31.9 / 47.6 | 2489 / 2444 | 68 | 21.5 | 1.48 | 10.18 | 218 |
| K=2 | 264/309 | 96 | 37.9 / 38.3 / 51.0 | 2474 / 2450 | 68 | 35.5 | 1.07 | 6.30 | 224 |
| K=3 | 252/295 | 96 | 40.1 / 40.7 / 51.7 | 2467 / 2450 | 69 | 37.4 | 1.07 | 6.07 | 227 |

Serving matrix: completed

| Case | Rows | Mean wall s | Mean aggregate tok/s | Mean TTFT s | Mean decode tok/s |
|---|---:|---:|---:|---:|---:|
| single8k | 5 | 17.084 | 29.969 | 4.343 | 40.106 |
| pair8k | 5 | 34.524 | 29.661 | 12.873 | 39.541 |
| single64k | 5 | 47.620 | 10.752 | 34.600 | 39.252 |
| pair64k | 5 | 95.140 | 10.763 | 58.359 | 26.292 |
| mixed | 5 | 64.630 | 15.844 | 25.827 | 39.747 |
