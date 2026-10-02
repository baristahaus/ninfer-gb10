## Flash-Next MoE microbench

- Commit: 42722673 (with local changes)
- GPU/driver: NVIDIA GB10, 580.173.02
- T=2 rows, 4 layer banks, distinct experts 10,20,40,64,80

```
flash_next_moe NVFP4 decode route: T=2 rows, 4 layer banks of 1.42 GB, 2.765 MB streamed per selected expert
distinct      us/layer       min us    routed GB/s
10               188.0        186.5          147.1
20               287.3        286.0          192.5
20               287.6        286.1          192.3
20               287.4        285.9          192.4
20               287.0        286.1          192.7
fit: 9.93 us per distinct expert (278 GB/s, 113% of 246 GB/s), 88.7 us fixed per layer
```

ncu (one eager pass, 2 layers, 80 distinct experts): see ncu-details.txt and moe-ncu.ncu-rep.
