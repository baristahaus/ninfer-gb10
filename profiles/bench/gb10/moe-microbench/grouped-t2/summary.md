## Flash-Next MoE microbench

- Commit: 59041aba (with local changes)
- GPU/driver: NVIDIA GB10, 580.173.02
- T=2 rows, 4 layer banks, distinct experts 10,20,40,64,80

```
flash_next_moe NVFP4 decode route: T=2 rows, 4 layer banks of 1.42 GB, 2.765 MB streamed per selected expert
wide_gate  distinct      us/layer       min us    routed GB/s
off        10               189.7        188.5          145.7
off        20               320.3        318.0          172.6
off        20               321.3        318.6          172.1
off        20               319.6        318.1          173.0
off        20               319.8        318.6          172.9
fit wide_gate=off: 13.06 us per distinct expert (212 GB/s, 86% of 246 GB/s), 59.1 us fixed per layer
on         10               189.8        188.8          145.7
on         20               319.6        317.9          173.0
on         20               319.4        318.4          173.1
on         20               320.1        318.5          172.8
on         20               319.6        318.1          173.0
fit wide_gate=on: 12.98 us per distinct expert (213 GB/s, 87% of 246 GB/s), 60.0 us fixed per layer
```

ncu (one eager pass, 2 layers, 80 distinct experts): see ncu-details.txt and moe-ncu.ncu-rep.
