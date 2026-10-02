## Flash-Next MoE microbench

- Commit: 59041aba (with local changes)
- GPU/driver: NVIDIA GB10, 580.173.02
- T=8 rows, 4 layer banks, distinct experts 10,20,40,64,80

```
flash_next_moe NVFP4 decode route: T=8 rows, 4 layer banks of 1.42 GB, 2.765 MB streamed per selected expert
wide_gate  distinct      us/layer       min us    routed GB/s
off        10               190.8        189.8          144.9
off        20               326.1        324.4          169.6
off        40               584.8        582.8          189.1
off        64               892.2        889.3          198.3
off        80              1097.0       1092.2          201.6
fit wide_gate=off: 12.92 us per distinct expert (214 GB/s, 87% of 246 GB/s), 65.3 us fixed per layer
on         10               191.1        190.0          144.7
on         20               326.2        324.7          169.5
on         40               583.7        581.7          189.5
on         64               891.8        888.5          198.4
on         80              1096.0       1092.9          201.8
fit wide_gate=on: 12.90 us per distinct expert (214 GB/s, 87% of 246 GB/s), 65.6 us fixed per layer
```

ncu (one eager pass, 2 layers, 80 distinct experts): see ncu-details.txt and moe-ncu.ncu-rep.
