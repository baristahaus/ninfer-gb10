## Flash-Next MoE microbench

- Commit: 59041aba (with local changes)
- GPU/driver: NVIDIA GB10, 580.173.02
- T=4 rows, 4 layer banks, distinct experts 10,20,40,64,80

```
flash_next_moe NVFP4 decode route: T=4 rows, 4 layer banks of 1.42 GB, 2.765 MB streamed per selected expert
wide_gate  distinct      us/layer       min us    routed GB/s
off        10               191.9        190.6          144.1
off        20               320.9        319.0          172.3
off        40               577.7        574.7          191.4
off        40               576.9        575.4          191.7
off        40               577.1        575.5          191.6
fit wide_gate=off: 12.84 us per distinct expert (215 GB/s, 88% of 246 GB/s), 63.8 us fixed per layer
on         10               191.6        190.3          144.3
on         20               320.6        319.1          172.5
on         40               577.1        575.4          191.6
on         40               576.9        575.3          191.7
on         40               577.0        575.2          191.7
fit wide_gate=on: 12.84 us per distinct expert (215 GB/s, 88% of 246 GB/s), 63.4 us fixed per layer
```

ncu (one eager pass, 2 layers, 80 distinct experts): see ncu-details.txt and moe-ncu.ncu-rep.
