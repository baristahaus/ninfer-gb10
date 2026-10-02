## Flash-Next MoE microbench

- Commit: a3b01e76 (with local changes)
- GPU/driver: NVIDIA GB10, 580.173.02
- T=8 rows, 4 layer banks, distinct experts 10,20,40,64,80

```
flash_next_moe NVFP4 decode route: T=8 rows, 4 layer banks of 1.42 GB, 2.765 MB streamed per selected expert
wide_gate  distinct      us/layer       min us    routed GB/s
off        10               189.8        189.0          145.6
off        20               320.9        319.7          172.3
off        40               576.9        574.4          191.7
off        64               883.3        882.0          200.3
off        80              1086.1       1084.5          203.7
fit wide_gate=off: 12.79 us per distinct expert (216 GB/s, 88% of 246 GB/s), 63.8 us fixed per layer
on         10               189.7        189.1          145.8
on         20               320.9        319.4          172.3
on         40               576.4        574.1          191.9
on         64               882.8        880.6          200.4
on         80              1086.1       1083.8          203.6
fit wide_gate=on: 12.79 us per distinct expert (216 GB/s, 88% of 246 GB/s), 63.7 us fixed per layer
```

ncu (one eager pass, 2 layers, 80 distinct experts): see ncu-details.txt and moe-ncu.ncu-rep.
