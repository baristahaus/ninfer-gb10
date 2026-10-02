## Flash-Next MoE microbench

- Commit: 1a44fe6f
- GPU/driver: NVIDIA GB10, 580.173.02
- T=8 rows, 4 layer banks, distinct experts 10,20,40,64,80

```
flash_next_moe NVFP4 decode route: T=8 rows, 4 layer banks of 1.42 GB, 2.765 MB streamed per selected expert
wide_gate  distinct      us/layer       min us    routed GB/s
off        10               295.0        293.6           93.7
off        20               778.1        774.5           71.1
off        40              1071.4       1069.4          103.2
off        64              1078.9       1076.1          164.0
off        80              1077.6       1075.1          205.3
fit wide_gate=off: 9.47 us per distinct expert (292 GB/s, 119% of 246 GB/s), 454.9 us fixed per layer
on         10               288.1        286.6           96.0
on         20               848.9        843.3           65.1
on         40              1081.4       1078.1          102.3
on         64              1087.9       1085.6          162.7
on         80              1087.8       1084.0          203.3
fit wide_gate=on: 9.22 us per distinct expert (300 GB/s, 122% of 246 GB/s), 484.1 us fixed per layer
```

ncu (one eager pass, 2 layers, 80 distinct experts): see ncu-details.txt and moe-ncu.ncu-rep.
