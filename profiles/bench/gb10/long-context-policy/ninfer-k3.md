### profiles/bench/gb10/long-context-policy/ninfer-k3.jsonl

| req | prompt | queue wait s | TTFT s | computed prefill | completion | prefill s | decode s | decode tok/s | rounds | tok/round | device wait ms/round | host exposed ms/round |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33397 | 4 ms | 15.52 | 33397 | 1 | 15.52 | 0.00 | nan | 0 | nan | nan | nan |
| 2 | 15638 | 3 ms | 7.21 | 15638 | 1536 | 7.20 | 24.93 | 61.6 | 517 | 2.97 | 48.2 | 0.02 |
| 3 | 15443 | 7 ms | 7.12 | 15443 | 1536 | 7.12 | 26.77 | 57.4 | 549 | 2.80 | 48.7 | 0.02 |
| 4 | 60481 | 13 ms | 28.25 | 60481 | 1536 | 28.24 | 25.10 | 61.2 | 513 | 2.99 | 48.9 | 0.02 |
| 5 | 60453 | 57 ms | 28.28 | 60453 | 1536 | 28.22 | 24.93 | 61.6 | 504 | 3.05 | 49.4 | 0.02 |
| 6 | 15509 | 26 ms | 7.15 | 15509 | 1536 | 7.13 | 25.14 | 61.1 | 511 | 3.01 | 49.2 | 0.05 |
| 7 | 60398 | 20109 ms | 48.27 | 60398 | 64 | 28.16 | 1.32 | 48.4 | 21 | 3.05 | 62.9 | 0.03 |

Totals: 7745 tokens, decode 128.2 s (60.4 tok/s), prefill 121.6 s, rounds 2615, 2.96 tok/round, device wait 49.0 ms/round, queue wait mean 2888 ms, host exposed 0.02 ms/round

