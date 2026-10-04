### profiles/bench/gb10/long-context-chunks/ninfer-k3-c1024.jsonl

| req | prompt | queue wait s | TTFT s | computed prefill | completion | prefill s | decode s | decode tok/s | rounds | tok/round | device wait ms/round | host exposed ms/round |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33397 | 4 ms | 21.08 | 33397 | 1 | 21.07 | 0.00 | nan | 0 | nan | nan | nan |
| 2 | 15638 | 2 ms | 9.86 | 15638 | 1536 | 9.86 | 24.06 | 63.8 | 502 | 3.06 | 47.9 | 0.02 |
| 3 | 15443 | 7 ms | 9.64 | 15443 | 1536 | 9.64 | 26.49 | 58.0 | 548 | 2.80 | 48.3 | 0.02 |
| 4 | 60481 | 14 ms | 39.07 | 60481 | 1536 | 39.05 | 24.93 | 61.6 | 514 | 2.99 | 48.5 | 0.02 |
| 5 | 60453 | 57 ms | 38.98 | 60453 | 1536 | 38.93 | 25.65 | 59.9 | 524 | 2.93 | 48.9 | 0.02 |
| 7 | 60398 | 124 ms | 38.96 | 60398 | 64 | 38.84 | 1.29 | 49.7 | 20 | 3.20 | 64.4 | 0.02 |
| 6 | 15509 | 26 ms | 9.77 | 15509 | 1536 | 9.74 | 24.60 | 62.4 | 505 | 3.04 | 48.7 | 0.05 |

Totals: 7745 tokens, decode 127.0 s (61.0 tok/s), prefill 167.1 s, rounds 2613, 2.96 tok/round, device wait 48.6 ms/round, queue wait mean 34 ms, host exposed 0.02 ms/round

