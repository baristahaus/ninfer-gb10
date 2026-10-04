### profiles/bench/gb10/long-context-chunks/ninfer-k3-c4096.jsonl

| req | prompt | queue wait s | TTFT s | computed prefill | completion | prefill s | decode s | decode tok/s | rounds | tok/round | device wait ms/round | host exposed ms/round |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33397 | 4 ms | 15.60 | 33397 | 1 | 15.59 | 0.00 | nan | 0 | nan | nan | nan |
| 2 | 15638 | 2 ms | 7.23 | 15638 | 1536 | 7.23 | 24.89 | 61.7 | 517 | 2.97 | 48.1 | 0.02 |
| 3 | 15443 | 7 ms | 7.15 | 15443 | 1536 | 7.14 | 26.68 | 57.6 | 549 | 2.80 | 48.6 | 0.02 |
| 4 | 60481 | 14 ms | 28.32 | 60481 | 1536 | 28.30 | 25.06 | 61.3 | 513 | 2.99 | 48.8 | 0.02 |
| 5 | 60453 | 58 ms | 28.33 | 60453 | 1536 | 28.27 | 24.80 | 61.9 | 504 | 3.05 | 49.2 | 0.02 |
| 7 | 60398 | 118 ms | 28.39 | 60398 | 64 | 28.27 | 1.32 | 48.5 | 21 | 3.05 | 62.8 | 0.02 |
| 6 | 15509 | 26 ms | 7.17 | 15509 | 1536 | 7.14 | 24.98 | 61.5 | 510 | 3.01 | 49.0 | 0.04 |

Totals: 7745 tokens, decode 127.7 s (60.6 tok/s), prefill 121.9 s, rounds 2614, 2.96 tok/round, device wait 48.8 ms/round, queue wait mean 33 ms, host exposed 0.02 ms/round

