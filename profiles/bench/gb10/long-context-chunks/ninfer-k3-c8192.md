### profiles/bench/gb10/long-context-chunks/ninfer-k3-c8192.jsonl

| req | prompt | queue wait s | TTFT s | computed prefill | completion | prefill s | decode s | decode tok/s | rounds | tok/round | device wait ms/round | host exposed ms/round |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33397 | 4 ms | 15.03 | 33397 | 1 | 15.03 | 0.00 | nan | 0 | nan | nan | nan |
| 2 | 15638 | 2 ms | 6.90 | 15638 | 1536 | 6.89 | 24.89 | 61.7 | 517 | 2.97 | 48.1 | 0.02 |
| 3 | 15443 | 7 ms | 6.78 | 15443 | 1536 | 6.77 | 26.70 | 57.5 | 549 | 2.80 | 48.6 | 0.02 |
| 4 | 60481 | 14 ms | 27.29 | 60481 | 1536 | 27.27 | 25.05 | 61.3 | 513 | 2.99 | 48.8 | 0.02 |
| 5 | 60453 | 57 ms | 27.32 | 60453 | 1536 | 27.26 | 24.80 | 61.9 | 504 | 3.05 | 49.2 | 0.02 |
| 7 | 60398 | 121 ms | 27.28 | 60398 | 64 | 27.16 | 1.33 | 48.0 | 21 | 3.05 | 63.5 | 0.03 |
| 6 | 15509 | 26 ms | 6.81 | 15509 | 1536 | 6.79 | 24.71 | 62.2 | 505 | 3.04 | 48.9 | 0.04 |

Totals: 7745 tokens, decode 127.5 s (60.7 tok/s), prefill 117.2 s, rounds 2609, 2.97 tok/round, device wait 48.9 ms/round, queue wait mean 33 ms, host exposed 0.02 ms/round

