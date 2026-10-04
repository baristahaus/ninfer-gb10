### profiles/bench/gb10/long-context-chunks/ninfer-k3-c2048.jsonl

| req | prompt | queue wait s | TTFT s | computed prefill | completion | prefill s | decode s | decode tok/s | rounds | tok/round | device wait ms/round | host exposed ms/round |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33397 | 4 ms | 17.76 | 33397 | 1 | 17.76 | 0.00 | nan | 0 | nan | nan | nan |
| 2 | 15638 | 2 ms | 8.22 | 15638 | 1536 | 8.22 | 24.10 | 63.7 | 502 | 3.06 | 48.0 | 0.02 |
| 3 | 15443 | 7 ms | 8.12 | 15443 | 1536 | 8.11 | 26.53 | 57.9 | 548 | 2.80 | 48.4 | 0.02 |
| 4 | 60481 | 14 ms | 32.56 | 60481 | 1536 | 32.55 | 26.62 | 57.7 | 549 | 2.80 | 48.5 | 0.02 |
| 5 | 60453 | 58 ms | 32.57 | 60453 | 1536 | 32.52 | 24.22 | 63.4 | 492 | 3.12 | 49.2 | 0.02 |
| 7 | 60398 | 124 ms | 32.56 | 60398 | 64 | 32.43 | 1.27 | 50.4 | 20 | 3.20 | 63.4 | 0.02 |
| 6 | 15509 | 26 ms | 8.15 | 15509 | 1536 | 8.13 | 24.54 | 62.6 | 506 | 3.04 | 48.5 | 0.04 |

Totals: 7745 tokens, decode 127.3 s (60.8 tok/s), prefill 139.7 s, rounds 2617, 2.96 tok/round, device wait 48.6 ms/round, queue wait mean 33 ms, host exposed 0.02 ms/round

