### profiles/bench/gb10/long-context-lookup/ninfer-k3-lookup.jsonl

| req | prompt | queue wait s | TTFT s | computed prefill | completion | prefill s | decode s | decode tok/s | rounds | tok/round | device wait ms/round | host exposed ms/round |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33397 | 4 ms | 15.54 | 33397 | 1 | 15.53 | 0.00 | nan | 0 | nan | nan | nan |
| 2 | 15638 | 2 ms | 7.20 | 15638 | 1536 | 7.19 | 25.40 | 60.5 | 522 | 2.94 | 48.6 | 0.02 |
| 3 | 15443 | 7 ms | 7.10 | 15443 | 1536 | 7.10 | 27.01 | 56.9 | 550 | 2.79 | 49.1 | 0.02 |
| 4 | 60481 | 14 ms | 28.20 | 60481 | 1536 | 28.19 | 25.30 | 60.7 | 514 | 2.99 | 49.2 | 0.02 |
| 5 | 60453 | 58 ms | 28.16 | 60453 | 1536 | 28.10 | 25.13 | 61.1 | 506 | 3.04 | 49.7 | 0.02 |
| 7 | 60398 | 120 ms | 28.14 | 60398 | 64 | 28.02 | 1.36 | 47.1 | 21 | 3.05 | 64.7 | 0.04 |
| 6 | 15509 | 26 ms | 7.12 | 15509 | 1536 | 7.10 | 25.41 | 60.5 | 512 | 3.00 | 49.6 | 0.04 |
| 9 | 1878 | 692 ms | 1.68 | 1878 | 64 | 0.99 | 1.02 | 62.5 | 21 | 3.05 | 48.7 | 0.04 |
| 8 | 60227 | 78 ms | 28.04 | 60227 | 64 | 27.96 | 1.18 | 54.3 | 24 | 2.67 | 49.1 | 0.04 |

Totals: 7873 tokens, decode 131.8 s (59.7 tok/s), prefill 150.2 s, rounds 2670, 2.95 tok/round, device wait 49.4 ms/round, queue wait mean 111 ms, host exposed 0.02 ms/round
Prompt lookup: 57 of 2665 drafted rounds, 95 of 171 drafts accepted

