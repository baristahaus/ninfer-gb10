# Concurrency sweep (2026-10-11)

Two `concurrency_sweep.py` runs at MTP K=3 on the FP8 MTP artifact.

Server (one server for every row):

    ninfer-serve out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer \
      --host 0.0.0.0 --port 8000 --max-context 73728 --kv-capacity auto \
      --max-concurrency 8 --kv-dtype fp8 --spec mtp --draft-tokens 3 \
      --lm-head-draft --preserve-thinking

Sweep:

    tools/gb10/concurrency_sweep.py http://127.0.0.1:8000 <out.json> \
      --n 1,2,4,6,8 --max-tokens 512 --prompt-chars 8000 [--ignore-eos]

Prompts: first 8000 chars of distinct perplexity-corpus streams; thinking disabled
(`reasoning_effort: none`). Aggregate tok/s = sum of completion tokens / batch wall.

- `sweep-eos.json`: stop at EOS or the 512-token limit (token counts vary 10–512).
- `sweep-ignore-eos.json`: `--ignore-eos`, exactly 512 tokens per request.

Aggregate tok/s:

| N | eos | ignore-eos |
|---:|---:|---:|
| 1 | 41.0 | 45.3 |
| 2 | 51.9 | 62.7 |
| 4 | 53.8 | 81.5 |
| 6 | 79.2 | 87.4 |
| 8 | 89.7 | 100.3 |

Status: below the front page's 2026-10-03 figures (1 request 59.9, 4 requests 121.4).
The gap is unexplained — it could be the protocol (one 8-lane / 73728 server for every
row, 8000-char prompts, `--ignore-eos`) or a code regression since 2026-10-03. No scaling
or regression claim until the control run (published protocol on current head) lands.
Recorded only as: MTP at 6–8 requests runs, 87.4 and 100.3 tok/s aggregate under this
protocol.
