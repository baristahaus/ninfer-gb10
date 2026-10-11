# GB10 control run and discriminator: the 2026-10-11 gap (control-2026-10-11)

2026-10-11 19:08-19:25 UTC, current head, `fp8mtp` artifact, MTP K=3. Two runs: the control run
(the published 2026-10-03 protocol on the current head) and the discriminator (the sweep's own
requests at C1 and C4 on a `--max-concurrency 4` server).

## Purpose

The 2026-10-11 MTP concurrency sweep (one server sized for eight requests, 8000-character prompts,
`--ignore-eos` to a fixed 512 tokens) measured, at the rows comparable to the published 2026-10-03
figures, 45.3 tok/s at one request and 81.5 at four — below the published 59.9 (one request) and
121.4 (four concurrent). (The sweep's six and eight-request rows are 87.4 and 100.3.) Two candidate
causes: a protocol difference (the sweep's workload differs from the published protocol) or a code
regression since 2026-10-03.

- The **control run** re-runs the published 2026-10-03 protocol on the current head to test for a
  regression.
- The **discriminator** re-runs the sweep's own requests (8000-character prompts, `--ignore-eos`
  to 512, the perplexity-corpus prompt set) at C1 and C4 on a server started with
  `--max-concurrency 4` (the sweep used 8) to test whether the 8-lane server sizing is the cause.

## Setup

- Tree: see `tree.txt` (current head).
- Artifact: `out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer` (4 volumes).
- Control run: server `--max-context 73728 --max-concurrency 4 --kv-dtype fp8 --preserve-thinking
  --spec mtp --draft-tokens 3 --lm-head-draft`, port 18087; client `~/dgpp/scripts/serve_load.py
  127.0.0.1 18087 --concurrency 1,2,4 --classes prose,code,json,math,chat --max-tokens 256
  --repeat 3` (DGPP at `4198955`); page cache dropped before the server start.
- Discriminator: server `--max-context 73728 --kv-capacity auto --max-concurrency 4 --kv-dtype fp8
  --spec mtp --draft-tokens 3 --lm-head-draft --preserve-thinking`, port 8001; client
  `tools/gb10/concurrency_sweep.py http://127.0.0.1:8001 --n 1,4 --max-tokens 512
  --prompt-chars 8000 --ignore-eos`.

## Control run result

Per-class wall tok/s (prefill included), median of three repeats, averaged over the five classes:

| C | control (this run) | published 2026-10-03 | delta |
|---|---:|---:|---:|
| 1 | 60.7 | 59.89 | +1.4% |
| 2 | 92.3 | 91.53 | +0.9% |
| 4 | 122.3 | 121.39 | +0.7% |

Per class (this run):

| class | c=1 | c=2 | c=4 |
|---|---:|---:|---:|
| prose | 47.3 | 69.8 | 99.3 |
| code | 64.2 | 99.9 | 132.3 |
| json | 76.2 | 113.6 | 144.4 |
| math | 70.7 | 106.0 | 130.2 |
| chat | 45.1 | 72.4 | 105.3 |

All 105 requests status 200, finish=length; no errors.

## Discriminator result

The sweep's own requests (8000-character prompts, `--ignore-eos` to 512) at C1 and C4 on a
`--max-concurrency 4` server:

| C | discriminator | sweep (8-lane) | published 2026-10-03 |
|---|---:|---:|---:|
| 1 | 45.0 | 45.3 | 59.89 |
| 4 | 77.5 | 81.5 | 121.39 |

The C1/C4 rates match the sweep's 8-lane rows (45.0 vs 45.3, 77.5 vs 81.5), not the published
figures. So the 8-lane server sizing (`--max-concurrency 8`) is **not** the cause of the gap.

## Conclusion

- **No regression:** the current head reproduces the published 2026-10-03 figures within about 1.4%
  at every concurrency. The code changes since 2026-10-03 did not move the served rate on the
  published protocol.
- **The gap is the sweep's protocol, not a regression.** The sweep differs from the published
  protocol in four ways at once: (1) one server sized for eight requests, (2) 8000-character
  prompts, (3) `--ignore-eos` to a fixed 512 tokens, and (4) a different prompt set (the perplexity
  corpus). The discriminator rules out (1): the 8-lane server sizing does not change the C1/C4
  rates. The remaining variables — (2) the 8000-character prompts, (3) the `--ignore-eos` fixed
  512-token length, and (4) the prompt set — are not isolated.
- No scaling or regression claim is made for the sweep.

## Code changes since 2026-10-03

The behavior-changing changes since the 2026-10-03 tree (`05ed1f2a`) are the two dnf (six-to-
eight-request MTP crash) fixes:
- `7e2ab697` fix(linear): keep captured BF16 GEMMs off cuBLAS (the vendor route now starts at 49
  columns);
- `1ab69dd9` fix(qsa): write zeros for inactive columns on the unsplit batched QSA route.

`a8f3455d` is the README lessons commit (documentation only). The other commits since 2026-10-03
are bench/test/docs records.

## Records

- `ninfer-k3.txt` / `ninfer-k3.json`: the control run's `serve_load.py` output and raw requests.
- `ninfer-k3.jsonl` / `ninfer-k3.md`: the control run's request log and its decode-only summary.
- `server.log`: the control run's server log.
- `control_run.sh`: the control run's reproducible script.
- `discr-k3.txt` / `discr-k3.json`: the discriminator's sweep output and raw requests.
- `discr-server.log`: the discriminator's server log.
- `discriminator.sh`: the discriminator's reproducible script.
