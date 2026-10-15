# SP-6 serving session — resident set under load (production flags)

- 2026-10-15 ~01:57–02:00Z, commit 8be2bb12 lineage (branch tip at session time), GB10 sm_121a
- Artifact: qwen3_8_flash_next_125b_a6b_nvfp4.ninfer (126 GB multi-volume)
- Serve flags: `--max-context 262144 --kv-capacity auto --max-concurrency 2 --kv-dtype fp8
  --spec mtp --draft-tokens 2 --lm-head-draft --vision` (the verified production spec)
- Load: 8k/64k single and pair phases, 512 outputs each, natural prefix reuse (no cache_salt),
  temperature 0. `/metrics` + `/proc/<pid>/status` + `free -g` sampled every 15 s (8 samples).

## Resident set

| State | Weights | Engine runtime | KV pool | Process VmRSS | System |
|---|---|---|---|---|---|
| Startup (idle) | 77.8 GiB (page-cache mapped) | 11.4 GiB | auto: 524,288 tokens, 8,192/8,192 pages | 11.47 GiB (RssAnon 1.93, RssFile 0.20) | 14.4 GiB free |
| After full session | same | same | 76,224 tokens retained (kept_prompts 7) | 11.63 GiB (RssAnon 1.97, RssFile 0.32) | 109 used / 3 free / 23 cache / 11 available |

- Host KV spill untouched under this load: `host_kv_occupied_bytes 0` of an 8 GiB allowance.
- Device state slots 4 occupied, host state slots 5 (session checkpoints) after the session.
- No swap at any point; no degraded/evicted private owners.

## Serving numbers at production flags (vs the cold matrix)

| Phase | Cold (matrix, no reuse) | This session (reuse on) |
|---|---|---|
| single8k | 17.1 s, TTFT 4.3 s | first 17.2 s (TTFT 4.6 s), then **12.7–12.8 s (TTFT 125–17 ms)** |
| pair8k | 34.5 s, **0 overlap** | **15.5–15.8 s, both lanes concurrent** (TTFT 219–371 ms, queue ≤340 ms) |
| single64k | 47.6 s, TTFT 34.6 s | first 48.2 s, then **13.2 s (TTFT 157 ms, 100% cached)** |
| pair64k | (503 in the cold matrix) | **16.4–16.6 s, both lanes concurrent** |
| Decode per lane | 40.4 tok/s (batch 1) | 31.4–33.0 tok/s at batch 2 → **~64 tok/s aggregate** |
| MTP acceptance | 73.8–74.4% | 74.8–78.8% |

## The concurrency finding (corrects the step 2 record)

The step 2 matrix (no reuse, cache_salt per request) showed zero overlap for any prompt over
the 4,096-token device-default prefill chunk: `defer_long_prefill` serializes long prefills
behind the running request. With prefix reuse ON at production flags, cached prompts re-enter
with prefill work below the chunk threshold, skip the long-prefill deferral entirely, and
co-admit: pair8k 15.7 s (vs 34.5 serialized), pair64k 16.6 s with batch-2 decode. **Prefix
reuse is the concurrency mechanism on GB10 for prompts over 4k tokens.** Cold (>4k) pairs
serialize; a same-time second cold 64k request still expires at the default 30 s pending
deadline (see the step 2 record).

## Artifacts

- `load.jsonl`, `load.log` — per-request walls and phases
- `load-samples.txt` — 8 × `/metrics` + process status + free samples during load
- `metrics-idle.txt`, `metrics-after.txt` — full Prometheus dumps
- `status-idle.txt`, `status-after.txt`, `free-*.txt` — resident-set evidence
- `server.log` — per-request lines (prompt/cached/TTFT/decode/MTP per request)
