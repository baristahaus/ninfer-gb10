# Step 7a baseline: lookup2-gb10-k3-lookup-r1

- Host: Linux 6.17.0-1029-nvidia aarch64; Ubuntu 24.04.4 LTS
- Commit: 1307107c (with local changes)
- nvcc: release 13.0, V13.0.88
- GPU/driver: NVIDIA GB10, 580.173.02
- Artifact: qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer, 119G (multi-volume total)

- Artifact: `/home/apollo11/ninfer-gb10/out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer`
- Serve args: --max-context 73728 --max-concurrency 1 --kv-dtype fp8 --spec mtp --draft-tokens 3 --lm-head-draft --prompt-lookup --preserve-thinking
- KV dtype (perplexity): fp8
- CPU: pinned to X925 cores 5-9,15-19 (taskset)

## 1. Perplexity token scores (fixed corpus, default protocol)

- report.json: missing or no overall section

Product table (stdout):

```
(missing)
```

## 2. MTP acceptance (serving run, greedy, 1024 tokens/stream)

- requests: 16, rounds: 6282
- drafted: 18832, accepted: 9909
- acceptance ratio (accepted/drafted): 0.526179
- fallback steps: 8
- per-position per-round: {'0': 0.705985, '1': 0.499204, '2': 0.372174}
- generated tokens: 16215 total, 855-1024 per stream
- wall time: 334s total

## 3. TEB hardmode (--seed 42; single trial by default, TEB_TRIALS overrides)

Report dir: `teb/`. Console output (tail):

```
(missing)
```
