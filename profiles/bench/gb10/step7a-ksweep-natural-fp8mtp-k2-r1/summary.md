# Step 7a baseline: ksweep-natural-fp8mtp-k2-r1

- Host: Linux 6.17.0-1029-nvidia aarch64; Ubuntu 24.04.4 LTS
- Commit: acf6f5f7
- nvcc: release 13.0, V13.0.88
- GPU/driver: NVIDIA GB10, 580.173.02
- Artifact: qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer, 119G (multi-volume total)

- Artifact: `/home/apollo11/ninfer-gb10/out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer`
- Serve args: --max-context 73728 --max-concurrency 1 --kv-dtype fp8 --spec mtp --draft-tokens 2 --lm-head-draft --preserve-thinking
- KV dtype (perplexity): fp8
- CPU: pinned to X925 cores 5-9,15-19 (taskset)

## 1. Perplexity token scores (fixed corpus, default protocol)

- report.json: missing or no overall section

Product table (stdout):

```
(missing)
```

## 2. MTP acceptance (serving run, greedy, 1024 tokens/stream)

- requests: 16, rounds: 7118
- drafted: 14235, accepted: 9074
- acceptance ratio (accepted/drafted): 0.637443
- fallback steps: 8
- per-position per-round: {'0': 0.73855, '1': 0.536246}
- generated tokens: 16215 total, 855-1024 per stream
- wall time: 332s total

## 3. TEB hardmode (--seed 42; single trial by default, TEB_TRIALS overrides)

Report dir: `teb/`. Console output (tail):

```
(missing)
```
