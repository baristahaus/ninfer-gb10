## MoE grouped decode (8+ rows): final merge run (2111f3f9)

- Oracle test (`ninfer_flash_next_moe_test`): PASS. T=1/2 on the per-row route,
  T=8/16/17 on the grouped route; 0.16-0.29% relative L2 across all T and tokens;
  8-rows batched vs one-by-one bitwise exact.
- Real test (`ninfer_qwen3_8_flash_next_real_test`): PASS, bitwise vs goldens,
  no change (83 s).

- C4 K=1 untraced (two batches of 4 x 256 tokens, 2000-char prompts, ignore-eos,
  page cache dropped before the load):
  26.3 tok/s (77.9 s decode) vs 25.5 tok/s (80.3 s) working-branch baseline = +3.1%.
- C4 K=1 traced (round-attribution protocol, `round-attribution-merged/`):
  26.3 tok/s (77.8 s) vs 25.9 tok/s (79.0 s) traced working baseline = +1.5%.
- moe.nvfp4 per call: 837.5 us (5387.1 ms / 6432) vs 896.3 us working = -6.6%;
  exp-branch run: 827.7 us.
- Host time (trace sqlite, launch APIs per round): merged 1.675 ms vs working
  1.773 ms (flat) vs exp-branch 2.135 ms (+0.36). Kernels/round: 535 merged
  vs 524 working vs 584 exp. Traced host-exposed (request log, noisy under nsys):
  2.19 merged vs 2.58 working vs 3.25 exp.

Finding: the exp branch's +0.67 ms/round host-exposed delta did not carry over.
The exp tree grouped 2..16-row calls, so sub-8-row MoE calls also paid the grouped
launch cost (~+60 kernels/round, +0.36 ms/round of launch-API host time). At the
8+ threshold those calls stay on the per-row route: host launch cost is flat,
while the MoE per-call gain (837.5 vs 896.3 us) and the +3.1% end-to-end remain.

- Commit: 2111f3f9 (2026-10-02T16:05Z), twoFour
