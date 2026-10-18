# MTP economics — draft-tokens sweep (2026-10-18)

Artifact: `out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer`, GB10,
single stream, greedy, `--max-new 512`, prompt "Explain how a transformer
attention layer works, step by step." Wall time minus ~21 s model load.

| Config | Wall (s) | ~Gen (s) | ~tok/s | vs no-MTP |
|---|---|---|---|---|
| `--spec mtp --draft-tokens 2 --lm-head-draft` | 34.7 | 13.7 | 37 | **0.95x** |
| `--spec mtp --draft-tokens 1 --lm-head-draft` | 33.7 | 12.7 | 40 | 1.03x |
| no MTP | 34.0 | 13.0 | 39 | 1.00x |

Context: the acceptance battery measured mean accept length 1.40 tok/round at
draft-tokens 2 (range 1.25-1.69), with the second draft position accepted only
rarely (run 1: 18 at position 1 vs 5 at position 2).

## Finding

At the measured acceptance, MTP with draft-tokens 2 is a net **loss** (0.95x)
and draft-tokens 1 is roughly neutral (1.03x) for single-stream greedy decode
on GB10. The per-round overhead (a full MTP-layer QSA+MoE forward plus the
wider verify batch) exceeds the acceptance gain. With 128-token prompts the
difference is inside the load noise; at 512 tokens it resolves to the numbers
above.

## Implications

- The cheapest lever is round overhead, not acceptance: the T=1 propose call
  and the verify-batch width dominate. Options: overlap the propose forward
  with the verify round, shrink the propose pass, or accept draft-tokens 1
  until the overhead drops.
- Position-2 acceptance is a secondary lever; the offline torch model
  (tools/gb10/torch_parity) can attribute proposal-head quality at position 2
  if that lever is pursued.

Raw times: `512-*.time` in this directory. Battery context:
`profiles/bench/gb10/acceptance-2026-10-18/summary.md`.
