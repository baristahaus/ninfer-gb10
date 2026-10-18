# Flash-Next MTP offline torch parity — final report (2026-10-18)

Scope: complete standalone torch replication of the Flash-Next MTP draft layer
(125B-A6B NVFP4+FP8 MTP artifact, GB10, `--no-cuda-graph --kv-dtype fp8 --spec
mtp --draft-tokens 2 --lm-head-draft --greedy`), verified against windowed
engine dumps (`NINFER_MTP_DUMP`, parity window `NINFER_MTP_PARITY_CALL`).

## Engine call structure (measured)

- Prefill runs TWO MTP core calls: T=18 (prompt chunk, non-final) then T=2
  (final chunk). The final chunk proposes the draft from `sample[:, 1]`.
- Decode rounds: verify call (T=3 after the first round) then a T=1 propose
  call; one proposal per round.

## Verified formulas

1. **Layer dataflow** — QSA and MoE run in PARALLEL on `block1` (the attention
   HC-mixer output); the MoE output folds into the hyper state through the MLP
   HC mixer; the final HC mixer produces the sample.
2. **HC mixers** — per-stream rmsnorm (`(w+1)`, eps 1e-6), fp8 down/up
   projections, `0.25*sigmoid` low-rank gate, branch mean, sigmoid injection
   fold (`2/(1+exp(-inj/4))` scale).
3. **QSA** — packed fp8 query/gate projection (24 heads x [q256, gate256]),
   per-head rmsnorm, generic MRoPE (rotary 64, theta 1e7, axis = pair%3, dims
   (p, p+32)), H256/16 Hadamard on queries AND on K/V at cache append,
   row-scaled FP8 E4M3 KV (scale = fp16(448/absmax)), attention scale 1/16 with
   fp32 softmax over all visible keys (dense route within the 2051 budget),
   sigmoid gate, fp8 row-scaled output projection (scale applied at the fp32
   reduction).
4. **MoE** — NVFP4 W4A4 grouped path with PER-EXPERT weight scale divisors
   (gu 1933..15910, down 3171..22378) and per-expert activation divisors
   (gate input 186.182, h 34.684); quantize = e4m3(div*maxabs/6) scale,
   e2m1 round-half-even sign-magnitude codes; epilogue divides the fp32
   reduction by (weight_div * input_div).
5. **Proposal** — Q4_G64_FP16 shortlist head (nibble (n^8)-8, fp16 scale per
   64-group) over 147456 rows, per-512-tile top-2, id remap, exact rescore with
   the fp8 row-scaled lm_head (248320 x 2560), argmax.

## Parity numbers (engine operands)

| Stage | Max diff | Relative |
|---|---|---|
| HC-mix block | 0.00003 | ~0 |
| QSA attention | 0.0625 | 0.36% |
| QSA destination | 0.0156 | 0.32% |
| MoE output | 0.0078 | 0.94% |
| Gate chain h (fp64) | 0.0156 | 1-2 bf16 ULP |
| Proposal-head logits | 0.031 | 0.19% |
| Proposal exact scores | 0.053 | ~0.3% |

x-quantization reproduces the engine's NVFP4 codes 100% byte-exact; h codes
99.3%.

## Fully self-contained forward (own routing + own quantization)

Last-bit bf16 rounding differences (fp32 accumulation order in the engine's
mma vs torch) flip fp8/nvfp4 codes that sit at quantization midpoints; the
flat-tail softmax amplifies them (per-token QSA rel up to ~0.4). Despite that,
the end-to-end proposal token matches the engine: **1156 == 1156** on the test
run (`torch_draft/final_chunk.py`).

## Artifacts

- `torch_draft/draft_forward.py` — consolidated standalone forward (HC mix,
  QSA, MoE, mixes, proposal).
- `torch_draft/final_chunk.py` — T=2 final-chunk replay, token verification.
- `torch_draft/parity_qsa.py`, `parity_moe_w4a4.py`, `parity_proposal.py`,
  `parity_full.py` — stage-level parity scripts.
- Engine harness: commit e9fe2c47 on `proposal-head-rows` (windowed dump
  instrumentation; no cost without `NINFER_MTP_DUMP`).
- Dumps: `/nfs/mtp-correction/oracle_dumps4/` (T=18 and T=2 call oracles).

## Residual limitations

- Bit-exact self-contained replication would require emulating the engine's
  mma tile accumulation order; the current boundary is quantization-decision
  chaos, not a formula error.
- The T=18 chunk stage dumps and the T=2 oracle come from separate runs; the
  harness now gates every dump on one selectable call.
