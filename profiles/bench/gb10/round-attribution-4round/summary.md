## Round attribution: C4, MTP draft 1

- Host: Linux 6.17.0-1029-nvidia aarch64; Ubuntu 24.04.4 LTS
- Commit: 0e8f7516 (with local changes)
- nvcc: release 13.0, V13.0.88
- GPU/driver: NVIDIA GB10, 580.173.02
- Artifact: qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer, 119G (multi-volume total)
- nsys: NVIDIA Nsight Systems version 2025.3.2.474-253236389321v0
- Load: two batches of 4 simultaneous greedy requests, 256 output tokens each
- Request log totals: Totals: 969 tokens, decode 36.5 s (26.5 tok/s), prefill 2.5 s, rounds 548, 1.77 tok/round, device wait 64.3 ms/round, queue wait mean 617 ms, host exposed 2.32 ms/round


Measured wall: 313.029 ms; GPU busy: 307.540 ms; attributed GPU work: 90.69%.

Estimates describe partial modeled work, not an achievable optimum. Bytes/FLOPs below are per call; time and share are aggregated.

| Phase / role | Stage | T / B / context envelope | Calls | GPU work ms | Phase share | Useful bytes min–max | BF16 / NVFP4 / FP32 FLOPs | Estimate ms min–max | Estimate efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | 8 / 0 / 0 | 192 | 171.422 | 55.7% | 35279520–228816640 | 99655680 / 786432000 / 0 | 27.535–178.589 | 16.1–104.2% |
| decode / target.verify | gdn.record | 8 / 4 / 0 | 144 | 46.765 | 15.2% | 71163904–71163904 | 926679040 / 0 / 37748736 | 41.657–41.657 | 89.1–89.1% |
| decode / target.verify | qsa.select | 8 / 4 / 129 | 48 | 24.207 | 7.9% | 55817216–55817216 | 823132160 / 0 / 0 | 10.891–10.891 | 45.0–45.0% |
| decode / target.verify | hyper.combine_mix | 8 / 0 / 0 | 376 | 20.685 | 6.7% | 7066368–7066368 | 105512960 / 0 / 0 | 10.801–10.801 | 52.2–52.2% |
| decode / target.verify | ple.record | 8 / 4 / 0 | 4 | 9.075 | 3.0% | 65884160–65884160 | 524288000 / 0 / 0 | 1.071–1.071 | 11.8–11.8% |
| decode / predictor | moe.nvfp4 | 8 / 0 / 0 | 4 | 3.561 | 1.2% | 35279520–228816640 | 99655680 / 786432000 / 0 | 0.574–3.721 | 16.1–104.5% |
| decode / predictor | qsa.select | 8 / 4 / 129 | 4 | 1.931 | 0.6% | 55817216–55817216 | 823132160 / 0 / 0 | 0.908–0.908 | 47.0–47.0% |
| decode / target.verify | hyper.mix | 8 / 0 / 0 | 8 | 0.413 | 0.1% | 6861504–6861504 | 105512960 / 0 / 0 | 0.223–0.223 | 54.0–54.0% |
| decode / predictor | hyper.combine_mix | 8 / 0 / 0 | 4 | 0.271 | 0.1% | 7066368–7066368 | 105512960 / 0 / 0 | 0.115–0.115 | 42.4–42.4% |
| decode / predictor | hyper.mix | 8 / 0 / 0 | 4 | 0.180 | 0.1% | 6861504–6861504 | 105512960 / 0 / 0 | 0.112–0.112 | 61.9–61.9% |
| decode / target.verify | hyper.mix | 8 / 0 / 0 | 4 | 0.178 | 0.1% | 6779520–6779520 | 104857600 / 0 / 0 | 0.110–0.110 | 62.1–62.1% |
| decode / predictor | hyper.mix | 8 / 0 / 0 | 4 | 0.176 | 0.1% | 6779520–6779520 | 104857600 / 0 / 0 | 0.110–0.110 | 62.8–62.8% |
| decode / target.verify | hyper.combine | 8 / 0 / 0 | 8 | 0.023 | 0.0% | 368704–368704 | 0 / 0 / 0 | 0.012–0.012 | 52.6–52.6% |
| decode / predictor | hyper.combine | 8 / 0 / 0 | 4 | 0.012 | 0.0% | 368704–368704 | 0 / 0 / 0 | 0.006–0.006 | 52.0–52.0% |
| decode / predictor | hyper.add | 8 / 0 / 0 | 4 | 0.009 | 0.0% | 368640–368640 | 0 / 0 / 0 | 0.006–0.006 | 70.2–70.2% |
| decode / target.verify | hyper.repeat | 8 / 0 / 0 | 4 | 0.008 | 0.0% | 204800–204800 | 0 / 0 / 0 | 0.003–0.003 | 39.6–39.6% |

Unattributed GPU work: 28.626 ms.

Per round over 4 rounds: host wall 78.257 ms, GPU work 76.885 ms, GPU busy 76.885 ms.

| Phase / role | Stage | GPU work ms/round | Share |
|---|---|---:|---:|
| decode / target.verify | moe.nvfp4 | 42.855 | 55.7% |
| decode / target.verify | gdn.record | 11.691 | 15.2% |
| decode / target.verify | qsa.select | 6.052 | 7.9% |
| decode / target.verify | hyper.combine_mix | 5.171 | 6.7% |
| decode / target.verify | ple.record | 2.269 | 3.0% |
| decode / predictor | moe.nvfp4 | 0.890 | 1.2% |
| decode / predictor | qsa.select | 0.483 | 0.6% |
| decode / target.verify | hyper.mix | 0.103 | 0.1% |
| decode / predictor | hyper.combine_mix | 0.068 | 0.1% |
| decode / predictor | hyper.mix | 0.045 | 0.1% |
| decode / target.verify | hyper.mix | 0.044 | 0.1% |
| decode / predictor | hyper.mix | 0.044 | 0.1% |
| decode / target.verify | hyper.combine | 0.006 | 0.0% |
| decode / predictor | hyper.combine | 0.003 | 0.0% |
| decode / predictor | hyper.add | 0.002 | 0.0% |
| decode / target.verify | hyper.repeat | 0.002 | 0.0% |
| unattributed | kernel recurrent_fold_kernel | 3.035 | 3.9% |
| unattributed | kernel fp8_a16_sliced_k_mma_kernel | 2.644 | 3.4% |
| unattributed | kernel q4_a16_sliced_k_mma_kernel | 0.895 | 1.2% |
| unattributed | kernel wait_staged_kernel | 0.188 | 0.2% |
| unattributed | kernel bf16_gemm_mma_kernel | 0.071 | 0.1% |
| unattributed | kernel bf16_small_t_inner_kernel | 0.058 | 0.1% |
| unattributed | kernel argmax_tiled_atomic_kernel | 0.054 | 0.1% |
| unattributed | kernel shortlist_exact_scores_fp8_kernel | 0.047 | 0.1% |
| unattributed | kernel speculative_sampling_partial_topk_kernel | 0.033 | 0.0% |
| unattributed | kernel shortlist_tile_candidates_kernel | 0.028 | 0.0% |
| unattributed | kernel ple_fold_kernel | 0.024 | 0.0% |
| unattributed | kernel rmsnorm_generic_kernel | 0.019 | 0.0% |
| unattributed | kernel speculative_sampling_group_finalize_kernel | 0.009 | 0.0% |
| unattributed | kernel embed_gather_dense_kernel | 0.008 | 0.0% |
| unattributed | memcpy memcpy | 0.008 | 0.0% |
| unattributed | kernel publish_ids_kernel | 0.007 | 0.0% |
| unattributed | kernel scatter_bf16x8_kernel | 0.006 | 0.0% |
| unattributed | kernel speculative_select_accepted_hidden_kernel | 0.006 | 0.0% |
| unattributed | kernel mtp_advance_round_kernel | 0.004 | 0.0% |
| unattributed | kernel rmsnorm_cta_bf16x2_kernel | 0.003 | 0.0% |
| unattributed | kernel expand_text_positions_kernel | 0.002 | 0.0% |
| unattributed | kernel mtp_prepare_next_round_kernel | 0.002 | 0.0% |
| unattributed | kernel speculative_prepare_verify_inputs_kernel | 0.002 | 0.0% |
| unattributed | kernel select_shared_indices_kernel | 0.002 | 0.0% |
| unattributed | kernel shortlist_exact_select_kernel | 0.001 | 0.0% |
| unattributed | kernel residual_add_bf16x8_kernel | 0.001 | 0.0% |
| unattributed | memset memset | 0.000 | 0.0% |

Host lookup work (may overlap GPU execution):

- ple.gather: 3.339 ms across 4 calls.
- ple.hash: 0.000 ms across 0 calls.

- Estimates are partial cold useful-work models, not distance from optimal.
- Token counts are execution envelopes; padded columns are not committed outputs.
- QSA estimates cover projections only; device-dependent scoring/attention is unmodeled.
- GDN prefill estimates omit chunked recurrence compute; PLE estimates omit convolution/state traffic.
- DRAM cache reuse, scratch traffic, nonlinear math and instruction geometry are unmodeled.
- Profiler overhead affects timings; use unprofiled paired benchmarks for speed claims.
- Unattributed GPU work is retained; no whole-model efficiency is reported.
- Serve window: GPU work launched inside 4 decode.mtp_round ranges with batch 4 (trim 0.00 of rounds at each end); work from other host threads launched during a round is included.
- An estimate exceeds measured time: inspect cache reuse, envelope work and hardware rates; percentages are not clamped.
