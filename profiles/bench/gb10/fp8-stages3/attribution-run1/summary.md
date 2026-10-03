## Round attribution: C4, MTP draft 1

- Host: Linux 6.17.0-1029-nvidia aarch64; Ubuntu 24.04.4 LTS
- Commit: 5912b05b (with local changes)
- nvcc: release 13.0, V13.0.88
- GPU/driver: NVIDIA GB10, 580.173.02
- Artifact: qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer, 119G (multi-volume total)
- nsys: NVIDIA Nsight Systems version 2025.3.2.474-253236389321v0
- Load: two batches of 4 simultaneous greedy requests, 256 output tokens each (ignore_eos)
- Request log totals: Totals: 2048 tokens, decode 69.4 s (29.5 tok/s), prefill 2.5 s, rounds 1168, 1.75 tok/round, device wait 58.0 ms/round, queue wait mean 1592 ms, host exposed 1.43 ms/round


Measured wall: 8743.402 ms; GPU busy: 8652.643 ms; attributed GPU work: 93.92%.

Estimates describe partial modeled work, not an achievable optimum. Bytes/FLOPs below are per call; time and share are aggregated.

| Phase / role | Stage | T / B / context envelope | Calls | GPU work ms | Phase share | Useful bytes min–max | BF16 / NVFP4 / FP32 FLOPs | Estimate ms min–max | Estimate efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | 8 / 0 / 0 | 6336 | 4807.694 | 55.6% | 35279520–228816640 | 99655680 / 786432000 / 0 | 908.663–5893.424 | 18.9–122.6% |
| decode / target.verify | gdn.record | 8 / 4 / 0 | 4752 | 1776.033 | 20.5% | 71163904–71163904 | 926679040 / 0 / 37748736 | 1374.678–1374.678 | 77.4–77.4% |
| decode / target.verify | hyper.combine_mix | 8 / 0 / 0 | 12408 | 715.676 | 8.3% | 7066368–7066368 | 105512960 / 0 / 0 | 356.421–356.421 | 49.8–49.8% |
| decode / target.verify | qsa.select | 8 / 4 / 129 | 1584 | 591.973 | 6.8% | 55817216–55817216 | 823132160 / 0 / 0 | 359.408–359.408 | 60.7–60.7% |
| decode / predictor | moe.nvfp4 | 8 / 0 / 0 | 132 | 99.720 | 1.2% | 35279520–228816640 | 99655680 / 786432000 / 0 | 18.930–122.780 | 19.0–123.1% |
| decode / predictor | qsa.select | 8 / 4 / 129 | 132 | 50.118 | 0.6% | 55817216–55817216 | 823132160 / 0 / 0 | 29.951–29.951 | 59.8–59.8% |
| decode / target.verify | ple.record | 8 / 4 / 0 | 132 | 42.925 | 0.5% | 65884160–65884160 | 524288000 / 0 / 0 | 35.352–35.352 | 82.4–82.4% |
| decode / target.verify | hyper.mix | 8 / 0 / 0 | 264 | 14.202 | 0.2% | 6861504–6861504 | 105512960 / 0 / 0 | 7.364–7.364 | 51.8–51.8% |
| decode / predictor | hyper.combine_mix | 8 / 0 / 0 | 132 | 9.342 | 0.1% | 7066368–7066368 | 105512960 / 0 / 0 | 3.792–3.792 | 40.6–40.6% |
| decode / predictor | hyper.mix | 8 / 0 / 0 | 132 | 6.028 | 0.1% | 6861504–6861504 | 105512960 / 0 / 0 | 3.682–3.682 | 61.1–61.1% |
| decode / target.verify | hyper.mix | 8 / 0 / 0 | 132 | 5.605 | 0.1% | 6779520–6779520 | 104857600 / 0 / 0 | 3.638–3.638 | 64.9–64.9% |
| decode / predictor | hyper.mix | 8 / 0 / 0 | 132 | 5.391 | 0.1% | 6779520–6779520 | 104857600 / 0 / 0 | 3.638–3.638 | 67.5–67.5% |
| decode / target.verify | hyper.combine | 8 / 0 / 0 | 264 | 0.754 | 0.0% | 368704–368704 | 0 / 0 / 0 | 0.396–0.396 | 52.5–52.5% |
| decode / predictor | hyper.combine | 8 / 0 / 0 | 132 | 0.377 | 0.0% | 368704–368704 | 0 / 0 / 0 | 0.198–0.198 | 52.4–52.4% |
| decode / predictor | hyper.add | 8 / 0 / 0 | 132 | 0.282 | 0.0% | 368640–368640 | 0 / 0 / 0 | 0.198–0.198 | 70.2–70.2% |
| decode / target.verify | hyper.repeat | 8 / 0 / 0 | 132 | 0.277 | 0.0% | 204800–204800 | 0 / 0 / 0 | 0.110–0.110 | 39.7–39.7% |

Unattributed GPU work: 526.257 ms.

Kernels per stage (ms/round, launches per round):

| Phase / role | Stage | Kernel | Launches | GPU work ms/round | Stage share |
|---|---|---|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 96.0 | 33.279 | 91.4% |
| decode / target.verify | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 96.0 | 1.255 | 3.4% |
| decode / target.verify | moe.nvfp4 | bf16_small_t_inner_kernel | 48.0 | 0.795 | 2.2% |
| decode / target.verify | moe.nvfp4 | route_kernel | 48.0 | 0.594 | 1.6% |
| decode / target.verify | moe.nvfp4 | quantize_grouped_kernel | 48.0 | 0.201 | 0.6% |
| decode / target.verify | moe.nvfp4 | gather_quantize_routes_kernel | 48.0 | 0.200 | 0.5% |
| decode / target.verify | moe.nvfp4 | reduce_grouped_kernel | 48.0 | 0.098 | 0.3% |
| decode / target.verify | gdn.record | fp8_a16_sliced_k_mma_kernel | 108.0 | 10.040 | 74.6% |
| decode / target.verify | gdn.record | recurrent_fold_record_kernel | 36.0 | 2.806 | 20.9% |
| decode / target.verify | gdn.record | project_control_gating_kernel | 36.0 | 0.294 | 2.2% |
| decode / target.verify | gdn.record | conv_replay_record_kernel | 36.0 | 0.246 | 1.8% |
| decode / target.verify | gdn.record | rmsnorm_warp_bf16x2_kernel | 36.0 | 0.068 | 0.5% |
| decode / target.verify | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 188.0 | 4.099 | 75.6% |
| decode / target.verify | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 94.0 | 1.323 | 24.4% |
| decode / target.verify | qsa.select | fp8_a16_sliced_k_mma_kernel | 24.0 | 2.439 | 54.4% |
| decode / target.verify | qsa.select | selected_attention_split_fp8_kernel | 12.0 | 1.016 | 22.6% |
| decode / target.verify | qsa.select | bf16_small_t_inner_kernel | 48.0 | 0.695 | 15.5% |
| decode / target.verify | qsa.select | append_cache_fp8_kernel | 12.0 | 0.065 | 1.5% |
| decode / target.verify | qsa.select | reduce_selected_attention_splits_kernel | 12.0 | 0.058 | 1.3% |
| decode / target.verify | qsa.select | rmsnorm_warp_bf16x2_kernel | 24.0 | 0.050 | 1.1% |
| decode / target.verify | qsa.select | rope_generic_kernel | 12.0 | 0.050 | 1.1% |
| decode / target.verify | qsa.select | compress_index_groups_kernel | 12.0 | 0.036 | 0.8% |
| decode / target.verify | qsa.select | prepare_index_query_kernel | 12.0 | 0.023 | 0.5% |
| decode / target.verify | qsa.select | split_query_gate_kernel | 12.0 | 0.020 | 0.4% |
| decode / target.verify | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 12.0 | 0.018 | 0.4% |
| decode / target.verify | qsa.select | hadamard_rows_kernel | 12.0 | 0.014 | 0.3% |
| decode / predictor | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 2.0 | 0.688 | 91.1% |
| decode / predictor | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.026 | 3.4% |
| decode / predictor | moe.nvfp4 | bf16_small_t_inner_kernel | 1.0 | 0.019 | 2.5% |
| decode / predictor | moe.nvfp4 | route_kernel | 1.0 | 0.012 | 1.6% |
| decode / predictor | moe.nvfp4 | quantize_grouped_kernel | 1.0 | 0.004 | 0.6% |
| decode / predictor | moe.nvfp4 | gather_quantize_routes_kernel | 1.0 | 0.004 | 0.6% |
| decode / predictor | moe.nvfp4 | reduce_grouped_kernel | 1.0 | 0.002 | 0.3% |
| decode / predictor | qsa.select | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.204 | 53.8% |
| decode / predictor | qsa.select | selected_attention_split_fp8_kernel | 1.0 | 0.083 | 21.9% |
| decode / predictor | qsa.select | bf16_small_t_inner_kernel | 4.0 | 0.062 | 16.4% |
| decode / predictor | qsa.select | append_cache_fp8_kernel | 1.0 | 0.007 | 1.9% |
| decode / predictor | qsa.select | reduce_selected_attention_splits_kernel | 1.0 | 0.005 | 1.3% |
| decode / predictor | qsa.select | rope_generic_kernel | 1.0 | 0.004 | 1.0% |
| decode / predictor | qsa.select | rmsnorm_warp_bf16x2_kernel | 2.0 | 0.004 | 1.0% |
| decode / predictor | qsa.select | compress_index_groups_kernel | 1.0 | 0.003 | 0.8% |
| decode / predictor | qsa.select | prepare_index_query_kernel | 1.0 | 0.002 | 0.5% |
| decode / predictor | qsa.select | split_query_gate_kernel | 1.0 | 0.002 | 0.4% |
| decode / predictor | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 1.0 | 0.002 | 0.4% |
| decode / predictor | qsa.select | dense_indices_batched_kernel | 1.0 | 0.001 | 0.3% |
| decode / predictor | qsa.select | hadamard_rows_kernel | 1.0 | 0.001 | 0.3% |
| decode / target.verify | ple.record | bf16_gemv_columns_kernel | 2.0 | 0.302 | 92.8% |
| decode / target.verify | ple.record | gate_kernel | 1.0 | 0.008 | 2.5% |
| decode / target.verify | ple.record | dilated_conv_record_kernel | 1.0 | 0.006 | 1.9% |
| decode / target.verify | ple.record | grouped_norm_kernel | 1.0 | 0.006 | 1.8% |
| decode / target.verify | ple.record | dequantize_embedding_kernel | 1.0 | 0.003 | 1.1% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 4.0 | 0.089 | 82.9% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 2.0 | 0.018 | 17.1% |
| decode / predictor | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.057 | 80.4% |
| decode / predictor | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 1.0 | 0.014 | 19.6% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.037 | 81.9% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.008 | 18.1% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.036 | 85.3% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.006 | 14.7% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.035 | 84.6% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.006 | 15.4% |
| decode / target.verify | hyper.combine | combine_kernel | 2.0 | 0.006 | 100.0% |
| decode / predictor | hyper.combine | combine_kernel | 1.0 | 0.003 | 100.0% |
| decode / predictor | hyper.add | add_repeated_kernel | 1.0 | 0.002 | 100.0% |
| decode / target.verify | hyper.repeat | repeat_kernel | 1.0 | 0.002 | 100.0% |

Per round over 132 rounds: host wall 66.238 ms, GPU work 65.550 ms, GPU busy 65.550 ms.

| Phase / role | Stage | GPU work ms/round | Share |
|---|---|---:|---:|
| decode / target.verify | moe.nvfp4 | 36.422 | 55.6% |
| decode / target.verify | gdn.record | 13.455 | 20.5% |
| decode / target.verify | hyper.combine_mix | 5.422 | 8.3% |
| decode / target.verify | qsa.select | 4.485 | 6.8% |
| decode / predictor | moe.nvfp4 | 0.755 | 1.2% |
| decode / predictor | qsa.select | 0.380 | 0.6% |
| decode / target.verify | ple.record | 0.325 | 0.5% |
| decode / target.verify | hyper.mix | 0.108 | 0.2% |
| decode / predictor | hyper.combine_mix | 0.071 | 0.1% |
| decode / predictor | hyper.mix | 0.046 | 0.1% |
| decode / target.verify | hyper.mix | 0.042 | 0.1% |
| decode / predictor | hyper.mix | 0.041 | 0.1% |
| decode / target.verify | hyper.combine | 0.006 | 0.0% |
| decode / predictor | hyper.combine | 0.003 | 0.0% |
| decode / predictor | hyper.add | 0.002 | 0.0% |
| decode / target.verify | hyper.repeat | 0.002 | 0.0% |
| unattributed | kernel fp8_a16_sliced_k_mma_kernel | 2.575 | 3.9% |
| unattributed | kernel q4_a16_sliced_k_mma_kernel | 0.882 | 1.3% |
| unattributed | kernel bf16_gemm_mma_kernel | 0.071 | 0.1% |
| unattributed | kernel wait_staged_kernel | 0.069 | 0.1% |
| unattributed | kernel bf16_small_t_inner_kernel | 0.059 | 0.1% |
| unattributed | kernel copy_record_rows_kernel | 0.058 | 0.1% |
| unattributed | kernel argmax_tiled_atomic_kernel | 0.054 | 0.1% |
| unattributed | kernel shortlist_exact_scores_fp8_kernel | 0.048 | 0.1% |
| unattributed | kernel speculative_sampling_partial_topk_kernel | 0.034 | 0.1% |
| unattributed | kernel shortlist_tile_candidates_kernel | 0.028 | 0.0% |
| unattributed | kernel ple_fold_kernel | 0.026 | 0.0% |
| unattributed | kernel rmsnorm_generic_kernel | 0.019 | 0.0% |
| unattributed | memcpy memcpy | 0.010 | 0.0% |
| unattributed | kernel speculative_sampling_group_finalize_kernel | 0.009 | 0.0% |
| unattributed | kernel embed_gather_dense_kernel | 0.008 | 0.0% |
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

- ple.gather: 117.242 ms across 132 calls.
- ple.hash: 0.000 ms across 0 calls.

- Estimates are partial cold useful-work models, not distance from optimal.
- Token counts are execution envelopes; padded columns are not committed outputs.
- QSA estimates cover projections only; device-dependent scoring/attention is unmodeled.
- GDN prefill estimates omit chunked recurrence compute; PLE estimates omit convolution/state traffic.
- DRAM cache reuse, scratch traffic, nonlinear math and instruction geometry are unmodeled.
- Profiler overhead affects timings; use unprofiled paired benchmarks for speed claims.
- Unattributed GPU work is retained; no whole-model efficiency is reported.
- Serve window: GPU work launched inside 132 decode.mtp_round ranges with batch 4 (trim 0.00 of rounds at each end); work from other host threads launched during a round is included.
- An estimate exceeds measured time: inspect cache reuse, envelope work and hardware rates; percentages are not clamped.
