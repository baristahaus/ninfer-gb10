# Flash-Next GPU work report

Measured wall: 10154.630 ms; GPU busy: 10017.843 ms; attributed GPU work: 89.42%.

Estimates describe partial modeled work, not an achievable optimum. Bytes/FLOPs below are per call; time and share are aggregated.

| Phase / role | Stage | T / B / context envelope | Calls | GPU work ms | Phase share | Useful bytes min–max | BF16 / NVFP4 / FP32 FLOPs | Estimate ms min–max | Estimate efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | 8 / 0 / 0 | 6432 | 5323.791 | 53.1% | 35279520–228816640 | 99655680 / 786432000 / 0 | 922.430–5982.718 | 17.3–112.4% |
| decode / target.verify | gdn.record | 8 / 4 / 0 | 4824 | 1567.502 | 15.6% | 71163904–71163904 | 926679040 / 0 / 37748736 | 1395.507–1395.507 | 89.0–89.0% |
| decode / target.verify | qsa.select | 8 / 4 / 129 | 1608 | 853.978 | 8.5% | 55817216–55817216 | 823132160 / 0 / 0 | 364.854–364.854 | 42.7–42.7% |
| decode / target.verify | hyper.combine_mix | 8 / 0 / 0 | 12596 | 682.466 | 6.8% | 7066368–7066368 | 105512960 / 0 / 0 | 361.821–361.821 | 53.0–53.0% |
| decode / target.verify | ple.record | 8 / 4 / 0 | 134 | 307.152 | 3.1% | 65884160–65884160 | 524288000 / 0 / 0 | 35.888–35.888 | 11.7–11.7% |
| decode / predictor | moe.nvfp4 | 8 / 0 / 0 | 134 | 110.246 | 1.1% | 35279520–228816640 | 99655680 / 786432000 / 0 | 19.217–124.640 | 17.4–113.1% |
| decode / predictor | qsa.select | 8 / 4 / 129 | 134 | 70.307 | 0.7% | 55817216–55817216 | 823132160 / 0 / 0 | 30.404–30.404 | 43.2–43.2% |
| decode / target.verify | hyper.mix | 8 / 0 / 0 | 268 | 14.456 | 0.1% | 6861504–6861504 | 105512960 / 0 / 0 | 7.475–7.475 | 51.7–51.7% |
| decode / predictor | hyper.combine_mix | 8 / 0 / 0 | 134 | 8.685 | 0.1% | 7066368–7066368 | 105512960 / 0 / 0 | 3.849–3.849 | 44.3–44.3% |
| decode / predictor | hyper.mix | 8 / 0 / 0 | 134 | 6.254 | 0.1% | 6861504–6861504 | 105512960 / 0 / 0 | 3.738–3.738 | 59.8–59.8% |
| decode / target.verify | hyper.mix | 8 / 0 / 0 | 134 | 5.862 | 0.1% | 6779520–6779520 | 104857600 / 0 / 0 | 3.693–3.693 | 63.0–63.0% |
| decode / predictor | hyper.mix | 8 / 0 / 0 | 134 | 5.848 | 0.1% | 6779520–6779520 | 104857600 / 0 / 0 | 3.693–3.693 | 63.2–63.2% |
| decode / target.verify | hyper.combine | 8 / 0 / 0 | 268 | 0.775 | 0.0% | 368704–368704 | 0 / 0 / 0 | 0.402–0.402 | 51.8–51.8% |
| decode / predictor | hyper.combine | 8 / 0 / 0 | 134 | 0.389 | 0.0% | 368704–368704 | 0 / 0 / 0 | 0.201–0.201 | 51.7–51.7% |
| decode / predictor | hyper.add | 8 / 0 / 0 | 134 | 0.292 | 0.0% | 368640–368640 | 0 / 0 / 0 | 0.201–0.201 | 68.8–68.8% |
| decode / target.verify | hyper.repeat | 8 / 0 / 0 | 134 | 0.283 | 0.0% | 204800–204800 | 0 / 0 / 0 | 0.112–0.112 | 39.4–39.4% |

Unattributed GPU work: 1059.570 ms.

Kernels per stage (ms/round, launches per round):

| Phase / role | Stage | Kernel | Launches | GPU work ms/round | Stage share |
|---|---|---|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 96.0 | 36.353 | 91.5% |
| decode / target.verify | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 96.0 | 1.287 | 3.2% |
| decode / target.verify | moe.nvfp4 | bf16_small_t_inner_kernel | 48.0 | 0.808 | 2.0% |
| decode / target.verify | moe.nvfp4 | route_kernel | 48.0 | 0.477 | 1.2% |
| decode / target.verify | moe.nvfp4 | quantize_grouped_kernel | 48.0 | 0.202 | 0.5% |
| decode / target.verify | moe.nvfp4 | gather_quantize_routes_kernel | 48.0 | 0.202 | 0.5% |
| decode / target.verify | moe.nvfp4 | count_routes_kernel | 48.0 | 0.120 | 0.3% |
| decode / target.verify | moe.nvfp4 | reduce_grouped_kernel | 48.0 | 0.097 | 0.2% |
| decode / target.verify | moe.nvfp4 | make_route_jobs_kernel | 48.0 | 0.078 | 0.2% |
| decode / target.verify | moe.nvfp4 | scan_routes_kernel | 48.0 | 0.068 | 0.2% |
| decode / target.verify | moe.nvfp4 | memset | 96.0 | 0.038 | 0.1% |
| decode / target.verify | gdn.record | fp8_a16_sliced_k_mma_kernel | 108.0 | 9.255 | 79.1% |
| decode / target.verify | gdn.record | recurrent_record_kernel | 36.0 | 1.917 | 16.4% |
| decode / target.verify | gdn.record | project_control_gating_kernel | 36.0 | 0.294 | 2.5% |
| decode / target.verify | gdn.record | conv_replay_record_kernel | 36.0 | 0.171 | 1.5% |
| decode / target.verify | gdn.record | rmsnorm_warp_bf16x2_kernel | 36.0 | 0.060 | 0.5% |
| decode / target.verify | qsa.select | selected_attention_split_fp8_kernel | 12.0 | 2.871 | 45.0% |
| decode / target.verify | qsa.select | fp8_a16_sliced_k_mma_kernel | 24.0 | 2.466 | 38.7% |
| decode / target.verify | qsa.select | bf16_small_t_inner_kernel | 48.0 | 0.698 | 11.0% |
| decode / target.verify | qsa.select | append_cache_fp8_kernel | 12.0 | 0.065 | 1.0% |
| decode / target.verify | qsa.select | reduce_selected_attention_splits_kernel | 12.0 | 0.058 | 0.9% |
| decode / target.verify | qsa.select | rope_generic_kernel | 12.0 | 0.050 | 0.8% |
| decode / target.verify | qsa.select | rmsnorm_warp_bf16x2_kernel | 24.0 | 0.050 | 0.8% |
| decode / target.verify | qsa.select | compress_index_groups_kernel | 12.0 | 0.038 | 0.6% |
| decode / target.verify | qsa.select | prepare_index_query_kernel | 12.0 | 0.023 | 0.4% |
| decode / target.verify | qsa.select | split_query_gate_kernel | 12.0 | 0.020 | 0.3% |
| decode / target.verify | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 12.0 | 0.018 | 0.3% |
| decode / target.verify | qsa.select | hadamard_rows_kernel | 12.0 | 0.014 | 0.2% |
| decode / target.verify | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 188.0 | 3.780 | 74.2% |
| decode / target.verify | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 94.0 | 1.313 | 25.8% |
| decode / target.verify | ple.record | bf16_gemv_kernel | 16.0 | 2.266 | 98.9% |
| decode / target.verify | ple.record | gate_kernel | 1.0 | 0.011 | 0.5% |
| decode / target.verify | ple.record | dilated_conv_record_kernel | 1.0 | 0.006 | 0.3% |
| decode / target.verify | ple.record | grouped_norm_kernel | 1.0 | 0.006 | 0.3% |
| decode / target.verify | ple.record | dequantize_embedding_kernel | 1.0 | 0.003 | 0.1% |
| decode / predictor | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 2.0 | 0.754 | 91.6% |
| decode / predictor | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.026 | 3.2% |
| decode / predictor | moe.nvfp4 | bf16_small_t_inner_kernel | 1.0 | 0.016 | 2.0% |
| decode / predictor | moe.nvfp4 | route_kernel | 1.0 | 0.010 | 1.2% |
| decode / predictor | moe.nvfp4 | gather_quantize_routes_kernel | 1.0 | 0.004 | 0.5% |
| decode / predictor | moe.nvfp4 | quantize_grouped_kernel | 1.0 | 0.004 | 0.5% |
| decode / predictor | moe.nvfp4 | count_routes_kernel | 1.0 | 0.002 | 0.3% |
| decode / predictor | moe.nvfp4 | reduce_grouped_kernel | 1.0 | 0.002 | 0.2% |
| decode / predictor | moe.nvfp4 | make_route_jobs_kernel | 1.0 | 0.002 | 0.2% |
| decode / predictor | moe.nvfp4 | scan_routes_kernel | 1.0 | 0.001 | 0.2% |
| decode / predictor | moe.nvfp4 | memset | 2.0 | 0.001 | 0.1% |
| decode / predictor | qsa.select | selected_attention_split_fp8_kernel | 1.0 | 0.229 | 43.6% |
| decode / predictor | qsa.select | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.204 | 38.9% |
| decode / predictor | qsa.select | bf16_small_t_inner_kernel | 4.0 | 0.061 | 11.7% |
| decode / predictor | qsa.select | append_cache_fp8_kernel | 1.0 | 0.007 | 1.4% |
| decode / predictor | qsa.select | reduce_selected_attention_splits_kernel | 1.0 | 0.005 | 0.9% |
| decode / predictor | qsa.select | rope_generic_kernel | 1.0 | 0.004 | 0.7% |
| decode / predictor | qsa.select | rmsnorm_warp_bf16x2_kernel | 2.0 | 0.004 | 0.7% |
| decode / predictor | qsa.select | compress_index_groups_kernel | 1.0 | 0.003 | 0.6% |
| decode / predictor | qsa.select | prepare_index_query_kernel | 1.0 | 0.002 | 0.4% |
| decode / predictor | qsa.select | split_query_gate_kernel | 1.0 | 0.002 | 0.3% |
| decode / predictor | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 1.0 | 0.002 | 0.3% |
| decode / predictor | qsa.select | dense_indices_batched_kernel | 1.0 | 0.001 | 0.2% |
| decode / predictor | qsa.select | hadamard_rows_kernel | 1.0 | 0.001 | 0.2% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 4.0 | 0.089 | 82.4% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 2.0 | 0.019 | 17.6% |
| decode / predictor | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.051 | 78.5% |
| decode / predictor | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 1.0 | 0.014 | 21.5% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.038 | 82.2% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.008 | 17.8% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.037 | 85.6% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.006 | 14.4% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.037 | 85.7% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.006 | 14.3% |
| decode / target.verify | hyper.combine | combine_kernel | 2.0 | 0.006 | 100.0% |
| decode / predictor | hyper.combine | combine_kernel | 1.0 | 0.003 | 100.0% |
| decode / predictor | hyper.add | add_repeated_kernel | 1.0 | 0.002 | 100.0% |
| decode / target.verify | hyper.repeat | repeat_kernel | 1.0 | 0.002 | 100.0% |

Per round over 134 rounds: host wall 75.781 ms, GPU work 74.760 ms, GPU busy 74.760 ms.

| Phase / role | Stage | GPU work ms/round | Share |
|---|---|---:|---:|
| decode / target.verify | moe.nvfp4 | 39.730 | 53.1% |
| decode / target.verify | gdn.record | 11.698 | 15.6% |
| decode / target.verify | qsa.select | 6.373 | 8.5% |
| decode / target.verify | hyper.combine_mix | 5.093 | 6.8% |
| decode / target.verify | ple.record | 2.292 | 3.1% |
| decode / predictor | moe.nvfp4 | 0.823 | 1.1% |
| decode / predictor | qsa.select | 0.525 | 0.7% |
| decode / target.verify | hyper.mix | 0.108 | 0.1% |
| decode / predictor | hyper.combine_mix | 0.065 | 0.1% |
| decode / predictor | hyper.mix | 0.047 | 0.1% |
| decode / target.verify | hyper.mix | 0.044 | 0.1% |
| decode / predictor | hyper.mix | 0.044 | 0.1% |
| decode / target.verify | hyper.combine | 0.006 | 0.0% |
| decode / predictor | hyper.combine | 0.003 | 0.0% |
| decode / predictor | hyper.add | 0.002 | 0.0% |
| decode / target.verify | hyper.repeat | 0.002 | 0.0% |
| unattributed | kernel recurrent_fold_kernel | 4.002 | 5.4% |
| unattributed | kernel fp8_a16_sliced_k_mma_kernel | 2.600 | 3.5% |
| unattributed | kernel q4_a16_sliced_k_mma_kernel | 0.898 | 1.2% |
| unattributed | kernel bf16_gemm_mma_kernel | 0.070 | 0.1% |
| unattributed | kernel bf16_small_t_inner_kernel | 0.058 | 0.1% |
| unattributed | kernel argmax_tiled_atomic_kernel | 0.054 | 0.1% |
| unattributed | kernel shortlist_exact_scores_fp8_kernel | 0.046 | 0.1% |
| unattributed | kernel speculative_sampling_partial_topk_kernel | 0.033 | 0.0% |
| unattributed | kernel ple_fold_kernel | 0.029 | 0.0% |
| unattributed | kernel shortlist_tile_candidates_kernel | 0.028 | 0.0% |
| unattributed | kernel rmsnorm_generic_kernel | 0.019 | 0.0% |
| unattributed | memcpy memcpy | 0.011 | 0.0% |
| unattributed | kernel speculative_sampling_group_finalize_kernel | 0.009 | 0.0% |
| unattributed | kernel embed_gather_dense_kernel | 0.009 | 0.0% |
| unattributed | kernel publish_ids_kernel | 0.007 | 0.0% |
| unattributed | kernel scatter_bf16x8_kernel | 0.006 | 0.0% |
| unattributed | kernel speculative_select_accepted_hidden_kernel | 0.006 | 0.0% |
| unattributed | kernel mtp_advance_round_kernel | 0.004 | 0.0% |
| unattributed | kernel rmsnorm_cta_bf16x2_kernel | 0.004 | 0.0% |
| unattributed | kernel expand_text_positions_kernel | 0.003 | 0.0% |
| unattributed | kernel mtp_prepare_next_round_kernel | 0.002 | 0.0% |
| unattributed | kernel speculative_prepare_verify_inputs_kernel | 0.002 | 0.0% |
| unattributed | kernel wait_staged_kernel | 0.002 | 0.0% |
| unattributed | kernel select_shared_indices_kernel | 0.002 | 0.0% |
| unattributed | kernel shortlist_exact_select_kernel | 0.001 | 0.0% |
| unattributed | kernel residual_add_bf16x8_kernel | 0.001 | 0.0% |
| unattributed | memset memset | 0.000 | 0.0% |

Host lookup work (may overlap GPU execution):

- ple.gather: 47.899 ms across 134 calls.
- ple.hash: 0.000 ms across 0 calls.

- Estimates are partial cold useful-work models, not distance from optimal.
- Token counts are execution envelopes; padded columns are not committed outputs.
- QSA estimates cover projections only; device-dependent scoring/attention is unmodeled.
- GDN prefill estimates omit chunked recurrence compute; PLE estimates omit convolution/state traffic.
- DRAM cache reuse, scratch traffic, nonlinear math and instruction geometry are unmodeled.
- Profiler overhead affects timings; use unprofiled paired benchmarks for speed claims.
- Unattributed GPU work is retained; no whole-model efficiency is reported.
- Serve window: GPU work launched inside 134 decode.mtp_round ranges with batch 4 (trim 0.00 of rounds at each end); work from other host threads launched during a round is included.
- An estimate exceeds measured time: inspect cache reuse, envelope work and hardware rates; percentages are not clamped.
