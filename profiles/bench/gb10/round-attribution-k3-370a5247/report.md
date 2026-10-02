# Flash-Next GPU work report

Measured wall: 9246.216 ms; GPU busy: 9165.739 ms; attributed GPU work: 92.65%.

Estimates describe partial modeled work, not an achievable optimum. Bytes/FLOPs below are per call; time and share are aggregated.

| Phase / role | Stage | T / B / context envelope | Calls | GPU work ms | Phase share | Useful bytes min–max | BF16 / NVFP4 / FP32 FLOPs | Estimate ms min–max | Estimate efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | 16 / 0 / 0 | 5088 | 5359.707 | 58.5% | 35361440–450083840 | 199311360 / 1572864000 / 0 | 731.378–9309.051 | 13.6–173.7% |
| decode / target.verify | gdn.record | 16 / 4 / 0 | 3816 | 1496.239 | 16.3% | 71543808–71543808 | 1853358080 / 0 / 75497472 | 1109.802–1109.802 | 74.2–74.2% |
| decode / target.verify | qsa.select | 16 / 4 / 131 | 1272 | 597.514 | 6.5% | 55899136–55899136 | 1646264320 / 0 / 0 | 289.039–289.039 | 48.4–48.4% |
| decode / target.verify | hyper.combine_mix | 16 / 0 / 0 | 9964 | 597.165 | 6.5% | 7476096–7476096 | 211025920 / 0 / 0 | 302.812–302.812 | 50.7–50.7% |
| decode / predictor | moe.nvfp4 | 16 / 0 / 0 | 106 | 108.497 | 1.2% | 35361440–450083840 | 199311360 / 1572864000 / 0 | 15.237–193.939 | 14.0–178.8% |
| decode / predictor | moe.nvfp4 | 4 / 0 / 0 | 212 | 100.285 | 1.1% | 35238560–118183040 | 49827840 / 393216000 / 0 | 30.368–101.849 | 30.3–101.6% |
| decode / target.verify | ple.record | 16 / 4 / 0 | 106 | 57.063 | 0.6% | 66232320–66232320 | 1048576000 / 0 / 0 | 28.539–28.539 | 50.0–50.0% |
| decode / predictor | qsa.select | 16 / 4 / 131 | 106 | 48.867 | 0.5% | 55899136–55899136 | 1646264320 / 0 / 0 | 24.087–24.087 | 49.3–49.3% |
| decode / predictor | qsa.reuse | 4 / 4 / 132 | 106 | 29.882 | 0.3% | 53154816–53154816 | 401080320 / 0 / 0 | 22.904–22.904 | 76.6–76.6% |
| decode / predictor | qsa.reuse | 4 / 4 / 133 | 106 | 29.852 | 0.3% | 53154816–53154816 | 401080320 / 0 / 0 | 22.904–22.904 | 76.7–76.7% |
| decode / predictor | hyper.combine_mix | 4 / 0 / 0 | 212 | 12.662 | 0.1% | 6861504–6861504 | 52756480 / 0 / 0 | 5.913–5.913 | 46.7–46.7% |
| decode / target.verify | hyper.mix | 16 / 0 / 0 | 212 | 11.936 | 0.1% | 7066368–7066368 | 211025920 / 0 / 0 | 6.090–6.090 | 51.0–51.0% |
| decode / predictor | hyper.mix | 4 / 0 / 0 | 212 | 9.067 | 0.1% | 6759072–6759072 | 52756480 / 0 / 0 | 5.825–5.825 | 64.2–64.2% |
| decode / predictor | hyper.mix | 4 / 0 / 0 | 212 | 8.557 | 0.1% | 6677120–6677120 | 52428800 / 0 / 0 | 5.754–5.754 | 67.2–67.2% |
| decode / predictor | hyper.combine_mix | 16 / 0 / 0 | 106 | 7.334 | 0.1% | 7476096–7476096 | 211025920 / 0 / 0 | 3.221–3.221 | 43.9–43.9% |
| decode / predictor | hyper.mix | 16 / 0 / 0 | 106 | 5.118 | 0.1% | 7066368–7066368 | 211025920 / 0 / 0 | 3.045–3.045 | 59.5–59.5% |
| decode / target.verify | hyper.mix | 16 / 0 / 0 | 106 | 4.804 | 0.1% | 6984320–6984320 | 209715200 / 0 / 0 | 3.010–3.010 | 62.7–62.7% |
| decode / predictor | hyper.mix | 16 / 0 / 0 | 106 | 4.761 | 0.1% | 6984320–6984320 | 209715200 / 0 / 0 | 3.010–3.010 | 63.2–63.2% |
| decode / target.verify | hyper.combine | 16 / 0 / 0 | 212 | 0.924 | 0.0% | 737408–737408 | 0 / 0 / 0 | 0.635–0.635 | 68.8–68.8% |
| decode / predictor | hyper.combine | 4 / 0 / 0 | 212 | 0.460 | 0.0% | 184352–184352 | 0 / 0 / 0 | 0.159–0.159 | 34.5–34.5% |
| decode / predictor | hyper.combine | 16 / 0 / 0 | 106 | 0.460 | 0.0% | 737408–737408 | 0 / 0 / 0 | 0.318–0.318 | 69.1–69.1% |
| decode / predictor | hyper.add | 16 / 0 / 0 | 106 | 0.352 | 0.0% | 737280–737280 | 0 / 0 / 0 | 0.318–0.318 | 90.1–90.1% |
| decode / target.verify | hyper.repeat | 16 / 0 / 0 | 106 | 0.346 | 0.0% | 409600–409600 | 0 / 0 / 0 | 0.176–0.176 | 51.0–51.0% |
| decode / predictor | hyper.add | 4 / 0 / 0 | 212 | 0.338 | 0.0% | 184320–184320 | 0 / 0 / 0 | 0.159–0.159 | 47.1–47.1% |

Unattributed GPU work: 673.583 ms.

Kernels per stage (ms/round, launches per round):

| Phase / role | Stage | Kernel | Launches | GPU work ms/round | Stage share |
|---|---|---|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 96.0 | 46.983 | 92.9% |
| decode / target.verify | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 96.0 | 1.289 | 2.5% |
| decode / target.verify | moe.nvfp4 | bf16_small_t_inner_kernel | 48.0 | 0.980 | 1.9% |
| decode / target.verify | moe.nvfp4 | route_kernel | 48.0 | 0.729 | 1.4% |
| decode / target.verify | moe.nvfp4 | gather_quantize_routes_kernel | 48.0 | 0.216 | 0.4% |
| decode / target.verify | moe.nvfp4 | quantize_grouped_kernel | 48.0 | 0.208 | 0.4% |
| decode / target.verify | moe.nvfp4 | reduce_grouped_kernel | 48.0 | 0.158 | 0.3% |
| decode / target.verify | gdn.record | fp8_a16_sliced_k_mma_kernel | 108.0 | 10.175 | 72.1% |
| decode / target.verify | gdn.record | recurrent_fold_record_kernel | 36.0 | 3.172 | 22.5% |
| decode / target.verify | gdn.record | project_control_gating_kernel | 36.0 | 0.385 | 2.7% |
| decode / target.verify | gdn.record | conv_replay_record_kernel | 36.0 | 0.294 | 2.1% |
| decode / target.verify | gdn.record | rmsnorm_warp_bf16x2_kernel | 36.0 | 0.089 | 0.6% |
| decode / target.verify | qsa.select | fp8_a16_sliced_k_mma_kernel | 24.0 | 2.539 | 45.0% |
| decode / target.verify | qsa.select | selected_attention_split_fp8_kernel | 12.0 | 1.907 | 33.8% |
| decode / target.verify | qsa.select | bf16_small_t_inner_kernel | 48.0 | 0.858 | 15.2% |
| decode / target.verify | qsa.select | append_cache_fp8_kernel | 12.0 | 0.065 | 1.2% |
| decode / target.verify | qsa.select | rmsnorm_warp_bf16x2_kernel | 24.0 | 0.052 | 0.9% |
| decode / target.verify | qsa.select | rope_generic_kernel | 12.0 | 0.050 | 0.9% |
| decode / target.verify | qsa.select | reduce_selected_attention_splits_kernel | 12.0 | 0.040 | 0.7% |
| decode / target.verify | qsa.select | compress_index_groups_kernel | 12.0 | 0.036 | 0.6% |
| decode / target.verify | qsa.select | split_query_gate_kernel | 12.0 | 0.030 | 0.5% |
| decode / target.verify | qsa.select | prepare_index_query_kernel | 12.0 | 0.024 | 0.4% |
| decode / target.verify | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 12.0 | 0.019 | 0.3% |
| decode / target.verify | qsa.select | hadamard_rows_kernel | 12.0 | 0.016 | 0.3% |
| decode / target.verify | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 188.0 | 4.251 | 75.5% |
| decode / target.verify | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 94.0 | 1.383 | 24.5% |
| decode / predictor | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 2.0 | 0.949 | 92.7% |
| decode / predictor | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.027 | 2.7% |
| decode / predictor | moe.nvfp4 | bf16_small_t_inner_kernel | 1.0 | 0.020 | 2.0% |
| decode / predictor | moe.nvfp4 | route_kernel | 1.0 | 0.015 | 1.5% |
| decode / predictor | moe.nvfp4 | gather_quantize_routes_kernel | 1.0 | 0.004 | 0.4% |
| decode / predictor | moe.nvfp4 | quantize_grouped_kernel | 1.0 | 0.004 | 0.4% |
| decode / predictor | moe.nvfp4 | reduce_grouped_kernel | 1.0 | 0.003 | 0.3% |
| decode / predictor | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 4.0 | 0.845 | 89.4% |
| decode / predictor | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 4.0 | 0.048 | 5.0% |
| decode / predictor | moe.nvfp4 | bf16_small_t_inner_kernel | 2.0 | 0.031 | 3.3% |
| decode / predictor | moe.nvfp4 | route_kernel | 2.0 | 0.015 | 1.6% |
| decode / predictor | moe.nvfp4 | quantize_decode_routes_kernel | 2.0 | 0.004 | 0.4% |
| decode / predictor | moe.nvfp4 | reduce_decode_routes_kernel | 2.0 | 0.003 | 0.3% |
| decode / target.verify | ple.record | bf16_gemv_columns_kernel | 2.0 | 0.513 | 95.3% |
| decode / target.verify | ple.record | gate_kernel | 1.0 | 0.009 | 1.7% |
| decode / target.verify | ple.record | dilated_conv_record_kernel | 1.0 | 0.007 | 1.3% |
| decode / target.verify | ple.record | grouped_norm_kernel | 1.0 | 0.006 | 1.1% |
| decode / target.verify | ple.record | dequantize_embedding_kernel | 1.0 | 0.004 | 0.7% |
| decode / predictor | qsa.select | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.207 | 44.8% |
| decode / predictor | qsa.select | selected_attention_split_fp8_kernel | 1.0 | 0.153 | 33.2% |
| decode / predictor | qsa.select | bf16_small_t_inner_kernel | 4.0 | 0.072 | 15.6% |
| decode / predictor | qsa.select | append_cache_fp8_kernel | 1.0 | 0.007 | 1.6% |
| decode / predictor | qsa.select | rmsnorm_warp_bf16x2_kernel | 2.0 | 0.004 | 0.8% |
| decode / predictor | qsa.select | rope_generic_kernel | 1.0 | 0.004 | 0.8% |
| decode / predictor | qsa.select | reduce_selected_attention_splits_kernel | 1.0 | 0.003 | 0.7% |
| decode / predictor | qsa.select | compress_index_groups_kernel | 1.0 | 0.003 | 0.6% |
| decode / predictor | qsa.select | split_query_gate_kernel | 1.0 | 0.002 | 0.5% |
| decode / predictor | qsa.select | prepare_index_query_kernel | 1.0 | 0.002 | 0.4% |
| decode / predictor | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 1.0 | 0.002 | 0.4% |
| decode / predictor | qsa.select | hadamard_rows_kernel | 1.0 | 0.001 | 0.3% |
| decode / predictor | qsa.select | dense_indices_batched_kernel | 1.0 | 0.001 | 0.3% |
| decode / predictor | qsa.reuse | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.189 | 67.1% |
| decode / predictor | qsa.reuse | bf16_small_t_inner_kernel | 3.0 | 0.035 | 12.4% |
| decode / predictor | qsa.reuse | selected_attention_batched_fp8_kernel | 1.0 | 0.031 | 11.1% |
| decode / predictor | qsa.reuse | reduce_selected_attention_splits_kernel | 1.0 | 0.010 | 3.7% |
| decode / predictor | qsa.reuse | rope_generic_kernel | 1.0 | 0.004 | 1.4% |
| decode / predictor | qsa.reuse | append_cache_fp8_kernel | 1.0 | 0.003 | 1.2% |
| decode / predictor | qsa.reuse | rmsnorm_warp_bf16x2_kernel | 2.0 | 0.003 | 1.0% |
| decode / predictor | qsa.reuse | compress_index_groups_kernel | 1.0 | 0.002 | 0.9% |
| decode / predictor | qsa.reuse | sigmoid_gate_mul_bf16x8_kernel | 1.0 | 0.002 | 0.5% |
| decode / predictor | qsa.reuse | split_query_gate_kernel | 1.0 | 0.001 | 0.5% |
| decode / predictor | qsa.reuse | hadamard_rows_kernel | 1.0 | 0.001 | 0.4% |
| decode / predictor | qsa.reuse | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.189 | 67.3% |
| decode / predictor | qsa.reuse | bf16_small_t_inner_kernel | 3.0 | 0.035 | 12.4% |
| decode / predictor | qsa.reuse | selected_attention_batched_fp8_kernel | 1.0 | 0.031 | 10.9% |
| decode / predictor | qsa.reuse | reduce_selected_attention_splits_kernel | 1.0 | 0.010 | 3.7% |
| decode / predictor | qsa.reuse | rope_generic_kernel | 1.0 | 0.004 | 1.4% |
| decode / predictor | qsa.reuse | append_cache_fp8_kernel | 1.0 | 0.003 | 1.2% |
| decode / predictor | qsa.reuse | rmsnorm_warp_bf16x2_kernel | 2.0 | 0.003 | 1.0% |
| decode / predictor | qsa.reuse | compress_index_groups_kernel | 1.0 | 0.002 | 0.9% |
| decode / predictor | qsa.reuse | sigmoid_gate_mul_bf16x8_kernel | 1.0 | 0.002 | 0.5% |
| decode / predictor | qsa.reuse | split_query_gate_kernel | 1.0 | 0.001 | 0.5% |
| decode / predictor | qsa.reuse | hadamard_rows_kernel | 1.0 | 0.001 | 0.4% |
| decode / predictor | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 4.0 | 0.091 | 76.4% |
| decode / predictor | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 2.0 | 0.028 | 23.6% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 4.0 | 0.093 | 82.3% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 2.0 | 0.020 | 17.7% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 4.0 | 0.069 | 80.2% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 2.0 | 0.017 | 19.8% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 4.0 | 0.068 | 84.3% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 2.0 | 0.013 | 15.7% |
| decode / predictor | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.055 | 79.4% |
| decode / predictor | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 1.0 | 0.014 | 20.6% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.040 | 82.4% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.008 | 17.6% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.039 | 86.3% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.006 | 13.7% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.039 | 86.2% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.006 | 13.8% |
| decode / target.verify | hyper.combine | combine_kernel | 2.0 | 0.009 | 100.0% |
| decode / predictor | hyper.combine | combine_kernel | 2.0 | 0.004 | 100.0% |
| decode / predictor | hyper.combine | combine_kernel | 1.0 | 0.004 | 100.0% |
| decode / predictor | hyper.add | add_repeated_kernel | 1.0 | 0.003 | 100.0% |
| decode / target.verify | hyper.repeat | repeat_kernel | 1.0 | 0.003 | 100.0% |
| decode / predictor | hyper.add | add_repeated_kernel | 2.0 | 0.003 | 100.0% |

Per round over 106 rounds: host wall 87.228 ms, GPU work 86.470 ms, GPU busy 86.469 ms.

| Phase / role | Stage | GPU work ms/round | Share |
|---|---|---:|---:|
| decode / target.verify | moe.nvfp4 | 50.563 | 58.5% |
| decode / target.verify | gdn.record | 14.115 | 16.3% |
| decode / target.verify | qsa.select | 5.637 | 6.5% |
| decode / target.verify | hyper.combine_mix | 5.634 | 6.5% |
| decode / predictor | moe.nvfp4 | 1.024 | 1.2% |
| decode / predictor | moe.nvfp4 | 0.946 | 1.1% |
| decode / target.verify | ple.record | 0.538 | 0.6% |
| decode / predictor | qsa.select | 0.461 | 0.5% |
| decode / predictor | qsa.reuse | 0.282 | 0.3% |
| decode / predictor | qsa.reuse | 0.282 | 0.3% |
| decode / predictor | hyper.combine_mix | 0.119 | 0.1% |
| decode / target.verify | hyper.mix | 0.113 | 0.1% |
| decode / predictor | hyper.mix | 0.086 | 0.1% |
| decode / predictor | hyper.mix | 0.081 | 0.1% |
| decode / predictor | hyper.combine_mix | 0.069 | 0.1% |
| decode / predictor | hyper.mix | 0.048 | 0.1% |
| decode / target.verify | hyper.mix | 0.045 | 0.1% |
| decode / predictor | hyper.mix | 0.045 | 0.1% |
| decode / target.verify | hyper.combine | 0.009 | 0.0% |
| decode / predictor | hyper.combine | 0.004 | 0.0% |
| decode / predictor | hyper.combine | 0.004 | 0.0% |
| decode / predictor | hyper.add | 0.003 | 0.0% |
| decode / target.verify | hyper.repeat | 0.003 | 0.0% |
| decode / predictor | hyper.add | 0.003 | 0.0% |
| unattributed | kernel fp8_a16_sliced_k_mma_kernel | 2.646 | 3.1% |
| unattributed | kernel q4_a16_sliced_k_mma_kernel | 2.640 | 3.1% |
| unattributed | kernel bf16_small_t_inner_kernel | 0.279 | 0.3% |
| unattributed | kernel copy_record_rows_kernel | 0.149 | 0.2% |
| unattributed | kernel shortlist_exact_scores_fp8_kernel | 0.138 | 0.2% |
| unattributed | kernel argmax_tiled_atomic_kernel | 0.107 | 0.1% |
| unattributed | kernel shortlist_tile_candidates_kernel | 0.085 | 0.1% |
| unattributed | kernel bf16_gemm_mma_kernel | 0.072 | 0.1% |
| unattributed | kernel rmsnorm_generic_kernel | 0.061 | 0.1% |
| unattributed | kernel speculative_sampling_partial_topk_kernel | 0.058 | 0.1% |
| unattributed | kernel ple_fold_kernel | 0.025 | 0.0% |
| unattributed | memcpy memcpy | 0.017 | 0.0% |
| unattributed | kernel embed_gather_dense_kernel | 0.016 | 0.0% |
| unattributed | kernel speculative_sampling_group_finalize_kernel | 0.014 | 0.0% |
| unattributed | kernel rmsnorm_cta_bf16x2_kernel | 0.006 | 0.0% |
| unattributed | kernel publish_ids_kernel | 0.006 | 0.0% |
| unattributed | kernel scatter_bf16x8_kernel | 0.006 | 0.0% |
| unattributed | kernel speculative_select_accepted_hidden_kernel | 0.006 | 0.0% |
| unattributed | kernel expand_text_positions_kernel | 0.005 | 0.0% |
| unattributed | kernel mtp_advance_round_kernel | 0.005 | 0.0% |
| unattributed | kernel shortlist_exact_select_kernel | 0.004 | 0.0% |
| unattributed | kernel mtp_prepare_next_round_kernel | 0.002 | 0.0% |
| unattributed | kernel speculative_prepare_verify_inputs_kernel | 0.002 | 0.0% |
| unattributed | kernel wait_staged_kernel | 0.002 | 0.0% |
| unattributed | kernel select_shared_indices_kernel | 0.002 | 0.0% |
| unattributed | kernel residual_add_bf16x8_kernel | 0.002 | 0.0% |
| unattributed | memset memset | 0.000 | 0.0% |

Host lookup work (may overlap GPU execution):

- ple.gather: 117.558 ms across 106 calls.
- ple.hash: 0.000 ms across 0 calls.

- Estimates are partial cold useful-work models, not distance from optimal.
- Token counts are execution envelopes; padded columns are not committed outputs.
- QSA estimates cover projections only; device-dependent scoring/attention is unmodeled.
- GDN prefill estimates omit chunked recurrence compute; PLE estimates omit convolution/state traffic.
- DRAM cache reuse, scratch traffic, nonlinear math and instruction geometry are unmodeled.
- Profiler overhead affects timings; use unprofiled paired benchmarks for speed claims.
- Unattributed GPU work is retained; no whole-model efficiency is reported.
- Serve window: GPU work launched inside 106 decode.mtp_round ranges with batch 4 (trim 0.00 of rounds at each end); work from other host threads launched during a round is included.
- An estimate exceeds measured time: inspect cache reuse, envelope work and hardware rates; percentages are not clamped.
