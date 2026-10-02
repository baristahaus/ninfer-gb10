# Flash-Next GPU work report

Measured wall: 9657.884 ms; GPU busy: 9527.333 ms; attributed GPU work: 94.62%.

Estimates describe partial modeled work, not an achievable optimum. Bytes/FLOPs below are per call; time and share are aggregated.

| Phase / role | Stage | T / B / context envelope | Calls | GPU work ms | Phase share | Useful bytes min–max | BF16 / NVFP4 / FP32 FLOPs | Estimate ms min–max | Estimate efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | 8 / 0 / 0 | 6192 | 5230.801 | 54.9% | 35279520–228816640 | 99655680 / 786432000 / 0 | 888.011–5759.482 | 17.0–110.1% |
| decode / target.verify | gdn.record | 8 / 4 / 0 | 4644 | 1745.511 | 18.3% | 71163904–71163904 | 926679040 / 0 / 37748736 | 1343.436–1343.436 | 77.0–77.0% |
| decode / target.verify | qsa.select | 8 / 4 / 129 | 1548 | 824.175 | 8.7% | 55817216–55817216 | 823132160 / 0 / 0 | 351.240–351.240 | 42.6–42.6% |
| decode / target.verify | hyper.combine_mix | 8 / 0 / 0 | 12126 | 703.340 | 7.4% | 7066368–7066368 | 105512960 / 0 / 0 | 348.320–348.320 | 49.5–49.5% |
| decode / target.verify | ple.record | 8 / 4 / 0 | 129 | 296.677 | 3.1% | 65884160–65884160 | 524288000 / 0 / 0 | 34.549–34.549 | 11.6–11.6% |
| decode / predictor | moe.nvfp4 | 8 / 0 / 0 | 129 | 105.212 | 1.1% | 35279520–228816640 | 99655680 / 786432000 / 0 | 18.500–119.989 | 17.6–114.0% |
| decode / predictor | qsa.select | 8 / 4 / 129 | 129 | 67.840 | 0.7% | 55817216–55817216 | 823132160 / 0 / 0 | 29.270–29.270 | 43.1–43.1% |
| decode / target.verify | hyper.mix | 8 / 0 / 0 | 258 | 13.873 | 0.1% | 6861504–6861504 | 105512960 / 0 / 0 | 7.196–7.196 | 51.9–51.9% |
| decode / predictor | hyper.combine_mix | 8 / 0 / 0 | 129 | 8.652 | 0.1% | 7066368–7066368 | 105512960 / 0 / 0 | 3.706–3.706 | 42.8–42.8% |
| decode / predictor | hyper.mix | 8 / 0 / 0 | 129 | 5.895 | 0.1% | 6861504–6861504 | 105512960 / 0 / 0 | 3.598–3.598 | 61.0–61.0% |
| decode / predictor | hyper.mix | 8 / 0 / 0 | 129 | 5.837 | 0.1% | 6779520–6779520 | 104857600 / 0 / 0 | 3.555–3.555 | 60.9–60.9% |
| decode / target.verify | hyper.mix | 8 / 0 / 0 | 129 | 5.691 | 0.1% | 6779520–6779520 | 104857600 / 0 / 0 | 3.555–3.555 | 62.5–62.5% |
| decode / target.verify | hyper.combine | 8 / 0 / 0 | 258 | 0.742 | 0.0% | 368704–368704 | 0 / 0 / 0 | 0.387–0.387 | 52.1–52.1% |
| decode / predictor | hyper.combine | 8 / 0 / 0 | 129 | 0.372 | 0.0% | 368704–368704 | 0 / 0 / 0 | 0.193–0.193 | 52.0–52.0% |
| decode / target.verify | hyper.repeat | 8 / 0 / 0 | 129 | 0.277 | 0.0% | 204800–204800 | 0 / 0 / 0 | 0.107–0.107 | 38.8–38.8% |
| decode / predictor | hyper.add | 8 / 0 / 0 | 129 | 0.275 | 0.0% | 368640–368640 | 0 / 0 / 0 | 0.193–0.193 | 70.3–70.3% |

Unattributed GPU work: 512.184 ms.

Kernels per stage (ms/round, launches per round):

| Phase / role | Stage | Kernel | Launches | GPU work ms/round | Stage share |
|---|---|---|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 96.0 | 37.173 | 91.7% |
| decode / target.verify | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 96.0 | 1.286 | 3.2% |
| decode / target.verify | moe.nvfp4 | bf16_small_t_inner_kernel | 48.0 | 0.803 | 2.0% |
| decode / target.verify | moe.nvfp4 | route_kernel | 48.0 | 0.480 | 1.2% |
| decode / target.verify | moe.nvfp4 | quantize_grouped_kernel | 48.0 | 0.202 | 0.5% |
| decode / target.verify | moe.nvfp4 | gather_quantize_routes_kernel | 48.0 | 0.202 | 0.5% |
| decode / target.verify | moe.nvfp4 | count_routes_kernel | 48.0 | 0.121 | 0.3% |
| decode / target.verify | moe.nvfp4 | reduce_grouped_kernel | 48.0 | 0.098 | 0.2% |
| decode / target.verify | moe.nvfp4 | make_route_jobs_kernel | 48.0 | 0.077 | 0.2% |
| decode / target.verify | moe.nvfp4 | scan_routes_kernel | 48.0 | 0.069 | 0.2% |
| decode / target.verify | moe.nvfp4 | memset | 96.0 | 0.038 | 0.1% |
| decode / target.verify | gdn.record | fp8_a16_sliced_k_mma_kernel | 108.0 | 10.089 | 74.6% |
| decode / target.verify | gdn.record | recurrent_fold_record_kernel | 36.0 | 2.831 | 20.9% |
| decode / target.verify | gdn.record | project_control_gating_kernel | 36.0 | 0.293 | 2.2% |
| decode / target.verify | gdn.record | conv_replay_record_kernel | 36.0 | 0.248 | 1.8% |
| decode / target.verify | gdn.record | rmsnorm_warp_bf16x2_kernel | 36.0 | 0.070 | 0.5% |
| decode / target.verify | qsa.select | selected_attention_split_fp8_kernel | 12.0 | 2.866 | 44.9% |
| decode / target.verify | qsa.select | fp8_a16_sliced_k_mma_kernel | 24.0 | 2.481 | 38.8% |
| decode / target.verify | qsa.select | bf16_small_t_inner_kernel | 48.0 | 0.706 | 11.1% |
| decode / target.verify | qsa.select | append_cache_fp8_kernel | 12.0 | 0.064 | 1.0% |
| decode / target.verify | qsa.select | reduce_selected_attention_splits_kernel | 12.0 | 0.059 | 0.9% |
| decode / target.verify | qsa.select | rmsnorm_warp_bf16x2_kernel | 24.0 | 0.051 | 0.8% |
| decode / target.verify | qsa.select | rope_generic_kernel | 12.0 | 0.050 | 0.8% |
| decode / target.verify | qsa.select | compress_index_groups_kernel | 12.0 | 0.037 | 0.6% |
| decode / target.verify | qsa.select | prepare_index_query_kernel | 12.0 | 0.023 | 0.4% |
| decode / target.verify | qsa.select | split_query_gate_kernel | 12.0 | 0.020 | 0.3% |
| decode / target.verify | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 12.0 | 0.018 | 0.3% |
| decode / target.verify | qsa.select | hadamard_rows_kernel | 12.0 | 0.014 | 0.2% |
| decode / target.verify | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 188.0 | 4.122 | 75.6% |
| decode / target.verify | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 94.0 | 1.330 | 24.4% |
| decode / target.verify | ple.record | bf16_gemv_kernel | 16.0 | 2.273 | 98.8% |
| decode / target.verify | ple.record | gate_kernel | 1.0 | 0.011 | 0.5% |
| decode / target.verify | ple.record | dilated_conv_record_kernel | 1.0 | 0.006 | 0.3% |
| decode / target.verify | ple.record | grouped_norm_kernel | 1.0 | 0.006 | 0.3% |
| decode / target.verify | ple.record | dequantize_embedding_kernel | 1.0 | 0.004 | 0.2% |
| decode / predictor | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 2.0 | 0.742 | 91.0% |
| decode / predictor | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.029 | 3.5% |
| decode / predictor | moe.nvfp4 | bf16_small_t_inner_kernel | 1.0 | 0.018 | 2.2% |
| decode / predictor | moe.nvfp4 | route_kernel | 1.0 | 0.010 | 1.2% |
| decode / predictor | moe.nvfp4 | quantize_grouped_kernel | 1.0 | 0.004 | 0.5% |
| decode / predictor | moe.nvfp4 | gather_quantize_routes_kernel | 1.0 | 0.004 | 0.5% |
| decode / predictor | moe.nvfp4 | count_routes_kernel | 1.0 | 0.002 | 0.3% |
| decode / predictor | moe.nvfp4 | reduce_grouped_kernel | 1.0 | 0.002 | 0.2% |
| decode / predictor | moe.nvfp4 | make_route_jobs_kernel | 1.0 | 0.002 | 0.2% |
| decode / predictor | moe.nvfp4 | scan_routes_kernel | 1.0 | 0.001 | 0.2% |
| decode / predictor | moe.nvfp4 | memset | 2.0 | 0.001 | 0.1% |
| decode / predictor | qsa.select | selected_attention_split_fp8_kernel | 1.0 | 0.233 | 44.4% |
| decode / predictor | qsa.select | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.201 | 38.2% |
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
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 4.0 | 0.089 | 82.6% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 2.0 | 0.019 | 17.4% |
| decode / predictor | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.053 | 79.1% |
| decode / predictor | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 1.0 | 0.014 | 20.9% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.037 | 81.7% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.008 | 18.3% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.039 | 86.0% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.006 | 14.0% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.038 | 85.9% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.006 | 14.1% |
| decode / target.verify | hyper.combine | combine_kernel | 2.0 | 0.006 | 100.0% |
| decode / predictor | hyper.combine | combine_kernel | 1.0 | 0.003 | 100.0% |
| decode / target.verify | hyper.repeat | repeat_kernel | 1.0 | 0.002 | 100.0% |
| decode / predictor | hyper.add | add_repeated_kernel | 1.0 | 0.002 | 100.0% |

Per round over 129 rounds: host wall 74.867 ms, GPU work 73.855 ms, GPU busy 73.855 ms.

| Phase / role | Stage | GPU work ms/round | Share |
|---|---|---:|---:|
| decode / target.verify | moe.nvfp4 | 40.549 | 54.9% |
| decode / target.verify | gdn.record | 13.531 | 18.3% |
| decode / target.verify | qsa.select | 6.389 | 8.7% |
| decode / target.verify | hyper.combine_mix | 5.452 | 7.4% |
| decode / target.verify | ple.record | 2.300 | 3.1% |
| decode / predictor | moe.nvfp4 | 0.816 | 1.1% |
| decode / predictor | qsa.select | 0.526 | 0.7% |
| decode / target.verify | hyper.mix | 0.108 | 0.1% |
| decode / predictor | hyper.combine_mix | 0.067 | 0.1% |
| decode / predictor | hyper.mix | 0.046 | 0.1% |
| decode / predictor | hyper.mix | 0.045 | 0.1% |
| decode / target.verify | hyper.mix | 0.044 | 0.1% |
| decode / target.verify | hyper.combine | 0.006 | 0.0% |
| decode / predictor | hyper.combine | 0.003 | 0.0% |
| decode / target.verify | hyper.repeat | 0.002 | 0.0% |
| decode / predictor | hyper.add | 0.002 | 0.0% |
| unattributed | kernel fp8_a16_sliced_k_mma_kernel | 2.608 | 3.5% |
| unattributed | kernel q4_a16_sliced_k_mma_kernel | 0.895 | 1.2% |
| unattributed | kernel bf16_gemm_mma_kernel | 0.072 | 0.1% |
| unattributed | kernel bf16_small_t_inner_kernel | 0.059 | 0.1% |
| unattributed | kernel copy_record_rows_kernel | 0.059 | 0.1% |
| unattributed | kernel argmax_tiled_atomic_kernel | 0.054 | 0.1% |
| unattributed | kernel shortlist_exact_scores_fp8_kernel | 0.047 | 0.1% |
| unattributed | kernel speculative_sampling_partial_topk_kernel | 0.034 | 0.0% |
| unattributed | kernel shortlist_tile_candidates_kernel | 0.028 | 0.0% |
| unattributed | kernel ple_fold_kernel | 0.026 | 0.0% |
| unattributed | kernel rmsnorm_generic_kernel | 0.019 | 0.0% |
| unattributed | memcpy memcpy | 0.011 | 0.0% |
| unattributed | kernel speculative_sampling_group_finalize_kernel | 0.009 | 0.0% |
| unattributed | kernel embed_gather_dense_kernel | 0.008 | 0.0% |
| unattributed | kernel publish_ids_kernel | 0.006 | 0.0% |
| unattributed | kernel scatter_bf16x8_kernel | 0.006 | 0.0% |
| unattributed | kernel speculative_select_accepted_hidden_kernel | 0.006 | 0.0% |
| unattributed | kernel mtp_advance_round_kernel | 0.004 | 0.0% |
| unattributed | kernel rmsnorm_cta_bf16x2_kernel | 0.003 | 0.0% |
| unattributed | kernel expand_text_positions_kernel | 0.003 | 0.0% |
| unattributed | kernel wait_staged_kernel | 0.003 | 0.0% |
| unattributed | kernel mtp_prepare_next_round_kernel | 0.002 | 0.0% |
| unattributed | kernel speculative_prepare_verify_inputs_kernel | 0.002 | 0.0% |
| unattributed | kernel select_shared_indices_kernel | 0.002 | 0.0% |
| unattributed | kernel shortlist_exact_select_kernel | 0.001 | 0.0% |
| unattributed | kernel residual_add_bf16x8_kernel | 0.001 | 0.0% |
| unattributed | memset memset | 0.000 | 0.0% |

Host lookup work (may overlap GPU execution):

- ple.gather: 81.624 ms across 129 calls.
- ple.hash: 0.000 ms across 0 calls.

- Estimates are partial cold useful-work models, not distance from optimal.
- Token counts are execution envelopes; padded columns are not committed outputs.
- QSA estimates cover projections only; device-dependent scoring/attention is unmodeled.
- GDN prefill estimates omit chunked recurrence compute; PLE estimates omit convolution/state traffic.
- DRAM cache reuse, scratch traffic, nonlinear math and instruction geometry are unmodeled.
- Profiler overhead affects timings; use unprofiled paired benchmarks for speed claims.
- Unattributed GPU work is retained; no whole-model efficiency is reported.
- Serve window: GPU work launched inside 129 decode.mtp_round ranges with batch 4 (trim 0.00 of rounds at each end); work from other host threads launched during a round is included.
- An estimate exceeds measured time: inspect cache reuse, envelope work and hardware rates; percentages are not clamped.
