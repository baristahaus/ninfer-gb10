# Flash-Next GPU work report

Measured wall: 8217.293 ms; GPU busy: 8128.547 ms; attributed GPU work: 93.96%.

Estimates describe partial modeled work, not an achievable optimum. Bytes/FLOPs below are per call; time and share are aggregated.

| Phase / role | Stage | T / B / context envelope | Calls | GPU work ms | Phase share | Useful bytes min–max | BF16 / NVFP4 / FP32 FLOPs | Estimate ms min–max | Estimate efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | 8 / 0 / 0 | 5952 | 4500.998 | 55.4% | 35279520–228816640 | 99655680 / 786432000 / 0 | 853.592–5536.247 | 19.0–123.0% |
| decode / target.verify | gdn.record | 8 / 4 / 0 | 4464 | 1679.928 | 20.7% | 71163904–71163904 | 926679040 / 0 / 37748736 | 1291.365–1291.365 | 76.9–76.9% |
| decode / target.verify | hyper.combine_mix | 8 / 0 / 0 | 11656 | 676.958 | 8.3% | 7066368–7066368 | 105512960 / 0 / 0 | 334.819–334.819 | 49.5–49.5% |
| decode / target.verify | qsa.select | 8 / 4 / 129 | 1488 | 560.341 | 6.9% | 55817216–55817216 | 823132160 / 0 / 0 | 337.626–337.626 | 60.3–60.3% |
| decode / predictor | moe.nvfp4 | 8 / 0 / 0 | 124 | 92.830 | 1.1% | 35279520–228816640 | 99655680 / 786432000 / 0 | 17.783–115.338 | 19.2–124.2% |
| decode / predictor | qsa.select | 8 / 4 / 129 | 124 | 46.357 | 0.6% | 55817216–55817216 | 823132160 / 0 / 0 | 28.136–28.136 | 60.7–60.7% |
| decode / target.verify | ple.record | 8 / 4 / 0 | 124 | 40.408 | 0.5% | 65884160–65884160 | 524288000 / 0 / 0 | 33.210–33.210 | 82.2–82.2% |
| decode / target.verify | hyper.mix | 8 / 0 / 0 | 248 | 13.355 | 0.2% | 6861504–6861504 | 105512960 / 0 / 0 | 6.917–6.917 | 51.8–51.8% |
| decode / predictor | hyper.combine_mix | 8 / 0 / 0 | 124 | 8.701 | 0.1% | 7066368–7066368 | 105512960 / 0 / 0 | 3.562–3.562 | 40.9–40.9% |
| decode / predictor | hyper.mix | 8 / 0 / 0 | 124 | 5.591 | 0.1% | 6861504–6861504 | 105512960 / 0 / 0 | 3.459–3.459 | 61.9–61.9% |
| decode / target.verify | hyper.mix | 8 / 0 / 0 | 124 | 5.557 | 0.1% | 6779520–6779520 | 104857600 / 0 / 0 | 3.417–3.417 | 61.5–61.5% |
| decode / predictor | hyper.mix | 8 / 0 / 0 | 124 | 5.230 | 0.1% | 6779520–6779520 | 104857600 / 0 / 0 | 3.417–3.417 | 65.3–65.3% |
| decode / target.verify | hyper.combine | 8 / 0 / 0 | 248 | 0.725 | 0.0% | 368704–368704 | 0 / 0 / 0 | 0.372–0.372 | 51.3–51.3% |
| decode / predictor | hyper.combine | 8 / 0 / 0 | 124 | 0.358 | 0.0% | 368704–368704 | 0 / 0 / 0 | 0.186–0.186 | 51.9–51.9% |
| decode / predictor | hyper.add | 8 / 0 / 0 | 124 | 0.265 | 0.0% | 368640–368640 | 0 / 0 / 0 | 0.186–0.186 | 70.2–70.2% |
| decode / target.verify | hyper.repeat | 8 / 0 / 0 | 124 | 0.262 | 0.0% | 204800–204800 | 0 / 0 / 0 | 0.103–0.103 | 39.4–39.4% |

Unattributed GPU work: 490.684 ms.

Kernels per stage (ms/round, launches per round):

| Phase / role | Stage | Kernel | Launches | GPU work ms/round | Stage share |
|---|---|---|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 96.0 | 33.072 | 91.1% |
| decode / target.verify | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 96.0 | 1.268 | 3.5% |
| decode / target.verify | moe.nvfp4 | bf16_small_t_inner_kernel | 48.0 | 0.794 | 2.2% |
| decode / target.verify | moe.nvfp4 | route_kernel | 48.0 | 0.595 | 1.6% |
| decode / target.verify | moe.nvfp4 | quantize_grouped_kernel | 48.0 | 0.202 | 0.6% |
| decode / target.verify | moe.nvfp4 | gather_quantize_routes_kernel | 48.0 | 0.200 | 0.6% |
| decode / target.verify | moe.nvfp4 | reduce_grouped_kernel | 48.0 | 0.098 | 0.3% |
| decode / target.verify | moe.nvfp4 | route_stats_kernel | 48.0 | 0.069 | 0.2% |
| decode / target.verify | gdn.record | fp8_a16_sliced_k_mma_kernel | 108.0 | 10.089 | 74.5% |
| decode / target.verify | gdn.record | recurrent_fold_record_kernel | 36.0 | 2.848 | 21.0% |
| decode / target.verify | gdn.record | project_control_gating_kernel | 36.0 | 0.292 | 2.2% |
| decode / target.verify | gdn.record | conv_replay_record_kernel | 36.0 | 0.249 | 1.8% |
| decode / target.verify | gdn.record | rmsnorm_warp_bf16x2_kernel | 36.0 | 0.069 | 0.5% |
| decode / target.verify | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 188.0 | 4.134 | 75.7% |
| decode / target.verify | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 94.0 | 1.326 | 24.3% |
| decode / target.verify | qsa.select | fp8_a16_sliced_k_mma_kernel | 24.0 | 2.473 | 54.7% |
| decode / target.verify | qsa.select | selected_attention_split_fp8_kernel | 12.0 | 1.008 | 22.3% |
| decode / target.verify | qsa.select | bf16_small_t_inner_kernel | 48.0 | 0.703 | 15.6% |
| decode / target.verify | qsa.select | append_cache_fp8_kernel | 12.0 | 0.065 | 1.4% |
| decode / target.verify | qsa.select | reduce_selected_attention_splits_kernel | 12.0 | 0.058 | 1.3% |
| decode / target.verify | qsa.select | rmsnorm_warp_bf16x2_kernel | 24.0 | 0.050 | 1.1% |
| decode / target.verify | qsa.select | rope_generic_kernel | 12.0 | 0.050 | 1.1% |
| decode / target.verify | qsa.select | compress_index_groups_kernel | 12.0 | 0.037 | 0.8% |
| decode / target.verify | qsa.select | prepare_index_query_kernel | 12.0 | 0.023 | 0.5% |
| decode / target.verify | qsa.select | split_query_gate_kernel | 12.0 | 0.020 | 0.4% |
| decode / target.verify | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 12.0 | 0.018 | 0.4% |
| decode / target.verify | qsa.select | hadamard_rows_kernel | 12.0 | 0.014 | 0.3% |
| decode / predictor | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 2.0 | 0.682 | 91.1% |
| decode / predictor | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.026 | 3.5% |
| decode / predictor | moe.nvfp4 | bf16_small_t_inner_kernel | 1.0 | 0.016 | 2.2% |
| decode / predictor | moe.nvfp4 | route_kernel | 1.0 | 0.012 | 1.6% |
| decode / predictor | moe.nvfp4 | quantize_grouped_kernel | 1.0 | 0.004 | 0.6% |
| decode / predictor | moe.nvfp4 | gather_quantize_routes_kernel | 1.0 | 0.004 | 0.6% |
| decode / predictor | moe.nvfp4 | reduce_grouped_kernel | 1.0 | 0.002 | 0.3% |
| decode / predictor | moe.nvfp4 | route_stats_kernel | 1.0 | 0.001 | 0.2% |
| decode / predictor | qsa.select | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.200 | 53.6% |
| decode / predictor | qsa.select | selected_attention_split_fp8_kernel | 1.0 | 0.083 | 22.2% |
| decode / predictor | qsa.select | bf16_small_t_inner_kernel | 4.0 | 0.060 | 16.1% |
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
| decode / target.verify | ple.record | dilated_conv_record_kernel | 1.0 | 0.006 | 1.8% |
| decode / target.verify | ple.record | grouped_norm_kernel | 1.0 | 0.006 | 1.8% |
| decode / target.verify | ple.record | dequantize_embedding_kernel | 1.0 | 0.004 | 1.1% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 4.0 | 0.089 | 82.7% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 2.0 | 0.019 | 17.3% |
| decode / predictor | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.056 | 80.1% |
| decode / predictor | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 1.0 | 0.014 | 19.9% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.037 | 81.6% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.008 | 18.4% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.039 | 86.0% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.006 | 14.0% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.036 | 85.3% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.006 | 14.7% |
| decode / target.verify | hyper.combine | combine_kernel | 2.0 | 0.006 | 100.0% |
| decode / predictor | hyper.combine | combine_kernel | 1.0 | 0.003 | 100.0% |
| decode / predictor | hyper.add | add_repeated_kernel | 1.0 | 0.002 | 100.0% |
| decode / target.verify | hyper.repeat | repeat_kernel | 1.0 | 0.002 | 100.0% |

Per round over 124 rounds: host wall 66.268 ms, GPU work 65.553 ms, GPU busy 65.553 ms.

| Phase / role | Stage | GPU work ms/round | Share |
|---|---|---:|---:|
| decode / target.verify | moe.nvfp4 | 36.298 | 55.4% |
| decode / target.verify | gdn.record | 13.548 | 20.7% |
| decode / target.verify | hyper.combine_mix | 5.459 | 8.3% |
| decode / target.verify | qsa.select | 4.519 | 6.9% |
| decode / predictor | moe.nvfp4 | 0.749 | 1.1% |
| decode / predictor | qsa.select | 0.374 | 0.6% |
| decode / target.verify | ple.record | 0.326 | 0.5% |
| decode / target.verify | hyper.mix | 0.108 | 0.2% |
| decode / predictor | hyper.combine_mix | 0.070 | 0.1% |
| decode / predictor | hyper.mix | 0.045 | 0.1% |
| decode / target.verify | hyper.mix | 0.045 | 0.1% |
| decode / predictor | hyper.mix | 0.042 | 0.1% |
| decode / target.verify | hyper.combine | 0.006 | 0.0% |
| decode / predictor | hyper.combine | 0.003 | 0.0% |
| decode / predictor | hyper.add | 0.002 | 0.0% |
| decode / target.verify | hyper.repeat | 0.002 | 0.0% |
| unattributed | kernel fp8_a16_sliced_k_mma_kernel | 2.620 | 4.0% |
| unattributed | kernel q4_a16_sliced_k_mma_kernel | 0.876 | 1.3% |
| unattributed | kernel bf16_gemm_mma_kernel | 0.070 | 0.1% |
| unattributed | kernel bf16_small_t_inner_kernel | 0.058 | 0.1% |
| unattributed | kernel copy_record_rows_kernel | 0.056 | 0.1% |
| unattributed | kernel argmax_tiled_atomic_kernel | 0.054 | 0.1% |
| unattributed | kernel shortlist_exact_scores_fp8_kernel | 0.047 | 0.1% |
| unattributed | kernel speculative_sampling_partial_topk_kernel | 0.034 | 0.1% |
| unattributed | kernel shortlist_tile_candidates_kernel | 0.028 | 0.0% |
| unattributed | kernel ple_fold_kernel | 0.026 | 0.0% |
| unattributed | kernel rmsnorm_generic_kernel | 0.019 | 0.0% |
| unattributed | memcpy memcpy | 0.010 | 0.0% |
| unattributed | kernel speculative_sampling_group_finalize_kernel | 0.009 | 0.0% |
| unattributed | kernel embed_gather_dense_kernel | 0.008 | 0.0% |
| unattributed | kernel publish_ids_kernel | 0.006 | 0.0% |
| unattributed | kernel scatter_bf16x8_kernel | 0.006 | 0.0% |
| unattributed | kernel speculative_select_accepted_hidden_kernel | 0.006 | 0.0% |
| unattributed | kernel wait_staged_kernel | 0.006 | 0.0% |
| unattributed | kernel mtp_advance_round_kernel | 0.004 | 0.0% |
| unattributed | kernel rmsnorm_cta_bf16x2_kernel | 0.003 | 0.0% |
| unattributed | kernel expand_text_positions_kernel | 0.002 | 0.0% |
| unattributed | kernel mtp_prepare_next_round_kernel | 0.002 | 0.0% |
| unattributed | kernel speculative_prepare_verify_inputs_kernel | 0.002 | 0.0% |
| unattributed | kernel select_shared_indices_kernel | 0.001 | 0.0% |
| unattributed | kernel shortlist_exact_select_kernel | 0.001 | 0.0% |
| unattributed | kernel residual_add_bf16x8_kernel | 0.001 | 0.0% |
| unattributed | memset memset | 0.000 | 0.0% |

Host lookup work (may overlap GPU execution):

- ple.gather: 74.129 ms across 124 calls.
- ple.hash: 0.000 ms across 0 calls.

- Estimates are partial cold useful-work models, not distance from optimal.
- Token counts are execution envelopes; padded columns are not committed outputs.
- QSA estimates cover projections only; device-dependent scoring/attention is unmodeled.
- GDN prefill estimates omit chunked recurrence compute; PLE estimates omit convolution/state traffic.
- DRAM cache reuse, scratch traffic, nonlinear math and instruction geometry are unmodeled.
- Profiler overhead affects timings; use unprofiled paired benchmarks for speed claims.
- Unattributed GPU work is retained; no whole-model efficiency is reported.
- Serve window: GPU work launched inside 124 decode.mtp_round ranges with batch 4 (trim 0.00 of rounds at each end); work from other host threads launched during a round is included.
- An estimate exceeds measured time: inspect cache reuse, envelope work and hardware rates; percentages are not clamped.
