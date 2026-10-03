# Flash-Next GPU work report

Measured wall: 8439.655 ms; GPU busy: 8351.311 ms; attributed GPU work: 93.77%.

Estimates describe partial modeled work, not an achievable optimum. Bytes/FLOPs below are per call; time and share are aggregated.

| Phase / role | Stage | T / B / context envelope | Calls | GPU work ms | Phase share | Useful bytes min–max | BF16 / NVFP4 / FP32 FLOPs | Estimate ms min–max | Estimate efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | 8 / 0 / 0 | 6096 | 4638.260 | 55.5% | 35279520–228816640 | 99655680 / 786432000 / 0 | 874.244–5670.188 | 18.8–122.2% |
| decode / target.verify | gdn.record | 8 / 4 / 0 | 4572 | 1711.914 | 20.5% | 71163904–71163904 | 926679040 / 0 / 37748736 | 1322.607–1322.607 | 77.3–77.3% |
| decode / target.verify | hyper.combine_mix | 8 / 0 / 0 | 11938 | 686.673 | 8.2% | 7066368–7066368 | 105512960 / 0 / 0 | 342.920–342.920 | 49.9–49.9% |
| decode / target.verify | qsa.select | 8 / 4 / 129 | 1524 | 569.791 | 6.8% | 55817216–55817216 | 823132160 / 0 / 0 | 345.794–345.794 | 60.7–60.7% |
| decode / predictor | moe.nvfp4 | 8 / 0 / 0 | 127 | 94.162 | 1.1% | 35279520–228816640 | 99655680 / 786432000 / 0 | 18.213–118.129 | 19.3–125.5% |
| decode / predictor | qsa.select | 8 / 4 / 129 | 127 | 47.807 | 0.6% | 55817216–55817216 | 823132160 / 0 / 0 | 28.816–28.816 | 60.3–60.3% |
| decode / target.verify | ple.record | 8 / 4 / 0 | 127 | 41.837 | 0.5% | 65884160–65884160 | 524288000 / 0 / 0 | 34.013–34.013 | 81.3–81.3% |
| decode / target.verify | hyper.mix | 8 / 0 / 0 | 254 | 13.757 | 0.2% | 6861504–6861504 | 105512960 / 0 / 0 | 7.085–7.085 | 51.5–51.5% |
| decode / predictor | hyper.combine_mix | 8 / 0 / 0 | 127 | 9.033 | 0.1% | 7066368–7066368 | 105512960 / 0 / 0 | 3.648–3.648 | 40.4–40.4% |
| decode / predictor | hyper.mix | 8 / 0 / 0 | 127 | 5.595 | 0.1% | 6861504–6861504 | 105512960 / 0 / 0 | 3.542–3.542 | 63.3–63.3% |
| decode / target.verify | hyper.mix | 8 / 0 / 0 | 127 | 5.510 | 0.1% | 6779520–6779520 | 104857600 / 0 / 0 | 3.500–3.500 | 63.5–63.5% |
| decode / predictor | hyper.mix | 8 / 0 / 0 | 127 | 5.401 | 0.1% | 6779520–6779520 | 104857600 / 0 / 0 | 3.500–3.500 | 64.8–64.8% |
| decode / target.verify | hyper.combine | 8 / 0 / 0 | 254 | 0.727 | 0.0% | 368704–368704 | 0 / 0 / 0 | 0.381–0.381 | 52.4–52.4% |
| decode / predictor | hyper.combine | 8 / 0 / 0 | 127 | 0.366 | 0.0% | 368704–368704 | 0 / 0 / 0 | 0.190–0.190 | 52.0–52.0% |
| decode / predictor | hyper.add | 8 / 0 / 0 | 127 | 0.276 | 0.0% | 368640–368640 | 0 / 0 / 0 | 0.190–0.190 | 68.9–68.9% |
| decode / target.verify | hyper.repeat | 8 / 0 / 0 | 127 | 0.264 | 0.0% | 204800–204800 | 0 / 0 / 0 | 0.106–0.106 | 40.0–40.0% |

Unattributed GPU work: 519.945 ms.

Kernels per stage (ms/round, launches per round):

| Phase / role | Stage | Kernel | Launches | GPU work ms/round | Stage share |
|---|---|---|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 96.0 | 33.358 | 91.3% |
| decode / target.verify | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 96.0 | 1.265 | 3.5% |
| decode / target.verify | moe.nvfp4 | bf16_small_t_inner_kernel | 48.0 | 0.800 | 2.2% |
| decode / target.verify | moe.nvfp4 | route_kernel | 48.0 | 0.594 | 1.6% |
| decode / target.verify | moe.nvfp4 | quantize_grouped_kernel | 48.0 | 0.203 | 0.6% |
| decode / target.verify | moe.nvfp4 | gather_quantize_routes_kernel | 48.0 | 0.200 | 0.5% |
| decode / target.verify | moe.nvfp4 | reduce_grouped_kernel | 48.0 | 0.102 | 0.3% |
| decode / target.verify | gdn.record | fp8_a16_sliced_k_mma_kernel | 108.0 | 10.005 | 74.2% |
| decode / target.verify | gdn.record | recurrent_fold_record_kernel | 36.0 | 2.862 | 21.2% |
| decode / target.verify | gdn.record | project_control_gating_kernel | 36.0 | 0.292 | 2.2% |
| decode / target.verify | gdn.record | conv_replay_record_kernel | 36.0 | 0.251 | 1.9% |
| decode / target.verify | gdn.record | rmsnorm_warp_bf16x2_kernel | 36.0 | 0.070 | 0.5% |
| decode / target.verify | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 188.0 | 4.081 | 75.5% |
| decode / target.verify | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 94.0 | 1.326 | 24.5% |
| decode / target.verify | qsa.select | fp8_a16_sliced_k_mma_kernel | 24.0 | 2.445 | 54.5% |
| decode / target.verify | qsa.select | selected_attention_split_fp8_kernel | 12.0 | 1.005 | 22.4% |
| decode / target.verify | qsa.select | bf16_small_t_inner_kernel | 48.0 | 0.701 | 15.6% |
| decode / target.verify | qsa.select | append_cache_fp8_kernel | 12.0 | 0.065 | 1.4% |
| decode / target.verify | qsa.select | reduce_selected_attention_splits_kernel | 12.0 | 0.059 | 1.3% |
| decode / target.verify | qsa.select | rmsnorm_warp_bf16x2_kernel | 24.0 | 0.050 | 1.1% |
| decode / target.verify | qsa.select | rope_generic_kernel | 12.0 | 0.050 | 1.1% |
| decode / target.verify | qsa.select | compress_index_groups_kernel | 12.0 | 0.037 | 0.8% |
| decode / target.verify | qsa.select | prepare_index_query_kernel | 12.0 | 0.023 | 0.5% |
| decode / target.verify | qsa.select | split_query_gate_kernel | 12.0 | 0.019 | 0.4% |
| decode / target.verify | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 12.0 | 0.018 | 0.4% |
| decode / target.verify | qsa.select | hadamard_rows_kernel | 12.0 | 0.014 | 0.3% |
| decode / predictor | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 2.0 | 0.676 | 91.2% |
| decode / predictor | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.026 | 3.6% |
| decode / predictor | moe.nvfp4 | bf16_small_t_inner_kernel | 1.0 | 0.016 | 2.2% |
| decode / predictor | moe.nvfp4 | route_kernel | 1.0 | 0.012 | 1.6% |
| decode / predictor | moe.nvfp4 | quantize_grouped_kernel | 1.0 | 0.004 | 0.6% |
| decode / predictor | moe.nvfp4 | gather_quantize_routes_kernel | 1.0 | 0.004 | 0.6% |
| decode / predictor | moe.nvfp4 | reduce_grouped_kernel | 1.0 | 0.002 | 0.3% |
| decode / predictor | qsa.select | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.204 | 54.2% |
| decode / predictor | qsa.select | selected_attention_split_fp8_kernel | 1.0 | 0.082 | 21.9% |
| decode / predictor | qsa.select | bf16_small_t_inner_kernel | 4.0 | 0.060 | 15.9% |
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
| decode / target.verify | ple.record | bf16_gemv_columns_kernel | 2.0 | 0.305 | 92.7% |
| decode / target.verify | ple.record | gate_kernel | 1.0 | 0.008 | 2.5% |
| decode / target.verify | ple.record | dilated_conv_record_kernel | 1.0 | 0.006 | 1.8% |
| decode / target.verify | ple.record | grouped_norm_kernel | 1.0 | 0.006 | 1.8% |
| decode / target.verify | ple.record | dequantize_embedding_kernel | 1.0 | 0.004 | 1.2% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 4.0 | 0.090 | 82.8% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 2.0 | 0.019 | 17.2% |
| decode / predictor | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.057 | 80.3% |
| decode / predictor | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 1.0 | 0.014 | 19.7% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.036 | 81.2% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.008 | 18.8% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.037 | 85.7% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.006 | 14.3% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.036 | 85.2% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.006 | 14.8% |
| decode / target.verify | hyper.combine | combine_kernel | 2.0 | 0.006 | 100.0% |
| decode / predictor | hyper.combine | combine_kernel | 1.0 | 0.003 | 100.0% |
| decode / predictor | hyper.add | add_repeated_kernel | 1.0 | 0.002 | 100.0% |
| decode / target.verify | hyper.repeat | repeat_kernel | 1.0 | 0.002 | 100.0% |

Per round over 127 rounds: host wall 66.454 ms, GPU work 65.758 ms, GPU busy 65.758 ms.

| Phase / role | Stage | GPU work ms/round | Share |
|---|---|---:|---:|
| decode / target.verify | moe.nvfp4 | 36.522 | 55.5% |
| decode / target.verify | gdn.record | 13.480 | 20.5% |
| decode / target.verify | hyper.combine_mix | 5.407 | 8.2% |
| decode / target.verify | qsa.select | 4.487 | 6.8% |
| decode / predictor | moe.nvfp4 | 0.741 | 1.1% |
| decode / predictor | qsa.select | 0.376 | 0.6% |
| decode / target.verify | ple.record | 0.329 | 0.5% |
| decode / target.verify | hyper.mix | 0.108 | 0.2% |
| decode / predictor | hyper.combine_mix | 0.071 | 0.1% |
| decode / predictor | hyper.mix | 0.044 | 0.1% |
| decode / target.verify | hyper.mix | 0.043 | 0.1% |
| decode / predictor | hyper.mix | 0.043 | 0.1% |
| decode / target.verify | hyper.combine | 0.006 | 0.0% |
| decode / predictor | hyper.combine | 0.003 | 0.0% |
| decode / predictor | hyper.add | 0.002 | 0.0% |
| decode / target.verify | hyper.repeat | 0.002 | 0.0% |
| unattributed | kernel fp8_a16_sliced_k_mma_kernel | 2.581 | 3.9% |
| unattributed | kernel q4_a16_sliced_k_mma_kernel | 0.884 | 1.3% |
| unattributed | kernel wait_staged_kernel | 0.171 | 0.3% |
| unattributed | kernel bf16_gemm_mma_kernel | 0.070 | 0.1% |
| unattributed | kernel bf16_small_t_inner_kernel | 0.058 | 0.1% |
| unattributed | kernel copy_record_rows_kernel | 0.058 | 0.1% |
| unattributed | kernel argmax_tiled_atomic_kernel | 0.054 | 0.1% |
| unattributed | kernel shortlist_exact_scores_fp8_kernel | 0.047 | 0.1% |
| unattributed | kernel speculative_sampling_partial_topk_kernel | 0.033 | 0.1% |
| unattributed | kernel shortlist_tile_candidates_kernel | 0.028 | 0.0% |
| unattributed | kernel ple_fold_kernel | 0.025 | 0.0% |
| unattributed | kernel rmsnorm_generic_kernel | 0.019 | 0.0% |
| unattributed | memcpy memcpy | 0.010 | 0.0% |
| unattributed | kernel speculative_sampling_group_finalize_kernel | 0.009 | 0.0% |
| unattributed | kernel embed_gather_dense_kernel | 0.008 | 0.0% |
| unattributed | kernel publish_ids_kernel | 0.006 | 0.0% |
| unattributed | kernel scatter_bf16x8_kernel | 0.006 | 0.0% |
| unattributed | kernel speculative_select_accepted_hidden_kernel | 0.006 | 0.0% |
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

- ple.gather: 127.263 ms across 127 calls.
- ple.hash: 0.000 ms across 0 calls.

- Estimates are partial cold useful-work models, not distance from optimal.
- Token counts are execution envelopes; padded columns are not committed outputs.
- QSA estimates cover projections only; device-dependent scoring/attention is unmodeled.
- GDN prefill estimates omit chunked recurrence compute; PLE estimates omit convolution/state traffic.
- DRAM cache reuse, scratch traffic, nonlinear math and instruction geometry are unmodeled.
- Profiler overhead affects timings; use unprofiled paired benchmarks for speed claims.
- Unattributed GPU work is retained; no whole-model efficiency is reported.
- Serve window: GPU work launched inside 127 decode.mtp_round ranges with batch 4 (trim 0.00 of rounds at each end); work from other host threads launched during a round is included.
- An estimate exceeds measured time: inspect cache reuse, envelope work and hardware rates; percentages are not clamped.
