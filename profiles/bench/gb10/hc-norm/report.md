# Flash-Next GPU work report

Measured wall: 7962.344 ms; GPU busy: 7879.952 ms; attributed GPU work: 93.62%.

Estimates describe partial modeled work, not an achievable optimum. Bytes/FLOPs below are per call; time and share are aggregated.

| Phase / role | Stage | T / B / context envelope | Calls | GPU work ms | Phase share | Useful bytes min–max | BF16 / NVFP4 / FP32 FLOPs | Estimate ms min–max | Estimate efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | 8 / 0 / 0 | 5808 | 4423.799 | 56.1% | 35279520–228816640 | 99655680 / 786432000 / 0 | 832.941–5402.305 | 18.8–122.1% |
| decode / target.verify | gdn.record | 8 / 4 / 0 | 4356 | 1640.401 | 20.8% | 71163904–71163904 | 926679040 / 0 / 37748736 | 1260.122–1260.122 | 76.8–76.8% |
| decode / target.verify | hyper.combine_mix | 8 / 0 / 0 | 11374 | 552.838 | 7.0% | 7066368–7066368 | 105512960 / 0 / 0 | 326.719–326.719 | 59.1–59.1% |
| decode / target.verify | qsa.select | 8 / 4 / 129 | 1452 | 546.587 | 6.9% | 55817216–55817216 | 823132160 / 0 / 0 | 329.458–329.458 | 60.3–60.3% |
| decode / predictor | moe.nvfp4 | 8 / 0 / 0 | 121 | 93.066 | 1.2% | 35279520–228816640 | 99655680 / 786432000 / 0 | 17.353–112.548 | 18.6–120.9% |
| decode / predictor | qsa.select | 8 / 4 / 129 | 121 | 46.154 | 0.6% | 55817216–55817216 | 823132160 / 0 / 0 | 27.455–27.455 | 59.5–59.5% |
| decode / target.verify | ple.record | 8 / 4 / 0 | 121 | 39.293 | 0.5% | 65884160–65884160 | 524288000 / 0 / 0 | 32.406–32.406 | 82.5–82.5% |
| decode / target.verify | hyper.mix | 8 / 0 / 0 | 242 | 11.936 | 0.2% | 6861504–6861504 | 105512960 / 0 / 0 | 6.750–6.750 | 56.5–56.5% |
| decode / predictor | hyper.combine_mix | 8 / 0 / 0 | 121 | 7.380 | 0.1% | 7066368–7066368 | 105512960 / 0 / 0 | 3.476–3.476 | 47.1–47.1% |
| decode / predictor | hyper.mix | 8 / 0 / 0 | 121 | 4.890 | 0.1% | 6861504–6861504 | 105512960 / 0 / 0 | 3.375–3.375 | 69.0–69.0% |
| decode / target.verify | hyper.mix | 8 / 0 / 0 | 121 | 4.821 | 0.1% | 6779520–6779520 | 104857600 / 0 / 0 | 3.335–3.335 | 69.2–69.2% |
| decode / predictor | hyper.mix | 8 / 0 / 0 | 121 | 4.670 | 0.1% | 6779520–6779520 | 104857600 / 0 / 0 | 3.335–3.335 | 71.4–71.4% |
| decode / target.verify | hyper.combine | 8 / 0 / 0 | 242 | 0.693 | 0.0% | 368704–368704 | 0 / 0 / 0 | 0.363–0.363 | 52.3–52.3% |
| decode / predictor | hyper.combine | 8 / 0 / 0 | 121 | 0.345 | 0.0% | 368704–368704 | 0 / 0 / 0 | 0.181–0.181 | 52.6–52.6% |
| decode / predictor | hyper.add | 8 / 0 / 0 | 121 | 0.259 | 0.0% | 368640–368640 | 0 / 0 / 0 | 0.181–0.181 | 70.1–70.1% |
| decode / target.verify | hyper.repeat | 8 / 0 / 0 | 121 | 0.258 | 0.0% | 204800–204800 | 0 / 0 / 0 | 0.101–0.101 | 39.0–39.0% |

Unattributed GPU work: 502.590 ms.

Kernels per stage (ms/round, launches per round):

| Phase / role | Stage | Kernel | Launches | GPU work ms/round | Stage share |
|---|---|---|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 96.0 | 33.382 | 91.3% |
| decode / target.verify | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 96.0 | 1.269 | 3.5% |
| decode / target.verify | moe.nvfp4 | bf16_small_t_inner_kernel | 48.0 | 0.813 | 2.2% |
| decode / target.verify | moe.nvfp4 | route_kernel | 48.0 | 0.595 | 1.6% |
| decode / target.verify | moe.nvfp4 | quantize_grouped_kernel | 48.0 | 0.203 | 0.6% |
| decode / target.verify | moe.nvfp4 | gather_quantize_routes_kernel | 48.0 | 0.201 | 0.5% |
| decode / target.verify | moe.nvfp4 | reduce_grouped_kernel | 48.0 | 0.099 | 0.3% |
| decode / target.verify | gdn.record | fp8_a16_sliced_k_mma_kernel | 108.0 | 10.118 | 74.6% |
| decode / target.verify | gdn.record | recurrent_fold_record_kernel | 36.0 | 2.825 | 20.8% |
| decode / target.verify | gdn.record | project_control_gating_kernel | 36.0 | 0.294 | 2.2% |
| decode / target.verify | gdn.record | conv_replay_record_kernel | 36.0 | 0.250 | 1.8% |
| decode / target.verify | gdn.record | rmsnorm_warp_bf16x2_kernel | 36.0 | 0.070 | 0.5% |
| decode / target.verify | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 188.0 | 4.179 | 91.5% |
| decode / target.verify | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 94.0 | 0.390 | 8.5% |
| decode / target.verify | qsa.select | fp8_a16_sliced_k_mma_kernel | 24.0 | 2.469 | 54.6% |
| decode / target.verify | qsa.select | selected_attention_split_fp8_kernel | 12.0 | 1.011 | 22.4% |
| decode / target.verify | qsa.select | bf16_small_t_inner_kernel | 48.0 | 0.702 | 15.6% |
| decode / target.verify | qsa.select | append_cache_fp8_kernel | 12.0 | 0.065 | 1.4% |
| decode / target.verify | qsa.select | reduce_selected_attention_splits_kernel | 12.0 | 0.058 | 1.3% |
| decode / target.verify | qsa.select | rmsnorm_warp_bf16x2_kernel | 24.0 | 0.051 | 1.1% |
| decode / target.verify | qsa.select | rope_generic_kernel | 12.0 | 0.050 | 1.1% |
| decode / target.verify | qsa.select | compress_index_groups_kernel | 12.0 | 0.037 | 0.8% |
| decode / target.verify | qsa.select | prepare_index_query_kernel | 12.0 | 0.023 | 0.5% |
| decode / target.verify | qsa.select | split_query_gate_kernel | 12.0 | 0.020 | 0.4% |
| decode / target.verify | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 12.0 | 0.018 | 0.4% |
| decode / target.verify | qsa.select | hadamard_rows_kernel | 12.0 | 0.014 | 0.3% |
| decode / predictor | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 2.0 | 0.704 | 91.5% |
| decode / predictor | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.026 | 3.4% |
| decode / predictor | moe.nvfp4 | bf16_small_t_inner_kernel | 1.0 | 0.016 | 2.1% |
| decode / predictor | moe.nvfp4 | route_kernel | 1.0 | 0.012 | 1.6% |
| decode / predictor | moe.nvfp4 | quantize_grouped_kernel | 1.0 | 0.004 | 0.6% |
| decode / predictor | moe.nvfp4 | gather_quantize_routes_kernel | 1.0 | 0.004 | 0.5% |
| decode / predictor | moe.nvfp4 | reduce_grouped_kernel | 1.0 | 0.002 | 0.3% |
| decode / predictor | qsa.select | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.207 | 54.4% |
| decode / predictor | qsa.select | selected_attention_split_fp8_kernel | 1.0 | 0.083 | 21.7% |
| decode / predictor | qsa.select | bf16_small_t_inner_kernel | 4.0 | 0.061 | 16.1% |
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
| decode / target.verify | ple.record | bf16_gemv_columns_kernel | 2.0 | 0.301 | 92.6% |
| decode / target.verify | ple.record | gate_kernel | 1.0 | 0.008 | 2.6% |
| decode / target.verify | ple.record | dilated_conv_record_kernel | 1.0 | 0.006 | 2.0% |
| decode / target.verify | ple.record | grouped_norm_kernel | 1.0 | 0.006 | 1.8% |
| decode / target.verify | ple.record | dequantize_embedding_kernel | 1.0 | 0.004 | 1.1% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 4.0 | 0.090 | 91.6% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 2.0 | 0.008 | 8.4% |
| decode / predictor | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.058 | 94.7% |
| decode / predictor | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 1.0 | 0.003 | 5.3% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.038 | 94.4% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.002 | 5.6% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.038 | 96.2% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.002 | 3.8% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.037 | 96.1% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.001 | 3.9% |
| decode / target.verify | hyper.combine | combine_kernel | 2.0 | 0.006 | 100.0% |
| decode / predictor | hyper.combine | combine_kernel | 1.0 | 0.003 | 100.0% |
| decode / predictor | hyper.add | add_repeated_kernel | 1.0 | 0.002 | 100.0% |
| decode / target.verify | hyper.repeat | repeat_kernel | 1.0 | 0.002 | 100.0% |

Per round over 121 rounds: host wall 65.804 ms, GPU work 65.124 ms, GPU busy 65.124 ms.

| Phase / role | Stage | GPU work ms/round | Share |
|---|---|---:|---:|
| decode / target.verify | moe.nvfp4 | 36.560 | 56.1% |
| decode / target.verify | gdn.record | 13.557 | 20.8% |
| decode / target.verify | hyper.combine_mix | 4.569 | 7.0% |
| decode / target.verify | qsa.select | 4.517 | 6.9% |
| decode / predictor | moe.nvfp4 | 0.769 | 1.2% |
| decode / predictor | qsa.select | 0.381 | 0.6% |
| decode / target.verify | ple.record | 0.325 | 0.5% |
| decode / target.verify | hyper.mix | 0.099 | 0.2% |
| decode / predictor | hyper.combine_mix | 0.061 | 0.1% |
| decode / predictor | hyper.mix | 0.040 | 0.1% |
| decode / target.verify | hyper.mix | 0.040 | 0.1% |
| decode / predictor | hyper.mix | 0.039 | 0.1% |
| decode / target.verify | hyper.combine | 0.006 | 0.0% |
| decode / predictor | hyper.combine | 0.003 | 0.0% |
| decode / predictor | hyper.add | 0.002 | 0.0% |
| decode / target.verify | hyper.repeat | 0.002 | 0.0% |
| unattributed | kernel fp8_a16_sliced_k_mma_kernel | 2.595 | 4.0% |
| unattributed | kernel q4_a16_sliced_k_mma_kernel | 0.899 | 1.4% |
| unattributed | kernel wait_staged_kernel | 0.197 | 0.3% |
| unattributed | kernel bf16_gemm_mma_kernel | 0.071 | 0.1% |
| unattributed | kernel bf16_small_t_inner_kernel | 0.059 | 0.1% |
| unattributed | kernel copy_record_rows_kernel | 0.057 | 0.1% |
| unattributed | kernel argmax_tiled_atomic_kernel | 0.054 | 0.1% |
| unattributed | kernel shortlist_exact_scores_fp8_kernel | 0.047 | 0.1% |
| unattributed | kernel speculative_sampling_partial_topk_kernel | 0.034 | 0.1% |
| unattributed | kernel shortlist_tile_candidates_kernel | 0.028 | 0.0% |
| unattributed | kernel ple_fold_kernel | 0.026 | 0.0% |
| unattributed | kernel rmsnorm_generic_kernel | 0.019 | 0.0% |
| unattributed | memcpy memcpy | 0.011 | 0.0% |
| unattributed | kernel speculative_sampling_group_finalize_kernel | 0.009 | 0.0% |
| unattributed | kernel embed_gather_dense_kernel | 0.008 | 0.0% |
| unattributed | kernel publish_ids_kernel | 0.007 | 0.0% |
| unattributed | kernel speculative_select_accepted_hidden_kernel | 0.006 | 0.0% |
| unattributed | kernel scatter_bf16x8_kernel | 0.006 | 0.0% |
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

- ple.gather: 142.754 ms across 121 calls.
- ple.hash: 0.000 ms across 0 calls.

- Estimates are partial cold useful-work models, not distance from optimal.
- Token counts are execution envelopes; padded columns are not committed outputs.
- QSA estimates cover projections only; device-dependent scoring/attention is unmodeled.
- GDN prefill estimates omit chunked recurrence compute; PLE estimates omit convolution/state traffic.
- DRAM cache reuse, scratch traffic, nonlinear math and instruction geometry are unmodeled.
- Profiler overhead affects timings; use unprofiled paired benchmarks for speed claims.
- Unattributed GPU work is retained; no whole-model efficiency is reported.
- Serve window: GPU work launched inside 121 decode.mtp_round ranges with batch 4 (trim 0.00 of rounds at each end); work from other host threads launched during a round is included.
- An estimate exceeds measured time: inspect cache reuse, envelope work and hardware rates; percentages are not clamped.
