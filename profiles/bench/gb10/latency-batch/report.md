# Flash-Next GPU work report

Measured wall: 8640.849 ms; GPU busy: 8551.256 ms; attributed GPU work: 93.53%.

Estimates describe partial modeled work, not an achievable optimum. Bytes/FLOPs below are per call; time and share are aggregated.

| Phase / role | Stage | T / B / context envelope | Calls | GPU work ms | Phase share | Useful bytes min–max | BF16 / NVFP4 / FP32 FLOPs | Estimate ms min–max | Estimate efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | 8 / 0 / 0 | 6336 | 4802.988 | 56.2% | 35279520–228816640 | 99655680 / 786432000 / 0 | 908.663–5893.424 | 18.9–122.7% |
| decode / target.verify | gdn.record | 8 / 4 / 0 | 4752 | 1789.335 | 20.9% | 71163904–71163904 | 926679040 / 0 / 37748736 | 1374.678–1374.678 | 76.8–76.8% |
| decode / target.verify | hyper.combine_mix | 8 / 0 / 0 | 12408 | 600.229 | 7.0% | 7066368–7066368 | 105512960 / 0 / 0 | 356.421–356.421 | 59.4–59.4% |
| decode / target.verify | qsa.select | 8 / 4 / 129 | 1584 | 577.353 | 6.8% | 55817216–55817216 | 823132160 / 0 / 0 | 359.408–359.408 | 62.3–62.3% |
| decode / predictor | moe.nvfp4 | 8 / 0 / 0 | 132 | 98.249 | 1.1% | 35279520–228816640 | 99655680 / 786432000 / 0 | 18.930–122.780 | 19.3–125.0% |
| decode / predictor | qsa.select | 8 / 4 / 129 | 132 | 48.323 | 0.6% | 55817216–55817216 | 823132160 / 0 / 0 | 29.951–29.951 | 62.0–62.0% |
| decode / target.verify | ple.record | 8 / 4 / 0 | 132 | 43.488 | 0.5% | 65884160–65884160 | 524288000 / 0 / 0 | 35.352–35.352 | 81.3–81.3% |
| decode / target.verify | hyper.mix | 8 / 0 / 0 | 264 | 12.767 | 0.1% | 6861504–6861504 | 105512960 / 0 / 0 | 7.364–7.364 | 57.7–57.7% |
| decode / predictor | hyper.combine_mix | 8 / 0 / 0 | 132 | 8.242 | 0.1% | 7066368–7066368 | 105512960 / 0 / 0 | 3.792–3.792 | 46.0–46.0% |
| decode / target.verify | hyper.mix | 8 / 0 / 0 | 132 | 5.434 | 0.1% | 6779520–6779520 | 104857600 / 0 / 0 | 3.638–3.638 | 66.9–66.9% |
| decode / predictor | hyper.mix | 8 / 0 / 0 | 132 | 5.263 | 0.1% | 6861504–6861504 | 105512960 / 0 / 0 | 3.682–3.682 | 70.0–70.0% |
| decode / predictor | hyper.mix | 8 / 0 / 0 | 132 | 4.984 | 0.1% | 6779520–6779520 | 104857600 / 0 / 0 | 3.638–3.638 | 73.0–73.0% |
| decode / target.verify | hyper.combine | 8 / 0 / 0 | 264 | 0.765 | 0.0% | 368704–368704 | 0 / 0 / 0 | 0.396–0.396 | 51.7–51.7% |
| decode / predictor | hyper.combine | 8 / 0 / 0 | 132 | 0.377 | 0.0% | 368704–368704 | 0 / 0 / 0 | 0.198–0.198 | 52.5–52.5% |
| decode / predictor | hyper.add | 8 / 0 / 0 | 132 | 0.282 | 0.0% | 368640–368640 | 0 / 0 / 0 | 0.198–0.198 | 70.2–70.2% |
| decode / target.verify | hyper.repeat | 8 / 0 / 0 | 132 | 0.279 | 0.0% | 204800–204800 | 0 / 0 / 0 | 0.110–0.110 | 39.4–39.4% |

Unattributed GPU work: 552.918 ms.

Kernels per stage (ms/round, launches per round):

| Phase / role | Stage | Kernel | Launches | GPU work ms/round | Stage share |
|---|---|---|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 96.0 | 33.239 | 91.3% |
| decode / target.verify | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 96.0 | 1.255 | 3.4% |
| decode / target.verify | moe.nvfp4 | bf16_small_t_inner_kernel | 48.0 | 0.794 | 2.2% |
| decode / target.verify | moe.nvfp4 | route_kernel | 48.0 | 0.597 | 1.6% |
| decode / target.verify | moe.nvfp4 | quantize_grouped_kernel | 48.0 | 0.202 | 0.6% |
| decode / target.verify | moe.nvfp4 | gather_quantize_routes_kernel | 48.0 | 0.201 | 0.6% |
| decode / target.verify | moe.nvfp4 | reduce_grouped_kernel | 48.0 | 0.098 | 0.3% |
| decode / target.verify | gdn.record | fp8_a16_sliced_k_mma_kernel | 108.0 | 10.103 | 74.5% |
| decode / target.verify | gdn.record | recurrent_fold_record_kernel | 36.0 | 2.841 | 21.0% |
| decode / target.verify | gdn.record | project_control_gating_kernel | 36.0 | 0.294 | 2.2% |
| decode / target.verify | gdn.record | conv_replay_record_kernel | 36.0 | 0.247 | 1.8% |
| decode / target.verify | gdn.record | rmsnorm_warp_bf16x2_kernel | 36.0 | 0.070 | 0.5% |
| decode / target.verify | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 188.0 | 4.155 | 91.4% |
| decode / target.verify | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 94.0 | 0.392 | 8.6% |
| decode / target.verify | qsa.select | fp8_a16_sliced_k_mma_kernel | 24.0 | 2.451 | 56.0% |
| decode / target.verify | qsa.select | selected_attention_split_fp8_kernel | 12.0 | 1.012 | 23.1% |
| decode / target.verify | qsa.select | bf16_small_t_group_kernel | 12.0 | 0.483 | 11.0% |
| decode / target.verify | qsa.select | bf16_small_t_inner_kernel | 12.0 | 0.109 | 2.5% |
| decode / target.verify | qsa.select | append_cache_fp8_kernel | 12.0 | 0.066 | 1.5% |
| decode / target.verify | qsa.select | reduce_selected_attention_splits_kernel | 12.0 | 0.059 | 1.4% |
| decode / target.verify | qsa.select | rope_generic_kernel | 12.0 | 0.050 | 1.1% |
| decode / target.verify | qsa.select | compress_index_groups_kernel | 12.0 | 0.037 | 0.8% |
| decode / target.verify | qsa.select | rmsnorm_warp_bf16x2_kernel | 24.0 | 0.032 | 0.7% |
| decode / target.verify | qsa.select | prepare_index_query_kernel | 12.0 | 0.023 | 0.5% |
| decode / target.verify | qsa.select | split_query_gate_kernel | 12.0 | 0.020 | 0.4% |
| decode / target.verify | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 12.0 | 0.018 | 0.4% |
| decode / target.verify | qsa.select | hadamard_rows_kernel | 12.0 | 0.014 | 0.3% |
| decode / predictor | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 2.0 | 0.679 | 91.2% |
| decode / predictor | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.026 | 3.5% |
| decode / predictor | moe.nvfp4 | bf16_small_t_inner_kernel | 1.0 | 0.016 | 2.2% |
| decode / predictor | moe.nvfp4 | route_kernel | 1.0 | 0.012 | 1.6% |
| decode / predictor | moe.nvfp4 | quantize_grouped_kernel | 1.0 | 0.004 | 0.6% |
| decode / predictor | moe.nvfp4 | gather_quantize_routes_kernel | 1.0 | 0.004 | 0.6% |
| decode / predictor | moe.nvfp4 | reduce_grouped_kernel | 1.0 | 0.002 | 0.3% |
| decode / predictor | qsa.select | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.205 | 56.0% |
| decode / predictor | qsa.select | selected_attention_split_fp8_kernel | 1.0 | 0.083 | 22.8% |
| decode / predictor | qsa.select | bf16_small_t_group_kernel | 1.0 | 0.040 | 11.0% |
| decode / predictor | qsa.select | bf16_small_t_inner_kernel | 1.0 | 0.008 | 2.2% |
| decode / predictor | qsa.select | append_cache_fp8_kernel | 1.0 | 0.007 | 1.9% |
| decode / predictor | qsa.select | reduce_selected_attention_splits_kernel | 1.0 | 0.005 | 1.3% |
| decode / predictor | qsa.select | rope_generic_kernel | 1.0 | 0.004 | 1.1% |
| decode / predictor | qsa.select | compress_index_groups_kernel | 1.0 | 0.003 | 0.8% |
| decode / predictor | qsa.select | rmsnorm_warp_bf16x2_kernel | 2.0 | 0.003 | 0.8% |
| decode / predictor | qsa.select | prepare_index_query_kernel | 1.0 | 0.002 | 0.5% |
| decode / predictor | qsa.select | split_query_gate_kernel | 1.0 | 0.002 | 0.4% |
| decode / predictor | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 1.0 | 0.002 | 0.4% |
| decode / predictor | qsa.select | dense_indices_batched_kernel | 1.0 | 0.001 | 0.3% |
| decode / predictor | qsa.select | hadamard_rows_kernel | 1.0 | 0.001 | 0.3% |
| decode / target.verify | ple.record | bf16_gemv_columns_kernel | 2.0 | 0.306 | 92.9% |
| decode / target.verify | ple.record | gate_kernel | 1.0 | 0.008 | 2.5% |
| decode / target.verify | ple.record | dilated_conv_record_kernel | 1.0 | 0.006 | 1.8% |
| decode / target.verify | ple.record | grouped_norm_kernel | 1.0 | 0.006 | 1.8% |
| decode / target.verify | ple.record | dequantize_embedding_kernel | 1.0 | 0.003 | 1.1% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 4.0 | 0.090 | 92.8% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 2.0 | 0.007 | 7.2% |
| decode / predictor | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.059 | 93.7% |
| decode / predictor | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 1.0 | 0.004 | 6.3% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.040 | 96.4% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.001 | 3.6% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.038 | 94.2% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.002 | 5.8% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.036 | 96.1% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.001 | 3.9% |
| decode / target.verify | hyper.combine | combine_kernel | 2.0 | 0.006 | 100.0% |
| decode / predictor | hyper.combine | combine_kernel | 1.0 | 0.003 | 100.0% |
| decode / predictor | hyper.add | add_repeated_kernel | 1.0 | 0.002 | 100.0% |
| decode / target.verify | hyper.repeat | repeat_kernel | 1.0 | 0.002 | 100.0% |

Per round over 132 rounds: host wall 65.461 ms, GPU work 64.782 ms, GPU busy 64.782 ms.

| Phase / role | Stage | GPU work ms/round | Share |
|---|---|---:|---:|
| decode / target.verify | moe.nvfp4 | 36.386 | 56.2% |
| decode / target.verify | gdn.record | 13.556 | 20.9% |
| decode / target.verify | hyper.combine_mix | 4.547 | 7.0% |
| decode / target.verify | qsa.select | 4.374 | 6.8% |
| decode / predictor | moe.nvfp4 | 0.744 | 1.1% |
| decode / predictor | qsa.select | 0.366 | 0.6% |
| decode / target.verify | ple.record | 0.329 | 0.5% |
| decode / target.verify | hyper.mix | 0.097 | 0.1% |
| decode / predictor | hyper.combine_mix | 0.062 | 0.1% |
| decode / target.verify | hyper.mix | 0.041 | 0.1% |
| decode / predictor | hyper.mix | 0.040 | 0.1% |
| decode / predictor | hyper.mix | 0.038 | 0.1% |
| decode / target.verify | hyper.combine | 0.006 | 0.0% |
| decode / predictor | hyper.combine | 0.003 | 0.0% |
| decode / predictor | hyper.add | 0.002 | 0.0% |
| decode / target.verify | hyper.repeat | 0.002 | 0.0% |
| unattributed | kernel fp8_a16_sliced_k_mma_kernel | 2.597 | 4.0% |
| unattributed | kernel q4_a16_sliced_k_mma_kernel | 0.896 | 1.4% |
| unattributed | kernel wait_staged_kernel | 0.236 | 0.4% |
| unattributed | kernel bf16_gemm_mma_kernel | 0.071 | 0.1% |
| unattributed | kernel copy_record_rows_kernel | 0.059 | 0.1% |
| unattributed | kernel bf16_small_t_inner_kernel | 0.058 | 0.1% |
| unattributed | kernel argmax_tiled_atomic_kernel | 0.054 | 0.1% |
| unattributed | kernel shortlist_exact_scores_fp8_kernel | 0.047 | 0.1% |
| unattributed | kernel speculative_sampling_partial_topk_kernel | 0.034 | 0.1% |
| unattributed | kernel shortlist_tile_candidates_kernel | 0.028 | 0.0% |
| unattributed | kernel ple_fold_kernel | 0.025 | 0.0% |
| unattributed | kernel rmsnorm_generic_kernel | 0.019 | 0.0% |
| unattributed | memcpy memcpy | 0.010 | 0.0% |
| unattributed | kernel speculative_sampling_group_finalize_kernel | 0.009 | 0.0% |
| unattributed | kernel embed_gather_dense_kernel | 0.009 | 0.0% |
| unattributed | kernel publish_ids_kernel | 0.006 | 0.0% |
| unattributed | kernel speculative_select_accepted_hidden_kernel | 0.006 | 0.0% |
| unattributed | kernel scatter_bf16x8_kernel | 0.006 | 0.0% |
| unattributed | kernel mtp_advance_round_kernel | 0.004 | 0.0% |
| unattributed | kernel rmsnorm_cta_bf16x2_kernel | 0.004 | 0.0% |
| unattributed | kernel expand_text_positions_kernel | 0.002 | 0.0% |
| unattributed | kernel mtp_prepare_next_round_kernel | 0.002 | 0.0% |
| unattributed | kernel speculative_prepare_verify_inputs_kernel | 0.002 | 0.0% |
| unattributed | kernel select_shared_indices_kernel | 0.002 | 0.0% |
| unattributed | kernel shortlist_exact_select_kernel | 0.001 | 0.0% |
| unattributed | kernel residual_add_bf16x8_kernel | 0.001 | 0.0% |
| unattributed | memset memset | 0.000 | 0.0% |

Host lookup work (may overlap GPU execution):

- ple.gather: 158.497 ms across 132 calls.
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
