# Flash-Next GPU work report

Measured wall: 12107.851 ms; GPU busy: 12021.079 ms; attributed GPU work: 77.06%.

Estimates describe partial modeled work, not an achievable optimum. Bytes/FLOPs below are per call; time and share are aggregated.

| Phase / role | Stage | T / B / context envelope | Calls | GPU work ms | Phase share | Useful bytes min–max | BF16 / NVFP4 / FP32 FLOPs | Estimate ms min–max | Estimate efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | 8 / 0 / 0 | 6288 | 5655.844 | 47.0% | 35279520–228816640 | 99655680 / 786432000 / 0 | 901.779–5848.777 | 15.9–103.4% |
| decode / target.verify | gdn.record | 8 / 4 / 0 | 4716 | 1542.370 | 12.8% | 71163904–71163904 | 926679040 / 0 / 37748736 | 1364.264–1364.264 | 88.5–88.5% |
| decode / target.verify | qsa.select | 8 / 4 / 129 | 1572 | 849.559 | 7.1% | 55817216–55817216 | 823132160 / 0 / 0 | 356.686–356.686 | 42.0–42.0% |
| decode / target.verify | hyper.combine_mix | 8 / 0 / 0 | 12314 | 688.491 | 5.7% | 7066368–7066368 | 105512960 / 0 / 0 | 353.721–353.721 | 51.4–51.4% |
| decode / target.verify | ple.record | 8 / 4 / 0 | 131 | 293.345 | 2.4% | 65884160–65884160 | 524288000 / 0 / 0 | 35.085–35.085 | 12.0–12.0% |
| decode / predictor | moe.nvfp4 | 8 / 0 / 0 | 131 | 122.391 | 1.0% | 35279520–228816640 | 99655680 / 786432000 / 0 | 18.787–121.850 | 15.4–99.6% |
| decode / predictor | qsa.select | 8 / 4 / 129 | 131 | 68.990 | 0.6% | 55817216–55817216 | 823132160 / 0 / 0 | 29.724–29.724 | 43.1–43.1% |
| decode / target.verify | hyper.mix | 8 / 0 / 0 | 262 | 14.152 | 0.1% | 6861504–6861504 | 105512960 / 0 / 0 | 7.308–7.308 | 51.6–51.6% |
| decode / predictor | hyper.combine_mix | 8 / 0 / 0 | 131 | 9.172 | 0.1% | 7066368–7066368 | 105512960 / 0 / 0 | 3.763–3.763 | 41.0–41.0% |
| decode / predictor | hyper.mix | 8 / 0 / 0 | 131 | 6.203 | 0.1% | 6861504–6861504 | 105512960 / 0 / 0 | 3.654–3.654 | 58.9–58.9% |
| decode / predictor | hyper.mix | 8 / 0 / 0 | 131 | 5.745 | 0.0% | 6779520–6779520 | 104857600 / 0 / 0 | 3.610–3.610 | 62.8–62.8% |
| decode / target.verify | hyper.mix | 8 / 0 / 0 | 131 | 5.670 | 0.0% | 6779520–6779520 | 104857600 / 0 / 0 | 3.610–3.610 | 63.7–63.7% |
| decode / target.verify | hyper.combine | 8 / 0 / 0 | 262 | 0.761 | 0.0% | 368704–368704 | 0 / 0 / 0 | 0.393–0.393 | 51.6–51.6% |
| decode / predictor | hyper.combine | 8 / 0 / 0 | 131 | 0.379 | 0.0% | 368704–368704 | 0 / 0 / 0 | 0.196–0.196 | 51.9–51.9% |
| decode / target.verify | hyper.repeat | 8 / 0 / 0 | 131 | 0.302 | 0.0% | 204800–204800 | 0 / 0 / 0 | 0.109–0.109 | 36.1–36.1% |
| decode / predictor | hyper.add | 8 / 0 / 0 | 131 | 0.282 | 0.0% | 368640–368640 | 0 / 0 / 0 | 0.196–0.196 | 69.7–69.7% |

Unattributed GPU work: 2757.539 ms.

Kernels per stage (ms/round, launches per round):

| Phase / role | Stage | Kernel | Launches | GPU work ms/round | Stage share |
|---|---|---|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 96.0 | 40.266 | 93.3% |
| decode / target.verify | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 96.0 | 1.303 | 3.0% |
| decode / target.verify | moe.nvfp4 | bf16_small_t_inner_kernel | 48.0 | 0.826 | 1.9% |
| decode / target.verify | moe.nvfp4 | route_kernel | 48.0 | 0.483 | 1.1% |
| decode / target.verify | moe.nvfp4 | quantize_decode_routes_kernel | 48.0 | 0.204 | 0.5% |
| decode / target.verify | moe.nvfp4 | reduce_decode_routes_kernel | 48.0 | 0.092 | 0.2% |
| decode / target.verify | gdn.record | fp8_a16_sliced_k_mma_kernel | 108.0 | 9.301 | 79.0% |
| decode / target.verify | gdn.record | recurrent_record_kernel | 36.0 | 1.928 | 16.4% |
| decode / target.verify | gdn.record | project_control_gating_kernel | 36.0 | 0.304 | 2.6% |
| decode / target.verify | gdn.record | conv_replay_record_kernel | 36.0 | 0.178 | 1.5% |
| decode / target.verify | gdn.record | rmsnorm_warp_bf16x2_kernel | 36.0 | 0.063 | 0.5% |
| decode / target.verify | qsa.select | selected_attention_split_fp8_kernel | 12.0 | 2.931 | 45.2% |
| decode / target.verify | qsa.select | fp8_a16_sliced_k_mma_kernel | 24.0 | 2.488 | 38.4% |
| decode / target.verify | qsa.select | bf16_small_t_inner_kernel | 48.0 | 0.720 | 11.1% |
| decode / target.verify | qsa.select | append_cache_fp8_kernel | 12.0 | 0.068 | 1.0% |
| decode / target.verify | qsa.select | reduce_selected_attention_splits_kernel | 12.0 | 0.060 | 0.9% |
| decode / target.verify | qsa.select | rope_generic_kernel | 12.0 | 0.052 | 0.8% |
| decode / target.verify | qsa.select | rmsnorm_warp_bf16x2_kernel | 24.0 | 0.051 | 0.8% |
| decode / target.verify | qsa.select | compress_index_groups_kernel | 12.0 | 0.038 | 0.6% |
| decode / target.verify | qsa.select | prepare_index_query_kernel | 12.0 | 0.024 | 0.4% |
| decode / target.verify | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 12.0 | 0.020 | 0.3% |
| decode / target.verify | qsa.select | split_query_gate_kernel | 12.0 | 0.020 | 0.3% |
| decode / target.verify | qsa.select | hadamard_rows_kernel | 12.0 | 0.014 | 0.2% |
| decode / target.verify | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 188.0 | 3.882 | 73.9% |
| decode / target.verify | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 94.0 | 1.373 | 26.1% |
| decode / target.verify | ple.record | bf16_gemv_kernel | 16.0 | 2.212 | 98.8% |
| decode / target.verify | ple.record | gate_kernel | 1.0 | 0.011 | 0.5% |
| decode / target.verify | ple.record | dilated_conv_record_kernel | 1.0 | 0.006 | 0.3% |
| decode / target.verify | ple.record | grouped_norm_kernel | 1.0 | 0.006 | 0.3% |
| decode / target.verify | ple.record | dequantize_embedding_kernel | 1.0 | 0.004 | 0.2% |
| decode / predictor | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 2.0 | 0.871 | 93.2% |
| decode / predictor | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.028 | 2.9% |
| decode / predictor | moe.nvfp4 | bf16_small_t_inner_kernel | 1.0 | 0.019 | 2.1% |
| decode / predictor | moe.nvfp4 | route_kernel | 1.0 | 0.010 | 1.1% |
| decode / predictor | moe.nvfp4 | quantize_decode_routes_kernel | 1.0 | 0.005 | 0.5% |
| decode / predictor | moe.nvfp4 | reduce_decode_routes_kernel | 1.0 | 0.002 | 0.2% |
| decode / predictor | qsa.select | selected_attention_split_fp8_kernel | 1.0 | 0.224 | 42.6% |
| decode / predictor | qsa.select | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.209 | 39.7% |
| decode / predictor | qsa.select | bf16_small_t_inner_kernel | 4.0 | 0.063 | 11.9% |
| decode / predictor | qsa.select | append_cache_fp8_kernel | 1.0 | 0.007 | 1.4% |
| decode / predictor | qsa.select | reduce_selected_attention_splits_kernel | 1.0 | 0.005 | 0.9% |
| decode / predictor | qsa.select | rope_generic_kernel | 1.0 | 0.004 | 0.8% |
| decode / predictor | qsa.select | rmsnorm_warp_bf16x2_kernel | 2.0 | 0.004 | 0.7% |
| decode / predictor | qsa.select | compress_index_groups_kernel | 1.0 | 0.003 | 0.6% |
| decode / predictor | qsa.select | prepare_index_query_kernel | 1.0 | 0.002 | 0.4% |
| decode / predictor | qsa.select | split_query_gate_kernel | 1.0 | 0.002 | 0.3% |
| decode / predictor | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 1.0 | 0.002 | 0.3% |
| decode / predictor | qsa.select | dense_indices_batched_kernel | 1.0 | 0.001 | 0.2% |
| decode / predictor | qsa.select | hadamard_rows_kernel | 1.0 | 0.001 | 0.2% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 4.0 | 0.088 | 81.4% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 2.0 | 0.020 | 18.6% |
| decode / predictor | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.056 | 79.9% |
| decode / predictor | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 1.0 | 0.014 | 20.1% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.039 | 81.6% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.009 | 18.4% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.038 | 85.7% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.006 | 14.3% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.037 | 85.4% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.006 | 14.6% |
| decode / target.verify | hyper.combine | combine_kernel | 2.0 | 0.006 | 100.0% |
| decode / predictor | hyper.combine | combine_kernel | 1.0 | 0.003 | 100.0% |
| decode / target.verify | hyper.repeat | repeat_kernel | 1.0 | 0.002 | 100.0% |
| decode / predictor | hyper.add | add_repeated_kernel | 1.0 | 0.002 | 100.0% |

Per round over 131 rounds: host wall 92.426 ms, GPU work 91.765 ms, GPU busy 91.764 ms.

| Phase / role | Stage | GPU work ms/round | Share |
|---|---|---:|---:|
| decode / target.verify | moe.nvfp4 | 43.174 | 47.0% |
| decode / target.verify | gdn.record | 11.774 | 12.8% |
| decode / target.verify | qsa.select | 6.485 | 7.1% |
| decode / target.verify | hyper.combine_mix | 5.256 | 5.7% |
| decode / target.verify | ple.record | 2.239 | 2.4% |
| decode / predictor | moe.nvfp4 | 0.934 | 1.0% |
| decode / predictor | qsa.select | 0.527 | 0.6% |
| decode / target.verify | hyper.mix | 0.108 | 0.1% |
| decode / predictor | hyper.combine_mix | 0.070 | 0.1% |
| decode / predictor | hyper.mix | 0.047 | 0.1% |
| decode / predictor | hyper.mix | 0.044 | 0.0% |
| decode / target.verify | hyper.mix | 0.043 | 0.0% |
| decode / target.verify | hyper.combine | 0.006 | 0.0% |
| decode / predictor | hyper.combine | 0.003 | 0.0% |
| decode / target.verify | hyper.repeat | 0.002 | 0.0% |
| decode / predictor | hyper.add | 0.002 | 0.0% |
| unattributed | kernel wait_staged_kernel | 13.094 | 14.3% |
| unattributed | kernel recurrent_fold_kernel | 3.994 | 4.4% |
| unattributed | kernel fp8_a16_sliced_k_mma_kernel | 2.644 | 2.9% |
| unattributed | kernel q4_a16_sliced_k_mma_kernel | 0.902 | 1.0% |
| unattributed | kernel bf16_gemm_mma_kernel | 0.073 | 0.1% |
| unattributed | kernel bf16_small_t_inner_kernel | 0.060 | 0.1% |
| unattributed | kernel argmax_tiled_atomic_kernel | 0.054 | 0.1% |
| unattributed | kernel shortlist_exact_scores_fp8_kernel | 0.048 | 0.1% |
| unattributed | kernel speculative_sampling_partial_topk_kernel | 0.033 | 0.0% |
| unattributed | kernel ple_fold_kernel | 0.030 | 0.0% |
| unattributed | kernel shortlist_tile_candidates_kernel | 0.028 | 0.0% |
| unattributed | kernel rmsnorm_generic_kernel | 0.019 | 0.0% |
| unattributed | memcpy memcpy | 0.014 | 0.0% |
| unattributed | kernel speculative_sampling_group_finalize_kernel | 0.010 | 0.0% |
| unattributed | kernel embed_gather_dense_kernel | 0.008 | 0.0% |
| unattributed | kernel publish_ids_kernel | 0.008 | 0.0% |
| unattributed | kernel scatter_bf16x8_kernel | 0.006 | 0.0% |
| unattributed | kernel speculative_select_accepted_hidden_kernel | 0.006 | 0.0% |
| unattributed | kernel mtp_advance_round_kernel | 0.004 | 0.0% |
| unattributed | kernel rmsnorm_cta_bf16x2_kernel | 0.004 | 0.0% |
| unattributed | kernel expand_text_positions_kernel | 0.003 | 0.0% |
| unattributed | kernel mtp_prepare_next_round_kernel | 0.002 | 0.0% |
| unattributed | kernel speculative_prepare_verify_inputs_kernel | 0.002 | 0.0% |
| unattributed | kernel select_shared_indices_kernel | 0.002 | 0.0% |
| unattributed | kernel shortlist_exact_select_kernel | 0.001 | 0.0% |
| unattributed | kernel residual_add_bf16x8_kernel | 0.001 | 0.0% |
| unattributed | memset memset | 0.000 | 0.0% |

Host lookup work (may overlap GPU execution):

- ple.gather: 1906.842 ms across 131 calls.
- ple.hash: 0.000 ms across 0 calls.

- Estimates are partial cold useful-work models, not distance from optimal.
- Token counts are execution envelopes; padded columns are not committed outputs.
- QSA estimates cover projections only; device-dependent scoring/attention is unmodeled.
- GDN prefill estimates omit chunked recurrence compute; PLE estimates omit convolution/state traffic.
- DRAM cache reuse, scratch traffic, nonlinear math and instruction geometry are unmodeled.
- Profiler overhead affects timings; use unprofiled paired benchmarks for speed claims.
- Unattributed GPU work is retained; no whole-model efficiency is reported.
- Serve window: GPU work launched inside 131 decode.mtp_round ranges with batch 4 (trim 0.00 of rounds at each end); work from other host threads launched during a round is included.
- An estimate exceeds measured time: inspect cache reuse, envelope work and hardware rates; percentages are not clamped.
