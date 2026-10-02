# Flash-Next GPU work report

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

Kernels per stage (ms/round, launches per round):

| Phase / role | Stage | Kernel | Launches | GPU work ms/round | Stage share |
|---|---|---|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 96.0 | 40.000 | 93.3% |
| decode / target.verify | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 96.0 | 1.291 | 3.0% |
| decode / target.verify | moe.nvfp4 | bf16_small_t_inner_kernel | 48.0 | 0.799 | 1.9% |
| decode / target.verify | moe.nvfp4 | route_kernel | 48.0 | 0.480 | 1.1% |
| decode / target.verify | moe.nvfp4 | quantize_decode_routes_kernel | 48.0 | 0.200 | 0.5% |
| decode / target.verify | moe.nvfp4 | reduce_decode_routes_kernel | 48.0 | 0.086 | 0.2% |
| decode / target.verify | gdn.record | fp8_a16_sliced_k_mma_kernel | 108.0 | 9.251 | 79.1% |
| decode / target.verify | gdn.record | recurrent_record_kernel | 36.0 | 1.917 | 16.4% |
| decode / target.verify | gdn.record | project_control_gating_kernel | 36.0 | 0.293 | 2.5% |
| decode / target.verify | gdn.record | conv_replay_record_kernel | 36.0 | 0.171 | 1.5% |
| decode / target.verify | gdn.record | rmsnorm_warp_bf16x2_kernel | 36.0 | 0.059 | 0.5% |
| decode / target.verify | qsa.select | fp8_a16_sliced_k_mma_kernel | 24.0 | 2.514 | 41.5% |
| decode / target.verify | qsa.select | selected_attention_split_fp8_kernel | 12.0 | 2.485 | 41.1% |
| decode / target.verify | qsa.select | bf16_small_t_inner_kernel | 48.0 | 0.713 | 11.8% |
| decode / target.verify | qsa.select | append_cache_fp8_kernel | 12.0 | 0.067 | 1.1% |
| decode / target.verify | qsa.select | reduce_selected_attention_splits_kernel | 12.0 | 0.059 | 1.0% |
| decode / target.verify | qsa.select | rmsnorm_warp_bf16x2_kernel | 24.0 | 0.054 | 0.9% |
| decode / target.verify | qsa.select | rope_generic_kernel | 12.0 | 0.050 | 0.8% |
| decode / target.verify | qsa.select | compress_index_groups_kernel | 12.0 | 0.035 | 0.6% |
| decode / target.verify | qsa.select | prepare_index_query_kernel | 12.0 | 0.023 | 0.4% |
| decode / target.verify | qsa.select | split_query_gate_kernel | 12.0 | 0.020 | 0.3% |
| decode / target.verify | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 12.0 | 0.018 | 0.3% |
| decode / target.verify | qsa.select | hadamard_rows_kernel | 12.0 | 0.014 | 0.2% |
| decode / target.verify | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 188.0 | 3.844 | 74.3% |
| decode / target.verify | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 94.0 | 1.328 | 25.7% |
| decode / target.verify | ple.record | bf16_gemv_kernel | 16.0 | 2.242 | 98.8% |
| decode / target.verify | ple.record | gate_kernel | 1.0 | 0.011 | 0.5% |
| decode / target.verify | ple.record | dilated_conv_record_kernel | 1.0 | 0.006 | 0.3% |
| decode / target.verify | ple.record | grouped_norm_kernel | 1.0 | 0.006 | 0.3% |
| decode / target.verify | ple.record | dequantize_embedding_kernel | 1.0 | 0.004 | 0.2% |
| decode / predictor | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 2.0 | 0.827 | 92.9% |
| decode / predictor | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.028 | 3.1% |
| decode / predictor | moe.nvfp4 | bf16_small_t_inner_kernel | 1.0 | 0.020 | 2.2% |
| decode / predictor | moe.nvfp4 | route_kernel | 1.0 | 0.010 | 1.1% |
| decode / predictor | moe.nvfp4 | quantize_decode_routes_kernel | 1.0 | 0.004 | 0.5% |
| decode / predictor | moe.nvfp4 | reduce_decode_routes_kernel | 1.0 | 0.002 | 0.2% |
| decode / predictor | qsa.select | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.202 | 41.8% |
| decode / predictor | qsa.select | selected_attention_split_fp8_kernel | 1.0 | 0.190 | 39.3% |
| decode / predictor | qsa.select | bf16_small_t_inner_kernel | 4.0 | 0.060 | 12.5% |
| decode / predictor | qsa.select | append_cache_fp8_kernel | 1.0 | 0.008 | 1.7% |
| decode / predictor | qsa.select | reduce_selected_attention_splits_kernel | 1.0 | 0.005 | 1.0% |
| decode / predictor | qsa.select | rope_generic_kernel | 1.0 | 0.004 | 0.8% |
| decode / predictor | qsa.select | rmsnorm_warp_bf16x2_kernel | 2.0 | 0.004 | 0.8% |
| decode / predictor | qsa.select | compress_index_groups_kernel | 1.0 | 0.003 | 0.6% |
| decode / predictor | qsa.select | prepare_index_query_kernel | 1.0 | 0.002 | 0.4% |
| decode / predictor | qsa.select | split_query_gate_kernel | 1.0 | 0.002 | 0.3% |
| decode / predictor | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 1.0 | 0.002 | 0.3% |
| decode / predictor | qsa.select | dense_indices_batched_kernel | 1.0 | 0.001 | 0.2% |
| decode / predictor | qsa.select | hadamard_rows_kernel | 1.0 | 0.001 | 0.2% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 4.0 | 0.085 | 82.1% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 2.0 | 0.018 | 17.9% |
| decode / predictor | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.054 | 79.4% |
| decode / predictor | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 1.0 | 0.014 | 20.6% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.037 | 81.8% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.008 | 18.2% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.038 | 86.1% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.006 | 13.9% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.038 | 85.7% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.006 | 14.3% |
| decode / target.verify | hyper.combine | combine_kernel | 2.0 | 0.006 | 100.0% |
| decode / predictor | hyper.combine | combine_kernel | 1.0 | 0.003 | 100.0% |
| decode / predictor | hyper.add | add_repeated_kernel | 1.0 | 0.002 | 100.0% |
| decode / target.verify | hyper.repeat | repeat_kernel | 1.0 | 0.002 | 100.0% |

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
