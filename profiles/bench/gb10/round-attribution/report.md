# Flash-Next GPU work report

Measured wall: 8765.458 ms; GPU busy: 8671.331 ms; attributed GPU work: 93.96%.

Estimates describe partial modeled work, not an achievable optimum. Bytes/FLOPs below are per call; time and share are aggregated.

| Phase / role | Stage | T / B / context envelope | Calls | GPU work ms | Phase share | Useful bytes min–max | BF16 / NVFP4 / FP32 FLOPs | Estimate ms min–max | Estimate efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | 8 / 0 / 0 | 6432 | 4761.105 | 54.9% | 35279520–228816640 | 99655680 / 786432000 / 0 | 922.430–5982.718 | 19.4–125.7% |
| decode / target.verify | gdn.record | 8 / 4 / 0 | 4824 | 1816.849 | 21.0% | 71163904–71163904 | 926679040 / 0 / 37748736 | 1395.507–1395.507 | 76.8–76.8% |
| decode / target.verify | hyper.combine_mix | 8 / 0 / 0 | 12596 | 731.914 | 8.4% | 7066368–7066368 | 105512960 / 0 / 0 | 361.821–361.821 | 49.4–49.4% |
| decode / target.verify | qsa.select | 8 / 4 / 129 | 1608 | 600.571 | 6.9% | 55817216–55817216 | 823132160 / 0 / 0 | 364.854–364.854 | 60.8–60.8% |
| decode / predictor | moe.nvfp4 | 8 / 0 / 0 | 134 | 99.793 | 1.2% | 35279520–228816640 | 99655680 / 786432000 / 0 | 19.217–124.640 | 19.3–124.9% |
| decode / predictor | qsa.select | 8 / 4 / 129 | 134 | 50.794 | 0.6% | 55817216–55817216 | 823132160 / 0 / 0 | 30.404–30.404 | 59.9–59.9% |
| decode / target.verify | ple.record | 8 / 4 / 0 | 134 | 42.982 | 0.5% | 65884160–65884160 | 524288000 / 0 / 0 | 35.888–35.888 | 83.5–83.5% |
| decode / target.verify | hyper.mix | 8 / 0 / 0 | 268 | 14.209 | 0.2% | 6861504–6861504 | 105512960 / 0 / 0 | 7.475–7.475 | 52.6–52.6% |
| decode / predictor | hyper.combine_mix | 8 / 0 / 0 | 134 | 9.492 | 0.1% | 7066368–7066368 | 105512960 / 0 / 0 | 3.849–3.849 | 40.5–40.5% |
| decode / predictor | hyper.mix | 8 / 0 / 0 | 134 | 6.268 | 0.1% | 6861504–6861504 | 105512960 / 0 / 0 | 3.738–3.738 | 59.6–59.6% |
| decode / predictor | hyper.mix | 8 / 0 / 0 | 134 | 5.893 | 0.1% | 6779520–6779520 | 104857600 / 0 / 0 | 3.693–3.693 | 62.7–62.7% |
| decode / target.verify | hyper.mix | 8 / 0 / 0 | 134 | 5.736 | 0.1% | 6779520–6779520 | 104857600 / 0 / 0 | 3.693–3.693 | 64.4–64.4% |
| decode / target.verify | hyper.combine | 8 / 0 / 0 | 268 | 0.770 | 0.0% | 368704–368704 | 0 / 0 / 0 | 0.402–0.402 | 52.2–52.2% |
| decode / predictor | hyper.combine | 8 / 0 / 0 | 134 | 0.386 | 0.0% | 368704–368704 | 0 / 0 / 0 | 0.201–0.201 | 52.1–52.1% |
| decode / predictor | hyper.add | 8 / 0 / 0 | 134 | 0.286 | 0.0% | 368640–368640 | 0 / 0 / 0 | 0.201–0.201 | 70.3–70.3% |
| decode / target.verify | hyper.repeat | 8 / 0 / 0 | 134 | 0.284 | 0.0% | 204800–204800 | 0 / 0 / 0 | 0.112–0.112 | 39.3–39.3% |

Unattributed GPU work: 524.001 ms.

Kernels per stage (ms/round, launches per round):

| Phase / role | Stage | Kernel | Launches | GPU work ms/round | Stage share |
|---|---|---|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 96.0 | 32.402 | 91.2% |
| decode / target.verify | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 96.0 | 1.252 | 3.5% |
| decode / target.verify | moe.nvfp4 | bf16_small_t_inner_kernel | 48.0 | 0.783 | 2.2% |
| decode / target.verify | moe.nvfp4 | route_kernel | 48.0 | 0.595 | 1.7% |
| decode / target.verify | moe.nvfp4 | quantize_grouped_kernel | 48.0 | 0.202 | 0.6% |
| decode / target.verify | moe.nvfp4 | gather_quantize_routes_kernel | 48.0 | 0.199 | 0.6% |
| decode / target.verify | moe.nvfp4 | reduce_grouped_kernel | 48.0 | 0.097 | 0.3% |
| decode / target.verify | gdn.record | fp8_a16_sliced_k_mma_kernel | 108.0 | 10.087 | 74.4% |
| decode / target.verify | gdn.record | recurrent_fold_record_kernel | 36.0 | 2.861 | 21.1% |
| decode / target.verify | gdn.record | project_control_gating_kernel | 36.0 | 0.292 | 2.2% |
| decode / target.verify | gdn.record | conv_replay_record_kernel | 36.0 | 0.250 | 1.8% |
| decode / target.verify | gdn.record | rmsnorm_warp_bf16x2_kernel | 36.0 | 0.069 | 0.5% |
| decode / target.verify | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 188.0 | 4.126 | 75.5% |
| decode / target.verify | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 94.0 | 1.336 | 24.5% |
| decode / target.verify | qsa.select | fp8_a16_sliced_k_mma_kernel | 24.0 | 2.433 | 54.3% |
| decode / target.verify | qsa.select | selected_attention_split_fp8_kernel | 12.0 | 1.013 | 22.6% |
| decode / target.verify | qsa.select | bf16_small_t_inner_kernel | 48.0 | 0.701 | 15.6% |
| decode / target.verify | qsa.select | append_cache_fp8_kernel | 12.0 | 0.065 | 1.4% |
| decode / target.verify | qsa.select | reduce_selected_attention_splits_kernel | 12.0 | 0.059 | 1.3% |
| decode / target.verify | qsa.select | rope_generic_kernel | 12.0 | 0.050 | 1.1% |
| decode / target.verify | qsa.select | rmsnorm_warp_bf16x2_kernel | 24.0 | 0.050 | 1.1% |
| decode / target.verify | qsa.select | compress_index_groups_kernel | 12.0 | 0.037 | 0.8% |
| decode / target.verify | qsa.select | prepare_index_query_kernel | 12.0 | 0.023 | 0.5% |
| decode / target.verify | qsa.select | split_query_gate_kernel | 12.0 | 0.020 | 0.4% |
| decode / target.verify | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 12.0 | 0.019 | 0.4% |
| decode / target.verify | qsa.select | hadamard_rows_kernel | 12.0 | 0.014 | 0.3% |
| decode / predictor | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 2.0 | 0.676 | 90.7% |
| decode / predictor | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.028 | 3.7% |
| decode / predictor | moe.nvfp4 | bf16_small_t_inner_kernel | 1.0 | 0.019 | 2.5% |
| decode / predictor | moe.nvfp4 | route_kernel | 1.0 | 0.012 | 1.6% |
| decode / predictor | moe.nvfp4 | quantize_grouped_kernel | 1.0 | 0.004 | 0.6% |
| decode / predictor | moe.nvfp4 | gather_quantize_routes_kernel | 1.0 | 0.004 | 0.6% |
| decode / predictor | moe.nvfp4 | reduce_grouped_kernel | 1.0 | 0.002 | 0.3% |
| decode / predictor | qsa.select | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.204 | 53.8% |
| decode / predictor | qsa.select | selected_attention_split_fp8_kernel | 1.0 | 0.083 | 21.9% |
| decode / predictor | qsa.select | bf16_small_t_inner_kernel | 4.0 | 0.062 | 16.3% |
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
| decode / target.verify | ple.record | bf16_gemv_columns_kernel | 2.0 | 0.297 | 92.5% |
| decode / target.verify | ple.record | gate_kernel | 1.0 | 0.008 | 2.6% |
| decode / target.verify | ple.record | dilated_conv_record_kernel | 1.0 | 0.006 | 1.9% |
| decode / target.verify | ple.record | grouped_norm_kernel | 1.0 | 0.006 | 1.8% |
| decode / target.verify | ple.record | dequantize_embedding_kernel | 1.0 | 0.004 | 1.2% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 4.0 | 0.087 | 82.4% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 2.0 | 0.019 | 17.6% |
| decode / predictor | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.057 | 80.1% |
| decode / predictor | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 1.0 | 0.014 | 19.9% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.038 | 82.2% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.008 | 17.8% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.038 | 85.8% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.006 | 14.2% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.037 | 85.4% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.006 | 14.6% |
| decode / target.verify | hyper.combine | combine_kernel | 2.0 | 0.006 | 100.0% |
| decode / predictor | hyper.combine | combine_kernel | 1.0 | 0.003 | 100.0% |
| decode / predictor | hyper.add | add_repeated_kernel | 1.0 | 0.002 | 100.0% |
| decode / target.verify | hyper.repeat | repeat_kernel | 1.0 | 0.002 | 100.0% |

Per round over 134 rounds: host wall 65.414 ms, GPU work 64.711 ms, GPU busy 64.711 ms.

| Phase / role | Stage | GPU work ms/round | Share |
|---|---|---:|---:|
| decode / target.verify | moe.nvfp4 | 35.531 | 54.9% |
| decode / target.verify | gdn.record | 13.559 | 21.0% |
| decode / target.verify | hyper.combine_mix | 5.462 | 8.4% |
| decode / target.verify | qsa.select | 4.482 | 6.9% |
| decode / predictor | moe.nvfp4 | 0.745 | 1.2% |
| decode / predictor | qsa.select | 0.379 | 0.6% |
| decode / target.verify | ple.record | 0.321 | 0.5% |
| decode / target.verify | hyper.mix | 0.106 | 0.2% |
| decode / predictor | hyper.combine_mix | 0.071 | 0.1% |
| decode / predictor | hyper.mix | 0.047 | 0.1% |
| decode / predictor | hyper.mix | 0.044 | 0.1% |
| decode / target.verify | hyper.mix | 0.043 | 0.1% |
| decode / target.verify | hyper.combine | 0.006 | 0.0% |
| decode / predictor | hyper.combine | 0.003 | 0.0% |
| decode / predictor | hyper.add | 0.002 | 0.0% |
| decode / target.verify | hyper.repeat | 0.002 | 0.0% |
| unattributed | kernel fp8_a16_sliced_k_mma_kernel | 2.562 | 4.0% |
| unattributed | kernel q4_a16_sliced_k_mma_kernel | 0.887 | 1.4% |
| unattributed | kernel bf16_gemm_mma_kernel | 0.071 | 0.1% |
| unattributed | kernel bf16_small_t_inner_kernel | 0.058 | 0.1% |
| unattributed | kernel copy_record_rows_kernel | 0.058 | 0.1% |
| unattributed | kernel argmax_tiled_atomic_kernel | 0.054 | 0.1% |
| unattributed | kernel shortlist_exact_scores_fp8_kernel | 0.046 | 0.1% |
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
| unattributed | kernel mtp_advance_round_kernel | 0.004 | 0.0% |
| unattributed | kernel rmsnorm_cta_bf16x2_kernel | 0.003 | 0.0% |
| unattributed | kernel expand_text_positions_kernel | 0.002 | 0.0% |
| unattributed | kernel wait_staged_kernel | 0.002 | 0.0% |
| unattributed | kernel mtp_prepare_next_round_kernel | 0.002 | 0.0% |
| unattributed | kernel speculative_prepare_verify_inputs_kernel | 0.002 | 0.0% |
| unattributed | kernel select_shared_indices_kernel | 0.002 | 0.0% |
| unattributed | kernel shortlist_exact_select_kernel | 0.001 | 0.0% |
| unattributed | kernel residual_add_bf16x8_kernel | 0.001 | 0.0% |
| unattributed | memset memset | 0.000 | 0.0% |

Host lookup work (may overlap GPU execution):

- ple.gather: 72.436 ms across 134 calls.
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
