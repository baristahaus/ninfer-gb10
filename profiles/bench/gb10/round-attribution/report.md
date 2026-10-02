# Flash-Next GPU work report

Measured wall: 9132.181 ms; GPU busy: 9039.789 ms; attributed GPU work: 94.17%.

Estimates describe partial modeled work, not an achievable optimum. Bytes/FLOPs below are per call; time and share are aggregated.

| Phase / role | Stage | T / B / context envelope | Calls | GPU work ms | Phase share | Useful bytes min–max | BF16 / NVFP4 / FP32 FLOPs | Estimate ms min–max | Estimate efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | 8 / 0 / 0 | 6384 | 4878.310 | 54.0% | 35279520–228816640 | 99655680 / 786432000 / 0 | 915.547–5938.071 | 18.8–121.7% |
| decode / target.verify | gdn.record | 8 / 4 / 0 | 4788 | 1802.845 | 19.9% | 71163904–71163904 | 926679040 / 0 / 37748736 | 1385.093–1385.093 | 76.8–76.8% |
| decode / target.verify | qsa.select | 8 / 4 / 129 | 1596 | 852.683 | 9.4% | 55817216–55817216 | 823132160 / 0 / 0 | 362.131–362.131 | 42.5–42.5% |
| decode / target.verify | hyper.combine_mix | 8 / 0 / 0 | 12502 | 721.586 | 8.0% | 7066368–7066368 | 105512960 / 0 / 0 | 359.121–359.121 | 49.8–49.8% |
| decode / predictor | moe.nvfp4 | 8 / 0 / 0 | 133 | 101.415 | 1.1% | 35279520–228816640 | 99655680 / 786432000 / 0 | 19.074–123.710 | 18.8–122.0% |
| decode / predictor | qsa.select | 8 / 4 / 129 | 133 | 70.210 | 0.8% | 55817216–55817216 | 823132160 / 0 / 0 | 30.178–30.178 | 43.0–43.0% |
| decode / target.verify | ple.record | 8 / 4 / 0 | 133 | 43.641 | 0.5% | 65884160–65884160 | 524288000 / 0 / 0 | 35.620–35.620 | 81.6–81.6% |
| decode / target.verify | hyper.mix | 8 / 0 / 0 | 266 | 14.413 | 0.2% | 6861504–6861504 | 105512960 / 0 / 0 | 7.419–7.419 | 51.5–51.5% |
| decode / predictor | hyper.combine_mix | 8 / 0 / 0 | 133 | 8.566 | 0.1% | 7066368–7066368 | 105512960 / 0 / 0 | 3.820–3.820 | 44.6–44.6% |
| decode / predictor | hyper.mix | 8 / 0 / 0 | 133 | 6.057 | 0.1% | 6861504–6861504 | 105512960 / 0 / 0 | 3.710–3.710 | 61.2–61.2% |
| decode / target.verify | hyper.mix | 8 / 0 / 0 | 133 | 5.778 | 0.1% | 6779520–6779520 | 104857600 / 0 / 0 | 3.665–3.665 | 63.4–63.4% |
| decode / predictor | hyper.mix | 8 / 0 / 0 | 133 | 5.674 | 0.1% | 6779520–6779520 | 104857600 / 0 / 0 | 3.665–3.665 | 64.6–64.6% |
| decode / target.verify | hyper.combine | 8 / 0 / 0 | 266 | 0.760 | 0.0% | 368704–368704 | 0 / 0 / 0 | 0.399–0.399 | 52.4–52.4% |
| decode / predictor | hyper.combine | 8 / 0 / 0 | 133 | 0.382 | 0.0% | 368704–368704 | 0 / 0 / 0 | 0.199–0.199 | 52.2–52.2% |
| decode / predictor | hyper.add | 8 / 0 / 0 | 133 | 0.283 | 0.0% | 368640–368640 | 0 / 0 / 0 | 0.199–0.199 | 70.4–70.4% |
| decode / target.verify | hyper.repeat | 8 / 0 / 0 | 133 | 0.278 | 0.0% | 204800–204800 | 0 / 0 / 0 | 0.111–0.111 | 39.9–39.9% |

Unattributed GPU work: 526.931 ms.

Kernels per stage (ms/round, launches per round):

| Phase / role | Stage | Kernel | Launches | GPU work ms/round | Stage share |
|---|---|---|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 96.0 | 33.505 | 91.3% |
| decode / target.verify | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 96.0 | 1.277 | 3.5% |
| decode / target.verify | moe.nvfp4 | bf16_small_t_inner_kernel | 48.0 | 0.796 | 2.2% |
| decode / target.verify | moe.nvfp4 | route_kernel | 48.0 | 0.597 | 1.6% |
| decode / target.verify | moe.nvfp4 | quantize_grouped_kernel | 48.0 | 0.204 | 0.6% |
| decode / target.verify | moe.nvfp4 | gather_quantize_routes_kernel | 48.0 | 0.201 | 0.5% |
| decode / target.verify | moe.nvfp4 | reduce_grouped_kernel | 48.0 | 0.098 | 0.3% |
| decode / target.verify | gdn.record | fp8_a16_sliced_k_mma_kernel | 108.0 | 10.157 | 74.9% |
| decode / target.verify | gdn.record | recurrent_fold_record_kernel | 36.0 | 2.823 | 20.8% |
| decode / target.verify | gdn.record | project_control_gating_kernel | 36.0 | 0.295 | 2.2% |
| decode / target.verify | gdn.record | conv_replay_record_kernel | 36.0 | 0.211 | 1.6% |
| decode / target.verify | gdn.record | rmsnorm_warp_bf16x2_kernel | 36.0 | 0.069 | 0.5% |
| decode / target.verify | qsa.select | selected_attention_split_fp8_kernel | 12.0 | 2.888 | 45.0% |
| decode / target.verify | qsa.select | fp8_a16_sliced_k_mma_kernel | 24.0 | 2.477 | 38.6% |
| decode / target.verify | qsa.select | bf16_small_t_inner_kernel | 48.0 | 0.709 | 11.1% |
| decode / target.verify | qsa.select | append_cache_fp8_kernel | 12.0 | 0.066 | 1.0% |
| decode / target.verify | qsa.select | reduce_selected_attention_splits_kernel | 12.0 | 0.059 | 0.9% |
| decode / target.verify | qsa.select | rmsnorm_warp_bf16x2_kernel | 24.0 | 0.050 | 0.8% |
| decode / target.verify | qsa.select | rope_generic_kernel | 12.0 | 0.050 | 0.8% |
| decode / target.verify | qsa.select | compress_index_groups_kernel | 12.0 | 0.037 | 0.6% |
| decode / target.verify | qsa.select | prepare_index_query_kernel | 12.0 | 0.023 | 0.4% |
| decode / target.verify | qsa.select | split_query_gate_kernel | 12.0 | 0.020 | 0.3% |
| decode / target.verify | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 12.0 | 0.018 | 0.3% |
| decode / target.verify | qsa.select | hadamard_rows_kernel | 12.0 | 0.014 | 0.2% |
| decode / target.verify | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 188.0 | 4.096 | 75.5% |
| decode / target.verify | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 94.0 | 1.329 | 24.5% |
| decode / predictor | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 2.0 | 0.693 | 90.9% |
| decode / predictor | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.029 | 3.8% |
| decode / predictor | moe.nvfp4 | bf16_small_t_inner_kernel | 1.0 | 0.017 | 2.3% |
| decode / predictor | moe.nvfp4 | route_kernel | 1.0 | 0.012 | 1.6% |
| decode / predictor | moe.nvfp4 | quantize_grouped_kernel | 1.0 | 0.004 | 0.6% |
| decode / predictor | moe.nvfp4 | gather_quantize_routes_kernel | 1.0 | 0.004 | 0.6% |
| decode / predictor | moe.nvfp4 | reduce_grouped_kernel | 1.0 | 0.002 | 0.3% |
| decode / predictor | qsa.select | selected_attention_split_fp8_kernel | 1.0 | 0.231 | 43.8% |
| decode / predictor | qsa.select | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.205 | 38.8% |
| decode / predictor | qsa.select | bf16_small_t_inner_kernel | 4.0 | 0.061 | 11.6% |
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
| decode / target.verify | ple.record | bf16_gemv_columns_kernel | 2.0 | 0.304 | 92.7% |
| decode / target.verify | ple.record | gate_kernel | 1.0 | 0.008 | 2.5% |
| decode / target.verify | ple.record | dilated_conv_record_kernel | 1.0 | 0.006 | 1.9% |
| decode / target.verify | ple.record | grouped_norm_kernel | 1.0 | 0.006 | 1.8% |
| decode / target.verify | ple.record | dequantize_embedding_kernel | 1.0 | 0.004 | 1.2% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 4.0 | 0.089 | 82.0% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 2.0 | 0.020 | 18.0% |
| decode / predictor | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.050 | 78.0% |
| decode / predictor | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 1.0 | 0.014 | 22.0% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.037 | 81.8% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.008 | 18.2% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.037 | 85.7% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.006 | 14.3% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.036 | 85.3% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.006 | 14.7% |
| decode / target.verify | hyper.combine | combine_kernel | 2.0 | 0.006 | 100.0% |
| decode / predictor | hyper.combine | combine_kernel | 1.0 | 0.003 | 100.0% |
| decode / predictor | hyper.add | add_repeated_kernel | 1.0 | 0.002 | 100.0% |
| decode / target.verify | hyper.repeat | repeat_kernel | 1.0 | 0.002 | 100.0% |

Per round over 133 rounds: host wall 68.663 ms, GPU work 67.969 ms, GPU busy 67.968 ms.

| Phase / role | Stage | GPU work ms/round | Share |
|---|---|---:|---:|
| decode / target.verify | moe.nvfp4 | 36.679 | 54.0% |
| decode / target.verify | gdn.record | 13.555 | 19.9% |
| decode / target.verify | qsa.select | 6.411 | 9.4% |
| decode / target.verify | hyper.combine_mix | 5.425 | 8.0% |
| decode / predictor | moe.nvfp4 | 0.763 | 1.1% |
| decode / predictor | qsa.select | 0.528 | 0.8% |
| decode / target.verify | ple.record | 0.328 | 0.5% |
| decode / target.verify | hyper.mix | 0.108 | 0.2% |
| decode / predictor | hyper.combine_mix | 0.064 | 0.1% |
| decode / predictor | hyper.mix | 0.046 | 0.1% |
| decode / target.verify | hyper.mix | 0.043 | 0.1% |
| decode / predictor | hyper.mix | 0.043 | 0.1% |
| decode / target.verify | hyper.combine | 0.006 | 0.0% |
| decode / predictor | hyper.combine | 0.003 | 0.0% |
| decode / predictor | hyper.add | 0.002 | 0.0% |
| decode / target.verify | hyper.repeat | 0.002 | 0.0% |
| unattributed | kernel fp8_a16_sliced_k_mma_kernel | 2.593 | 3.8% |
| unattributed | kernel q4_a16_sliced_k_mma_kernel | 0.904 | 1.3% |
| unattributed | kernel bf16_gemm_mma_kernel | 0.069 | 0.1% |
| unattributed | kernel bf16_small_t_inner_kernel | 0.061 | 0.1% |
| unattributed | kernel copy_record_rows_kernel | 0.056 | 0.1% |
| unattributed | kernel argmax_tiled_atomic_kernel | 0.054 | 0.1% |
| unattributed | kernel shortlist_exact_scores_fp8_kernel | 0.048 | 0.1% |
| unattributed | kernel speculative_sampling_partial_topk_kernel | 0.034 | 0.0% |
| unattributed | kernel shortlist_tile_candidates_kernel | 0.028 | 0.0% |
| unattributed | kernel ple_fold_kernel | 0.027 | 0.0% |
| unattributed | kernel rmsnorm_generic_kernel | 0.020 | 0.0% |
| unattributed | memcpy memcpy | 0.010 | 0.0% |
| unattributed | kernel speculative_sampling_group_finalize_kernel | 0.009 | 0.0% |
| unattributed | kernel embed_gather_dense_kernel | 0.008 | 0.0% |
| unattributed | kernel publish_ids_kernel | 0.007 | 0.0% |
| unattributed | kernel speculative_select_accepted_hidden_kernel | 0.006 | 0.0% |
| unattributed | kernel scatter_bf16x8_kernel | 0.004 | 0.0% |
| unattributed | kernel mtp_advance_round_kernel | 0.004 | 0.0% |
| unattributed | kernel wait_staged_kernel | 0.004 | 0.0% |
| unattributed | kernel rmsnorm_cta_bf16x2_kernel | 0.004 | 0.0% |
| unattributed | kernel expand_text_positions_kernel | 0.002 | 0.0% |
| unattributed | kernel mtp_prepare_next_round_kernel | 0.002 | 0.0% |
| unattributed | kernel speculative_prepare_verify_inputs_kernel | 0.002 | 0.0% |
| unattributed | kernel select_shared_indices_kernel | 0.002 | 0.0% |
| unattributed | kernel shortlist_exact_select_kernel | 0.001 | 0.0% |
| unattributed | kernel residual_add_bf16x8_kernel | 0.001 | 0.0% |
| unattributed | memset memset | 0.000 | 0.0% |

Host lookup work (may overlap GPU execution):

- ple.gather: 78.537 ms across 133 calls.
- ple.hash: 0.000 ms across 0 calls.

- Estimates are partial cold useful-work models, not distance from optimal.
- Token counts are execution envelopes; padded columns are not committed outputs.
- QSA estimates cover projections only; device-dependent scoring/attention is unmodeled.
- GDN prefill estimates omit chunked recurrence compute; PLE estimates omit convolution/state traffic.
- DRAM cache reuse, scratch traffic, nonlinear math and instruction geometry are unmodeled.
- Profiler overhead affects timings; use unprofiled paired benchmarks for speed claims.
- Unattributed GPU work is retained; no whole-model efficiency is reported.
- Serve window: GPU work launched inside 133 decode.mtp_round ranges with batch 4 (trim 0.00 of rounds at each end); work from other host threads launched during a round is included.
- An estimate exceeds measured time: inspect cache reuse, envelope work and hardware rates; percentages are not clamped.
