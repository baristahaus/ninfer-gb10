## Round attribution: C4, MTP draft 3

- Host: Linux 6.17.0-1029-nvidia aarch64; Ubuntu 24.04.4 LTS
- Commit: a5e4a219 (with local changes)
- nvcc: release 13.0, V13.0.88
- GPU/driver: NVIDIA GB10, 580.173.02
- Artifact: qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer, 119G (multi-volume total)
- nsys: NVIDIA Nsight Systems version 2025.3.2.474-253236389321v0
- Load: two batches of 4 simultaneous greedy requests, 256 output tokens each (ignore_eos)
- Request log totals: Totals: 2048 tokens, decode 67.6 s (30.3 tok/s), prefill 2.5 s, rounds 838, 2.44 tok/round, device wait 79.3 ms/round, queue wait mean 1437 ms, host exposed 1.43 ms/round


Measured wall: 7357.911 ms; GPU busy: 7297.665 ms; attributed GPU work: 93.24%.

Estimates describe partial modeled work, not an achievable optimum. Bytes/FLOPs below are per call; time and share are aggregated.

| Phase / role | Stage | T / B / context envelope | Calls | GPU work ms | Phase share | Useful bytes min–max | BF16 / NVFP4 / FP32 FLOPs | Estimate ms min–max | Estimate efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | 16 / 0 / 0 | 3792 | 4181.426 | 57.3% | 35361440–450083840 | 199311360 / 1572864000 / 0 | 545.084–6937.878 | 13.0–165.9% |
| decode / target.verify | gdn.record | 16 / 4 / 0 | 2844 | 1112.216 | 15.2% | 71543808–71543808 | 1853358080 / 0 / 75497472 | 827.116–827.116 | 74.4–74.4% |
| decode / target.verify | qsa.select | 16 / 4 / 131 | 948 | 715.262 | 9.8% | 55899136–55899136 | 1646264320 / 0 / 0 | 215.416–215.416 | 30.1–30.1% |
| decode / target.verify | hyper.combine_mix | 16 / 0 / 0 | 7426 | 444.870 | 6.1% | 7476096–7476096 | 211025920 / 0 / 0 | 225.681–225.681 | 50.7–50.7% |
| decode / predictor | moe.nvfp4 | 16 / 0 / 0 | 79 | 87.114 | 1.2% | 35361440–450083840 | 199311360 / 1572864000 / 0 | 11.356–144.539 | 13.0–165.9% |
| decode / predictor | moe.nvfp4 | 4 / 0 / 0 | 158 | 77.302 | 1.1% | 35238560–118183040 | 49827840 / 393216000 / 0 | 22.633–75.906 | 29.3–98.2% |
| decode / predictor | qsa.select | 16 / 4 / 131 | 79 | 48.955 | 0.7% | 55899136–55899136 | 1646264320 / 0 / 0 | 17.951–17.951 | 36.7–36.7% |
| decode / target.verify | ple.record | 16 / 4 / 0 | 79 | 43.224 | 0.6% | 66232320–66232320 | 1048576000 / 0 / 0 | 21.270–21.270 | 49.2–49.2% |
| decode / predictor | qsa.reuse | 4 / 4 / 133 | 79 | 22.196 | 0.3% | 53154816–53154816 | 401080320 / 0 / 0 | 17.070–17.070 | 76.9–76.9% |
| decode / predictor | qsa.reuse | 4 / 4 / 132 | 79 | 22.111 | 0.3% | 53154816–53154816 | 401080320 / 0 / 0 | 17.070–17.070 | 77.2–77.2% |
| decode / target.verify | hyper.mix | 16 / 0 / 0 | 158 | 8.956 | 0.1% | 7066368–7066368 | 211025920 / 0 / 0 | 4.539–4.539 | 50.7–50.7% |
| decode / predictor | hyper.combine_mix | 4 / 0 / 0 | 158 | 8.861 | 0.1% | 6861504–6861504 | 52756480 / 0 / 0 | 4.407–4.407 | 49.7–49.7% |
| decode / predictor | hyper.mix | 4 / 0 / 0 | 158 | 6.714 | 0.1% | 6759072–6759072 | 52756480 / 0 / 0 | 4.341–4.341 | 64.7–64.7% |
| decode / predictor | hyper.mix | 4 / 0 / 0 | 158 | 6.410 | 0.1% | 6677120–6677120 | 52428800 / 0 / 0 | 4.289–4.289 | 66.9–66.9% |
| decode / predictor | hyper.combine_mix | 16 / 0 / 0 | 79 | 5.481 | 0.1% | 7476096–7476096 | 211025920 / 0 / 0 | 2.401–2.401 | 43.8–43.8% |
| decode / predictor | hyper.mix | 16 / 0 / 0 | 79 | 3.770 | 0.1% | 7066368–7066368 | 211025920 / 0 / 0 | 2.269–2.269 | 60.2–60.2% |
| decode / target.verify | hyper.mix | 16 / 0 / 0 | 79 | 3.643 | 0.0% | 6984320–6984320 | 209715200 / 0 / 0 | 2.243–2.243 | 61.6–61.6% |
| decode / predictor | hyper.mix | 16 / 0 / 0 | 79 | 3.554 | 0.0% | 6984320–6984320 | 209715200 / 0 / 0 | 2.243–2.243 | 63.1–63.1% |
| decode / target.verify | hyper.combine | 16 / 0 / 0 | 158 | 0.677 | 0.0% | 737408–737408 | 0 / 0 / 0 | 0.474–0.474 | 70.0–70.0% |
| decode / predictor | hyper.combine | 4 / 0 / 0 | 158 | 0.341 | 0.0% | 184352–184352 | 0 / 0 / 0 | 0.118–0.118 | 34.7–34.7% |
| decode / predictor | hyper.combine | 16 / 0 / 0 | 79 | 0.338 | 0.0% | 737408–737408 | 0 / 0 / 0 | 0.237–0.237 | 70.1–70.1% |
| decode / predictor | hyper.add | 16 / 0 / 0 | 79 | 0.260 | 0.0% | 737280–737280 | 0 / 0 / 0 | 0.237–0.237 | 91.0–91.0% |
| decode / target.verify | hyper.repeat | 16 / 0 / 0 | 79 | 0.257 | 0.0% | 409600–409600 | 0 / 0 / 0 | 0.132–0.132 | 51.1–51.1% |
| decode / predictor | hyper.add | 4 / 0 / 0 | 158 | 0.251 | 0.0% | 184320–184320 | 0 / 0 / 0 | 0.118–0.118 | 47.2–47.2% |

Unattributed GPU work: 493.483 ms.

Kernels per stage (ms/round, launches per round):

| Phase / role | Stage | Kernel | Launches | GPU work ms/round | Stage share |
|---|---|---|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 96.0 | 49.346 | 93.2% |
| decode / target.verify | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 96.0 | 1.288 | 2.4% |
| decode / target.verify | moe.nvfp4 | bf16_small_t_inner_kernel | 48.0 | 0.981 | 1.9% |
| decode / target.verify | moe.nvfp4 | route_kernel | 48.0 | 0.727 | 1.4% |
| decode / target.verify | moe.nvfp4 | gather_quantize_routes_kernel | 48.0 | 0.216 | 0.4% |
| decode / target.verify | moe.nvfp4 | quantize_grouped_kernel | 48.0 | 0.209 | 0.4% |
| decode / target.verify | moe.nvfp4 | reduce_grouped_kernel | 48.0 | 0.163 | 0.3% |
| decode / target.verify | gdn.record | fp8_a16_sliced_k_mma_kernel | 108.0 | 10.167 | 72.2% |
| decode / target.verify | gdn.record | recurrent_fold_record_kernel | 36.0 | 3.193 | 22.7% |
| decode / target.verify | gdn.record | project_control_gating_kernel | 36.0 | 0.385 | 2.7% |
| decode / target.verify | gdn.record | conv_replay_record_kernel | 36.0 | 0.247 | 1.8% |
| decode / target.verify | gdn.record | rmsnorm_warp_bf16x2_kernel | 36.0 | 0.087 | 0.6% |
| decode / target.verify | qsa.select | selected_attention_split_fp8_kernel | 12.0 | 5.335 | 58.9% |
| decode / target.verify | qsa.select | fp8_a16_sliced_k_mma_kernel | 24.0 | 2.530 | 27.9% |
| decode / target.verify | qsa.select | bf16_small_t_inner_kernel | 48.0 | 0.858 | 9.5% |
| decode / target.verify | qsa.select | append_cache_fp8_kernel | 12.0 | 0.066 | 0.7% |
| decode / target.verify | qsa.select | rmsnorm_warp_bf16x2_kernel | 24.0 | 0.052 | 0.6% |
| decode / target.verify | qsa.select | rope_generic_kernel | 12.0 | 0.050 | 0.5% |
| decode / target.verify | qsa.select | reduce_selected_attention_splits_kernel | 12.0 | 0.039 | 0.4% |
| decode / target.verify | qsa.select | compress_index_groups_kernel | 12.0 | 0.036 | 0.4% |
| decode / target.verify | qsa.select | split_query_gate_kernel | 12.0 | 0.028 | 0.3% |
| decode / target.verify | qsa.select | prepare_index_query_kernel | 12.0 | 0.024 | 0.3% |
| decode / target.verify | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 12.0 | 0.020 | 0.2% |
| decode / target.verify | qsa.select | hadamard_rows_kernel | 12.0 | 0.016 | 0.2% |
| decode / target.verify | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 188.0 | 4.278 | 76.0% |
| decode / target.verify | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 94.0 | 1.353 | 24.0% |
| decode / predictor | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 2.0 | 1.026 | 93.0% |
| decode / predictor | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.029 | 2.6% |
| decode / predictor | moe.nvfp4 | bf16_small_t_inner_kernel | 1.0 | 0.021 | 1.9% |
| decode / predictor | moe.nvfp4 | route_kernel | 1.0 | 0.015 | 1.4% |
| decode / predictor | moe.nvfp4 | gather_quantize_routes_kernel | 1.0 | 0.004 | 0.4% |
| decode / predictor | moe.nvfp4 | quantize_grouped_kernel | 1.0 | 0.004 | 0.4% |
| decode / predictor | moe.nvfp4 | reduce_grouped_kernel | 1.0 | 0.003 | 0.3% |
| decode / predictor | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 4.0 | 0.876 | 89.5% |
| decode / predictor | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 4.0 | 0.049 | 5.0% |
| decode / predictor | moe.nvfp4 | bf16_small_t_inner_kernel | 2.0 | 0.032 | 3.3% |
| decode / predictor | moe.nvfp4 | route_kernel | 2.0 | 0.015 | 1.6% |
| decode / predictor | moe.nvfp4 | quantize_decode_routes_kernel | 2.0 | 0.004 | 0.4% |
| decode / predictor | moe.nvfp4 | reduce_decode_routes_kernel | 2.0 | 0.003 | 0.3% |
| decode / predictor | qsa.select | selected_attention_split_fp8_kernel | 1.0 | 0.315 | 50.8% |
| decode / predictor | qsa.select | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.203 | 32.8% |
| decode / predictor | qsa.select | bf16_small_t_inner_kernel | 4.0 | 0.072 | 11.7% |
| decode / predictor | qsa.select | append_cache_fp8_kernel | 1.0 | 0.007 | 1.2% |
| decode / predictor | qsa.select | rmsnorm_warp_bf16x2_kernel | 2.0 | 0.004 | 0.6% |
| decode / predictor | qsa.select | rope_generic_kernel | 1.0 | 0.004 | 0.6% |
| decode / predictor | qsa.select | reduce_selected_attention_splits_kernel | 1.0 | 0.003 | 0.5% |
| decode / predictor | qsa.select | compress_index_groups_kernel | 1.0 | 0.003 | 0.5% |
| decode / predictor | qsa.select | split_query_gate_kernel | 1.0 | 0.002 | 0.4% |
| decode / predictor | qsa.select | prepare_index_query_kernel | 1.0 | 0.002 | 0.3% |
| decode / predictor | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 1.0 | 0.002 | 0.3% |
| decode / predictor | qsa.select | hadamard_rows_kernel | 1.0 | 0.001 | 0.2% |
| decode / predictor | qsa.select | dense_indices_batched_kernel | 1.0 | 0.001 | 0.2% |
| decode / target.verify | ple.record | bf16_gemv_columns_kernel | 2.0 | 0.522 | 95.3% |
| decode / target.verify | ple.record | gate_kernel | 1.0 | 0.009 | 1.6% |
| decode / target.verify | ple.record | dilated_conv_record_kernel | 1.0 | 0.007 | 1.3% |
| decode / target.verify | ple.record | grouped_norm_kernel | 1.0 | 0.006 | 1.1% |
| decode / target.verify | ple.record | dequantize_embedding_kernel | 1.0 | 0.004 | 0.7% |
| decode / predictor | qsa.reuse | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.189 | 67.2% |
| decode / predictor | qsa.reuse | bf16_small_t_inner_kernel | 3.0 | 0.035 | 12.5% |
| decode / predictor | qsa.reuse | selected_attention_batched_fp8_kernel | 1.0 | 0.030 | 10.7% |
| decode / predictor | qsa.reuse | reduce_selected_attention_splits_kernel | 1.0 | 0.010 | 3.7% |
| decode / predictor | qsa.reuse | rope_generic_kernel | 1.0 | 0.004 | 1.4% |
| decode / predictor | qsa.reuse | append_cache_fp8_kernel | 1.0 | 0.003 | 1.2% |
| decode / predictor | qsa.reuse | rmsnorm_warp_bf16x2_kernel | 2.0 | 0.003 | 1.0% |
| decode / predictor | qsa.reuse | compress_index_groups_kernel | 1.0 | 0.002 | 0.9% |
| decode / predictor | qsa.reuse | sigmoid_gate_mul_bf16x8_kernel | 1.0 | 0.002 | 0.6% |
| decode / predictor | qsa.reuse | split_query_gate_kernel | 1.0 | 0.001 | 0.5% |
| decode / predictor | qsa.reuse | hadamard_rows_kernel | 1.0 | 0.001 | 0.4% |
| decode / predictor | qsa.reuse | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.188 | 67.3% |
| decode / predictor | qsa.reuse | bf16_small_t_inner_kernel | 3.0 | 0.035 | 12.5% |
| decode / predictor | qsa.reuse | selected_attention_batched_fp8_kernel | 1.0 | 0.030 | 10.8% |
| decode / predictor | qsa.reuse | reduce_selected_attention_splits_kernel | 1.0 | 0.010 | 3.7% |
| decode / predictor | qsa.reuse | rope_generic_kernel | 1.0 | 0.004 | 1.4% |
| decode / predictor | qsa.reuse | append_cache_fp8_kernel | 1.0 | 0.003 | 1.2% |
| decode / predictor | qsa.reuse | rmsnorm_warp_bf16x2_kernel | 2.0 | 0.003 | 1.0% |
| decode / predictor | qsa.reuse | compress_index_groups_kernel | 1.0 | 0.002 | 0.8% |
| decode / predictor | qsa.reuse | sigmoid_gate_mul_bf16x8_kernel | 1.0 | 0.002 | 0.5% |
| decode / predictor | qsa.reuse | split_query_gate_kernel | 1.0 | 0.001 | 0.5% |
| decode / predictor | qsa.reuse | hadamard_rows_kernel | 1.0 | 0.001 | 0.4% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 4.0 | 0.094 | 82.8% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 2.0 | 0.019 | 17.2% |
| decode / predictor | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 4.0 | 0.084 | 74.8% |
| decode / predictor | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 2.0 | 0.028 | 25.2% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 4.0 | 0.068 | 80.3% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 2.0 | 0.017 | 19.7% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 4.0 | 0.069 | 84.5% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 2.0 | 0.013 | 15.5% |
| decode / predictor | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.055 | 79.6% |
| decode / predictor | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 1.0 | 0.014 | 20.4% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.039 | 82.5% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.008 | 17.5% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.040 | 86.6% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.006 | 13.4% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.039 | 86.3% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.006 | 13.7% |
| decode / target.verify | hyper.combine | combine_kernel | 2.0 | 0.009 | 100.0% |
| decode / predictor | hyper.combine | combine_kernel | 2.0 | 0.004 | 100.0% |
| decode / predictor | hyper.combine | combine_kernel | 1.0 | 0.004 | 100.0% |
| decode / predictor | hyper.add | add_repeated_kernel | 1.0 | 0.003 | 100.0% |
| decode / target.verify | hyper.repeat | repeat_kernel | 1.0 | 0.003 | 100.0% |
| decode / predictor | hyper.add | add_repeated_kernel | 2.0 | 0.003 | 100.0% |

Per round over 79 rounds: host wall 93.138 ms, GPU work 92.376 ms, GPU busy 92.376 ms.

| Phase / role | Stage | GPU work ms/round | Share |
|---|---|---:|---:|
| decode / target.verify | moe.nvfp4 | 52.929 | 57.3% |
| decode / target.verify | gdn.record | 14.079 | 15.2% |
| decode / target.verify | qsa.select | 9.054 | 9.8% |
| decode / target.verify | hyper.combine_mix | 5.631 | 6.1% |
| decode / predictor | moe.nvfp4 | 1.103 | 1.2% |
| decode / predictor | moe.nvfp4 | 0.979 | 1.1% |
| decode / predictor | qsa.select | 0.620 | 0.7% |
| decode / target.verify | ple.record | 0.547 | 0.6% |
| decode / predictor | qsa.reuse | 0.281 | 0.3% |
| decode / predictor | qsa.reuse | 0.280 | 0.3% |
| decode / target.verify | hyper.mix | 0.113 | 0.1% |
| decode / predictor | hyper.combine_mix | 0.112 | 0.1% |
| decode / predictor | hyper.mix | 0.085 | 0.1% |
| decode / predictor | hyper.mix | 0.081 | 0.1% |
| decode / predictor | hyper.combine_mix | 0.069 | 0.1% |
| decode / predictor | hyper.mix | 0.048 | 0.1% |
| decode / target.verify | hyper.mix | 0.046 | 0.0% |
| decode / predictor | hyper.mix | 0.045 | 0.0% |
| decode / target.verify | hyper.combine | 0.009 | 0.0% |
| decode / predictor | hyper.combine | 0.004 | 0.0% |
| decode / predictor | hyper.combine | 0.004 | 0.0% |
| decode / predictor | hyper.add | 0.003 | 0.0% |
| decode / target.verify | hyper.repeat | 0.003 | 0.0% |
| decode / predictor | hyper.add | 0.003 | 0.0% |
| unattributed | kernel fp8_a16_sliced_k_mma_kernel | 2.609 | 2.8% |
| unattributed | kernel q4_a16_sliced_k_mma_kernel | 2.571 | 2.8% |
| unattributed | kernel bf16_small_t_inner_kernel | 0.275 | 0.3% |
| unattributed | kernel copy_record_rows_kernel | 0.151 | 0.2% |
| unattributed | kernel shortlist_exact_scores_fp8_kernel | 0.139 | 0.2% |
| unattributed | kernel argmax_tiled_atomic_kernel | 0.107 | 0.1% |
| unattributed | kernel shortlist_tile_candidates_kernel | 0.085 | 0.1% |
| unattributed | kernel bf16_gemm_mma_kernel | 0.071 | 0.1% |
| unattributed | kernel rmsnorm_generic_kernel | 0.060 | 0.1% |
| unattributed | kernel speculative_sampling_partial_topk_kernel | 0.059 | 0.1% |
| unattributed | kernel ple_fold_kernel | 0.026 | 0.0% |
| unattributed | memcpy memcpy | 0.016 | 0.0% |
| unattributed | kernel embed_gather_dense_kernel | 0.016 | 0.0% |
| unattributed | kernel speculative_sampling_group_finalize_kernel | 0.014 | 0.0% |
| unattributed | kernel publish_ids_kernel | 0.006 | 0.0% |
| unattributed | kernel rmsnorm_cta_bf16x2_kernel | 0.006 | 0.0% |
| unattributed | kernel speculative_select_accepted_hidden_kernel | 0.006 | 0.0% |
| unattributed | kernel expand_text_positions_kernel | 0.005 | 0.0% |
| unattributed | kernel mtp_advance_round_kernel | 0.005 | 0.0% |
| unattributed | kernel shortlist_exact_select_kernel | 0.004 | 0.0% |
| unattributed | kernel scatter_bf16x8_kernel | 0.004 | 0.0% |
| unattributed | kernel wait_staged_kernel | 0.003 | 0.0% |
| unattributed | kernel mtp_prepare_next_round_kernel | 0.002 | 0.0% |
| unattributed | kernel speculative_prepare_verify_inputs_kernel | 0.002 | 0.0% |
| unattributed | kernel select_shared_indices_kernel | 0.002 | 0.0% |
| unattributed | kernel residual_add_bf16x8_kernel | 0.002 | 0.0% |
| unattributed | memset memset | 0.000 | 0.0% |

Host lookup work (may overlap GPU execution):

- ple.gather: 88.828 ms across 79 calls.
- ple.hash: 0.000 ms across 0 calls.

- Estimates are partial cold useful-work models, not distance from optimal.
- Token counts are execution envelopes; padded columns are not committed outputs.
- QSA estimates cover projections only; device-dependent scoring/attention is unmodeled.
- GDN prefill estimates omit chunked recurrence compute; PLE estimates omit convolution/state traffic.
- DRAM cache reuse, scratch traffic, nonlinear math and instruction geometry are unmodeled.
- Profiler overhead affects timings; use unprofiled paired benchmarks for speed claims.
- Unattributed GPU work is retained; no whole-model efficiency is reported.
- Serve window: GPU work launched inside 79 decode.mtp_round ranges with batch 4 (trim 0.00 of rounds at each end); work from other host threads launched during a round is included.
- An estimate exceeds measured time: inspect cache reuse, envelope work and hardware rates; percentages are not clamped.
