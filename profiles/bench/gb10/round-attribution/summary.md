## Round attribution: C4, MTP draft 1

- Host: Linux 6.17.0-1029-nvidia aarch64; Ubuntu 24.04.4 LTS
- Commit: 42722673 (with local changes)
- nvcc: release 13.0, V13.0.88
- GPU/driver: NVIDIA GB10, 580.173.02
- Artifact: qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer, 119G (multi-volume total)
- nsys: NVIDIA Nsight Systems version 2025.3.2.474-253236389321v0
- Load: two batches of 4 simultaneous greedy requests, 256 output tokens each (ignore_eos)
- Request log totals: Totals: 2048 tokens, decode 74.5 s (27.5 tok/s), prefill 2.5 s, rounds 1168, 1.75 tok/round, device wait 62.5 ms/round, queue wait mean 1669 ms, host exposed 1.28 ms/round


Measured wall: 9458.502 ms; GPU busy: 9367.460 ms; attributed GPU work: 94.19%.

Estimates describe partial modeled work, not an achievable optimum. Bytes/FLOPs below are per call; time and share are aggregated.

| Phase / role | Stage | T / B / context envelope | Calls | GPU work ms | Phase share | Useful bytes min–max | BF16 / NVFP4 / FP32 FLOPs | Estimate ms min–max | Estimate efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | 8 / 0 / 0 | 6336 | 4942.956 | 52.8% | 35279520–228816640 | 99655680 / 786432000 / 0 | 908.663–5893.424 | 18.4–119.2% |
| decode / target.verify | gdn.record | 8 / 4 / 0 | 4752 | 1801.501 | 19.2% | 71163904–71163904 | 926679040 / 0 / 37748736 | 1374.678–1374.678 | 76.3–76.3% |
| decode / target.verify | qsa.select | 8 / 4 / 129 | 1584 | 846.226 | 9.0% | 55817216–55817216 | 823132160 / 0 / 0 | 359.408–359.408 | 42.5–42.5% |
| decode / target.verify | hyper.combine_mix | 8 / 0 / 0 | 12408 | 714.702 | 7.6% | 7066368–7066368 | 105512960 / 0 / 0 | 356.421–356.421 | 49.9–49.9% |
| decode / target.verify | ple.record | 8 / 4 / 0 | 132 | 304.980 | 3.3% | 65884160–65884160 | 524288000 / 0 / 0 | 35.352–35.352 | 11.6–11.6% |
| decode / predictor | moe.nvfp4 | 8 / 0 / 0 | 132 | 102.472 | 1.1% | 35279520–228816640 | 99655680 / 786432000 / 0 | 18.930–122.780 | 18.5–119.8% |
| decode / predictor | qsa.select | 8 / 4 / 129 | 132 | 68.611 | 0.7% | 55817216–55817216 | 823132160 / 0 / 0 | 29.951–29.951 | 43.7–43.7% |
| decode / target.verify | hyper.mix | 8 / 0 / 0 | 264 | 14.086 | 0.2% | 6861504–6861504 | 105512960 / 0 / 0 | 7.364–7.364 | 52.3–52.3% |
| decode / predictor | hyper.combine_mix | 8 / 0 / 0 | 132 | 8.703 | 0.1% | 7066368–7066368 | 105512960 / 0 / 0 | 3.792–3.792 | 43.6–43.6% |
| decode / predictor | hyper.mix | 8 / 0 / 0 | 132 | 6.150 | 0.1% | 6861504–6861504 | 105512960 / 0 / 0 | 3.682–3.682 | 59.9–59.9% |
| decode / predictor | hyper.mix | 8 / 0 / 0 | 132 | 5.744 | 0.1% | 6779520–6779520 | 104857600 / 0 / 0 | 3.638–3.638 | 63.3–63.3% |
| decode / target.verify | hyper.mix | 8 / 0 / 0 | 132 | 5.721 | 0.1% | 6779520–6779520 | 104857600 / 0 / 0 | 3.638–3.638 | 63.6–63.6% |
| decode / target.verify | hyper.combine | 8 / 0 / 0 | 264 | 0.781 | 0.0% | 368704–368704 | 0 / 0 / 0 | 0.396–0.396 | 50.7–50.7% |
| decode / predictor | hyper.combine | 8 / 0 / 0 | 132 | 0.377 | 0.0% | 368704–368704 | 0 / 0 / 0 | 0.198–0.198 | 52.4–52.4% |
| decode / predictor | hyper.add | 8 / 0 / 0 | 132 | 0.281 | 0.0% | 368640–368640 | 0 / 0 / 0 | 0.198–0.198 | 70.4–70.4% |
| decode / target.verify | hyper.repeat | 8 / 0 / 0 | 132 | 0.277 | 0.0% | 204800–204800 | 0 / 0 / 0 | 0.110–0.110 | 39.7–39.7% |

Unattributed GPU work: 543.912 ms.

Kernels per stage (ms/round, launches per round):

| Phase / role | Stage | Kernel | Launches | GPU work ms/round | Stage share |
|---|---|---|---:|---:|---:|
| decode / target.verify | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 96.0 | 34.263 | 91.5% |
| decode / target.verify | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 96.0 | 1.286 | 3.4% |
| decode / target.verify | moe.nvfp4 | bf16_small_t_inner_kernel | 48.0 | 0.795 | 2.1% |
| decode / target.verify | moe.nvfp4 | route_kernel | 48.0 | 0.596 | 1.6% |
| decode / target.verify | moe.nvfp4 | quantize_grouped_kernel | 48.0 | 0.204 | 0.5% |
| decode / target.verify | moe.nvfp4 | gather_quantize_routes_kernel | 48.0 | 0.201 | 0.5% |
| decode / target.verify | moe.nvfp4 | reduce_grouped_kernel | 48.0 | 0.102 | 0.3% |
| decode / target.verify | gdn.record | fp8_a16_sliced_k_mma_kernel | 108.0 | 10.196 | 74.7% |
| decode / target.verify | gdn.record | recurrent_fold_record_kernel | 36.0 | 2.839 | 20.8% |
| decode / target.verify | gdn.record | project_control_gating_kernel | 36.0 | 0.294 | 2.2% |
| decode / target.verify | gdn.record | conv_replay_record_kernel | 36.0 | 0.249 | 1.8% |
| decode / target.verify | gdn.record | rmsnorm_warp_bf16x2_kernel | 36.0 | 0.070 | 0.5% |
| decode / target.verify | qsa.select | selected_attention_split_fp8_kernel | 12.0 | 2.881 | 44.9% |
| decode / target.verify | qsa.select | fp8_a16_sliced_k_mma_kernel | 24.0 | 2.482 | 38.7% |
| decode / target.verify | qsa.select | bf16_small_t_inner_kernel | 48.0 | 0.709 | 11.1% |
| decode / target.verify | qsa.select | append_cache_fp8_kernel | 12.0 | 0.066 | 1.0% |
| decode / target.verify | qsa.select | reduce_selected_attention_splits_kernel | 12.0 | 0.059 | 0.9% |
| decode / target.verify | qsa.select | rmsnorm_warp_bf16x2_kernel | 24.0 | 0.051 | 0.8% |
| decode / target.verify | qsa.select | rope_generic_kernel | 12.0 | 0.051 | 0.8% |
| decode / target.verify | qsa.select | compress_index_groups_kernel | 12.0 | 0.036 | 0.6% |
| decode / target.verify | qsa.select | prepare_index_query_kernel | 12.0 | 0.023 | 0.4% |
| decode / target.verify | qsa.select | split_query_gate_kernel | 12.0 | 0.020 | 0.3% |
| decode / target.verify | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 12.0 | 0.018 | 0.3% |
| decode / target.verify | qsa.select | hadamard_rows_kernel | 12.0 | 0.015 | 0.2% |
| decode / target.verify | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 188.0 | 4.084 | 75.4% |
| decode / target.verify | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 94.0 | 1.330 | 24.6% |
| decode / target.verify | ple.record | bf16_gemv_kernel | 16.0 | 2.284 | 98.8% |
| decode / target.verify | ple.record | gate_kernel | 1.0 | 0.011 | 0.5% |
| decode / target.verify | ple.record | dilated_conv_record_kernel | 1.0 | 0.006 | 0.3% |
| decode / target.verify | ple.record | grouped_norm_kernel | 1.0 | 0.006 | 0.3% |
| decode / target.verify | ple.record | dequantize_embedding_kernel | 1.0 | 0.004 | 0.2% |
| decode / predictor | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 2.0 | 0.710 | 91.5% |
| decode / predictor | moe.nvfp4 | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.027 | 3.4% |
| decode / predictor | moe.nvfp4 | bf16_small_t_inner_kernel | 1.0 | 0.017 | 2.1% |
| decode / predictor | moe.nvfp4 | route_kernel | 1.0 | 0.012 | 1.6% |
| decode / predictor | moe.nvfp4 | quantize_grouped_kernel | 1.0 | 0.004 | 0.5% |
| decode / predictor | moe.nvfp4 | gather_quantize_routes_kernel | 1.0 | 0.004 | 0.5% |
| decode / predictor | moe.nvfp4 | reduce_grouped_kernel | 1.0 | 0.002 | 0.3% |
| decode / predictor | qsa.select | selected_attention_split_fp8_kernel | 1.0 | 0.228 | 43.9% |
| decode / predictor | qsa.select | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.202 | 38.8% |
| decode / predictor | qsa.select | bf16_small_t_inner_kernel | 4.0 | 0.060 | 11.6% |
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
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 4.0 | 0.088 | 82.7% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 2.0 | 0.018 | 17.3% |
| decode / predictor | hyper.combine_mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.052 | 78.8% |
| decode / predictor | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 1.0 | 0.014 | 21.2% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.038 | 82.1% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.008 | 17.9% |
| decode / predictor | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.037 | 85.6% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.006 | 14.4% |
| decode / target.verify | hyper.mix | fp8_a16_sliced_k_mma_kernel | 2.0 | 0.037 | 85.4% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.006 | 14.6% |
| decode / target.verify | hyper.combine | combine_kernel | 2.0 | 0.006 | 100.0% |
| decode / predictor | hyper.combine | combine_kernel | 1.0 | 0.003 | 100.0% |
| decode / predictor | hyper.add | add_repeated_kernel | 1.0 | 0.002 | 100.0% |
| decode / target.verify | hyper.repeat | repeat_kernel | 1.0 | 0.002 | 100.0% |

Per round over 132 rounds: host wall 71.655 ms, GPU work 70.966 ms, GPU busy 70.966 ms.

| Phase / role | Stage | GPU work ms/round | Share |
|---|---|---:|---:|
| decode / target.verify | moe.nvfp4 | 37.447 | 52.8% |
| decode / target.verify | gdn.record | 13.648 | 19.2% |
| decode / target.verify | qsa.select | 6.411 | 9.0% |
| decode / target.verify | hyper.combine_mix | 5.414 | 7.6% |
| decode / target.verify | ple.record | 2.310 | 3.3% |
| decode / predictor | moe.nvfp4 | 0.776 | 1.1% |
| decode / predictor | qsa.select | 0.520 | 0.7% |
| decode / target.verify | hyper.mix | 0.107 | 0.2% |
| decode / predictor | hyper.combine_mix | 0.066 | 0.1% |
| decode / predictor | hyper.mix | 0.047 | 0.1% |
| decode / predictor | hyper.mix | 0.044 | 0.1% |
| decode / target.verify | hyper.mix | 0.043 | 0.1% |
| decode / target.verify | hyper.combine | 0.006 | 0.0% |
| decode / predictor | hyper.combine | 0.003 | 0.0% |
| decode / predictor | hyper.add | 0.002 | 0.0% |
| decode / target.verify | hyper.repeat | 0.002 | 0.0% |
| unattributed | kernel fp8_a16_sliced_k_mma_kernel | 2.574 | 3.6% |
| unattributed | kernel q4_a16_sliced_k_mma_kernel | 0.882 | 1.2% |
| unattributed | kernel wait_staged_kernel | 0.201 | 0.3% |
| unattributed | kernel bf16_gemm_mma_kernel | 0.071 | 0.1% |
| unattributed | kernel bf16_small_t_inner_kernel | 0.059 | 0.1% |
| unattributed | kernel copy_record_rows_kernel | 0.057 | 0.1% |
| unattributed | kernel argmax_tiled_atomic_kernel | 0.054 | 0.1% |
| unattributed | kernel shortlist_exact_scores_fp8_kernel | 0.048 | 0.1% |
| unattributed | kernel speculative_sampling_partial_topk_kernel | 0.034 | 0.0% |
| unattributed | kernel shortlist_tile_candidates_kernel | 0.028 | 0.0% |
| unattributed | kernel ple_fold_kernel | 0.027 | 0.0% |
| unattributed | kernel rmsnorm_generic_kernel | 0.019 | 0.0% |
| unattributed | memcpy memcpy | 0.012 | 0.0% |
| unattributed | kernel speculative_sampling_group_finalize_kernel | 0.009 | 0.0% |
| unattributed | kernel embed_gather_dense_kernel | 0.008 | 0.0% |
| unattributed | kernel speculative_select_accepted_hidden_kernel | 0.006 | 0.0% |
| unattributed | kernel scatter_bf16x8_kernel | 0.006 | 0.0% |
| unattributed | kernel publish_ids_kernel | 0.006 | 0.0% |
| unattributed | kernel mtp_advance_round_kernel | 0.004 | 0.0% |
| unattributed | kernel rmsnorm_cta_bf16x2_kernel | 0.003 | 0.0% |
| unattributed | kernel expand_text_positions_kernel | 0.003 | 0.0% |
| unattributed | kernel mtp_prepare_next_round_kernel | 0.002 | 0.0% |
| unattributed | kernel speculative_prepare_verify_inputs_kernel | 0.002 | 0.0% |
| unattributed | kernel select_shared_indices_kernel | 0.002 | 0.0% |
| unattributed | kernel shortlist_exact_select_kernel | 0.001 | 0.0% |
| unattributed | kernel residual_add_bf16x8_kernel | 0.001 | 0.0% |
| unattributed | memset memset | 0.000 | 0.0% |

Host lookup work (may overlap GPU execution):

- ple.gather: 156.360 ms across 132 calls.
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
