## Step 3 — decode attribution and PLE residency

- Host: Linux 6.17.0-1029-nvidia aarch64; Ubuntu 24.04.4 LTS
- Commit: 262cd09e
- nvcc: release 13.0, V13.0.88
- GPU/driver: NVIDIA GB10, 580.173.02
- Artifact: qwen3_8_flash_next_125b_a6b_nvfp4.ninfer, 126G (multi-volume total)
- nsys: NVIDIA Nsight Systems version 2025.3.2.474-253236389321v0
- Hardware profile: tools/bench/hardware/gb10.json (246 GB/s measured; derived compute rates)

### mtp0


Measured wall: 6802.283 ms; GPU busy: 6696.950 ms; attributed GPU work: 97.42%.

Estimates describe partial modeled work, not an achievable optimum. Bytes/FLOPs below are per call; time and share are aggregated.

| Phase / role | Stage | T / B / context envelope | Calls | GPU work ms | Phase share | Useful bytes min–max | BF16 / NVFP4 / FP32 FLOPs | Estimate ms min–max | Estimate efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prefill / target.prefill | gdn.prefill | 8192 / 1 / 0 | 36 | 1852.111 | 35.7% | 206012416–206012416 | 948919336960 / 0 / 0 | 273.289–273.289 | 14.8–14.8% |
| prefill / target.prefill | hyper.combine_mix | 8192 / 0 / 0 | 94 | 1189.998 | 22.9% | 432750592–432750592 | 108045271040 / 0 / 0 | 165.360–165.360 | 13.9–13.9% |
| prefill / target.prefill | qsa.select | 8192 / 1 / 8192 | 12 | 1155.133 | 22.3% | 186777600–186777600 | 842887331840 / 0 / 0 | 80.917–80.917 | 7.0–7.0% |
| prefill / target.prefill | moe.nvfp4 | 8192 / 0 / 0 | 48 | 905.576 | 17.4% | 123991200–1511928832 | 102047416320 / 805306368000 / 0 | 77.309–295.011 | 8.5–32.6% |
| decode / target.verify | gdn.update | 1 / 1 / 0 | 1152 | 602.654 | 40.0% | 122136576–122136576 | 115834880 / 0 / 4718592 | 571.957–571.957 | 95.0–95.0% |
| decode / target.verify | moe.nvfp4 | 1 / 0 / 0 | 1536 | 319.793 | 21.2% | 40115360–40115360 | 12456960 / 98304000 / 0 | 250.476–250.476 | 78.4–78.4% |
| decode / target.verify | qsa.select | 1 / 1 / 8192 | 384 | 210.015 | 13.9% | 102901760–102901760 | 102891520 / 0 / 0 | 160.627–160.627 | 76.5–76.5% |
| decode / target.verify | hyper.combine_mix | 1 / 0 / 0 | 3008 | 194.393 | 12.9% | 13240336–13240336 | 13189120 / 0 / 0 | 161.898–161.898 | 83.4–83.4% |
| prefill / target.prefill | ple.prefill | 8192 / 1 / 0 | 1 | 39.776 | 0.8% | 422051840–422051840 | 536870912000 / 0 / 0 | 4.295–4.295 | 10.8–10.8% |
| prefill / target.prefill | hyper.mix | 8192 / 0 / 0 | 2 | 23.455 | 0.5% | 222969856–222969856 | 108045271040 / 0 / 0 | 1.813–1.813 | 7.7–7.7% |
| prefill / target.prefill | hyper.mix | 8192 / 0 / 0 | 1 | 11.536 | 0.2% | 222822400–222822400 | 107374182400 / 0 / 0 | 0.906–0.906 | 7.9–7.9% |
| decode / target.verify | ple.update | 1 / 1 / 0 | 32 | 9.934 | 0.7% | 65579520–65579520 | 65536000 / 0 / 0 | 8.531–8.531 | 86.0–86.0% |
| decode / target.verify | hyper.mix | 1 / 0 / 0 | 64 | 4.127 | 0.3% | 13214728–13214728 | 13189120 / 0 / 0 | 3.438–3.438 | 83.3–83.3% |
| prefill / target.prefill | hyper.combine | 8192 / 0 / 0 | 2 | 4.079 | 0.1% | 377552896–377552896 | 0 / 0 / 0 | 3.070–3.070 | 75.2–75.2% |
| decode / target.verify | hyper.mix | 1 / 0 / 0 | 32 | 1.933 | 0.1% | 13132800–13132800 | 13107200 / 0 / 0 | 1.708–1.708 | 88.4–88.4% |
| prefill / target.prefill | hyper.repeat | 8192 / 0 / 0 | 1 | 0.991 | 0.0% | 209715200–209715200 | 0 / 0 / 0 | 0.853–0.853 | 86.0–86.0% |
| decode / target.verify | hyper.combine | 1 / 0 / 0 | 64 | 0.105 | 0.0% | 46088–46088 | 0 / 0 / 0 | 0.012–0.012 | 11.4–11.4% |
| decode / target.verify | hyper.repeat | 1 / 0 / 0 | 32 | 0.038 | 0.0% | 25600–25600 | 0 / 0 / 0 | 0.003–0.003 | 8.8–8.8% |

Unattributed GPU work: 172.590 ms.

Kernels per stage (ms, launches per capture):

| Phase / role | Stage | Kernel | Launches | GPU work ms | Stage share |
|---|---|---|---:|---:|---:|
| prefill / target.prefill | gdn.prefill | bf16_gemm_mma_kernel | 180.0 | 1662.825 | 89.8% |
| prefill / target.prefill | gdn.prefill | chunk_recurrence_kernel | 36.0 | 56.957 | 3.1% |
| prefill / target.prefill | gdn.prefill | causal_conv1d_prefill_pairs_kernel | 36.0 | 50.706 | 2.7% |
| prefill / target.prefill | gdn.prefill | rmsnorm_warp_bf16x2_kernel | 36.0 | 44.088 | 2.4% |
| prefill / target.prefill | gdn.prefill | prepare_kernel | 36.0 | 37.096 | 2.0% |
| prefill / target.prefill | gdn.prefill | gdn_gating_kernel | 36.0 | 0.343 | 0.0% |
| prefill / target.prefill | gdn.prefill | causal_conv1d_prefill_state_kernel | 36.0 | 0.097 | 0.0% |
| prefill / target.prefill | hyper.combine_mix | bf16_gemm_mma_kernel | 188.0 | 807.180 | 67.8% |
| prefill / target.prefill | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 94.0 | 214.663 | 18.0% |
| prefill / target.prefill | hyper.combine_mix | gate_mix_injection_kernel | 94.0 | 164.894 | 13.9% |
| prefill / target.prefill | hyper.combine_mix | scaled_silu_kernel | 94.0 | 3.261 | 0.3% |
| prefill / target.prefill | qsa.select | selected_attention_batched_fp8_kernel | 12.0 | 577.052 | 50.0% |
| prefill / target.prefill | qsa.select | bf16_gemm_mma_kernel | 72.0 | 487.590 | 42.2% |
| prefill / target.prefill | qsa.select | split_query_gate_kernel | 12.0 | 24.373 | 2.1% |
| prefill / target.prefill | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 12.0 | 15.562 | 1.3% |
| prefill / target.prefill | qsa.select | score_groups_mma_kernel | 12.0 | 13.720 | 1.2% |
| prefill / target.prefill | qsa.select | rmsnorm_warp_bf16x2_kernel | 24.0 | 10.309 | 0.9% |
| prefill / target.prefill | qsa.select | hadamard_rows_kernel | 12.0 | 9.980 | 0.9% |
| prefill / target.prefill | qsa.select | select_top_groups_kernel | 12.0 | 5.204 | 0.5% |
| prefill / target.prefill | qsa.select | expand_indices_batched_kernel | 12.0 | 4.765 | 0.4% |
| prefill / target.prefill | qsa.select | rope_generic_kernel | 12.0 | 3.597 | 0.3% |
| prefill / target.prefill | qsa.select | append_cache_fp8_kernel | 12.0 | 1.629 | 0.1% |
| prefill / target.prefill | qsa.select | prepare_index_query_kernel | 12.0 | 1.181 | 0.1% |
| prefill / target.prefill | qsa.select | compress_index_groups_kernel | 12.0 | 0.172 | 0.0% |
| prefill / target.prefill | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 96.0 | 507.766 | 56.1% |
| prefill / target.prefill | moe.nvfp4 | bf16_gemm_mma_kernel | 192.0 | 210.288 | 23.2% |
| prefill / target.prefill | moe.nvfp4 | reduce_grouped_kernel | 48.0 | 105.099 | 11.6% |
| prefill / target.prefill | moe.nvfp4 | gather_quantize_routes_kernel | 48.0 | 34.877 | 3.9% |
| prefill / target.prefill | moe.nvfp4 | quantize_grouped_kernel | 48.0 | 25.737 | 2.8% |
| prefill / target.prefill | moe.nvfp4 | route_kernel | 48.0 | 12.537 | 1.4% |
| prefill / target.prefill | moe.nvfp4 | silu_and_mul_kernel | 48.0 | 4.694 | 0.5% |
| prefill / target.prefill | moe.nvfp4 | count_routes_kernel | 48.0 | 3.964 | 0.4% |
| prefill / target.prefill | moe.nvfp4 | pack_routes_kernel | 48.0 | 0.336 | 0.0% |
| prefill / target.prefill | moe.nvfp4 | make_route_jobs_kernel | 48.0 | 0.128 | 0.0% |
| prefill / target.prefill | moe.nvfp4 | scan_routes_kernel | 48.0 | 0.090 | 0.0% |
| prefill / target.prefill | moe.nvfp4 | memset | 96.0 | 0.059 | 0.0% |
| decode / target.verify | gdn.update | bf16_gemv_kernel | 3456.0 | 570.510 | 94.7% |
| decode / target.verify | gdn.update | recurrent_batch_update_kernel | 1152.0 | 19.979 | 3.3% |
| decode / target.verify | gdn.update | project_control_gating_kernel | 1152.0 | 6.448 | 1.1% |
| decode / target.verify | gdn.update | causal_conv1d_snapshot_decode_kernel | 1152.0 | 3.660 | 0.6% |
| decode / target.verify | gdn.update | rmsnorm_warp_bf16x2_kernel | 1152.0 | 2.056 | 0.3% |
| decode / target.verify | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 3072.0 | 201.911 | 63.1% |
| decode / target.verify | moe.nvfp4 | bf16_gemv_kernel | 3072.0 | 49.099 | 15.4% |
| decode / target.verify | moe.nvfp4 | bf16_shared_swiglu_decode_kernel | 1536.0 | 47.442 | 14.8% |
| decode / target.verify | moe.nvfp4 | route_kernel | 1536.0 | 13.245 | 4.1% |
| decode / target.verify | moe.nvfp4 | quantize_decode_routes_kernel | 1536.0 | 6.331 | 2.0% |
| decode / target.verify | moe.nvfp4 | reduce_decode_routes_kernel | 1536.0 | 1.765 | 0.6% |
| decode / target.verify | qsa.select | bf16_gemv_kernel | 2304.0 | 172.978 | 82.4% |
| decode / target.verify | qsa.select | reduce_selected_attention_splits_kernel | 384.0 | 11.986 | 5.7% |
| decode / target.verify | qsa.select | selected_attention_batched_fp8_kernel | 384.0 | 7.868 | 3.7% |
| decode / target.verify | qsa.select | select_top_groups_kernel | 384.0 | 6.724 | 3.2% |
| decode / target.verify | qsa.select | score_groups_batched_kernel | 384.0 | 2.836 | 1.4% |
| decode / target.verify | qsa.select | append_cache_fp8_kernel | 384.0 | 1.905 | 0.9% |
| decode / target.verify | qsa.select | rope_generic_kernel | 384.0 | 1.632 | 0.8% |
| decode / target.verify | qsa.select | rmsnorm_warp_bf16x2_kernel | 768.0 | 1.049 | 0.5% |
| decode / target.verify | qsa.select | prepare_index_query_kernel | 384.0 | 0.772 | 0.4% |
| decode / target.verify | qsa.select | compress_index_groups_kernel | 384.0 | 0.656 | 0.3% |
| decode / target.verify | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 384.0 | 0.630 | 0.3% |
| decode / target.verify | qsa.select | expand_indices_batched_kernel | 384.0 | 0.560 | 0.3% |
| decode / target.verify | qsa.select | hadamard_rows_kernel | 384.0 | 0.419 | 0.2% |
| decode / target.verify | hyper.combine_mix | bf16_gemv_kernel | 6016.0 | 180.324 | 92.8% |
| decode / target.verify | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 3008.0 | 9.046 | 4.7% |
| decode / target.verify | hyper.combine_mix | gate_mix_injection_decode_kernel | 3008.0 | 5.023 | 2.6% |
| prefill / target.prefill | ple.prefill | bf16_gemm_mma_kernel | 2.0 | 28.204 | 70.9% |
| prefill / target.prefill | ple.prefill | dilated_conv_kernel | 1.0 | 7.592 | 19.1% |
| prefill / target.prefill | ple.prefill | gate_kernel | 1.0 | 2.250 | 5.7% |
| prefill / target.prefill | ple.prefill | grouped_norm_kernel | 1.0 | 1.390 | 3.5% |
| prefill / target.prefill | ple.prefill | dequantize_embedding_kernel | 1.0 | 0.340 | 0.9% |
| prefill / target.prefill | hyper.mix | bf16_gemm_mma_kernel | 4.0 | 17.049 | 72.7% |
| prefill / target.prefill | hyper.mix | gate_mix_injection_kernel | 2.0 | 3.408 | 14.5% |
| prefill / target.prefill | hyper.mix | grouped_rmsnorm_kernel | 2.0 | 2.937 | 12.5% |
| prefill / target.prefill | hyper.mix | scaled_silu_kernel | 2.0 | 0.061 | 0.3% |
| prefill / target.prefill | hyper.mix | bf16_gemm_mma_kernel | 2.0 | 8.426 | 73.0% |
| prefill / target.prefill | hyper.mix | gate_mix_kernel | 1.0 | 1.647 | 14.3% |
| prefill / target.prefill | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 1.434 | 12.4% |
| prefill / target.prefill | hyper.mix | scaled_silu_kernel | 1.0 | 0.029 | 0.3% |
| decode / target.verify | ple.update | bf16_gemv_kernel | 64.0 | 9.173 | 92.3% |
| decode / target.verify | ple.update | gate_kernel | 32.0 | 0.268 | 2.7% |
| decode / target.verify | ple.update | grouped_norm_kernel | 32.0 | 0.196 | 2.0% |
| decode / target.verify | ple.update | dequantize_embedding_kernel | 32.0 | 0.184 | 1.9% |
| decode / target.verify | ple.update | dilated_conv_batch_kernel | 32.0 | 0.112 | 1.1% |
| decode / target.verify | hyper.mix | bf16_gemv_kernel | 128.0 | 3.800 | 92.1% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 64.0 | 0.219 | 5.3% |
| decode / target.verify | hyper.mix | gate_mix_injection_decode_kernel | 64.0 | 0.108 | 2.6% |
| prefill / target.prefill | hyper.combine | combine_kernel | 2.0 | 4.079 | 100.0% |
| decode / target.verify | hyper.mix | bf16_gemv_kernel | 64.0 | 1.829 | 94.6% |
| decode / target.verify | hyper.mix | gate_mix_kernel | 32.0 | 0.056 | 2.9% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 32.0 | 0.048 | 2.5% |
| prefill / target.prefill | hyper.repeat | repeat_kernel | 1.0 | 0.991 | 100.0% |
| decode / target.verify | hyper.combine | combine_kernel | 64.0 | 0.105 | 100.0% |
| decode / target.verify | hyper.repeat | repeat_kernel | 32.0 | 0.038 | 100.0% |

Host lookup work (may overlap GPU execution):

- ple.gather: 73.826 ms across 33 calls.
- ple.hash: 0.124 ms across 33 calls.

- Estimates are partial cold useful-work models, not distance from optimal.
- Token counts are execution envelopes; padded columns are not committed outputs.
- QSA estimates cover projections only; device-dependent scoring/attention is unmodeled.
- GDN prefill estimates omit chunked recurrence compute; PLE estimates omit convolution/state traffic.
- DRAM cache reuse, scratch traffic, nonlinear math and instruction geometry are unmodeled.
- Profiler overhead affects timings; use unprofiled paired benchmarks for speed claims.
- Unattributed GPU work is retained; no whole-model efficiency is reported.

### mtp2


Measured wall: 6278.802 ms; GPU busy: 6174.914 ms; attributed GPU work: 97.83%.

Estimates describe partial modeled work, not an achievable optimum. Bytes/FLOPs below are per call; time and share are aggregated.

| Phase / role | Stage | T / B / context envelope | Calls | GPU work ms | Phase share | Useful bytes min–max | BF16 / NVFP4 / FP32 FLOPs | Estimate ms min–max | Estimate efficiency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prefill / target.prefill | gdn.prefill | 8192 / 1 / 0 | 36 | 1872.555 | 34.2% | 206012416–206012416 | 948919336960 / 0 / 0 | 273.289–273.289 | 14.6–14.6% |
| prefill / target.prefill | hyper.combine_mix | 8192 / 0 / 0 | 94 | 1209.249 | 22.1% | 432750592–432750592 | 108045271040 / 0 / 0 | 165.360–165.360 | 13.7–13.7% |
| prefill / target.prefill | qsa.select | 8192 / 1 / 8192 | 12 | 1155.283 | 21.1% | 186777600–186777600 | 842887331840 / 0 / 0 | 80.917–80.917 | 7.0–7.0% |
| prefill / target.prefill | moe.nvfp4 | 8192 / 0 / 0 | 48 | 912.582 | 16.7% | 123991200–1511928832 | 102047416320 / 805306368000 / 0 | 77.309–295.011 | 8.5–32.3% |
| decode / target.verify | gdn.record | 3 / 1 / 0 | 396 | 215.082 | 31.0% | 119123072–119123072 | 347504640 / 0 / 14155776 | 191.759–191.759 | 89.2–89.2% |
| decode / target.verify | moe.nvfp4 | 3 / 0 / 0 | 528 | 205.354 | 29.6% | 40135840–95432160 | 37370880 / 294912000 / 0 | 86.145–204.830 | 42.0–99.8% |
| prefill / predictor | qsa.select | 8192 / 1 / 8192 | 1 | 106.186 | 1.9% | 186777600–186777600 | 842887331840 / 0 / 0 | 6.743–6.743 | 6.4–6.4% |
| decode / target.verify | qsa.select | 3 / 1 / 8194 | 132 | 79.710 | 11.5% | 102922240–102922240 | 308674560 / 0 / 0 | 55.227–55.227 | 69.3–69.3% |
| decode / target.verify | hyper.combine_mix | 3 / 0 / 0 | 1034 | 71.595 | 10.3% | 13342768–13342768 | 39567360 / 0 / 0 | 56.083–56.083 | 78.4–78.4% |
| prefill / predictor | moe.bf16 | 8192 / 0 / 0 | 1 | 49.786 | 0.9% | 194647040–5129507840 | 907353784320 / 0 / 0 | 7.259–20.852 | 14.6–41.9% |
| prefill / target.prefill | ple.prefill | 8192 / 1 / 0 | 1 | 39.706 | 0.7% | 422051840–422051840 | 536870912000 / 0 / 0 | 4.295–4.295 | 10.8–10.8% |
| prefill / target.prefill | hyper.mix | 8192 / 0 / 0 | 2 | 23.558 | 0.4% | 222969856–222969856 | 108045271040 / 0 / 0 | 1.813–1.813 | 7.7–7.7% |
| decode / predictor | moe.bf16 | 3 / 0 / 0 | 11 | 13.653 | 2.0% | 110791680–307399680 | 332282880 / 0 / 0 | 4.954–13.746 | 36.3–100.7% |
| prefill / predictor | hyper.combine_mix | 8192 / 0 / 0 | 1 | 12.658 | 0.2% | 432750592–432750592 | 108045271040 / 0 / 0 | 1.759–1.759 | 13.9–13.9% |
| prefill / predictor | hyper.mix | 8192 / 0 / 0 | 1 | 12.372 | 0.2% | 222822400–222822400 | 107374182400 / 0 / 0 | 0.906–0.906 | 7.3–7.3% |
| prefill / target.prefill | hyper.mix | 8192 / 0 / 0 | 1 | 12.088 | 0.2% | 222822400–222822400 | 107374182400 / 0 / 0 | 0.906–0.906 | 7.5–7.5% |
| prefill / predictor | hyper.mix | 8192 / 0 / 0 | 1 | 12.004 | 0.2% | 222969856–222969856 | 108045271040 / 0 / 0 | 0.906–0.906 | 7.6–7.6% |
| decode / predictor | qsa.select | 3 / 1 / 8194 | 11 | 6.765 | 1.0% | 102922240–102922240 | 308674560 / 0 / 0 | 4.602–4.602 | 68.1–68.1% |
| decode / predictor | qsa.reuse | 1 / 1 / 8195 | 11 | 5.232 | 0.8% | 100280320–100280320 | 100270080 / 0 / 0 | 4.484–4.484 | 85.8–85.8% |
| decode / predictor | moe.bf16 | 1 / 0 / 0 | 11 | 5.195 | 0.7% | 110771200–110771200 | 110760960 / 0 / 0 | 4.953–4.953 | 95.4–95.4% |
| prefill / target.prefill | hyper.combine | 8192 / 0 / 0 | 2 | 3.895 | 0.1% | 377552896–377552896 | 0 / 0 / 0 | 3.070–3.070 | 78.8–78.8% |
| decode / target.verify | ple.record | 3 / 1 / 0 | 11 | 3.485 | 0.5% | 65666560–65666560 | 196608000 / 0 / 0 | 2.936–2.936 | 84.3–84.3% |
| prefill / predictor | hyper.add | 8192 / 0 / 0 | 1 | 2.301 | 0.0% | 377487360–377487360 | 0 / 0 / 0 | 1.535–1.535 | 66.7–66.7% |
| prefill / predictor | hyper.combine | 8192 / 0 / 0 | 1 | 1.864 | 0.0% | 377552896–377552896 | 0 / 0 / 0 | 1.535–1.535 | 82.3–82.3% |
| decode / target.verify | hyper.mix | 3 / 0 / 0 | 22 | 1.573 | 0.2% | 13265944–13265944 | 39567360 / 0 / 0 | 1.186–1.186 | 75.4–75.4% |
| prefill / target.prefill | hyper.repeat | 8192 / 0 / 0 | 1 | 1.036 | 0.0% | 209715200–209715200 | 0 / 0 / 0 | 0.853–0.853 | 82.3–82.3% |
| decode / predictor | hyper.mix | 3 / 0 / 0 | 11 | 0.836 | 0.1% | 13265944–13265944 | 39567360 / 0 / 0 | 0.593–0.593 | 71.0–71.0% |
| decode / predictor | hyper.combine_mix | 3 / 0 / 0 | 11 | 0.829 | 0.1% | 13342768–13342768 | 39567360 / 0 / 0 | 0.597–0.597 | 72.1–72.1% |
| decode / predictor | hyper.mix | 3 / 0 / 0 | 11 | 0.780 | 0.1% | 13184000–13184000 | 39321600 / 0 / 0 | 0.590–0.590 | 75.6–75.6% |
| decode / target.verify | hyper.mix | 3 / 0 / 0 | 11 | 0.767 | 0.1% | 13184000–13184000 | 39321600 / 0 / 0 | 0.590–0.590 | 76.9–76.9% |
| decode / predictor | hyper.combine_mix | 1 / 0 / 0 | 11 | 0.666 | 0.1% | 13240336–13240336 | 13189120 / 0 / 0 | 0.592–0.592 | 89.9–89.9% |
| decode / predictor | hyper.mix | 1 / 0 / 0 | 11 | 0.653 | 0.1% | 13214728–13214728 | 13189120 / 0 / 0 | 0.591–0.591 | 90.5–90.5% |
| decode / predictor | hyper.mix | 1 / 0 / 0 | 11 | 0.635 | 0.1% | 13132800–13132800 | 13107200 / 0 / 0 | 0.587–0.587 | 92.5–92.5% |
| prefill / predictor | qsa.select | 1 / 1 / 8193 | 1 | 0.525 | 0.0% | 102901760–102901760 | 102891520 / 0 / 0 | 0.418–0.418 | 79.7–79.7% |
| prefill / predictor | moe.bf16 | 1 / 0 / 0 | 1 | 0.453 | 0.0% | 110771200–110771200 | 110760960 / 0 / 0 | 0.450–0.450 | 99.3–99.3% |
| prefill / predictor | hyper.mix | 1 / 0 / 0 | 1 | 0.095 | 0.0% | 13214728–13214728 | 13189120 / 0 / 0 | 0.054–0.054 | 56.3–56.3% |
| prefill / predictor | hyper.combine_mix | 1 / 0 / 0 | 1 | 0.063 | 0.0% | 13240336–13240336 | 13189120 / 0 / 0 | 0.054–0.054 | 85.9–85.9% |
| prefill / predictor | hyper.mix | 1 / 0 / 0 | 1 | 0.056 | 0.0% | 13132800–13132800 | 13107200 / 0 / 0 | 0.053–0.053 | 94.9–94.9% |
| decode / target.verify | hyper.combine | 3 / 0 / 0 | 22 | 0.048 | 0.0% | 138264–138264 | 0 / 0 / 0 | 0.012–0.012 | 25.7–25.7% |
| decode / predictor | hyper.combine | 3 / 0 / 0 | 11 | 0.036 | 0.0% | 138264–138264 | 0 / 0 / 0 | 0.006–0.006 | 17.2–17.2% |
| decode / predictor | hyper.combine | 1 / 0 / 0 | 11 | 0.024 | 0.0% | 46088–46088 | 0 / 0 / 0 | 0.002–0.002 | 8.7–8.7% |
| decode / predictor | hyper.add | 3 / 0 / 0 | 11 | 0.020 | 0.0% | 138240–138240 | 0 / 0 / 0 | 0.006–0.006 | 30.7–30.7% |
| decode / target.verify | hyper.repeat | 3 / 0 / 0 | 11 | 0.016 | 0.0% | 76800–76800 | 0 / 0 / 0 | 0.003–0.003 | 21.8–21.8% |
| decode / predictor | hyper.add | 1 / 0 / 0 | 11 | 0.013 | 0.0% | 46080–46080 | 0 / 0 / 0 | 0.002–0.002 | 15.6–15.6% |
| prefill / predictor | hyper.combine | 1 / 0 / 0 | 1 | 0.002 | 0.0% | 46088–46088 | 0 / 0 / 0 | 0.000–0.000 | 9.9–9.9% |
| prefill / predictor | hyper.add | 1 / 0 / 0 | 1 | 0.001 | 0.0% | 46080–46080 | 0 / 0 / 0 | 0.000–0.000 | 17.2–17.2% |

Unattributed GPU work: 134.211 ms.

Kernels per stage (ms, launches per capture):

| Phase / role | Stage | Kernel | Launches | GPU work ms | Stage share |
|---|---|---|---:|---:|---:|
| prefill / target.prefill | gdn.prefill | bf16_gemm_mma_kernel | 180.0 | 1679.218 | 89.7% |
| prefill / target.prefill | gdn.prefill | chunk_recurrence_kernel | 36.0 | 59.404 | 3.2% |
| prefill / target.prefill | gdn.prefill | causal_conv1d_prefill_pairs_kernel | 36.0 | 50.855 | 2.7% |
| prefill / target.prefill | gdn.prefill | rmsnorm_warp_bf16x2_kernel | 36.0 | 43.695 | 2.3% |
| prefill / target.prefill | gdn.prefill | prepare_kernel | 36.0 | 38.948 | 2.1% |
| prefill / target.prefill | gdn.prefill | gdn_gating_kernel | 36.0 | 0.319 | 0.0% |
| prefill / target.prefill | gdn.prefill | causal_conv1d_prefill_state_kernel | 36.0 | 0.115 | 0.0% |
| prefill / target.prefill | hyper.combine_mix | bf16_gemm_mma_kernel | 188.0 | 832.013 | 68.8% |
| prefill / target.prefill | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 94.0 | 217.010 | 17.9% |
| prefill / target.prefill | hyper.combine_mix | gate_mix_injection_kernel | 94.0 | 157.092 | 13.0% |
| prefill / target.prefill | hyper.combine_mix | scaled_silu_kernel | 94.0 | 3.134 | 0.3% |
| prefill / target.prefill | qsa.select | selected_attention_batched_fp8_kernel | 12.0 | 578.362 | 50.1% |
| prefill / target.prefill | qsa.select | bf16_gemm_mma_kernel | 72.0 | 486.497 | 42.1% |
| prefill / target.prefill | qsa.select | split_query_gate_kernel | 12.0 | 23.400 | 2.0% |
| prefill / target.prefill | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 12.0 | 15.891 | 1.4% |
| prefill / target.prefill | qsa.select | score_groups_mma_kernel | 12.0 | 13.878 | 1.2% |
| prefill / target.prefill | qsa.select | rmsnorm_warp_bf16x2_kernel | 24.0 | 10.344 | 0.9% |
| prefill / target.prefill | qsa.select | hadamard_rows_kernel | 12.0 | 10.023 | 0.9% |
| prefill / target.prefill | qsa.select | select_top_groups_kernel | 12.0 | 5.208 | 0.5% |
| prefill / target.prefill | qsa.select | expand_indices_batched_kernel | 12.0 | 4.653 | 0.4% |
| prefill / target.prefill | qsa.select | rope_generic_kernel | 12.0 | 3.646 | 0.3% |
| prefill / target.prefill | qsa.select | append_cache_fp8_kernel | 12.0 | 1.667 | 0.1% |
| prefill / target.prefill | qsa.select | prepare_index_query_kernel | 12.0 | 1.536 | 0.1% |
| prefill / target.prefill | qsa.select | compress_index_groups_kernel | 12.0 | 0.178 | 0.0% |
| prefill / target.prefill | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 96.0 | 507.238 | 55.6% |
| prefill / target.prefill | moe.nvfp4 | bf16_gemm_mma_kernel | 192.0 | 213.188 | 23.4% |
| prefill / target.prefill | moe.nvfp4 | reduce_grouped_kernel | 48.0 | 110.770 | 12.1% |
| prefill / target.prefill | moe.nvfp4 | gather_quantize_routes_kernel | 48.0 | 33.511 | 3.7% |
| prefill / target.prefill | moe.nvfp4 | quantize_grouped_kernel | 48.0 | 25.946 | 2.8% |
| prefill / target.prefill | moe.nvfp4 | route_kernel | 48.0 | 12.948 | 1.4% |
| prefill / target.prefill | moe.nvfp4 | silu_and_mul_kernel | 48.0 | 4.582 | 0.5% |
| prefill / target.prefill | moe.nvfp4 | count_routes_kernel | 48.0 | 3.841 | 0.4% |
| prefill / target.prefill | moe.nvfp4 | pack_routes_kernel | 48.0 | 0.312 | 0.0% |
| prefill / target.prefill | moe.nvfp4 | make_route_jobs_kernel | 48.0 | 0.120 | 0.0% |
| prefill / target.prefill | moe.nvfp4 | scan_routes_kernel | 48.0 | 0.072 | 0.0% |
| prefill / target.prefill | moe.nvfp4 | memset | 96.0 | 0.054 | 0.0% |
| decode / target.verify | gdn.record | bf16_small_t_inner_kernel | 1188.0 | 200.487 | 93.2% |
| decode / target.verify | gdn.record | recurrent_fold_record_kernel | 396.0 | 9.620 | 4.5% |
| decode / target.verify | gdn.record | project_control_gating_kernel | 396.0 | 2.280 | 1.1% |
| decode / target.verify | gdn.record | conv_replay_record_kernel | 396.0 | 1.946 | 0.9% |
| decode / target.verify | gdn.record | rmsnorm_warp_bf16x2_kernel | 396.0 | 0.748 | 0.3% |
| decode / target.verify | moe.nvfp4 | nvfp4_w4a4_mma_kernel | 1056.0 | 161.410 | 78.6% |
| decode / target.verify | moe.nvfp4 | bf16_small_t_inner_kernel | 2112.0 | 35.498 | 17.3% |
| decode / target.verify | moe.nvfp4 | route_kernel | 528.0 | 4.686 | 2.3% |
| decode / target.verify | moe.nvfp4 | quantize_decode_routes_kernel | 528.0 | 2.167 | 1.1% |
| decode / target.verify | moe.nvfp4 | reduce_decode_routes_kernel | 528.0 | 0.825 | 0.4% |
| decode / target.verify | moe.nvfp4 | silu_and_mul_kernel | 528.0 | 0.767 | 0.4% |
| prefill / predictor | qsa.select | bf16_gemm_mma_kernel | 6.0 | 50.389 | 47.5% |
| prefill / predictor | qsa.select | selected_attention_batched_fp8_kernel | 1.0 | 48.374 | 45.6% |
| prefill / predictor | qsa.select | split_query_gate_kernel | 1.0 | 1.874 | 1.8% |
| prefill / predictor | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 1.0 | 1.306 | 1.2% |
| prefill / predictor | qsa.select | score_groups_mma_kernel | 1.0 | 1.147 | 1.1% |
| prefill / predictor | qsa.select | rmsnorm_warp_bf16x2_kernel | 2.0 | 0.838 | 0.8% |
| prefill / predictor | qsa.select | hadamard_rows_kernel | 1.0 | 0.812 | 0.8% |
| prefill / predictor | qsa.select | select_top_groups_kernel | 1.0 | 0.488 | 0.5% |
| prefill / predictor | qsa.select | expand_indices_batched_kernel | 1.0 | 0.416 | 0.4% |
| prefill / predictor | qsa.select | rope_generic_kernel | 1.0 | 0.296 | 0.3% |
| prefill / predictor | qsa.select | append_cache_fp8_kernel | 1.0 | 0.135 | 0.1% |
| prefill / predictor | qsa.select | prepare_index_query_kernel | 1.0 | 0.095 | 0.1% |
| prefill / predictor | qsa.select | compress_index_groups_kernel | 1.0 | 0.015 | 0.0% |
| decode / target.verify | qsa.select | bf16_small_t_inner_kernel | 396.0 | 58.173 | 73.0% |
| decode / target.verify | qsa.select | selected_attention_batched_fp8_kernel | 132.0 | 5.865 | 7.4% |
| decode / target.verify | qsa.select | bf16_small_t_group_kernel | 132.0 | 5.402 | 6.8% |
| decode / target.verify | qsa.select | select_top_groups_kernel | 132.0 | 2.620 | 3.3% |
| decode / target.verify | qsa.select | reduce_selected_attention_splits_kernel | 132.0 | 2.157 | 2.7% |
| decode / target.verify | qsa.select | score_groups_batched_kernel | 132.0 | 1.305 | 1.6% |
| decode / target.verify | qsa.select | order_groups_like_persistent_topk_kernel | 132.0 | 1.255 | 1.6% |
| decode / target.verify | qsa.select | append_cache_fp8_kernel | 132.0 | 0.646 | 0.8% |
| decode / target.verify | qsa.select | rope_generic_kernel | 132.0 | 0.557 | 0.7% |
| decode / target.verify | qsa.select | rmsnorm_warp_bf16x2_kernel | 264.0 | 0.368 | 0.5% |
| decode / target.verify | qsa.select | compress_index_groups_kernel | 132.0 | 0.347 | 0.4% |
| decode / target.verify | qsa.select | prepare_index_query_kernel | 132.0 | 0.258 | 0.3% |
| decode / target.verify | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 132.0 | 0.213 | 0.3% |
| decode / target.verify | qsa.select | split_query_gate_kernel | 132.0 | 0.209 | 0.3% |
| decode / target.verify | qsa.select | expand_indices_batched_kernel | 132.0 | 0.187 | 0.2% |
| decode / target.verify | qsa.select | hadamard_rows_kernel | 132.0 | 0.148 | 0.2% |
| decode / target.verify | hyper.combine_mix | bf16_small_t_inner_kernel | 2068.0 | 66.692 | 93.2% |
| decode / target.verify | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 1034.0 | 3.166 | 4.4% |
| decode / target.verify | hyper.combine_mix | gate_mix_injection_decode_kernel | 1034.0 | 1.738 | 2.4% |
| prefill / predictor | moe.bf16 | bf16_grouped_gemm_mma_kernel | 2.0 | 39.263 | 78.9% |
| prefill / predictor | moe.bf16 | bf16_gemm_mma_kernel | 4.0 | 4.282 | 8.6% |
| prefill / predictor | moe.bf16 | gather_routes_bf16_kernel | 1.0 | 2.406 | 4.8% |
| prefill / predictor | moe.bf16 | reduce_grouped_kernel | 1.0 | 2.177 | 4.4% |
| prefill / predictor | moe.bf16 | grouped_silu_kernel | 1.0 | 1.213 | 2.4% |
| prefill / predictor | moe.bf16 | route_kernel | 1.0 | 0.265 | 0.5% |
| prefill / predictor | moe.bf16 | silu_and_mul_kernel | 1.0 | 0.095 | 0.2% |
| prefill / predictor | moe.bf16 | count_routes_kernel | 1.0 | 0.079 | 0.2% |
| prefill / predictor | moe.bf16 | make_route_jobs_kernel | 1.0 | 0.003 | 0.0% |
| prefill / predictor | moe.bf16 | scan_routes_kernel | 1.0 | 0.001 | 0.0% |
| prefill / predictor | moe.bf16 | memset | 2.0 | 0.001 | 0.0% |
| prefill / target.prefill | ple.prefill | bf16_gemm_mma_kernel | 2.0 | 27.472 | 69.2% |
| prefill / target.prefill | ple.prefill | dilated_conv_kernel | 1.0 | 8.182 | 20.6% |
| prefill / target.prefill | ple.prefill | gate_kernel | 1.0 | 2.345 | 5.9% |
| prefill / target.prefill | ple.prefill | grouped_norm_kernel | 1.0 | 1.366 | 3.4% |
| prefill / target.prefill | ple.prefill | dequantize_embedding_kernel | 1.0 | 0.340 | 0.9% |
| prefill / target.prefill | hyper.mix | bf16_gemm_mma_kernel | 4.0 | 17.295 | 73.4% |
| prefill / target.prefill | hyper.mix | gate_mix_injection_kernel | 2.0 | 3.364 | 14.3% |
| prefill / target.prefill | hyper.mix | grouped_rmsnorm_kernel | 2.0 | 2.837 | 12.0% |
| prefill / target.prefill | hyper.mix | scaled_silu_kernel | 2.0 | 0.062 | 0.3% |
| decode / predictor | moe.bf16 | routed_gate_up_kernel | 11.0 | 8.623 | 63.2% |
| decode / predictor | moe.bf16 | routed_down_kernel | 11.0 | 4.108 | 30.1% |
| decode / predictor | moe.bf16 | bf16_small_t_inner_kernel | 44.0 | 0.775 | 5.7% |
| decode / predictor | moe.bf16 | route_kernel | 11.0 | 0.131 | 1.0% |
| decode / predictor | moe.bf16 | silu_and_mul_kernel | 11.0 | 0.016 | 0.1% |
| prefill / predictor | hyper.combine_mix | bf16_gemm_mma_kernel | 2.0 | 8.702 | 68.7% |
| prefill / predictor | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 1.0 | 2.202 | 17.4% |
| prefill / predictor | hyper.combine_mix | gate_mix_injection_kernel | 1.0 | 1.723 | 13.6% |
| prefill / predictor | hyper.combine_mix | scaled_silu_kernel | 1.0 | 0.031 | 0.2% |
| prefill / predictor | hyper.mix | bf16_gemm_mma_kernel | 2.0 | 9.093 | 73.5% |
| prefill / predictor | hyper.mix | gate_mix_kernel | 1.0 | 1.905 | 15.4% |
| prefill / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 1.330 | 10.7% |
| prefill / predictor | hyper.mix | scaled_silu_kernel | 1.0 | 0.044 | 0.4% |
| prefill / target.prefill | hyper.mix | bf16_gemm_mma_kernel | 2.0 | 9.072 | 75.1% |
| prefill / target.prefill | hyper.mix | gate_mix_kernel | 1.0 | 1.675 | 13.9% |
| prefill / target.prefill | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 1.310 | 10.8% |
| prefill / target.prefill | hyper.mix | scaled_silu_kernel | 1.0 | 0.031 | 0.3% |
| prefill / predictor | hyper.mix | bf16_gemm_mma_kernel | 2.0 | 8.754 | 72.9% |
| prefill / predictor | hyper.mix | gate_mix_injection_kernel | 1.0 | 1.725 | 14.4% |
| prefill / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 1.494 | 12.5% |
| prefill / predictor | hyper.mix | scaled_silu_kernel | 1.0 | 0.030 | 0.3% |
| decode / predictor | qsa.select | bf16_small_t_inner_kernel | 33.0 | 4.878 | 72.1% |
| decode / predictor | qsa.select | selected_attention_batched_fp8_kernel | 11.0 | 0.469 | 6.9% |
| decode / predictor | qsa.select | bf16_small_t_group_kernel | 11.0 | 0.461 | 6.8% |
| decode / predictor | qsa.select | select_top_groups_kernel | 11.0 | 0.253 | 3.7% |
| decode / predictor | qsa.select | reduce_selected_attention_splits_kernel | 11.0 | 0.179 | 2.6% |
| decode / predictor | qsa.select | score_groups_batched_kernel | 11.0 | 0.114 | 1.7% |
| decode / predictor | qsa.select | order_groups_like_persistent_topk_kernel | 11.0 | 0.103 | 1.5% |
| decode / predictor | qsa.select | append_cache_fp8_kernel | 11.0 | 0.074 | 1.1% |
| decode / predictor | qsa.select | rmsnorm_warp_bf16x2_kernel | 22.0 | 0.062 | 0.9% |
| decode / predictor | qsa.select | rope_generic_kernel | 11.0 | 0.055 | 0.8% |
| decode / predictor | qsa.select | compress_index_groups_kernel | 11.0 | 0.029 | 0.4% |
| decode / predictor | qsa.select | prepare_index_query_kernel | 11.0 | 0.021 | 0.3% |
| decode / predictor | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 11.0 | 0.021 | 0.3% |
| decode / predictor | qsa.select | split_query_gate_kernel | 11.0 | 0.018 | 0.3% |
| decode / predictor | qsa.select | expand_indices_batched_kernel | 11.0 | 0.015 | 0.2% |
| decode / predictor | qsa.select | hadamard_rows_kernel | 11.0 | 0.013 | 0.2% |
| decode / predictor | qsa.reuse | bf16_gemv_kernel | 55.0 | 4.599 | 87.9% |
| decode / predictor | qsa.reuse | reduce_selected_attention_splits_kernel | 11.0 | 0.281 | 5.4% |
| decode / predictor | qsa.reuse | selected_attention_batched_fp8_kernel | 11.0 | 0.193 | 3.7% |
| decode / predictor | qsa.reuse | rope_generic_kernel | 11.0 | 0.047 | 0.9% |
| decode / predictor | qsa.reuse | append_cache_fp8_kernel | 11.0 | 0.034 | 0.6% |
| decode / predictor | qsa.reuse | rmsnorm_warp_bf16x2_kernel | 22.0 | 0.030 | 0.6% |
| decode / predictor | qsa.reuse | compress_index_groups_kernel | 11.0 | 0.020 | 0.4% |
| decode / predictor | qsa.reuse | sigmoid_gate_mul_bf16x8_kernel | 11.0 | 0.018 | 0.3% |
| decode / predictor | qsa.reuse | hadamard_rows_kernel | 11.0 | 0.013 | 0.2% |
| decode / predictor | moe.bf16 | routed_gate_up_kernel | 11.0 | 2.963 | 57.0% |
| decode / predictor | moe.bf16 | routed_down_kernel | 11.0 | 1.549 | 29.8% |
| decode / predictor | moe.bf16 | bf16_shared_swiglu_decode_kernel | 11.0 | 0.311 | 6.0% |
| decode / predictor | moe.bf16 | bf16_gemv_kernel | 22.0 | 0.288 | 5.5% |
| decode / predictor | moe.bf16 | route_kernel | 11.0 | 0.083 | 1.6% |
| prefill / target.prefill | hyper.combine | combine_kernel | 2.0 | 3.895 | 100.0% |
| decode / target.verify | ple.record | bf16_gemv_columns_kernel | 22.0 | 3.243 | 93.1% |
| decode / target.verify | ple.record | gate_kernel | 11.0 | 0.091 | 2.6% |
| decode / target.verify | ple.record | grouped_norm_kernel | 11.0 | 0.066 | 1.9% |
| decode / target.verify | ple.record | dilated_conv_record_kernel | 11.0 | 0.045 | 1.3% |
| decode / target.verify | ple.record | dequantize_embedding_kernel | 11.0 | 0.039 | 1.1% |
| prefill / predictor | hyper.add | add_repeated_kernel | 1.0 | 2.301 | 100.0% |
| prefill / predictor | hyper.combine | combine_kernel | 1.0 | 1.864 | 100.0% |
| decode / target.verify | hyper.mix | bf16_small_t_inner_kernel | 44.0 | 1.457 | 92.6% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 22.0 | 0.075 | 4.8% |
| decode / target.verify | hyper.mix | gate_mix_injection_decode_kernel | 22.0 | 0.041 | 2.6% |
| prefill / target.prefill | hyper.repeat | repeat_kernel | 1.0 | 1.036 | 100.0% |
| decode / predictor | hyper.mix | bf16_small_t_inner_kernel | 22.0 | 0.763 | 91.3% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 11.0 | 0.053 | 6.4% |
| decode / predictor | hyper.mix | gate_mix_injection_decode_kernel | 11.0 | 0.019 | 2.3% |
| decode / predictor | hyper.combine_mix | bf16_small_t_inner_kernel | 22.0 | 0.768 | 92.7% |
| decode / predictor | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 11.0 | 0.042 | 5.1% |
| decode / predictor | hyper.combine_mix | gate_mix_injection_decode_kernel | 11.0 | 0.018 | 2.2% |
| decode / predictor | hyper.mix | bf16_small_t_inner_kernel | 22.0 | 0.714 | 91.5% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 11.0 | 0.038 | 4.9% |
| decode / predictor | hyper.mix | gate_mix_kernel | 11.0 | 0.028 | 3.6% |
| decode / target.verify | hyper.mix | bf16_small_t_inner_kernel | 22.0 | 0.727 | 94.9% |
| decode / target.verify | hyper.mix | grouped_rmsnorm_kernel | 11.0 | 0.021 | 2.8% |
| decode / target.verify | hyper.mix | gate_mix_kernel | 11.0 | 0.018 | 2.4% |
| decode / predictor | hyper.combine_mix | bf16_gemv_kernel | 22.0 | 0.606 | 91.0% |
| decode / predictor | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 11.0 | 0.042 | 6.3% |
| decode / predictor | hyper.combine_mix | gate_mix_injection_decode_kernel | 11.0 | 0.018 | 2.7% |
| decode / predictor | hyper.mix | bf16_gemv_kernel | 22.0 | 0.604 | 92.5% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 11.0 | 0.025 | 3.8% |
| decode / predictor | hyper.mix | gate_mix_injection_decode_kernel | 11.0 | 0.024 | 3.7% |
| decode / predictor | hyper.mix | bf16_gemv_kernel | 22.0 | 0.593 | 93.4% |
| decode / predictor | hyper.mix | gate_mix_kernel | 11.0 | 0.026 | 4.0% |
| decode / predictor | hyper.mix | grouped_rmsnorm_kernel | 11.0 | 0.017 | 2.6% |
| prefill / predictor | qsa.select | bf16_gemv_kernel | 6.0 | 0.427 | 81.4% |
| prefill / predictor | qsa.select | reduce_selected_attention_splits_kernel | 1.0 | 0.031 | 5.9% |
| prefill / predictor | qsa.select | select_top_groups_kernel | 1.0 | 0.025 | 4.7% |
| prefill / predictor | qsa.select | selected_attention_batched_fp8_kernel | 1.0 | 0.020 | 3.8% |
| prefill / predictor | qsa.select | score_groups_batched_kernel | 1.0 | 0.005 | 0.9% |
| prefill / predictor | qsa.select | append_cache_fp8_kernel | 1.0 | 0.004 | 0.8% |
| prefill / predictor | qsa.select | rope_generic_kernel | 1.0 | 0.004 | 0.8% |
| prefill / predictor | qsa.select | rmsnorm_warp_bf16x2_kernel | 2.0 | 0.003 | 0.5% |
| prefill / predictor | qsa.select | prepare_index_query_kernel | 1.0 | 0.002 | 0.4% |
| prefill / predictor | qsa.select | sigmoid_gate_mul_bf16x8_kernel | 1.0 | 0.001 | 0.3% |
| prefill / predictor | qsa.select | expand_indices_batched_kernel | 1.0 | 0.001 | 0.2% |
| prefill / predictor | qsa.select | hadamard_rows_kernel | 1.0 | 0.001 | 0.2% |
| prefill / predictor | qsa.select | compress_index_groups_kernel | 1.0 | 0.001 | 0.2% |
| prefill / predictor | moe.bf16 | routed_gate_up_kernel | 1.0 | 0.265 | 58.5% |
| prefill / predictor | moe.bf16 | routed_down_kernel | 1.0 | 0.129 | 28.4% |
| prefill / predictor | moe.bf16 | bf16_shared_swiglu_decode_kernel | 1.0 | 0.027 | 5.9% |
| prefill / predictor | moe.bf16 | bf16_gemv_kernel | 2.0 | 0.025 | 5.6% |
| prefill / predictor | moe.bf16 | route_kernel | 1.0 | 0.007 | 1.6% |
| prefill / predictor | hyper.mix | bf16_gemv_kernel | 2.0 | 0.082 | 86.2% |
| prefill / predictor | hyper.mix | gate_mix_injection_decode_kernel | 1.0 | 0.007 | 7.4% |
| prefill / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.006 | 6.4% |
| prefill / predictor | hyper.combine_mix | bf16_gemv_kernel | 2.0 | 0.058 | 93.3% |
| prefill / predictor | hyper.combine_mix | combine_grouped_rmsnorm_kernel | 1.0 | 0.003 | 4.1% |
| prefill / predictor | hyper.combine_mix | gate_mix_injection_decode_kernel | 1.0 | 0.002 | 2.6% |
| prefill / predictor | hyper.mix | bf16_gemv_kernel | 2.0 | 0.053 | 94.5% |
| prefill / predictor | hyper.mix | gate_mix_kernel | 1.0 | 0.002 | 3.0% |
| prefill / predictor | hyper.mix | grouped_rmsnorm_kernel | 1.0 | 0.001 | 2.5% |
| decode / target.verify | hyper.combine | combine_kernel | 22.0 | 0.048 | 100.0% |
| decode / predictor | hyper.combine | combine_kernel | 11.0 | 0.036 | 100.0% |
| decode / predictor | hyper.combine | combine_kernel | 11.0 | 0.024 | 100.0% |
| decode / predictor | hyper.add | add_repeated_kernel | 11.0 | 0.020 | 100.0% |
| decode / target.verify | hyper.repeat | repeat_kernel | 11.0 | 0.016 | 100.0% |
| decode / predictor | hyper.add | add_repeated_kernel | 11.0 | 0.013 | 100.0% |
| prefill / predictor | hyper.combine | combine_kernel | 1.0 | 0.002 | 100.0% |
| prefill / predictor | hyper.add | add_repeated_kernel | 1.0 | 0.001 | 100.0% |

Host lookup work (may overlap GPU execution):

- ple.gather: 90.943 ms across 12 calls.
- ple.hash: 0.102 ms across 1 calls.

- Estimates are partial cold useful-work models, not distance from optimal.
- Token counts are execution envelopes; padded columns are not committed outputs.
- QSA estimates cover projections only; device-dependent scoring/attention is unmodeled.
- GDN prefill estimates omit chunked recurrence compute; PLE estimates omit convolution/state traffic.
- DRAM cache reuse, scratch traffic, nonlinear math and instruction geometry are unmodeled.
- Profiler overhead affects timings; use unprofiled paired benchmarks for speed claims.
- Unattributed GPU work is retained; no whole-model efficiency is reported.
- An estimate exceeds measured time: inspect cache reuse, envelope work and hardware rates; percentages are not clamped.

### PLE residency (unannotated build, 8K prompt, 512 decode outputs, 1 warmup + 3 runs)

System-wide major page faults during the run: 1 over 138s (includes model load)

```
[ninfer_bench] test 1/1 pp8192+tg512: warmup=1 reps=3
  config:     max_context=16384 prefill_chunk=8192 concurrency=1 constraint=none kv_cache=fp8-e4m3-row256 spec=none draft_tokens=0 proposal_head=full sampling=greedy decode_path=cuda_graph graph_prime=3 outputs repetitions=3 warmup=1
test          n_prompt  n_gen      prefill t/s  decode out t/s  decode eng t/s  batch out t/s  spec acc  spec round/fb  work peak
pp8192+tg512      8192    512  1537.52 ± 0.37   21.43 ± 0.01   21.43 ± 0.01  17.56 ± 0.01       n/a            n/a   1.52 GiB
	Elapsed (wall clock) time (h:mm:ss or m:ss): 2:17.72
	Maximum resident set size (kbytes): 2082028
	Major (requiring I/O) page faults: 0

free -g before:
               total        used        free      shared  buff/cache   available
Mem:             121          10         104           0          13         111
Swap:             21           0          21
free -g after:
               total        used        free      shared  buff/cache   available
Mem:             121           9         105           0          13         111
Swap:             21           0          21
```
