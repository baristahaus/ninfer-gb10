## Step 0 — build and tests

- Host: Linux 6.17.0-1029-nvidia aarch64; Ubuntu 24.04.4 LTS
- Commit: 033cb2fb (with local changes)
- nvcc: release 13.0, V13.0.88
- GPU/driver: NVIDIA GB10, 580.173.02
- Artifact: qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer, 119G (multi-volume total)
- Configure: exit 0; build: exit 0

Full ctest (exit 0):

- Result: 100% tests passed, 0 tests failed out of 140
- Failed (0): none
- Skipped (11): ninfer_qwen3_5_dflash2_real_test, ninfer_qwen3_5_dflash_real_test, ninfer_qwen3_5_loading_real_test, ninfer_qwen3_5_moe_real_test, ninfer_qwen3_5_prefix_real_test, ninfer_qwen3_5_score_real_test, ninfer_qwen3_5_vision_workspace_test, ninfer_qwen3_8_flash_next_fault_test, ninfer_qwen3_8_flash_next_frontend_test, ninfer_qwen3_8_flash_next_load_plan_test, ninfer_qwen3_8_flash_next_real_test

Flash-Next real-artifact tests (exit 0):

- Result: 100% tests passed, 0 tests failed out of 4
- Failed (0): none
- Skipped (0): none
