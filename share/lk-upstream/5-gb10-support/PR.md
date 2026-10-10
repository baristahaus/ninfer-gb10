**Title:** GB10 (DGX Spark, sm_121a) support

**Target:** `lkarlslund/ninfer6000` `master` (`8f574ee4`).
**Head:** `baristahaus/ninfer-gb10` branch `upstream/ninfer6000/gb10-support` (four commits).
Independent of the other PRs in this set; together with the MTP RoPE fix it is what a GB10 needs to
serve Flash-Next with MTP at `--max-concurrency` 3 or more.

## Problem and scope

Related Issue: none

NVIDIA GB10 (DGX Spark) is a Blackwell `sm_121a` part with 48 SMs and 121.6 GiB of unified LPDDR5X
shared with the CPU. `master` does not build for it, and once built it fails three ways:

1. **Build gate.** CMake accepts only `120a`, and both family runtimes require compute capability
   12.0.
2. **Startup sizing.** On an integrated device `cudaMemGetInfo` counts reclaimable page cache as used.
   With the artifact cached it reports about 10 GiB free while `cudaMalloc` can obtain 104 GiB, so the
   weights check refuses the 125B model or the KV pool is planned far too small. A server restarted
   right after another one exited also sees memory still being returned.
3. **Fused HyperConnection fallback.** `fused_mix` allocates its staging, partials and square sums
   (about 25.7 KB per token) before `launch_fused_mix` finds the cooperative grid does not fit, and
   then returns false with them still held. Below 160 SMs (GB10: 48 SMs give 54 columns a CTA against
   the 16 the Up tile allows) every mix takes the general route on top of those buffers, which
   `hyperconnection_mix_workspace_capacity_bytes` does not count. The op test fails with
   `std::bad_alloc` in `finish_mix`; the full model only survives on its shared-arena slack.

A fourth change is performance, not correctness. Seven sites size a resident wave from the RTX
5090's 170 SMs (170, 510, 680, 1020, 5440). That oversubscribes GB10 3.5 times, and leaves SMs idle
on an RTX PRO 6000 (188).

## Implementation

- **`build(gb10)`:** CMake accepts `120a` or `121a`; both runtimes accept compute capability 12.0 or
  12.1. No kernel uses a feature 12.1 lacks.
- **`fix(runtime)`:** an integrated device sizes from `/proc/meminfo` `MemAvailable`, less a 6 GiB
  host reserve and the pending pinned Host KV arena. Startup first waits, up to 60 s, until
  `MemAvailable` stops rising. Discrete devices are unchanged.
- **`perf(ops)`:** `device_sm_count()` queries the device once (falling back to 170) and the seven
  sites scale from it: RMSNorm prefetch gate, RoPE wave capacity, GDN chunked output, sparse-MoE
  prefill, QSA score and the Flash-Next MoE grouped grids. Each of these kernels strides its work by
  `gridDim.x`, so only speed changes.
- **`fix(ops)`:** `fused_mix` checks `fused_mix_grid()` before allocating. Devices that take the fused
  route are unchanged.

## Verification

On NVIDIA GB10 (`sm_121a`, CUDA 13.0.88), `master` plus the first three commits plus the MTP RoPE fix:

- builds; ctest 129/135. The six that do not pass are environment or Qwen3.5 artifact gaps: one skip
  needs an explicit artifact; three need DFlash, DFlash2 or the 35B artifact, which are not on the
  machine; one compares 27B template-token goldens. The sixth is `hyperconnection`, the
  `std::bad_alloc` the fourth commit fixes.
- the Flash-Next real-artifact tests pass, `ninfer_qwen3_8_flash_next_real_test` and
  `ninfer_qwen3_8_flash_next_graph_real_test`;
- it served full load runs: 105 requests per arm, five prompt classes, at C1, C2 and C4, at K=1 and
  K=3 and with adaptive drafts.

The fourth commit is compile-checked on `master` (`sm_121a`); its op test run is part of this set's
validation run. On RTX PRO 6000 the third commit changes grid sizes (188 SMs instead of 170); it has
not run there.
