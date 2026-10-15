## GB10 memory probe

- Device: NVIDIA GB10, compute capability 12.1, 48 SMs, L2 24.0 MiB, bus width 256 bits
- CUDA driver 13000, runtime 13000; buffer 1.00 GiB; 20 CPU threads; 0.50 s per measurement

### 1. What CUDA reports

| Attribute | Value |
|---|---:|
| integrated | 1 |
| pageableMemoryAccess | 1 |
| pageableMemoryAccessUsesHostPageTables | 1 |
| concurrentManagedAccess | 1 |
| directManagedMemAccessFromHost | 0 |
| hostNativeAtomicSupported | 1 |
| canUseHostPointerForRegisteredMem | 1 |
| unifiedAddressing | 1 |
| memoryPoolsSupported | 1 |
| GENERIC_COMPRESSION_SUPPORTED (driver API) | 1 |

### 2. Generic compression (cuMemCreate)

| Allocation | Compression granted | Read GB/s, zero-filled | Read GB/s, random |
|---|---|---:|---:|
| plain | no | 236.1 | 236.3 |
| compression requested | yes | 2066.2 | 232.8 |

Compression is doing something only if the granted, zero-filled row reads clearly faster than the plain rows.

### 2b. Generic compression on weight samples

Each sample is the tensor's leading bytes, repeated to fill the 1.00 GiB buffer.

| Sample | Sample MiB | Plain GB/s | Compressed GB/s | Gain |
|---|---:|---:|---:|---:|
| bf16-gdn-in_proj_qkv | 50.0 | 237.0 | 236.7 | 1.00x |
| bf16-hc-input_mix_down | 6.2 | 236.7 | 234.7 | 0.99x |
| bf16-lm_head | 64.0 | 236.9 | 236.3 | 1.00x |
| nvfp4-experts-gate_up | 64.0 | 235.8 | 234.8 | 1.00x |
| fp8-ple-table | 64.0 | 236.9 | 236.0 | 1.00x |
| q4-proposal-head | 64.0 | 236.1 | 235.2 | 1.00x |

### 3. GPU read bandwidth by memory kind

| Source | GB/s |
|---|---:|
| cudaMalloc | 236.0 |
| cudaHostAlloc (pinned, mapped), read in place | 209.0 |
| malloc (pageable), read in place | 158.2 |
| cudaMemcpy H2D from pinned (payload; DRAM traffic is ~2x) | 59.3 |
| cudaMemcpy H2D from pageable (payload) | 59.2 |

### 4. CPU bandwidth and CPU/GPU contention

| Measurement | GB/s |
|---|---:|
| CPU read alone (20 threads) | 132.3 |
| GPU read alone | 234.4 |
| GPU read while CPU reads | 170.0 |
| CPU read while GPU reads (whole window, includes 100 ms CPU-only lead-in) | 64.6 |
| Sum while contending | 234.6 |

