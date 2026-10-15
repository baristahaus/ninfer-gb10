# File-backed page GPU-read probe — 2026-10-15T01:55:34Z

scratch 512 MiB (page-cache-backed file); evictor /home/apollo11/models/qwen3_8_flash_next_v3_fork/qwen3_8_flash_next_125b_a6b_nvfp4.ninfer.part-0003 (125 GiB read); row = 2560 B
at a 4 KiB-aligned offset; row offsets deterministic (splitmix64, fixed seed).
Decision gate: device-side PLE gather (plan step 6.3) — what does a GPU read of a
file-backed page cost when resident vs. cold (NVMe page-in via the fault path)?

## phase: prep
scratch file profiles/bench/gb10/file_page_probe/scratch.bin: 536870912 bytes (131072 pages), pattern written + fsynced

## phase: warm
host warm read: 131072 pages,  5.650 ms,    95.02 GB/s (acc=17722799057980459228)

## phase: res-seq
phase res-seq: 100000 rows x 2560 B = 250000 KiB, 48 SMs
| measurement | GPU ms | wall ms |
|---|---|---|
| sequential full-file read |     3.304 ms |   290.610 ms |
effective bandwidth:   162.48 GB/s
sink sanity: 0x0

## phase: res-rows
phase res-rows: 100000 rows x 2560 B = 250000 KiB, 48 SMs
| measurement | GPU ms | wall ms |
|---|---|---|
| random 2560 B rows |     1.517 ms |   125.917 ms |
effective bandwidth: 168712.31 MB/s | per-row (amortized):     0.02 us
sink sanity: 0x0

## phase: evict
evictor /home/apollo11/models/qwen3_8_flash_next_v3_fork/qwen3_8_flash_next_125b_a6b_nvfp4.ninfer.part-0003: 128000 MiB read in   32.0 s (    4.19 GB/s, acc=1484129505) — scratch mapping should now be out of the page cache

## phase: cold-serial
phase cold-serial: 512 rows x 2560 B = 1280 KiB, 48 SMs
| measurement | GPU ms | wall ms |
|---|---|---|
| serial cold rows (1 thread) |    58.815 ms |    58.826 ms |
per-row:    114.87 us  (serialized page-fault path, NVMe page-in if cold)
sink sanity: 0x0

## phase: cold-rows
phase cold-rows: 100000 rows x 2560 B = 250000 KiB, 48 SMs
| measurement | GPU ms | wall ms |
|---|---|---|
| random 2560 B rows |   119.443 ms |   119.454 ms |
effective bandwidth:  2143.28 MB/s | per-row (amortized):     1.19 us
sink sanity: 0x0

## phase: res-rows-again
phase res-rows-again: 100000 rows x 2560 B = 250000 KiB, 48 SMs
| measurement | GPU ms | wall ms |
|---|---|---|
| random 2560 B rows |     1.498 ms |   116.115 ms |
effective bandwidth: 170903.65 MB/s | per-row (amortized):     0.01 us
sink sanity: 0x0

## cleanup
scratch removed
