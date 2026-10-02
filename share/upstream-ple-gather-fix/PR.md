**Title:** perf(flash-next): overlap PLE row page-ins in the host gather

**Target:** lkarlsund's NInfer fork, `master` at `2aa87467`.
**Head:** `baristahaus/ninfer-gb10` branch `upstream/flash-next-ple-gather-pagein` (`f46d2ae4`).
Independent of the other three fixes.

## Problem and scope

Related Issue: none

The PLE table is a 51.2 GB file mapping (320,001,536 rows × 160 B). Its row IDs are hashed
bigram/trigram lookups, spread uniformly over the file. `gather_ple_fp8` copies the rows of a call
one by one from a mapping with default advice, so:
- every row that is not page-cached is a demand page fault, and the faults are served one at a
  time, each waiting on one storage read;
- default fault readahead reads around each scattered row, which evicts other rows.

This costs time whenever the table is not fully cached: cold prefill, and decode on a host that
cannot keep 51 GB resident. On a unified-memory GB10 (121 GB shared by the weights and the page
cache), decode generating new n-grams faulted ~135 rows per round at C4. That is 14.4 ms per
gather, all spent waiting on the device.

## Implementation

- **`artifact::MappedRange`** gains two hints. Both are best effort and never change contents:
  - `advise_random()` sets `MADV_RANDOM` once on the PLE mapping, so faults stop reading ahead;
  - `prefetch(offset, bytes)` issues `MADV_WILLNEED` on the covering pages. It starts their
    reads and does not wait.
- **`gather_ple_fp8`** prefetches the next 512 rows ahead of each copy. A call's page-ins then
  overlap instead of running one after another; the window bounds the hints in flight for a long
  prefill chunk.
- **Output:** gathered bytes are identical.

## Verification

- **On `master`:** compile-checked (`sm_121a`) against `master`'s own headers. The artifact reader
  test now also prefetches a PLE-width row across a shard boundary; it builds and passes.
- **Downstream fork (`baristahaus/ninfer-gb10`), NVIDIA GB10 (unified memory, NVMe):** the same
  change; `fp8_mtp` artifact, C4, MTP draft 1, 256-token generations, untraced. The real-artifact
  test stays bitwise identical. Kernel times are unchanged in the traced attribution: only host
  behaviour moved.

| | before | after |
|---|---:|---:|
| decode tok/s | 21.6 | 25.5 (+18%) |
| major page faults per decode round | 135.4 | 0.1 |
| host PLE gather, traced (ms per call) | 14.4 | 0.64 |
| device wait for the PLE stage, traced (ms per round) | 13.09 | 0.002 |
| prefill wall | 7.5 s | 2.5 s |

**Expected effect on a discrete GPU with large host RAM:**
- once the table is cached, decode should be unchanged;
- cold prefill, and runs where host memory is tight, should benefit.

None of this has been measured on RTX PRO 6000.
