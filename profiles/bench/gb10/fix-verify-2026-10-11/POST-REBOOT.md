# dnf post-reboot run list (2026-10-11)

## State at reboot
- dnf is OPEN. The cuBLAS GEMM-routing fix (7e2ab697) closed the 24-row crash
  (A/B at 6 K3, same server flags: crashed without the fix, served with it).
  But the production **8 K3 at a 73,728-token context still crashes at startup**
  with `cudaErrorIllegalAddress` (an async OOB caught in
  `DeviceContext::synchronize`, device.cu:182, during CUDA graph capture).
- The GPU node was left holding a dead CUDA context (96 % phantom util at
  ~17 W, no process) from the crashing serve run. A node reboot is required to
  clear it; `--gpu-reset` is refused (root + dead context).
- The crash needs BOTH a large context and >24 rows:
  - 6 K3 at the production context passes (24 rows OK).
  - 8 K3 at the production context fails (32 rows = 8 lanes x K3 verify width 4).
  - 8 K3 at max_context 512 passes.
  At 512 every graph uses the dense QSA route; at 73728 sparse-QSA graphs are
  also built (top-group select + other kernels). Likely location: sparse-QSA
  sizing for 32 rows, or the MoE routing step (a bad shared-memory read there
  also reports as an illegal address without memcheck).
- The regression test was rewritten (3a5f0030) to build the engine in the
  serving shape (73728 context, auto KV, chunk 4096, 8 lanes, K=3) and require
  8 concurrent requests to complete through MTP. It crashes at startup until
  dnf is fixed — that is the intended behavior of this regression test. It has
  only been syntax-checked, not built.

## Run list (one GPU job at a time; memcheck FIRST — each crash needs another
## reboot, so name the kernel before re-crashing)
1. `compute-sanitizer --tool memcheck --print-limit 20 build/apps/ninfer-serve
   <fp8mtp entry> --max-concurrency 8 --spec mtp --draft-tokens 3
   --max-context 73728 --kv-dtype fp8 --lm-head-draft --prefill-chunk 4096`
   Send the first report: kernel name, source line, thread/block, address, and
   the host backtrace.
2. Only if step 1 does not name a kernel: 7 K3 at 73728, to show whether
   28 rows already fail.

## After step 1
- If it points at sparse QSA: read that kernel against the 32-row geometry.
- If it points at the MoE routing kernel again: find which earlier op makes
  row N non-finite.

## Boundaries
- Flash-Next with MTP at `--max-concurrency` 7-8 stays off-limits at
  production context until dnf is closed.
- 6 is fine with the fix but has only been tested at K=3; K=5 at concurrency 4
  has not been rerun.
