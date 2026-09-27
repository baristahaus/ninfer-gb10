# Step 6.1 — inter-round gap breakdown (mtp2.sqlite)

Question from the step-6 plan: the median between-round gap is ~1.5–1.7 ms and the PLE
gather is 9.2 µs warm (0.7%) — what fills the rest? This is the runtime-call + NVTX
breakdown of the gaps, from `profiles/bench/gb10/step3/mtp2.sqlite` (nsys capture, K=2
MTP, the two decode phases of the run).

## Method

- Gap = `start − LAG(end)` over `CUPTI_ACTIVITY_KIND_KERNEL` ordered by start; 65 gaps
  >200 µs total: 49 in the two decode phases, 11 in prefill, 5 run-boundary/transition
  gaps >3 ms excluded (graph re-instantiation between runs — `program.submit` spans ~3 s
  there).
- Per gap: the next kernel's enqueuing CUDA call is resolved exactly via
  `kernel.correlationId = runtime.correlationId`; host work is read from NVTX ranges
  (`engine.commit_output`, `program.submit`, `decode.mtp.submit/wait`, `cuda_graph.launch`,
  `ninfer.host|ple.hash/ple.gather`) and CUPTI `SYNCHRONIZATION`/`MEMCPY` rows.
- "Host portion" = next-kernel enqueue call start − gap start (g0, the moment the GPU
  went idle). Positive ⇒ the GPU sat idle waiting for the host.

## Headline

**43 of 49 decode-phase gaps are host-bound: the next kernel's enqueue call starts
0.32–2.01 ms after the GPU went idle.** Launch → kernel start is ~0–7 µs for eager
kernels; for the graph, the first kernel starts during the `cudaGraphLaunch` call.
The GPU is not the thing that is late — the host is.

| Quantity | n | mean | p50 | range |
|---|---|---|---|---|
| gap total | 49 | 1542 µs | 1444 µs | 330–2585 µs |
| host portion (g0 → enqueue), host-bound gaps | 43 | 1140 µs | 1142 µs | 324–2008 µs |
| CUPTI stream-sync return − g0 (completion lag) | 43 | 1438 µs | 1474 µs | 262–2935 µs |

Two stall-point types, cleanly separated by (prev kernel → next kernel):

| | Type A: `qsa.select → GDN` (eager) | Type B: `ple_fold → verify graph` (graph) |
|---|---|---|
| n | 22 | 21 |
| gap mean / p50 | 1357 / 1321 µs | 1872 / 1845 µs |
| enqueue via | `cudaLaunchKernel` (one eager kernel at a time) | `cudaGraphLaunch` (940–970 µs call) |
| host portion mean / p50 | 1320 / 1279 µs | 1127 / 1142 µs |
| enqueue − g0 range | 651–2008 µs | 324–1756 µs |
| first kernel start | 0–7 µs after enqueue | 110–1390 µs before the launch call returns (during it) |

## What is actually in the gap

Per-gap dumps (several examples) show the same shape for both types:

1. **g0 → sync return ≈ 0.3–2.9 ms (the dominant piece).** A CUPTI stream sync ends
   ~1.4 ms (p50 1.47 ms) *after* the last kernel it waited on has already completed.
   No GPU work completes in between (no kernel starts inside the gap). The host's
   blocking wait is returning long after GPU completion — completion-detection
   latency of the engine's wait path. Mechanism to confirm in the engine source:
   poll/sleep granularity of the device wait loop, or stream work enqueued after the
   last kernel (D2H of the round's output / accept transaction) that the sync covers.
2. **Sync return → enqueue ≈ 30–110 µs (the real work).** `engine.commit_output`
   ~8 µs, `program.submit` ~84 µs (eager rounds), `ple.hash` ~2 µs, `ple.gather`
   ~9 µs, small H2D staging (7.7 KB PLE rows + <1 KB), then the launch call.
3. **Type B extra:** the `cudaGraphLaunch` call itself runs 940–970 µs and the first
   graph kernel starts ~800 µs after the call starts (i.e. mid-call). The launch call
   duration is therefore not itself on the critical path, but the call-start →
   kernel-start interval is. A ~1 ms `cudaGraphLaunch` is worth a follow-up check
   (node count / instantiation state) if the wait-path fix alone doesn't land the gain.

PLE total is ~11 µs per round: 0.7% of the gap. Confirms Opus's correction — the 134 µs
plan figure was an averaging artifact (decode + one 8k-token prefill gather).

## Consequences for the step-6 ordering

- **Step 6.3 (device PLE gather): ~11 µs/round of steady-state throughput.** Its value
  remains the cold-tail fix (host cold gather 72 ms outlier → device faults parallel,
  ~100 µs/faulted row) and removing PLE from the host round path. Not a gap fix.
- **Step 6.2 (overlap commit/submit/PLE with the previous round): recovers the visible
  ~100–200 µs/round** (commit + submit + PLE + launch overhead, and for type B the
  pre-commit transaction that currently serializes after the round's sync returns).
- **New top target (candidate 6.0): the ~1.4 ms sync-return completion lag.** If the
  device wait loop polls with a sleep backoff, replacing it with a blocking event
  sync (or a much tighter loop) should return ~1 ms/round ≈ most of the gap, at far
  lower implementation cost than 6.2/6.3. Needs the engine's wait-path source to
  confirm the mechanism before sizing.

## Caveats

- Measured under nsys (CUPTI instrumentation on); a µs-level per-call overhead, not
  1 ms — the completion lag is not a profiling artifact.
- The matched sync ("sync ending nearest g0 within ±3 ms") is the right sync in most
  rows (enqueue 30–110 µs after its return); in a minority of rows the enqueue
  precedes the matched sync return (different sync gates the enqueue, or the match
  caught another stream's sync).
- Only gaps >200 µs are analyzed; rounds whose stall points stayed <200 µs are the
  pipelined ones (next kernel already enqueued before the GPU finished) — they show
  the wait path *can* hit completion in time, which supports the wait-granularity
  interpretation over a structural serialization.
- 5 run-boundary gaps (>3 ms, graph re-instantiation between the two measured runs)
  are excluded from all statistics.
