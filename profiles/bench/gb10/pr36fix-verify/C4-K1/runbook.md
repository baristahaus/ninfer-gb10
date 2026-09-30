# C4 K=1 A/B runbook — PR #36 speed decision (2026-09-30)

Verbatim brief from Opus (via the user), plus the machine mapping used to run it.

## Brief

**PR #36: decide the speed question (the C4 K=1 A/B)**

**Why.** The C-DECODE run can't decide whether #36 is faster or slower:
- **Run order:** the fix ran first with a cold page cache (82 % host time during its first
  prefill against 13.8 % on the base, and #36 changes no host code).
- **Batch widths differed:** the base stalled at 2 running with 6 waiting for about 15 s.
- **Coverage:** at 8 requests with draft 2, each round verifies 24 rows. That path is
  identical on both trees; #36 only changes rounds of 16 rows or fewer.

Correctness is settled; only speed is open.

**Trees.**
- Fix = `a1a43667` (engine code the same as `eb9e87fa`).
- Base = `ed6525fa`.
- Artifact: fp8_mtp.

**Workload.** The same one as I9 NInfer K=1, which Block J profiled:
- `ninfer-serve --max-context 73728 --max-concurrency 4 --kv-dtype fp8 --preserve-thinking
  --spec mtp --draft-tokens 1 --lm-head-draft --request-log-jsonl <out>.jsonl`
- Load: DGPP's `scripts/serve_load.py 127.0.0.1 <port> --concurrency 4 --classes prose
  --max-tokens 256 --repeat 3 --json-out <out>.json`
- This gives 4 verify rows + 4 draft rows per round, the case whose 5.7 ms QSA stage
  motivated the change.

**Protocol.**
1. Run in the order fix, base, base, fix, restarting the server for each run. One GPU job
   at a time.
2. After each server start, do one throwaway warm-up pass of the same load before the
   measured pass. This warms the page cache for the file-backed PLE rows.
3. Don't trace the measured passes.
4. Afterwards, run one nsys pair: fix and base, one rep each, same setup as Block J. Report
   the attention + QSA-selection stage time per step at 8 rows. The base value was 5.7 ms,
   against 0.6 for DGPP.

**Report.**
- For each of the 4 runs:
  - aggregate tok/s;
  - per-round device wait and host exposure (`request_log_summary.py` on the jsonl);
  - accepted tokens per round.
- Mean and spread for each tree.
- The attention/QSA stage time from the nsys pair.
- **Width check:** confirm from the serve logs that each measured pass ran 4 requests at
  once (look for `running 4`). If a pass shows `running 2 | waiting 2`, flag it and include
  that log. That's the admission stall below, and it invalidates that pass.

**Cache check (quick, before the A/B).** Restart the fix server with a warm page cache and
repeat the 1-request run from C-DECODE. If prefill for the 1,882-token request drops from
4.84 s to about 1.3 s, the run-order explanation holds.

**Admission stall (only if time allows, after the A/B).**
- **Seen:** in the C-DECODE base log (19:37:09–19:37:19), the server held `running 2 |
  waiting 6` with no prefill until the response-replay request finished.
- **Setup:** base tree, the same server flags as C-DECODE, 8 requests.
- **Reproduce:**
  1. Send one request.
  2. Once it finishes, send 8 at once, with the first request's prompt placed second so the
     replay isn't admitted first.
- **Capture:** the serve log (at debug level if available) plus the request jsonl.
- **Expected:**
  - If the stall reproduces, admission is being held by a state fork that never settles
    during MTP decode.
  - Report which request held it and for how long.

**What happens next.**
- If the fix is at parity or better and the QSA stage drops substantially: merge #36.
- If it's worse: I rework the split sizing before merging.

## Machine mapping (this box, 2026-09-30)

- Fix tree: `/home/apollo11/ninfer-gb10` (checked out at `a1a43667` for the runs; serve
  binary `build/apps/ninfer-serve`, engine code = `eb9e87fa`, built 18:03Z).
- Base tree: `/home/apollo11/ninfer-gb10-base` (detached `ed6525fa`; serve binary
  `build/apps/ninfer-serve`, built 18:43Z).
- Artifact: `/home/apollo11/ninfer-gb10/out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer`
  (30 GB; recipe `qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp-v3`).
- Load tool: `/home/apollo11/dgpp/scripts/serve_load.py` (greedy, thinking off by default,
  4 distinct prose prompts per wave; `--repeat N` repeats the wave with the same prompts;
  `--warm 1` default sends one 64-token warm request before the waves).
- Port: 18087 (checked free before each server start).
- Page-cache warm-up: full read of the 30 GB artifact (`dd if=... of=/dev/null bs=16M`)
  before the first server start; the artifact is shared by both trees.
- nsys: 2025.3.2 (same version as Block J). Capture:
  `nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none
  -o <out> <ninfer-serve ...>`, load via `serve_load.py --concurrency 4 --classes prose
  --max-tokens 256 --repeat 3` (the Block J workload).
- Stage analysis: per-tree sqlite export of the nsys rep; stage = GPU time of the QSA op
  kernels (attention: `selected_attention_split_fp8_kernel`, `selected_attention_batched_fp8_kernel`,
  `reduce_selected_attention_splits_kernel`; selection/indexing: `select_top_groups_kernel`,
  `score_groups_batched_kernel`, `score_groups_mma_kernel`, `compress_index_groups_kernel`,
  `expand_indices_batched_kernel`, `dense_indices_batched_kernel`,
  `order_groups_like_persistent_topk_kernel`, `hierarchical_top_groups_kernel`,
  `prepare_index_query_kernel`); per step = stage total over decode rounds from the
  request log. Block J reference (its own window-based method, `profiles/nsys/block-j/`):
  attention + QSA select 5.7 ms/step at 8 rows (DGPP 0.6 ms).
- Width check: the serve log prints a throughput line every 5 s
  (`running N (decode-ready N) | waiting M | batch X | host Y%`); measured passes must show
  `running 4` in decode; `running 2 ... waiting 2` flags the admission stall.
- Outputs: `profiles/bench/gb10/pr36fix-verify/C4-K1/{run1-fix,run2-base,run3-base,run4-fix,
  cache-check,nsys-fix,nsys-base,stall-repro}/`.
