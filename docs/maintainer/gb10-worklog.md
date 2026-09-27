# GB10 port worklog — 24 / twofour

Timestamped process record of the GB10 (`sm_121a`) port work done on this machine
(`172.30.30.24`, NVIDIA GB10 Grace Blackwell). It captures measurements, campaign
runs with return codes and wall times, failures with root causes, and operational
decisions with their rationale — including what did not work out — so the history
stays auditable as the port evolves. Stable performance results graduate to
`docs/performance/qwen3.8-flash-next-125b-a6b.md`; the active plan is
`docs/maintainer/plan-2026-09-gb10.md`. All times are UTC. Raw (gitignored)
artifacts live under `profiles/bench/gb10/<step>/`.

## Machine baseline (established 2026-09-27)

| Item | Value |
|---|---|
| Machine | NVIDIA GB10 (Grace Blackwell): aarch64 20-core Arm, 128 GB unified LPDDR5x, 21 GB swap |
| GPU | compute capability 12.1 → `sm_121a`, 48 SM, 24 MiB L2, 256-bit bus, advertised 273 GB/s |
| Driver / toolkit | 580.173.02 / CUDA 13.1 (nvcc release 13.0, V13.0.88) |
| nsys | Nsight Systems 2025.3.2.474 |
| Python / ninja / uv | 3.12.3 / 1.11.1 / uv present |
| Repo | fork `baristahaus/ninfer-gb10` (cloned 18:07); port baseline `sm_120a` → `sm_121a` |
| Artifact | `qwen3_8_flash_next_125b_a6b_nvfp4.ninfer` under `~/models/qwen3_8_flash_next_v3_fork/`: 5 volumes, 134.7 G total (entry 32 G + part-0001..0004); weights 83.3 G, PLE 51 G FP8 file-backed (mmap, CPU gather) |
| Config | `tools/gb10/config.local.sh`: port 18087, KV fp8, MTP draft 2, RUN_SERVING=0 |
| Disk / RAM at campaign time | 346 G free / 111 G available |

Fork state: cloned 18:07 at `9eef6f72`; upstream PR 5–8 merged on GitHub by 18:27;
local reset to `origin/master` `a24542f0` at 18:38 (tools/docs-only changes → no
rebuild needed; judged from the diff).

Step 0 (18:07–18:32): configure + build `build/` (untraced, `sm_121a`), ctest
131/131, Flash-Next real-artifact tests 3/3.

## 2026-09-27 timeline

### 18:20 — beads initialized

`bd init` on the clone (embedded dolt DB). See logistics below for the local-only
decision.

### Memory probe, run 1 (before 18:35; clock time not recorded)

`tools/gb10/probe_memory.sh`, idle GB10, 1 GiB buffers. Key results (full tables in
the plan doc):

- plain GPU read ≈ 246 GB/s;
- pinned / `malloc` host memory read in place ≈ 174 GB/s (71% of device speed) —
  in-place host reads beat a host-to-device copy (59.2 GB/s);
- compression-granted read on zero-filled data faster than plain (granter active on
  the bus), so the question for real data was left open.

Decision carried: test KV/activation buffers and real weights before believing the
compression-granted rate.

### 18:35 — memory probe, run 2 (+ weight samples)

With `ART` set to the 125B-A6B NVFP4 fork artifact, sample the artifact's own bytes
additionally — executing the explicit instruction from the pre-PR #8 Opus conversation
("next time on the GB10: set `ART` and rerun the probe, ~15 s").
`tools/gb10/weight_samples.py` (PR #8) extracts 64 MiB of each weight class through
the Flash-Next bindings; `probe_memory.sh` runs it automatically when `ART` is set and
removes the samples afterwards. `hbm_bandwidth_probe.cu` section 2b:

- plain GPU read 236–240 GB/s (≈3% run-to-run variance);
- compression granted on zero-filled data 2024.9 GB/s (granter works when data is
  compressible);
- pinned host in-place 216.4 GB/s (91% of device speed this run; `malloc` 159.3);
- host-to-device copy 59.2 GB/s — in-place host reads beat copies by ≈3×;
- CPU-only read (20 threads) 132.8 GB/s; CPU + GPU together 245.7/171.6 GB/s — CPU
  and GPU share one budget, benchmarks must run on an otherwise idle machine;
- **real weights: 1.00–1.01× compression across all six byte classes** (BF16 GDN
  50 MiB, BF16 HyperConnection 6.2 MiB, BF16 `lm_head`, NVFP4 experts, FP8 PLE
  table, Q4 proposal head) → generic compression does not help real weight bytes and
  stays out of the decode path.

### 18:42:24 — campaign 1 (bg_16): steps 1→3 + report

Pipeline `step1_json_check.sh → step2_baseline.sh → step3_attribution.sh →
report.sh`; wall 367 s; final rc `step1=0 step2=1 step3=0 report=0`.

- 18:42:38 step 1: server on 18087 (fp8 KV, MTP draft 2, ctx 73728). TEB
  **5/6 PASS, 83/100**; TC-68 (schema violation resistance) FAIL 0/2 — output not
  valid JSON (genuine model-behavior data point for the plan's step 4 decision).
  The three `response_format` JSON checks reported FAIL because of bug B2 — invalid.
- 18:44:46 step 2: bandwidth probe built + run (below), then aborted at the first
  bench run: `ninfer_bench: invalid speculative backend: none` (bug B1). No baseline
  matrix.
- 18:45–18:48:45 step 3: rc 0, but the `--spec none` paths (mtp0 capture, PLE
  residency run) hit B1 and were discarded runs; both attribution reports failed with
  B3. The mtp2 capture + bench JSON are valid (preview below).
- 18:48: `report.md` assembled (18,786 B) with the gaps noted above.

Valid data from campaign 1:

- **Bandwidth** (`tools/hbm_bandwidth_probe.cu`, sm_121a, 4 GiB buffers, 5 trials):
  best sustained **248.4 GB/s** (91.0% of 273, `cudaMemcpyAsync` D2D); kernel
  `uint4` read 244.4 (89.5%), write 199.5 (73.1%).
- **MTP2 preview** (8K prompt, fp8 KV, `--spec mtp --draft-tokens 2 --lm-head-draft`,
  from the bench's own JSON): prefill **1447 tok/s** (5.66 s/8192), decode **43.2
  output tok/s** (0.741 s/32), MTP acceptance 1.0 (accept length 2.91, 11 rounds),
  load 20.0 s / upload 15.4 s, weights 83.26 G resident. Already far above the plan's
  ~25 tok/s naive bytes-per-step ceiling.

### 18:46 — logistics commits

Beads config (local-only dolt) committed as `d08f230e` (identity `baristahaus`,
pre-commit hook path verified end-to-end) and pushed to
`twoFour/beads-local-only`.

### 18:52:56 — campaign 2 (bg_17): fixed pipeline re-run

Same config; fixes B1–B4 applied (below).

- 18:52:56 step 1 re-run — **all three `response_format` checks PASS**: streamed
  json_schema (66 events, `{"title":"Casablanca","year":1942}`), non-streamed
  json_schema, streamed json_object (118 events, `{"city":"Paris","country":"France"}`);
  B2 fixed and verified. TEB again 83/100 with TC-68 FAIL 0/2 (seed 42) — the TC-68
  failure reproduces across runs: genuine model behavior, not a harness artifact.
- 18:55 step 2 **failed again within ~15 s** (245-byte log, same B1 signature): the
  page-cache warm call had `--spec none` hardcoded — missed by the B1 fix. Recorded
  as bug B5, fixed at 18:57 (this run predated the fix).
- 18:55–18:59 step 3 with `--profile-measured` (B3 fix) and the `--spec` fixes:
  mtp0/mtp2 captures OK (both bench JSONs written); PLE residency valid this run:
  **49 major page faults over 147 s** (includes model load; page cache was warm from
  earlier runs — cold-start PLE faulting is not measured here). Attribution reports
  still failed with `no selected measured range` despite the flag: bug B6 — the
  `ninfer_bench` TU compiles without the trace macro.
- 18:59 bg_17 done: rc `step1=0 step2=1 step3=0 report=0`, wall 365 s; `report.md`
  21,488 B. B4 verified (summaries now state the multi-volume total: 126 G disk
  usage, 134.7 G apparent).

### 18:58 — bug B6 fix; campaign 3 (bg_18) launched

B6 fixed by defining the macro on the bench target
(`bench/inference/benchmarks.cmake`). Campaign 3 (bg_18) runs step 3 (rebuilds the
`build-trace/` bench, re-captures, attribution, residency) → step 2 full matrix (B5
fix) → `report.sh`, serial to keep the machine idle for the timing runs. Results
pending at time of writing.

### 19:32 — campaign 3 (bg_18) results: all green

bg_18 done: rc `step3=0 step2=0 report=0`, wall 1,839 s (30.7 min); final
`profiles/bench/gb10/report.md` 25,214 B (2026-09-27T19:32Z). **B5 and B6
verified.** Note: the report's step 0 section reuses campaign 1's step 0 output, so
its artifact line still reads 30 G (pre-B4); steps 1–3 show the correct 126 G
multi-volume total.

Step 2 matrix (KV fp8-e4m3-row256, max-ctx 73728, prefill-chunk 8192, 1 warmup + 5
measured; in-run bandwidth probe 247.1 GB/s, 90.5% of spec):

| Config | Prefill 8K / 64K tok/s | Decode out 8K / 64K tok/s | Mean accepted |
|---|---:|---:|---:|
| none | 1,513.7 / 1,473.3 | 19.3 / 19.1 | — |
| mtp K=2 | 1,435.0 / 1,400.0 | 44.6 / 43.6 | 2.99 / 2.98 |
| mtp K=3 | 1,435.0 / 1,397.5 | 52.0 / 51.1 | 3.96 / 3.96 |

Decode is context-insensitive (−1.7% 8K→64K at K=3); K=3 is +16.5% over K=2 and 2.7×
no-spec. Prefill is essentially context-independent (≤2.7% drop). KV capacity
73,728 tokens; 17.3 GiB free after startup.

Step 3 attribution (measured region: 8192-token prefill + 32 decode steps; 97.5%
(mtp0) / 97.8% (mtp2) of GPU work attributed):

- Decode (mtp0): `gdn.update` 38.8% of GPU work at 95.9% of the bandwidth estimate;
  `moe.nvfp4` 21.3% at 76.4%; `hyper.combine_mix` 14.8% at 71.3%; `ple.update` 0.6%
  at 90.0%. The decode path is bandwidth-saturated — little headroom left in the
  decode stages.
- Decode (mtp2): `moe.nvfp4` 30.2%, `gdn.record` 28.5% at 94.3%,
  `hyper.combine_mix` 11.5%, `qsa.select` 10.8% at 71.7%; the MTP predictor's decode
  cost is ~3.5% of total GPU work.
- Prefill (both): `gdn.prefill` ~37% at ~13% estimate efficiency,
  `hyper.combine_mix` ~21% at ~13.6%, `qsa.select` ~20% at **6.9%** (843 GFLOP —
  the largest single inefficiency in the trace), `moe.nvfp4` ~16% at 8.5–32%.
- Unattributed GPU work: 171.6 ms (mtp0) / 141.4 ms (mtp2).
- PLE host lookup (mtp0): `ple.gather` 134 µs/step, `ple.hash` 0.5 µs/step — <0.3% of
  a 51.8 ms decode step.

PLE residency (campaign 3): **0 major page faults over 147 s** (campaign 2: 49) —
the PLE working set is now fully page-cached. Together with the host-gather cost
above, the plan's PLE residency question is settled: keep the PLE on the file +
page cache; no device-resident copy needed.

TC-68 failed again (0/2, seed 42) — three consecutive reproducible runs; genuine
model behavior (step 4 decision input).

Next (plan): step 4 KV read-share at the served context (the 64K row above informs
it); steps 5–6 optimization targets point at prefill — `qsa.select` (6.9% estimate
efficiency), then `gdn.prefill` / `hyper.combine_mix` (~13%) — not the decode path.

## Bug history (first live run exposed all of them)

| ID | Symptom | Root cause | Fix | Status |
|---|---|---|---|---|
| B1 | `ninfer_bench: invalid speculative backend: none` — step 2 matrix, step 3 mtp0 capture, PLE residency run | the bench CLI takes `--spec <mtp\|dflash\|dflash2>`; `none` is the default, not a passable value; scripts passed `--spec none` | omit the flag for the no-spec case (`step2:38`, `step3:39`, `step3:50`) | fixed; verified in campaign 2 (step 1 parse + step 3 captures; the step 2 warm run failed on B5 instead) |
| B2 | step 1 JSON checks: `model_not_found: model 'ninfer' not found`; all three FAIL | `step1_json_check.sh` hardcoded `"model":"ninfer"`; the server's model id is `qwen3.8-flash-next-125b-a6b` (TEB queries `/v1/models`, which is why it passed) | model id derived from `$BASE_URL/v1/models` at runtime (`step1:8-10,14,21-24`) | fixed; verified campaign 2 (all three checks PASS) |
| B3 | attribution: `no selected measured range` for both captures | the `ninfer.region/1\|measured` scope is only emitted under `--profile-measured` (`bench/inference/ninfer_bench.cpp:219`); captures never passed it. The trace itself is fully annotated: 1,892 `ninfer.work/1\|` op scopes, 27 `ninfer.region/1\|` (predictor/target.verify/target.prefill), 58 `ninfer.host/1\|ple.hash\|gather` — engine `NINFER_PERFORMANCE_TRACE` wiring is intact | add `--profile-measured` to the capture (`step3:32`); its preconditions (exactly one test, `-r 1`) were already met | flag fix correct; superseded by B6 (macro missing in the bench TU); end-to-end verified campaign 3 via the B6 fix |
| B4 | reports state the artifact as 30 G | `machine_summary` used `du -h "$ART"` — the entry volume only; the artifact is 5 volumes, 134.7 G | sum `ART` + `ART.part-*` (`common.sh:105-107`) | fixed; verified campaign 2 (126 G multi-volume total in summaries) |
| B5 | campaign 2 step 2 failed again, same B1 signature, ~15 s in | `step2:32` page-cache warm call had `--spec none` hardcoded — missed by the B1 fix | remove the flag (`step2:32`) | fixed 18:57; **verified campaign 3** (full 6-cell matrix completed) |
| B6 | attribution still `no selected measured range` with `--profile-measured` passed (campaign 2) | the `ninfer_bench` TU compiles without `NINFER_PERFORMANCE_TRACE`: core's `target_compile_definitions(PUBLIC ...)` does not propagate through `ninfer_engine`'s private link, so the bench's measured-region scope compiled to a no-op (string absent from the build-trace binary) | explicit `target_compile_definitions(ninfer_bench PRIVATE NINFER_PERFORMANCE_TRACE=1)` under the option (`bench/inference/benchmarks.cmake`) | fixed 18:58; **verified campaign 3** (both captures attributed 97.5–97.8%) |

Note on B3: the first trace inspection looked like the annotations were missing
entirely; the query had an inner JOIN on `textId` and silently dropped rows that
carry their text inline in the `text` column. The coalesced query (`COALESCE(text,
StringIds.value)`) shows the full annotation set. Kept here because the same trap
will bite any future nsys-sqlite inspection.

### Pre-existing bug fixed upstream (verified here)

The hardware profile `tools/bench/hardware/gb10.json` was named "NVIDIA GB10 Grace
Blackwell (DGX Spark class)"; the attribution script requires the profile name to match
`cudaDeviceProp.name`, which the benchmark reports as "NVIDIA GB10". The old name would
have stopped step 3's attribution at the name check
(`flash_next_performance.py:382`). Found in the pre-PR #8 Opus conversation (CPU side,
no GPU); fixed in PR #8 (`b7e770d8`, merged 18:27) together with the measured 246 GB/s
bandwidth value. Verified here: `gb10.json` now reads `name: "NVIDIA GB10"`,
`dram_gbps: 246.0`, and the campaign 2 attribution runs passed the name check (they
failed later, on the missing measured region — B6).

## Operational logistics (timestamped)

- **18:34 deploy key + SSH wiring.** Repo-scoped GitHub deploy key
  `~/.ssh/ninfer-gb10-deploy`; `~/.ssh/config` routes `github.com` to it
  (`IdentitiesOnly`); `github.com` host key pinned in `known_hosts` (18:37).
  `origin` uses the SSH URL. Verified: fetch, push of test branch
  `twoFour/deploy-key-test` (rc=0), test branch deleted.
- **18:41 beads dolt: local-only.** `bd` refuses to add a Dolt remote that matches
  the git origin (dolt data would publish/collide with the project's refs), and this
  is a single-machine box. Decision: `dolt.local-only=true`, `federation.remote`
  unset → `bd sync` is a clean no-op (`{"status":"disabled"}`, rc=0); the residual
  `federation.remote: required for Dolt sync` validate warning is expected under
  local-only. Issue data stays on the machine; the git-tracked `.beads/` files ride
  the same git/SSH route (committed 18:46, `d08f230e`).
- **18:46 git identities.** Repo default `baristahaus <baristahaus@users.noreply.github.com>`
  (everything committed here targets the GitHub fork; a pushed commit must carry a
  GitHub identity); local-only commits use `TwoFour <24@twofour.gb10.local>` via
  `git -c user.name=... -c user.email=...`; local WIP is re-authored to baristahaus
  before pushing. Verified by `d08f230e` (hooks + commit + push). Policy recorded in
  `AGENTS.md` (Device and role).
- **before 18:42 TEB.** `tool-eval-bench 2.7.1.dev7` installed via
  `uv tool install git+https://github.com/SeraphimSerapis/tool-eval-bench.git` (step 1
  prerequisite).
- **port 18087** (config.local.sh); confirmed free before campaigns. `RUN_SERVING=0`:
  the serving matrix is opt-in (~+1 h, needs transformers + a local tokenizer).
- **LLM backend concurrency** (2026-09-27, user instruction): the provider has tight
  concurrency limits — cap at main session + at most one subagent while running long
  context; long tasks run as plain background shell jobs (no LLM concurrency).
  Recorded in `AGENTS.md`.

- **19:37 — repo hygiene after campaign 3.** `git status` showed uncommitted edits
  alongside my own commits. Ownership settled by mtime + worklog evidence: `AGENTS.md`
  (18:47, designation/logistics/beads block), `.gitignore` (18:20, bd init), and
  `README.md` + plan doc (18:40, probe run-2 results) are this machine's own
  uncommitted work — committed in `b5156f35`. The eight maintainer-doc "English
  clarification" sections (all written 18:26:09 within 100 ms — a scripted batch,
  absent from this worklog) and the bd-init agent integrations (`.codex/`, `CLAUDE.md`,
  `.claude/`, `.cursor/`, `.agents/`) are not this session's work — kept local,
  uncommitted, for their owner to ship.
- **19:37 — beads PR subsumed.** `twoFour/beads-local-only` is subsumed: its commit
  `d08f230e` is on `master` and in `twoFour/gb10-campaign`; the PR can be closed
  without merging (a push-only deploy key cannot close it from here).

## Data inventory so far

| Data | Source | Where |
|---|---|---|
| Probe runs 1–2 (bandwidth behavior, compression on real weights) | `probe_memory.sh` 18:35 | plan doc (Memory probe section) |
| Bandwidth 248.4 GB/s sustained | `hbm_bandwidth_probe.cu` in step 2 | `profiles/bench/gb10/step2/bandwidth.txt` |
| TEB 83/100 (TC-68 fail) | `tool-eval-bench` seed 42 | `profiles/bench/gb10/step1/` |
| MTP2 preview 1447 prefill / 43.2 decode tok/s, acceptance 1.0 | `ninfer_bench` JSON | `profiles/bench/gb10/step3/mtp2-bench.json` (campaign 1) |
| Step 0: ctest 131/131, real-artifact 3/3 | `step0_build_test.sh` | `profiles/bench/gb10/step0/` |
| Baseline matrix: none 19.3, K=2 44.6, K=3 52.0 tok/s decode (8K/512) | step 2, campaign 3 | `profiles/bench/gb10/step2/summary.md` |
| PLE residency: 49 faults (campaign 2, warm) → 0 faults (campaign 3); host gather 134 µs/step | step 3, campaigns 2–3 | `profiles/bench/gb10/step3/residency.log` |
| Per-stage attribution (mtp0/mtp2, 97.5–97.8% attributed) | step 3, campaign 3 | `profiles/bench/gb10/step3/{mtp0,mtp2}-report.md` |
| Final consolidated report (25,214 B, 2026-09-27T19:32Z) | `report.sh`, campaign 3 | `profiles/bench/gb10/report.md` |
| Idle-time gap attribution; step 6 task decision | kernel-gap bucketing on `profiles/bench/gb10/step3/{mtp0,mtp2}.sqlite` (no new capture) | `gb10-worklog.md` § Idle-time attribution; brief folded into `plan-2026-09-gb10.md` § step 6 |
| File-backed page GPU-read probe (device-PLE gate) | `file_page_probe.cu` + `probe_file_pages.sh` | `profiles/bench/gb10/file_page_probe/summary.md` |
| Step 6.1 gap breakdown (43/49 gaps host-bound; ~1.4 ms sync-return completion lag is the dominant component) | sqlite analysis of `profiles/bench/gb10/step3/mtp2.sqlite` | `plan-2026-09-gb10.md` § 3 (between-round gap bullets) |
| Round-boundary sync probe (blocking 0.4–1.2 ms vs yield/spin/auto 2.8 µs; no per-node graph cost at 4000 nodes) | `probe_sync.sh` + `sync_probe.cu`, PIN_CPUS="10 19" | `profiles/bench/gb10/sync_probe/summary.md` |

## Idle-time attribution (2026-09-27, mtp0/mtp2 sqlite, no new capture)

Kernel-gap bucketing on `CUPTI_ACTIVITY_KIND_KERNEL`
(`LAG("end") OVER (ORDER BY start)`, single stream 13 / ctx 1 in both captures;
each capture = one 163 ms warmup burst + 2 concatenated benchmark runs; the 79 ms /
8.4 ms gaps at ~1% and ~50% span are run-boundary pauses, not round stalls).

| Bucket | mtp0 (MTP off, diagnostic) | mtp2 (production, draft K=2) |
|---|---|---|
| kernels | 88,775 — **100% eager** (`graphNodeId` = 0 for all) | 47,408 — **85% in CUDA graphs** (40,446 graph / 6,962 eager) |
| span / busy / idle | 14,469 / 14,088 / 381.7 ms (≈191 ms/run) | 13,613 / 13,355 / 258 ms (≈129 ms/run) |
| gaps >1 ms | 60 / 174.2 ms | 59 / 233.7 ms (**90.6% of idle**) |
| gaps 100 µs–1 ms | 19 / 9.1 ms | 8 / 3.7 ms |
| gaps 10–100 µs | 176 / 3.7 ms | 80 / 2.1 ms |
| gaps <10 µs | 88,519 / **194.6 ms** | 37,552 / 19.7 ms (7.6% of idle) |

Findings:

1. mtp0's 194.6 ms of sub-10 µs "small gaps" are **eager-launch latency**: with MTP
   off, the decode path runs 100% eager on one stream (no graph launches at all in
   the capture). That is an artifact of the attribution config, not of the
   production path.
2. In the production config (mtp2) decode is 85% graph-captured and in-round gaps
   collapse to 19.7 ms. **>90% of production idle is between-round host gaps** —
   round-boundary signature: `shortlist_exact_select` / `ple_fold` /
   `embed_gather_dense` (last kernels of round N) → host (PLE gather, sampling
   readback, scalar set) → `set_i32_scalar` / `speculative_prepare_verify_inputs`
   (first kernels of round N+1). Median ≈1.7 ms; outliers 26.5 ms and 14.2 ms at
   the MTP verify-inputs prep boundary (after `shortlist_exact_select`).
   Excluding run-boundary outliers: ≈73 ms/run ≈ 2.3 ms/decode round ≈ ~20% of
   the ~12 ms/token gap.
3. **Step 6 first task: overlap the between-round host work with the previous
   round's GPU execution** (pipelined commit path: verify-inputs prep and PLE
   gather of round N+1 in flight while round N's GPU work runs; sampling readback
   off the critical path). PDL (knoopx fork `e0a6d18c`) is demoted to second:
   its remaining target is the 19.7 ms of in-round gaps plus the ~6,962
   eager kernels/run (PLE fold/gather, scalar ops, verify prep), and it would have
   to coexist with graph capture — a harder combination than eager-only PDL.

## File-backed page GPU-read probe (2026-09-27, step 6.3 gate)

Plan step 6.3 asks whether a device kernel can gather PLE rows directly from the
file-backed mapping, and what a GPU pays for a file-backed page that is resident vs.
cold (page-cache miss → NVMe page-in via the fault path). New standalone probe
(`tools/gb10/file_page_probe.cu`, runner `tools/gb10/probe_file_pages.sh`; 512 MiB
scratch file, 2560 B rows at 4 KiB-aligned offsets, deterministic offsets; each phase a
separate invocation so a hung cold fault would time out without hanging the run).

| Phase | Result |
|---|---|
| host warm read (CPU, page cache) | 97.5 GB/s |
| GPU sequential full read, resident | 161.9 GB/s |
| GPU random 2560 B rows ×100k, resident | 168.9 GB/s (1.5 ms) |
| evict: 125 GiB host read (LRU recycle) | 76.6 s at 1.75 GB/s |
| GPU serial cold rows, 1 thread ×512 | **96.4 µs/row** (fault round-trip) |
| GPU parallel cold rows ×100k | 102.2 ms, 2.5 GB/s, **1.02 µs/row** amortized |
| same rows re-read, resident (sanity) | 168.1 GB/s — matches resident phase |

Verdict — **gate passed**:

1. Device faults on file-backed (page-cache) pages work on GB10 sm_121: no hangs, no
   errors, all 100k cold rows completed.
2. Steady-state (resident) device PLE gather is ~free: 48 rows/round (3 tokens ×
   16 heads × 160 B = 7.7 KB) at 168 GB/s is ~45 ns; the residency campaign already
   showed 0 faults over 147 s warm. Note the 162–169 GB/s is ~30% below the ~246 GB/s
   the GPU gets from its own allocations — irrelevant at a latency-bound gather of a
   few KB, but the plan should not claim "no penalty".
3. Cold tail is ~100 µs per faulted row, and faults parallelize across SMs
   (1.02 µs/row amortized at 100k rows) — vs. the host gather's 72 ms cold outlier
   (single-threaded serialized faults). The device gather is ~700× better in the cold case.

Caveats: the 125 GiB evicting read recycles the whole page cache — PLE table residency
is gone until the next campaign's warm run re-establishes it (the campaign warmup
handles it; campaign 2 showed convergence 49 faults → 0). The sink magic-check prints
0x0 by construction (best-effort anti-DCE); the bandwidth and completion evidence stands.

## Step 6.1 — inter-round gap breakdown (2026-09-27, mtp2.sqlite)

Opus's step-6 ordering question: the PLE gather is 9.2 µs warm (0.7% of the gap) —
what fills the rest of the ~1.5–1.7 ms? Full analysis now in `plan-2026-09-gb10.md`
(step-3 attribution, between-round gap bullets; the dated file was removed once the plan
carried it; method: LAG(end) gaps over the kernel table, next-kernel enqueue resolved
exactly via `kernel.correlationId = runtime.correlationId`, NVTX + CUPTI sync/memcpy per
gap; 49 decode-phase gaps >200 µs, 5 run-boundary gaps >3 ms excluded).

Headline:

1. **43/49 gaps are host-bound** — the next kernel's enqueue call starts 0.32–2.01 ms
   after the GPU went idle. The GPU is waiting for the host.
2. Two stall points per round: type A `qsa.select → GDN` (eager, one kernel at a time,
   gap p50 1.32 ms) and type B `ple_fold → verify graph` (graph, gap p50 1.85 ms).
3. Dominant component: a CUPTI stream sync returns **~1.4 ms (p50 1.47 ms) after the
   last kernel it waited on has already completed** — completion-detection latency of
   the engine's wait path (poll/sleep granularity or post-kernel stream work; to
   confirm in the engine source).
4. The real between-round work (commit ~8 µs, submit ~84 µs, PLE 11 µs, H2D, launch)
   is only ~30–110 µs after the sync returns.
5. `cudaGraphLaunch` runs 940–970 µs with the first kernel starting mid-call — the
   call duration is not itself critical, but the ~800 µs call-start → kernel-start
   interval is; flagged for follow-up.

Consequence: step 6.3 (device PLE) is ~11 µs/round of steady-state (cold-tail value
only); step 6.2 as designed recovers ~100–200 µs/round; the **new top target is the
~1.4 ms sync-return lag** (candidate 6.0 — blocking event sync / tighter wait loop),
which needs the engine's wait-path source to confirm the mechanism.

## Round-boundary sync probe (2026-09-27, plan step 6 item 1 "the wake-up")

`tools/gb10/probe_sync.sh` with `PIN_CPUS="10 19"` (cpu10 = Cortex-A725 2.8 GHz,
cpu19 = Cortex-X925 3.9 GHz; governor performance). Results in
`profiles/bench/gb10/sync_probe/summary.md`.

| schedule | sync return lag p50 (kernel 50 µs / 1 ms / 10 ms / 60 ms) |
|---|---|
| blocking (unpinned) | 401 / 670 / 956 / 1181 µs |
| yield | **2.7–2.8 µs** at every duration |
| spin | 2.8 µs at every duration |
| auto | 2.8 µs (auto spins here: 20 cores > 1 GPU) |
| blocking pinned cpu10 (A725) | 138 / 138 / 675 / 596 µs |
| blocking pinned cpu19 (X925) | 394 / 607 / 734 / 976 µs |

- Confirms the ~1.4 ms completion lag from the gap breakdown, growing with kernel
  duration (a 60 ms decode round → 1.18 ms). Core type does not change blocking's
  order of magnitude.
- Graph launch: 100/1000/4000 nodes → 1.6–14 µs call, 2.5–14.5 µs to first kernel.
  **Per-node submission cost is not confirmed** — the 940–970 µs `cudaGraphLaunch` call
  in the trace is not node-count-driven; likely CUPTI overhead or per-round graph exec
  update (`install_graph_profile`). Step 6 item 2 (fewer/larger nodes) is not indicated
  by this evidence; investigate under item 3's instrumentation.
- Decision: `src/core/device.cu` switches `cudaDeviceScheduleBlockingSync` →
  `cudaDeviceScheduleYield`: yield matches spin's 2.8 µs wake-up without holding a core
  at 100% (shared CPU/GPU power budget); auto is excluded because it spins on this
  machine. Gate: step 2 re-run against 44.6 tokens/s (running).

- Ops note: after the file-page probe's 125 GiB evictor read, 101 GB of page cache left
  only 16 GB free, so the first step-2 load failed the `current_free_device_bytes`
  check. Non-root page-cache drop/reclaim is not permitted on this box; clear it with
  `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'` (passwordless sudo is available)
  before load-heavy runs following any probe that reads large artifact volumes.
