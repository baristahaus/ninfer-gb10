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

Same setup, plus samples from the artifact's own bytes with `ART` set to the 125B-A6B
NVFP4 fork artifact:

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

## Bug history (first live run exposed all of them)

| ID | Symptom | Root cause | Fix | Status |
|---|---|---|---|---|
| B1 | `ninfer_bench: invalid speculative backend: none` — step 2 matrix, step 3 mtp0 capture, PLE residency run | the bench CLI takes `--spec <mtp\|dflash\|dflash2>`; `none` is the default, not a passable value; scripts passed `--spec none` | omit the flag for the no-spec case (`step2:38`, `step3:39`, `step3:50`) | fixed; verified in campaign 2 (step 1 parse + step 3 captures; the step 2 warm run failed on B5 instead) |
| B2 | step 1 JSON checks: `model_not_found: model 'ninfer' not found`; all three FAIL | `step1_json_check.sh` hardcoded `"model":"ninfer"`; the server's model id is `qwen3.8-flash-next-125b-a6b` (TEB queries `/v1/models`, which is why it passed) | model id derived from `$BASE_URL/v1/models` at runtime (`step1:8-10,14,21-24`) | fixed; verified campaign 2 (all three checks PASS) |
| B3 | attribution: `no selected measured range` for both captures | the `ninfer.region/1\|measured` scope is only emitted under `--profile-measured` (`bench/inference/ninfer_bench.cpp:219`); captures never passed it. The trace itself is fully annotated: 1,892 `ninfer.work/1\|` op scopes, 27 `ninfer.region/1\|` (predictor/target.verify/target.prefill), 58 `ninfer.host/1\|ple.hash\|gather` — engine `NINFER_PERFORMANCE_TRACE` wiring is intact | add `--profile-measured` to the capture (`step3:32`); its preconditions (exactly one test, `-r 1`) were already met | flag fix correct; superseded by B6 (macro missing in the bench TU); verification pending (campaign 3) |
| B4 | reports state the artifact as 30 G | `machine_summary` used `du -h "$ART"` — the entry volume only; the artifact is 5 volumes, 134.7 G | sum `ART` + `ART.part-*` (`common.sh:105-107`) | fixed; verified campaign 2 (126 G multi-volume total in summaries) |
| B5 | campaign 2 step 2 failed again, same B1 signature, ~15 s in | `step2:32` page-cache warm call had `--spec none` hardcoded — missed by the B1 fix | remove the flag (`step2:32`) | fixed 18:57; verification pending (campaign 3 step 2) |
| B6 | attribution still `no selected measured range` with `--profile-measured` passed (campaign 2) | the `ninfer_bench` TU compiles without `NINFER_PERFORMANCE_TRACE`: core's `target_compile_definitions(PUBLIC ...)` does not propagate through `ninfer_engine`'s private link, so the bench's measured-region scope compiled to a no-op (string absent from the build-trace binary) | explicit `target_compile_definitions(ninfer_bench PRIVATE NINFER_PERFORMANCE_TRACE=1)` under the option (`bench/inference/benchmarks.cmake`) | fixed 18:58; verification pending (campaign 3 step 3) |

Note on B3: the first trace inspection looked like the annotations were missing
entirely; the query had an inner JOIN on `textId` and silently dropped rows that
carry their text inline in the `text` column. The coalesced query (`COALESCE(text,
StringIds.value)`) shows the full annotation set. Kept here because the same trap
will bite any future nsys-sqlite inspection.

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

## Data inventory so far

| Data | Source | Where |
|---|---|---|
| Probe runs 1–2 (bandwidth behavior, compression on real weights) | `probe_memory.sh` 18:35 | plan doc (Memory probe section) |
| Bandwidth 248.4 GB/s sustained | `hbm_bandwidth_probe.cu` in step 2 | `profiles/bench/gb10/step2/bandwidth.txt` |
| TEB 83/100 (TC-68 fail) | `tool-eval-bench` seed 42 | `profiles/bench/gb10/step1/` |
| MTP2 preview 1447 prefill / 43.2 decode tok/s, acceptance 1.0 | `ninfer_bench` JSON | `profiles/bench/gb10/step3/mtp2-bench.json` (campaign 1) |
| Step 0: ctest 131/131, real-artifact 3/3 | `step0_build_test.sh` | `profiles/bench/gb10/step0/` |
| Baseline matrix (8K/64K × MTP0/2/3, 512 decode) | step 2 | pending (B5 fix) |
| PLE residency: 49 major page faults / 147 s (page cache warm) | step 3, campaign 2 | `profiles/bench/gb10/step3/residency.log` |
| Per-stage attribution | step 3 | pending (campaign 3, B6 fix) |
