# Upstream PLE fork-fix verification on GB10

Verification of the upstream PR commit `e5c45144` (`fix(flash-next): read PLE
history from the fork source during prefill`) against its parent `2aa87467`
(upstream master) on the GB10 workstation, for the PR evidence package.

- Date: 2026-10-02 (UTC)
- Machine: GB10 (aarch64, CUDA compute capability 12.1 / `sm_121a`,
  driver 580.173.02, CUDA toolkit 13.0.88, ~121 GiB unified LPDDR5x)
- Upstream worktree: `lkarlslund/ninfer`, detached at each commit under test
- Local verification deltas (uncommitted in the worktree, marked
  "Do not commit"): CMake arch `sm_121a`; device gate accepts
  12.0 | 12.1. Semantics-neutral (compile arch + device capability gate).
- Artifact: `$HOME/models/qwen3_8_flash_next_v3_fork/qwen3_8_flash_next_125b_a6b_nvfp4.ninfer`
  (NVFP4 125B-A6B v3 fork artifact, ~77 GiB)
- Serve config: `--kv-dtype fp8 --preserve-thinking --no-cuda-graph
  --max-context 73728` (`--no-cuda-graph` everywhere because of upstream
  defect A below)
- Binaries: base serve sha256 `017ba409...` (2aa87467 + local deltas),
  post-fix serve sha256 `4dfbd975...` (e5c45144 + local deltas)

## Result

The fix is verified on GB10. On the parent commit the probe output depends on
`max_concurrency` (slot reuse) exactly as the commit message describes; with
the fix the output is identical across concurrency for both MTP on and off.

1. **PLE op test** (upstream `test_flash_next_ple`, post-fix tree; covers the
   in-place alias form and the fork source/destination form): PASS.
2. **Real artifact test** (upstream `ninfer_qwen3_8_flash_next_real_test`,
   bitwise goldens: MTP + prefix, concurrent state, vision, plain greedy):
   PASS on `e5c45144`.
3. **C1/C4 identity probe** (73-token prose, greedy, 256 completion tokens,
   no CUDA graph, no logprobs; fresh serve per config; `text_sha` over
   reasoning + content):

   | commit | MTP ON C1 | MTP ON C4 | MTP OFF C1 | MTP OFF C4 |
   |---|---|---|---|---|
   | `2aa87467` (base) | `125743ed` (1126) | `d60ae8ad` (1289) — SPLIT | `3e90f66c` (1274) | `ec14aafb` (1255) — SPLIT |
   | `e5c45144` (fix) | `05aa1613` (1229) | `05aa1613` (1229) — IDENTICAL | `5ff7b06e` (1265) | `5ff7b06e` (1265) — IDENTICAL |

   The base split appears with MTP off as well: the prefill PLE slot bug is
   independent of the decode schedule (consistent with the commit message:
   "Output then depended on slot reuse, and so on max_concurrency").

4. **Decode speed** (`e5c45144`, eager/no-graph): C1 sequential MTP
   (draft 1, 4 x 256-token prose): 24.0 tok/s decode, 1.72 tok/round,
   46.2 ms/round device, 25.6 ms/round host, 2.34 GiB runtime. The C4
   concurrent MTP speed run could not complete (upstream defect A below),
   so there is no C4 MTP speed number for the upstream tree on GB10.

## Defects found (separate from the PLE fix; flag for upstream)

- **A. Multi-in-flight MTP decode fails on the upstream tree.**
  `view element count mismatch: requested 18, available 6`, as an HTTP 500
  on the first multi-request MTP batch. Repro: `e5c45144` (or the base),
  v3_fork artifact, `--max-concurrency 4 --spec mtp --draft-tokens 1
  --lm-head-draft --no-cuda-graph`, 4 concurrent chat requests (256 tokens).
  The same error also aborts default CUDA graph preparation at C4+MTP.
  Single-in-flight MTP decode works (probes above; C1 load: 24.0 tok/s).
  The fork tree runs the same 4 x 256 x 3 MTP load without issue on GB10
  (K2/K4 campaigns), so this is an upstream-vs-fork delta in the MTP decode
  frame/slot sizing, not a problem with the PLE fix. The shape is derived
  from continuation-slot counts rather than SM count, so it is expected to
  fire on the RTX PRO 6000 too — needs maintainer confirmation.
- **B. No token-logprobs surface upstream.** The upstream serve has no
  `--token-logprobs` flag (a fork addition) and rejects `"logprobs": true`
  in the request body. The probe therefore uses the no-logprobs variant
  (`route_probe_nologprobs.py`); identity is still judged on the generated
  text sha, which logprobs do not affect.
- **C. Upstream binder rejects the fork's fp8_mtp artifact.**
  `input_mix_weight_down.weight: representation does not match its
  mathematical value type` (the `exact_format` check in `src/artifact/
  binder.cpp`). Known fork-vs-upstream representation delta; the PR evidence
  uses the v3_fork NVFP4 artifact. (The first before-battery run hit this
  before the artifact was corrected; those logs were discarded.)

## Files

- `route_probe_nologprobs.py` — C1/C4 identity probe (no logprobs)
- `run_upstream_before.sh` / `run_upstream_after.sh` / `run_upstream_c1k1.sh`
  — battery scripts (paths relative to `$HOME`)
- `up-*` — post-fix serve logs, probe json, request logs, load outputs,
  op/real test outputs (`up_ple_test.out`, `up_real_test.out`)
- `base-*` — base-serve probe logs/json for the before/after table
