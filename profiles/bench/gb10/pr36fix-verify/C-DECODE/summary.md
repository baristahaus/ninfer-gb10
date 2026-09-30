# PR #36 golden-commit verification + decode speed — results (2026-09-30)

Trees:
- fix = `a1a43667` (`claude/peaceful-cori-e9myku`): `d1efb24f` perf(qsa): skip index
  selection when every visible key fits the budget + `eb9e87fa` fix(ops): size the QSA
  workspace for every call up to its row count + `a1a43667` test(flash-next): one greedy
  golden (test-only commit; engine code identical to `eb9e87fa`).
- base = `ed6525fa` (pre-PR), worktree `/home/apollo11/ninfer-gb10-base`.
- Both trees built for `sm_121a` (fix build 19:28Z; base build 18:43Z). Machine: GB10,
  CUDA 13.0.88, driver 580.173.02. GPU idle before each run (single-instance discipline).

## C1 — Real suite on a1a43667 + fp8_mtp artifact (must-pass gate): PASS 5/5

Artifact `out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer`
(recipe `qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp-v3`), ctest, ~19:31–19:34Z:

    Start 132: ninfer_flash_next_qsa_test ................. Passed 2.54 sec
    Start 134: ninfer_qwen3_8_flash_next_real_test ........ Passed 79.66 sec
    Start 135: ninfer_qwen3_8_flash_next_fault_test ....... Passed 72.84 sec
    Start 138: ninfer_qwen3_8_flash_next_load_plan_test ... Passed 0.04 sec
    Start 139: ninfer_qwen3_8_flash_next_frontend_test .... Passed 0.53 sec
    100% tests passed, 0 tests failed out of 5

The single greedy golden holds on the fp8_mtp artifact for both the MTP verify and the
plain decode path — the golden merge in `a1a43667` is confirmed on the target machine.

## C2 — Real suite on a1a43667 + BF16-dense artifact: PASS 5/5

Artifact `/home/apollo11/models/qwen3_8_flash_next_v3_fork/qwen3_8_flash_next_125b_a6b_nvfp4.ninfer`,
same ctest set, after C1:

    Start 132: ninfer_flash_next_qsa_test ................. Passed 2.49 sec
    Start 134: ninfer_qwen3_8_flash_next_real_test ........ Passed 85.66 sec
    Start 135: ninfer_qwen3_8_flash_next_fault_test ....... Passed 77.22 sec
    Start 138: ninfer_qwen3_8_flash_next_load_plan_test ... Passed 0.04 sec
    Start 139: ninfer_qwen3_8_flash_next_frontend_test .... Passed 0.61 sec
    100% tests passed, 0 tests failed out of 5

The BF16-dense artifact shares the single golden as well. The `fp8_projections` artifact
was not run: `~/models/fp8/` was scrubbed from this machine on 2026-09-30 (machine AGENTS
note); the BF16-dense artifact is the available second artifact.

## C3 — Decode speed at MC=1 and MC=8, a1a43667 vs ed6525fa (fp8_mtp)

Method: `tools/gb10/pr36fix_decode_sweep.sh` — `ninfer-serve` per tree, identical flags
(`--max-context 73728 --max-concurrency 8 --kv-dtype fp8 --spec mtp --draft-tokens 2
--lm-head-draft --preserve-thinking`, fp8_mtp artifact), `concurrency_sweep.py --n 1,8`
(greedy, 512 max tokens, 8K-char prompts from the PPL corpus, streams wikitext-00..03 +
pg19-00..03), decode-only tok/s from the structured request log (`--request-log-jsonl`,
`request_log_summary.py`; prefill and transport excluded). One sweep per (tree, N);
single-shot, no repeats. FIX sweep 19:34:24Z–19:36:01Z, BASE sweep 19:36:27Z–19:37:58Z.

### MC=1 (single active request, stream wikitext-00, prompt 1882)

| tree | prefill s | completion | decode s | decode tok/s | rounds | tok/round |
|---|---:|---:|---:|---:|---:|---:|
| fix a1a43667 | 4.84 | 412 | 10.02 | 41.1 | 201 | 2.05 |
| base ed6525fa | 1.28 | 317 | 7.34 | 43.2 | 151 | 2.10 |

Decode-only: base +5.1%. Different stop positions (412 vs 317) — greedy sensitivity
between trees on the same prompt. Both are the first request after server start; the
fix-side prefill (4.84 s) is 3.8x the base-side (1.28 s).

### MC=8 (8 active requests) — decode-only tok/s per stream

| stream | fix (completion) | base (completion) | fix vs base |
|---|---:|---:|---:|
| wikitext-00 (prefix replay) | 16.4 (512) | 29.6 (442) | -45% |
| wikitext-01 | 15.5 (385) | 23.8 (512) | -35% |
| wikitext-02 | 16.5 (112) | 17.1 (109) | -3% |
| wikitext-03 | 11.6 (10) | 13.8 (46) | -16% |
| pg19-00 | 16.0 (512) | 18.7 (512) | -14% |
| pg19-01 | 15.5 (508) | 19.5 (512) | -21% |
| pg19-02 | 12.8 (188) | 14.6 (218) | -12% |
| pg19-03 | 15.0 (459) | 20.3 (512) | -26% |

Long-run requests (>= 385 completion tokens, alive to the end of the batch): base
18.7-29.6 tok/s vs fix 15.0-16.4 tok/s. Steady-state per-round device wait while all
long runs are active: base 98-101 ms/round vs fix 125-132 ms/round (+26-33%).
Wall-including-prefill aggregates: base 54.5 vs fix 49.9 tok/s over the batch (52.6 s vs
53.8 s walls; 2863 vs 2686 completion tokens).

### Prefill (~1.9-2.1K token prompts, MC=8 batch, non-replay requests)

base 1.28-1.46 s vs fix 2.20-3.32 s per prompt — base is ~2x faster.

### MTP acceptance

Similar on both trees: 1.87-2.27 (base) vs 1.94-2.12 (fix) accepted tokens per round on
the long runs; draft window 2, no route change between the trees.

## Reading

- The QSA route change (the PR's engine content) is a decode-speed cost at concurrency on
  GB10 at short context: ~-5% at MC=1, ~-15% to -45% per-request at MC=8 (per-round
  device time +26-33% in steady state), plus a ~2x slower prefill. It is neutral to small
  at MC=1. This is on top of the correctness gain the PR buys (the greedy golden unifies;
  the exact logit tie resolves to the plain-decode choice).
- The `d1efb24f` commit is labeled `perf` (skip index selection when every visible key
  fits the budget). At the 2-3K contexts served here, the skip-selection path does dense
  attention over all visible keys instead of a selected subset; the measurement says the
  selected-subset route was the faster one on GB10 at these lengths. Per-commit
  attribution (d1efb24f vs eb9e87fa) and the kernel-level ncu comparison are the pending
  follow-ups; this A/B covers the combined range as instructed.

## Caveats

- Single sweep per (tree, N); no repeat runs. The MC=1 pair agreeing within 5% argues
  against thermal/drift bias; the MC=8 and prefill gaps are well beyond plausible noise.
- Greedy completions differ between trees (different stop positions), so the MC=8 batches
  carry different completion mixes; the per-stream table and the steady-state per-round
  numbers are the comparable quantities.
- The N=8 batch re-sends stream 0 (already served in the N=1 phase); its prompt is served
  from the engine's prefix cache (1875/1882 tokens, `prefix_reuse_path:
  private_response_replay`) on both trees. Decode-only rates are unaffected; the 16.4 vs
  29.6 row is still the most directly paired long-run comparison (identical prompt).
- Corpus streams share no text prefix (verified: 0-1 common characters between
  neighbours), so the cache hit comes only from the N=1 run's own prompt.
- One harness slip, no product effect: the first C1 ctest invocation used a relative
  artifact path, which does not resolve from ctest's build-directory CWD (0/4 with
  "open: No such file or directory", ~19:31Z); re-run immediately with the absolute path,
  which is what C1 reports.

## Provenance

- Fix tree build: `a1a43667`, incremental from the verify-branch build (test-only delta:
  recompile + relink of the real test, 3.5 s, BUILD EXIT 0, 19:28Z); `ninfer-serve`
  18:03Z (engine code = `eb9e87fa`, unchanged by the golden commit).
- Base tree: `/home/apollo11/ninfer-gb10-base` detached at `ed6525fa`, `ninfer-serve`
  18:43Z, `sm_121a`.
- Raw data: `C-DECODE/FIX/` and `C-DECODE/BASE/` (server.log, sweep.json, request.jsonl,
  request_summary.md). Sweep runner: `tools/gb10/pr36fix_decode_sweep.sh`.
