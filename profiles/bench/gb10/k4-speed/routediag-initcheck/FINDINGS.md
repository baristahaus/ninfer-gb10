# initcheck on the MTP decode egress — findings

Date: 2026-10-01 (UTC). Machine: GB10 (sm_121a, aarch64, unified 121 GiB).
Toolchain: CUDA 13.0.88, compute-sanitizer --tool initcheck --launch-timeout 300.
Binary: `build/apps/ninfer-serve` (K4b tree, sha256 80951608… — see serve.log headers).
Artifact: `out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer`.

## Setup

- Serve: `--max-context 512 --max-concurrency 4 --kv-dtype fp8 --preserve-thinking
  --spec mtp --draft-tokens 1 --lm-head-draft --no-cuda-graph`.
  (ctx 512 instead of 73728: the sanitizer shrinks the free memory the engine
  sees, so the 2.8 GiB ctx-73728 reservation does not fit; the 73-token probe
  prefill is identical.)
- One request per run: 73-token prompt, max_tokens=1, temperature 0,
  logprobs on/off per variant.
- Run 1: no `--token-logprobs` → `routediag-initcheck/serve.log`.
- Run 2: `--token-logprobs` added → `routediag-initcheck-lp/serve.log`.
  (Run 2 needed a SIGKILL fallback: the serve hangs on SIGTERM under the
  sanitizer, 2026-10-01 23:2xZ; GPU memory released only after SIGKILL.)

## Run 1 (no --token-logprobs): 100 errors, one DtoH

All 100 errors are sub-chunks of one 8,320-byte `cuMemcpyDtoHAsync`
(`frame.egress.data` → `mtp_host_egress`, mtp_impl.h:322), source base
`0xefdbb9221e00`:

- 100 "Uninitialized access" blocks, 32 B each, contiguous from the buffer
  base: offsets [0, 3200). Each block's API context is the same 8,320-byte DtoH.
- Request still completed: `completion_tokens=1`, `finish=length`.

MtpDecodeEgress layout (offsetof-verified, sizeof = 8,320, matches the DtoH):

```
licensed_tokens   [0, 448)
licensed_counts   [192, 224)
accepted_drafts   [224, 256)
next_drafts       [256, 384)
next_extents      [416, 448)
token_logprobs    [448, 640)
top_ids           [640, 4480)
top_logprobs      [4480, 8320)
```

Flagged (never-written per initcheck): [0, 3200) = all structural fields
(licensed_tokens, licensed_counts, accepted_drafts, next_drafts, next_extents)
+ token_logprobs + top_ids prefix.
Not flagged ("written"): [3200, 8320) = top_ids tail + all of top_logprobs.

## Run 2 (--token-logprobs): 0 errors

Same shape, same request. The same egress DtoH reports no errors at all.

## Why this is a sanitizer artifact, not an engine bug

1. Single-stream ordering. Every op in the decode body and the egress DtoH
   run on `state.execution.device.stream` (mtp_impl.h:184-327). The accept
   kernels' writes complete before the DtoH read in stream order.
2. Consistent bindings. All egress fields bind by offsetof into the same span
   the DtoH copies (round_state.cpp:315-329); kernel write targets and DtoH
   source are the same storage. No host-side write to the device egress
   exists (the only host→device ops are the ingress H2D and set_device_i32,
   both cudaMemcpyAsync; program_impl.h:11270-11273).
3. The host validated the flagged copy. After the DtoH + device.synchronize
   (program_impl.h:12504), the engine checks strict invariants on exactly the
   copied bytes (program_impl.h:12522-12536): count in (0, width],
   accepted+1 == count, next in [0, draft_window], budget/capacity bounds,
   plus validate_licensed_tokens (token-domain check vs 248k vocab). Random
   uninitialized bytes pass all of these with probability ~1e-30. The run
   passed them: the bytes were genuinely written by the accept kernels.
4. Unexplained "written" tail. In run 1, [3200, 8320) is reported written,
   but with the flag off no op in the body writes that region:
   target_logprobs (mtp_impl.h:226-242) is the only writer of the report
   fields and it is gated on io.report_token_logprobs. The sanitizer saw a
   write that the code does not issue → its write tracking for this buffer is
   demonstrably inconsistent in run 1.
5. Flag dependence. The accept kernels are identical in both runs (same
   request, same sampling); only target_logprobs differs. The visibility of
   the structural fields' writes flipped 100 errors → 0 errors.

Conclusion: compute-sanitizer initcheck mis-tracks this arena on GB10
unified memory (CUDA 13.0.88). It is not evidence of an engine bug, and its
absence of errors is not clean evidence either.

## Consequences

- The F1 prefill split (state dumps in `routediag/`, layers 1-35 divergent
  C4 vs C1, layer 0 identical) cannot be attributed to — or cleared of — an
  uninitialized read via initcheck: run 1 showed no prefill errors, but the
  demonstrated tracking unreliability makes that absence non-informative.
- Do not use initcheck as a correctness gate for the eager MTP decode path
  on this machine. A definitive check of egress write coverage is a code
  audit of the accept kernels' write coverage (target_verify_accept and
  mtp_prepare_next_round next_extents), not a runtime sanitizer run.
