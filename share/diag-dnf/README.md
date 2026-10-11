# Locating the bad verify rows behind ninfer-gb10-dnf

A 24-row verify (K=3 at six requests, K=5 at four) faults in the MoE routing pass: an expert id
outside 0-511 at `flash_next_moe.cu:247` for rows 22-23. The MoE and HyperConnection op tests pass
at 24 rows against their oracles, so those rows arrive non-finite from earlier in the pass.

`nonfinite-scan.patch` is a temporary diagnostic, not product code. Applied on the branch head it:

- scans BF16 activations for non-finite values after each stage of every target layer and of the
  MTP head, and inside the MoE op its router input and 512-wide router scores, recording per call
  width, layer and stage the first bad token column;
- prints from the routing kernel whenever its top-10 step finds no valid expert for a row: the row,
  the rank, the invalid id, the row's count of non-finite scores and its largest score. It then
  substitutes a valid id so graph preparation completes;
- after graph preparation, prints the record, the number of scans enqueued per call width and the
  number skipped.

Stages: 0 input, 1 after PLE, 2 hyper state before the mixer, 3 mixer input, 4 mixer output,
5 MoE input, 6 MoE output. Layer 48 is the MTP head (0 its hidden input, 1 its hyper state). Layer
49 is inside the MoE op, for every MoE call: 0 router input, 1 router scores.

v1 (2026-10-11) recorded no non-finite activation at 24 columns. Its top-10 clamp substituted
invalid ids silently, so it could not show whether they occurred; v2 prints them.

## Run (one GPU job; reboot first if a crashed context is still held)

```bash
git apply share/diag-dnf/nonfinite-scan.patch
cmake --build build -j --target ninfer-serve
build/apps/ninfer-serve <fp8mtp entry> --port 8001 --max-concurrency 6 --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 --lm-head-draft --prefill-chunk 4096 2>&1 | tee diag.log
# stop the server once "[diag after graph preparation]" has printed
git checkout -- . && git clean -fd include/ninfer/ops src/ops/launcher   # drop the patch
```

Send every line containing `[diag`, including the `top10` lines from the routing kernel. The first stage, in layer order, whose `columns=24` record names
column 22 or below is where the bad rows are produced.

v2 (2026-10-11) recorded nothing either: no non-finite scan at 24 columns, no `top10` line, no
fault. Compute-sanitizer initcheck on the unpatched server (2026-10-11) found no device-side
uninitialized read. Its only reports are the whole-struct copies of the MTP decode egress, whose
rows past the batch and logprob arrays are never written by design. initcheck does not see loads
issued by TMA or `cp.async`, so the absence of device reports is weak evidence.

## Suspect: the cuBLAS route for T > 16 (2026-10-11)

`ops::linear(x, w, out, stream, bf16_gemm)` sends every BF16 projection with more than 16 tokens
to cuBLAS (`src/ops/linear/linear.cpp`), although the context's contract says graph shapes stay
on NInfer's kernels. Flash-Next passes its context to the MoE router ([512,2560], BF16 in every
artifact) and the PLE projections, so a verify graph with more than 16 rows runs cuBLAS inside the
capture. No test covered that overload: the MoE op test passes no context. The failures fit a
cuBLAS kernel choice that differs by row count: 16 rows (native) and 18 and 20 rows (cuBLAS) work,
while 24 and 32 rows fail.

- `ninfer_linear_bf16_a16_test` now runs the context overload at the router and PLE shapes, T = 1
  to 48, eager and captured. The output starts as NaN, and every element must be finite and match
  the FP64 oracle on sampled rows.
- `cublas-above-48.patch` keeps T <= 48 (every decode and verify graph: 8 rows x width 6) on the
  native kernels. It changes kernel choice only, not the workspace layout: the native A16 route
  takes no workspace.

## Run (one GPU job at a time; reboot first if a crashed context is still held)

```bash
# 1. Op test on the branch head: is the cuBLAS route wrong at some T?
cmake --build build -j --target ninfer_linear_bf16_a16_test
ctest --test-dir build -R ninfer_linear_bf16_a16_test --output-on-failure
# 2. A/B on the server that faults deterministically today
git apply share/diag-dnf/cublas-above-48.patch
cmake --build build -j --target ninfer-serve
build/apps/ninfer-serve <fp8mtp entry> --port 8001 --max-concurrency 6 --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 --lm-head-draft --prefill-chunk 4096
#    starts ("CUDA graphs ready")? then six concurrent greedy chat requests, 256 tokens each
# 3. Only if 2 starts: the patch's cost where cuBLAS ran before, at C5 K3 (20-row verify),
#    patched vs unpatched, the usual natural-corpus decode tok/s
git checkout -- src/ops/linear/linear.cpp   # drop the patch
```

Send: step 1's summary, plus the first failing lines if any (they name T, eager or graph, and the
shape). Step 2: whether startup completes and the six requests finish, and if it faults, the
fault line. Step 3: tok/s for both builds.
