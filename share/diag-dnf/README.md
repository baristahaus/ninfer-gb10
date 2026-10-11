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
