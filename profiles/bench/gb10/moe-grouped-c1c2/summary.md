## MoE grouped decode: C1/C2 K=1 end-to-end (branch 2e54cfe0)

- Protocol: DGPP serve_load.py, 5 classes, greedy, thinking off, 256 tokens, 3 repeats;
  page cache dropped before the load. Baselines: the working-branch I9 rerun of 2026-10-02 (45.05 / 66.64).
- C1 (2 rows): 44.22 tok/s (3840 tok / 86.8 s) = -1.8% vs working branch
- C2 (4 rows): 66.33 tok/s (7680 tok / 115.8 s) = -0.5% vs working branch (flat)
- C4 (8 rows): +3.1% (separate record, moe-grouped-check)
- Switch-over between 4 and 8 rows.

- Commit: 2e54cfe0 (2026-10-02T15:27:27Z)
