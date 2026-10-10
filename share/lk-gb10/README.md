# lkarlslund/ninfer master on GB10

Six patches that make `lkarlslund/ninfer` master (`8f574ee4`, 2026-10-07) build and run on GB10, so its
Flash-Next work can be measured beside this fork before anything is cherry-picked. The first three change no
behaviour on an RTX PRO 6000 except grid sizes there (188 SMs instead of the 5090's 170); 0004 fixes a
startup crash that only shows with a batch-3 MTP decode graph; 0005 and 0006 are the fixes found during
the 2026-10-10 campaign. 0001-0003, 0005 and 0006 are also the upstream PRs in `share/lk-upstream/`.

| patch | what | from this fork |
|---|---|---|
| 0001 | CMake accepts `121a`; both family runtimes accept compute capability 12.1 | `589e4df1` |
| 0002 | integrated devices size startup memory from `MemAvailable` less a 6 GiB reserve and the pending Host KV arena, after waiting up to 60 s for a just-exited process's memory | `7e23083c`, `7bfa75fb`, `b61f9867` |
| 0003 | persistent grids sized from the device's SM count instead of the 5090's 170 (RMSNorm, RoPE, GDN chunked output, sparse-MoE prefill, QSA score, Flash-Next MoE grouped) | `589e4df1` |
| 0004 | Flash-Next MTP RoPE layout decided by the element count, not a shape test on `ne[1]` (the shape test misread `{width,batch}` text positions as three-axis MRoPE whenever the decode batch was 3; the batch-3 MTP graph capture then threw a view element-count mismatch at startup) | `11f9e7a6` (already in this fork; lk master never received the fix) |
| 0005 | the fused HyperConnection route is decided before its workspace is allocated (below 160 SMs it fell back with ~25.7 KB a token still held; the op test failed with `std::bad_alloc` on GB10) | new |
| 0006 | Flash-Next template accepts `reasoning_effort` `none` with disabled thinking (OpenAI requests with `none` got HTTP 400) | new |

Deliberately not ported:
- our `yield` synchronize, because lk's low-latency round wait (`a9ccbabe`) answers the same wake-up cost;
- the 4096-token prefill default, which the run list passes as `--prefill-chunk 4096`;
- the scheduling policies (long-prefill wait, decode share, yield, backfill).

lk master already has the converter's chunked reads (the GB10 2 GiB read clamp). His new cooperative
HyperConnection kernel sizes its grid from the device and falls back to the general route when the grid
does not fit (on 48 SMs it always does), so it never deadlocks; 0005 fixes the workspace that fallback
leaked.

0001-0004 built and served the 2026-10-10 campaign on GB10. 0005 and 0006 passed the same day's
validation run in `share/lk-upstream/README.md`.

## Apply and build

```bash
git clone https://github.com/lkarlslund/ninfer lk-ninfer && cd lk-ninfer
git checkout 8f574ee4
git am /path/to/ninfer-gb10/share/lk-gb10/*.patch
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=121a
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Artifact

lk's binder picks its profile from the PLE table's format and reads each projection by its stored
format. Our `fp8mtp` artifact uses the same `fp8_e4m3fn_row_bf16` format name, but lk's load plan
requires the shared expert's down projection as exact BF16 and `fp8mtp` stores it as row-scaled
FP8, so his binder refuses it (confirmed 2026-10-10). Try it first (a
refusal costs seconds). If it is refused, convert with lk's converter from the same RadixArk source,
with his equivalent of `fp8mtp`:

```bash
python3 -m tools.convert.qwen3_8_flash_next_125b_a6b.convert \
  --model /path/to/RadixArk-Qwen3.8-Flash-Next-NVFP4 --out out/lk_fp8.ninfer \
  --projection-format fp8_e4m3fn_row_bf16 --mtp-expert-format nvfp4
```

The source's PLE table is already FP8, so no `--ple-format` is needed. The result is about 119 GB; check
free disk first.

## Run list (one GPU job at a time)

1. Build and `ctest` as above. Also run the Flash-Next real-artifact test with
   `NINFER_QWEN38_FLASH_NEXT_WEIGHTS` pointing at the lk artifact.
2. Parity protocol (DGPP's `serve_load.py`, five classes, 256 tokens, greedy, three reps, C1/C2/C4),
   with the same serve flags as our October 3 rows (`--kv-dtype fp8 --lm-head-draft --host-kv-mib 0`,
   `--prefill-chunk 4096`), in four arms:
   - lk, `--spec mtp --draft-tokens 3`: like-for-like against our K=3 rows;
   - lk, `--spec mtp` (adaptive 3..7 drafts, his default);
   - lk, `--spec mtp --draft-tokens 1`: against DGPP depth 1 and our K=1;
   - our branch at its current head, K=3: same day, same box.
3. `tools/gb10/k_sweep.sh` on the natural corpus for the lk arms (acceptance per position and tok/s),
   with the lk server binary.
4. Batch invariance: on the lk build at C4, the same 4-request replay twice; compare outputs (his
   `2fa4c756` claims a row's result does not depend on its batch). Our branch is not invariant at C4.

## 8-bit arm (second campaign)

The 2026-10-10 campaign (`profiles/bench/gb10/lk-parity-2026-10-10/`) compared the engines on the
NVFP4 `v3_fork` entry, whose dense projections are BF16. Each engine's own 8-bit path was not
measured, though that is where both spent their recent work. This arm measures each on its own
8-bit artifact.

1. **Convert lk's 8-bit artifact** on the lk build (the converter uses the GPU: one job at a time,
   about 130 GB of disk):
   ```bash
   cd ~/lk-ninfer && python3 -m tools.convert.qwen3_8_flash_next_125b_a6b.convert      --model <the RadixArk NVFP4 source the v3_fork entry came from>      --out <dir>/lk_fp8.ninfer      --projection-format fp8_e4m3fn_row_bf16 --mtp-expert-format nvfp4
   ```
   Our counterpart is the existing `fp8mtp` entry. The two differ in what is 8-bit:
   - lk keeps the shared-expert down projection, the GDN control and the HyperConnection injection
     in BF16;
   - `fp8mtp` makes the shared-expert down FP8 too, and its MTP layer uses the main layers'
     formats.

   So acceptance and quality can differ as well as speed.
2. **Load gate:** `ninfer_qwen3_8_flash_next_load_plan_test` and the two Flash-Next real-artifact
   tests on the lk build, with `NINFER_QWEN38_FLASH_NEXT_WEIGHTS` pointing at `lk_fp8`.
3. **Parity,** with `profiles/bench/gb10/lk-parity-2026-10-10/parity_arm.sh` (same flags, C1/C2/C4,
   five classes, three repeats):
   - `lk8-k3`: the lk binary, `lk_fp8`, `--spec mtp --draft-tokens 3`;
   - `lk8-auto`: the lk binary, `lk_fp8`, `--spec mtp`;
   - `ours8-k3`: our binary at head, `fp8mtp`, `--spec mtp --draft-tokens 3` (a same-day rerun of
     the October 3 rows, 59.89/91.53/121.39).
4. **Acceptance:** `tools/gb10/k_sweep.sh` with `KS="3"` for both, with `SERVE_BIN` set per arm.
5. **Quality check:** each engine's `ninfer-perplexity` on its own artifact, over the same 4K-window
   corpus as our gate (`fp8mtp`: 3.9982). A speed lead bought with worse perplexity is not a lead.
6. **If `lk8` trails at C4:** one nsys trace of a C4 K=3 round on the lk build. His FP8
   HyperConnection relies on the fused route, which widens FP8 at the MMA. On GB10 the cooperative
   grid does not fit (48 SMs; the route needs 160), so every mix takes the general route. That route
   widens the whole 320×10240 FP8 Up matrix to BF16 in workspace on every call. If the trace shows
   that, the gap is a GB10 routing artifact of his tree, not his FP8 design, and a small Up-widening
   fix for his general route would be the useful upstream contribution.

## Cherry-pick candidates (pending the measurements above)

Correctness:
- `535e9f28` (GDN prefill chunk-invariance) does not apply here. This fork took upstream's two-stage GDN
  in the `e31bc99b` merge, which has no "chunked prefix plus recurrent tail" split: a call of 16 or more
  tokens runs entirely chunked, a shorter call runs recurrent. The same class of question is open here
  instead. `tests/ops/test_gated_delta_net.cpp` now checks bitwise that chunk-aligned splits (4096+313,
  4×1024+313, 256+57) reproduce one call. It records, without requiring, the one known divergence: a
  final piece under 16 tokens (4096+4) takes the recurrent route.
- `68409b45` ordinary CUDA graph replay and `3509dcde` chunked GDN precision were written against lk's
  older GDN and graph code. Check each against this fork's merged code before porting; `3509dcde` would
  need re-deriving on the two-stage kernels and then our perplexity and drift gate.

Features, decided by step 2:
- `532778a5` and `d96e6315`: adaptive MTP, 1..7 drafts with a graph family per width. Porting it means
  merging with our device-resident round frame (K2) and in-graph PLE stage (K3).
- `2fa4c756`: batch-invariant verify up to 16 tokens. It depends on `1cd1b917` (HyperConnection mix up to
  16 tokens) and on `97a033a2` and `d24f9c2c` (FP8 MoE entry and routing overlap for 2..8 tokens).
- `8df59b0b`: NVFP4 MTP experts with calibrated divisors. We have our own (7d, max-abs); compare
  acceptance.

Kernels that overlap ours, decided per kernel against our microbenches:
- MoE latency pieces, the 3.2 ms "other" share of a C4 round: `e6c1c3f2`, `42fd08d7`, `647be071`,
  `528023a8`, `52a604a4`;
- QSA: `2b01139f`, `1f3ab2c5`, `a92df9a8`, `0f31336e`;
- HyperConnection: `37e781f1`, `a18aa31b`, `b43ccf5a`, `d06914e4`;
- FP8 K-split double buffering: `8a38df31` (our three-stage attempt was reverted);
- PLE: `f202d3f2` and `b66d6ecc`, against our overlapped page-ins.

Duplicates of our FP8 work (`87163868`, `d2f42756`, `bd7c391c`, `9b80b069`). Keep ours unless his
kernels measure faster. Artifact compatibility decides how much of either side can move.

Not relevant to the GB10 product:
- activation capture and steering (`f74ced1d`, `742a38e7`);
- INT8 KV (`fef9f976`): capacity only, since decode is flat across context here;
- docs and bench commits.

Swift 1.5 support (`2dcc5b13`, `6565a2b2`) is optional: a fine-tune that thinks about 60% less, which
may suit the operations workload.
