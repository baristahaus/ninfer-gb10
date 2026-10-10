# lkarlslund/ninfer master on GB10

Three patches that make `lkarlslund/ninfer` master (`8f574ee4`, 2026-10-07) build and run on GB10, so its
Flash-Next work can be measured beside this fork before anything is cherry-picked. They change no
behaviour on an RTX PRO 6000 except grid sizes there (188 SMs instead of the 5090's 170).

| patch | what | from this fork |
|---|---|---|
| 0001 | CMake accepts `121a`; both family runtimes accept compute capability 12.1 | `589e4df1` |
| 0002 | integrated devices size startup memory from `MemAvailable` less a 6 GiB reserve and the pending Host KV arena, after waiting up to 60 s for a just-exited process's memory | `7e23083c`, `7bfa75fb`, `b61f9867` |
| 0003 | persistent grids sized from the device's SM count instead of the 5090's 170 (RMSNorm, RoPE, GDN chunked output, sparse-MoE prefill, QSA score, Flash-Next MoE grouped) | `589e4df1` |

Deliberately not ported:
- our `yield` synchronize, because lk's low-latency round wait (`a9ccbabe`) answers the same wake-up cost;
- the 4096-token prefill default, which the run list passes as `--prefill-chunk 4096`;
- the scheduling policies (long-prefill wait, decode share, yield, backfill).

lk master already has the converter's chunked reads (the GB10 2 GiB read clamp). His new cooperative
HyperConnection kernel sizes its grid from the device and falls back to the general route when the grid
does not fit, so it is safe on 48 SMs.

Checked in a sandbox without a GPU: the seven touched `.cu` files compile with nvcc 13.0.88 for
`sm_121a`, and `model_instance.cpp` passes a syntax check. The full build has not run.

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
format. Our `fp8mtp` artifact uses the same `fp8_e4m3fn_row_bf16` format name, but stores the shared
expert's gate and up as two halves of one parent, so expect the binder to refuse it. Try it first (a
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

## Cherry-pick candidates (pending the measurements above)

Correctness, to take regardless of the benchmark:
- `535e9f28` GDN prefill chunk-invariance. A prompt's first token could depend on how prefill was
  chunked, because full chunks staged normalized q/k while tail-only calls normalized in-kernel. Our tree
  has the same code. It matters more now that our default chunk is 4096 and a yielded prefill resumes
  mid-prompt.
- `68409b45` ordinary CUDA graph replay. Compare it with our own capture-safe valid-columns fix before
  taking it.
- `3509dcde` chunked GDN precision (FP16 U/v_new, rounded TF32 WY inverse). This changes numerics, so it
  needs our perplexity and drift gate.

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
