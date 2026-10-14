# EXL3 Trellis Contract (strategy SP-7)

Status: active reference for SP-7. This document is the semantic contract for EXL3-encoded weights
in NInfer's v3 container: the format definition, the two oracles, the container extension, the
tiering change that per-expert widths require, the converter routes, the kernel contracts, and
acceptance. The delegated work (§9) executes against it. The fork owns every change to what
"correct" means.

## 0. Decisions

- **EXL3 compatibility.** The format is bit-compatible with EXL3 as written by exllamav3 v1.6.1
  (turboderp-org/exllamav3, commit `3a513ac9`, 2026-10-11). NInfer decodes any in-scope tensor that
  version writes to the same FP16 inner weights that exllamav3's own reconstruction produces, and
  the fork's converter writes the same format. §2 states the format completely; the pinned commit is
  the external ground truth for the oracle fixtures (§3.3), not a second definition. A later
  exllamav3 format change is adopted only by revising this contract and re-pinning.
- **Per-expert widths.** Every routed expert of a bank carries its own bitrate. This is native to
  the format, because EXL3 stores each expert as its own tensor. exllamav3 v1.6.1 assigns one
  bitrate per group of linears (all experts of a projection in a layer) and its MoE fast paths
  require one bitrate and codebook across a projection's experts (its `MultiLinear` asserts it), so
  mixed-width banks come from the fork's quantize route (§6.2) or another producer of the same
  format, not from exllamav3. Per-expert widths make expert byte extents unequal, which changes the
  SP-4 tiering machinery (§5).
- **Transforms are per expert and per projection.** EXL3 derives each linear's input and output
  vectors from that linear's own weights, so a routed expert's gate, up and down projections each
  carry their own. Sharing is not part of the format.

## 1. Scope

In scope for SP-7: the routed expert banks of the Qwen3.6 35B-A3B MoE (hidden 2048, 256 experts,
expert intermediate 512, top-8), with every other tensor in the artifact keeping its existing
representation. Dense projections, the shared expert, attention/GDN projections, the embedding and
lm_head follow in later steps under this same contract. The 125B Flash-Next banks (512-way MoE) are
a later step that adds shapes, not semantics.

Out of scope, and rejected at import: exllamav3's n-gram embedding codec (160-element tiles),
quantized dimensions padded to a multiple of 128 (`trim_padded_out`), linear biases, the obsolete
`scale` tensor, half-integer step masks other than `0xAAAA`, and mixed codebooks within one bank.

## 2. The format

### 2.1 One EXL3 linear

An EXL3 linear with `k` input features and `n` output features (both multiples of 128) stores:

| Tensor | Type and shape | Meaning |
|---|---|---|
| `trellis` | int16 `[k/16, n/16, 16·K]` | packed trellis, one ring per 16×16 tile (§2.3, §2.4) |
| `suh` | FP16 `[k]` | input-side vector: sign × channel scale, signed |
| `svh` | FP16 `[n]` | output-side vector: sign × channel scale, signed |
| `mcg` or `mul1` | int32 scalar, optional | codebook marker (§2.5); absent means 3INST |

The bitrate `K` is read from the trellis shape: `K = trellis.shape[-1] / 16`. Legacy packed-sign
tensors `su`/`sv` (int16 bitfields) are accepted at import and expanded exactly to `±1` FP16
vectors: bit set → `-1`, bit clear → `+1`, bit `b` of element `w` giving entry `16·w + b`.

Let `Ŵ` be the FP16 **inner weight**, shape `[k, n]` (input-major), decoded from `trellis` by
§2.3–§2.5. Let `Hn_m` be the block-diagonal matrix of `m/128` copies of `H128 / √128`, where `H128`
is the Sylvester Hadamard matrix in natural order, `H128[i][j] = (−1)^popcount(i AND j)`. For an
input row `x` of length `k`, the linear's mathematical output is

```
u = (x ⊙ suh) · Hn_k
z = u · Ŵ
y = (z · Hn_n) ⊙ svh
```

Equivalently `y = x · W` with `W = diag(suh) · Hn_k · Ŵ · Hn_n · diag(svh)` (Hn is symmetric). `suh`
already includes EXL3's codebook scale and global scale; nothing else multiplies `Ŵ`.

### 2.2 The routed MoE with EXL3 banks

For routed expert `e`, `gate_e` and `up_e` are EXL3 linears with `k = 2048`, `n = 512`, and `down_e`
has `k = 512`, `n = 2048`. The SparseMoe mathematics is unchanged: `h = SiLU(x·W_gate,e) ⊙
(x·W_up,e)`, `y_e = h·W_down,e`, and the layer output is the routing-weighted sum over the selected
experts plus the shared expert. EXL3 changes only how each `W` is represented. Gate and up are
separate EXL3 linears with distinct `suh`. They are not fused into one weight, because one fused
weight cannot carry two input-side vectors.

### 2.3 Tiles and element order

`Ŵ` is cut into 16×16 tiles. Tile `(a, b)` covers rows `16a … 16a+15` of the input dimension and
columns `16b … 16b+15` of the output dimension, and is stored at `trellis[a][b]`. A tile's 256
weights form one trellis sequence in tensor-core element order: sequence position `i` (0…255) holds
tile element `(r, c)` with

```
t = i >> 3,  j = i & 7
r = 2·(t & 3) + (j & 1) + 8·((j >> 1) & 1)
c = (t >> 2) + 8·((j >> 2) & 1)
```

so `Ŵ[16a + r][16b + c]` is the value decoded at position `i` of tile `(a, b)`.

### 2.4 The trellis ring

Each tile is an independent tail-biting ring of `R = 256·K` bits. No initial state is stored.

- **Storage.** The tile's `16·K` int16 elements are read as `8·K` little-endian uint32 words
  `w[0 … 8K−1]`. Ring bit `p` (`0 ≤ p < R`) is bit `31 − (p mod 32)` of `w[p / 32]`: the ring is
  most-significant-bit first within each 32-bit word. In int16 terms, element `2m+1` holds ring bits
  `32m … 32m+15` and element `2m` holds `32m+16 … 32m+31`, each most-significant-bit first.
- **Step widths.** Position `i` contributes `D(i)` new bits. For integer `K` (1…8), `D(i) = K`. For
  a half-integer `K = KA + ½` (`K` ∈ {1.5, 2.5, 3.5}, the rates exllamav3 v1.6.1 writes; MUL1
  codebook only), `D(i) = KA + ((0xAAAA >> (i mod 16)) & 1)`, alternating `KA`- and `(KA+1)`-bit
  steps. Let `S(i) = D(0) + … + D(i)`, so `S(255) = R`.
- **State.** The state of position `i` is the 16-bit unsigned integer formed by ring bits
  `S(i) − 16 … S(i) − 1` (indices mod `R`), the first bit most significant. Consecutive states
  therefore satisfy the bitshift constraint `s_i = ((s_{i−1} << D(i)) | new bits) & 0xFFFF`, and
  the windows of positions near the start wrap to the end of the ring.

The bitrate is `K` bits per weight: a tile stores `32·K` bytes. Each linear adds `2·(k + n)` bytes
of `suh`/`svh`.

### 2.5 Codebooks

A codebook maps a state `s` to the FP16 inner weight. All integer arithmetic is on uint32 modulo
`2^32`. `half(v)` reinterprets the 16-bit pattern `v` as IEEE binary16. `⊕` is binary16 addition
rounded to nearest-even. `fma16(a, b, c)` is `a·b + c` computed exactly and rounded once to
binary16, nearest-even.

| Codebook | Marker | Value of state `s` |
|---|---|---|
| 3INST | none | `x = s·89226354 + 64248484`; `x = (x AND 0x8FFF8FFF) XOR 0x3B603B60`; `half(x AND 0xFFFF) ⊕ half(x >> 16)` |
| MCG | `mcg = 0xCBAC1FED` | `x = s·0xCBAC1FED`; `x = (x AND 0x8FFF8FFF) XOR 0x3B603B60`; `half(x AND 0xFFFF) ⊕ half(x >> 16)` |
| MUL1 | `mul1 = 0x83DCD12D` | `x = s·0x83DCD12D`; `b` = sum of the four bytes of `x` (unsigned); `fma16(half(0x6400 + b), half(0x1EEE), half(0xC931))` |

A marker's stored value must equal its multiplier; any other value is rejected. The binary16
rounding inside these formulas is part of the format. It is not a production staging choice, and the
exact codec oracle (§3.1) reproduces it.

## 3. Oracles

Two oracles, per `op-development.md` §6.1: an exact oracle for the codec, and an FP64 oracle for the
Op that consumes it.

### 3.1 Exact codec oracle

- **Input**: the stored `trellis` elements (exact bits), `K`, and the codebook.
- **Evaluation**: §2.3–§2.5 literally, per tile and per position: bit extraction from the ring,
  state formation, the codebook's integer operations and its specified binary16 rounding.
- **Output criterion**: bit equality of every decoded FP16 inner weight `Ŵ`.
- **Use**: converter payload verification (§6), the checks against exllamav3 (§3.3, §8.3), and any
  route that materializes `Ŵ` (a standalone decode or a reconstruct-then-GEMM route).

### 3.2 Op oracle (SparseMoe with EXL3 banks)

- **Input**: the represented BF16 activations, the routing inputs, the exact decoded `Ŵ` of every
  selected expert projection (from §3.1), its `suh`/`svh`, and the shared expert's existing weights.
- **Evaluation**: §2.1–§2.2 in FP64, including both Hadamard transforms of every projection. It does
  not reuse a production kernel's fusion, folding, staging casts or reduction order.
- **Output criterion**: the existing SparseMoe A16 criterion (`kSparseMoeA16Tolerance` in
  `tests/ops/test_sparse_moe.cpp`: relative L2 ≤ 1.2e-2, gross absolute error ≤ 4.0e-3). That bound
  belongs to the BF16-in/BF16-out compute profile, not to a codec. An EXL3 route that needs a wider
  bound must justify it as a precision profile with measured maxima before the bound changes.
- **Fused kernels** never materialize `Ŵ`. They are qualified against this oracle only. §3.1
  qualifies the codec itself.

### 3.3 Independence and ground truth

- The exact codec oracle exists twice, both written from this contract alone. The Python
  implementation under `tools/convert/<target>` serves converter verification. The C++ test-owned
  decoder serves `tests/ops` kernel qualification. Neither calls production code.
- **External fixture.** A small fixture generated once with the pinned exllamav3 commit: for every
  `K` in 1…8 and {1.5, 2.5, 3.5} and each codebook (MUL1 only for half-integer `K`), a few random
  tiles with their exllamav3-reconstructed FP16 `Ŵ` (`ext.reconstruct`), and one full 128×128 linear
  with its `suh`/`svh` and exllamav3's `get_weight_tensor` result. Both oracle implementations must
  reproduce the reconstructed tiles bit-exactly. The full-weight composition, which exllamav3
  evaluates in FP16, is supplementary evidence for §2.1's transform order and is compared within
  FP16 tolerance, not bit-exactly. The fixture, its generation command and the pinned commit are
  committed with the oracle.
- **Shapes**: the 35B banks (gate/up `k=2048, n=512`; down `k=512, n=2048`; 256 experts) with
  uniform `K` ∈ {2, 3, 4, 5, 6} and a mixed bank whose experts span integer and half-integer `K`;
  plus the single 16×16 tile and the 128×128 linear for the codec oracle.

## 4. Container extension (v3)

EXL3 is a bounded extension of the existing binding and resource model, not a new container version.
The persistent names below are fixed by this contract. `tensor-formats.md` and `storage-layouts.md`
gain their entries when the codec lands.

- **Format** `exl3`: numeric decode per §2.3–§2.5. Object scalar: the codebook (`3inst`, `mcg`,
  `mul1`), uniform within a bank.
- **Layout** `exl3_expert_bank`: one object per (layer, projection), i.e. `routed_gate`, `routed_up`
  and `routed_down`, each with logical shape `[E·n, k]` so the existing row convention and the tier
  plan's `experts × rows_per_expert` check (`rows_per_expert = n`) hold unchanged. Planes:

| Plane | Content |
|---|---|
| `experts` | `E` entries of 16 bytes: u64 byte offset of the expert's trellis extent, u32 extent bytes, u8 `2K` (2…16; 3, 5 or 7 for the half-integer rates), three zero bytes |
| `trellis` | each expert's exllamav3 `trellis` bytes verbatim (int16, little-endian), in expert id order; expert `e`'s extent is `k·n·K_e/8` bytes, zero-padded to a multiple of 256 bytes |
| `suh` | FP16 `[E, k]`, row `e` = expert `e`'s `suh` |
| `svh` | FP16 `[E, n]`, row `e` = expert `e`'s `svh` |

- **Load validation**: offsets increase in expert id order, are 256-byte aligned, and match the
  extents implied by `2K`. An odd `2K` requires MUL1 and `K` ∈ {1.5, 2.5, 3.5}. Both `k` and `n`
  are multiples of 128, and the bank's codebook is uniform.
- **Provenance** (descriptive only): the source exllamav3 version and checkpoint identity, or the
  fork converter's recipe and bitrate map. Nothing needed for execution lives in provenance.

## 5. Tiering with variable expert extents (SP-4 change)

SP-4 today assumes a constant per-expert byte size per bank: a cache slot "costs its per-expert
bytes in every bank", and a residency entry's compacted rank addresses an expert directly.
Per-expert widths break that assumption for the `trellis` plane. The change is generic (an extent
per expert), not an EXL3 branch, and the existing codecs are its constant-extent case:

- **Mini-banks.** The hot (device) and warm (pinned host) mini-banks pack their experts in expert id
  order at their own 256-byte-aligned extents. Fixed-stride planes (`suh`, `svh`) keep their stride.
- **Extent table.** Per tiered bank, a device table maps compacted rank to (byte offset, `2K`). The
  residency entry keeps its meaning (tier and compacted rank). Kernels resolve address and bitrate
  through the extent table.
- **Cache tail.** Slots stay equal in count across a layer's banks. Each slot is sized to the bank's
  largest expert extent, and that size is the per-slot cost the device budget uses. A promotion
  copies the expert's actual extent and updates its extent-table entry together with the residency
  flip, at the existing promotion boundaries only.
- **Graph stability.** The extent tables live at startup-fixed addresses. Their contents change only
  at promotion boundaries, as residency does today.
- **No regression.** Constant-extent banks (groupwise, NVFP4) keep their current addressing cost and
  measured performance. The 27B baseline is untiered and unaffected.

## 6. Converter routes (`tools/convert/<target>`)

### 6.1 Import route (first deliverable)

1. Read an EXL3 checkpoint whose `config.json` carries `quantization_config.quant_method = "exl3"`
   and whose `version` is the pinned one (or a version this contract has re-pinned).
2. For each layer and expert, read `model.layers.{i}.mlp.experts.{e}.{gate_proj,up_proj,down_proj}`
   with its `trellis`, `suh` or `su`, `svh` or `sv`, and marker. Validate shapes, `K`, markers,
   bank-uniform codebooks, and the §1 rejections.
3. Repack verbatim into the §4 planes. Trellis bytes and `suh`/`svh` are copied unchanged; packed
   `su`/`sv` are expanded per §2.1.
4. Verify the emitted planes with the Python exact oracle: for every expert, the first and last tile
   plus 14 deterministically seeded tiles decode from the emitted `experts`/`trellis` planes
   bit-identically to the same tiles decoded from the source tensors, and every emitted `suh`/`svh`
   row equals its source. This catches repacking and offset errors; the comparison with exllamav3's
   own reconstruction is acceptance item §8.3.
5. Take every non-routed tensor from the BF16 source checkpoint through the existing recipe, and
   check that the EXL3 checkpoint's architecture configuration matches that source.

### 6.2 Quantize route (per-expert widths)

The fork's converter emits the same format by EXL3's method. The steps:

1. **Hessians**: per expert, from the calibration tokens routed to it, using the frequency-corpus
   discipline. An expert with fewer routed calibration tokens than a stated threshold uses the
   layer's pooled routed-token Hessian. Damping is applied before LDL as in EXL3.
2. **Incoherence**: EXL3's derivation. Random signs, output-channel scales into `svh` when the
   out-scales mode applies them, the output-side Hadamard, block-RMS input-channel scales folded
   with `−1/codebook_scale` into `suh`, the input-side Hadamard, and the global-scale search folded
   into `suh`.
3. **Quantization**: LDLQ over 16-row blocks, with each 16×16 tile in tensor-core order quantized as
   one tail-biting ring by Viterbi over the bitshift trellis at the expert's `K` and the bank's
   codebook.
4. **Bitrate map**: a converter input assigning `K` per (layer, projection, expert), integer or
   half-integer (half-integer only with MUL1).
5. **Verification**: the same exact-oracle check as §6.1 step 4. The output must also pass the
   import route's validation unchanged.

## 7. Kernel contracts (`src/ops`)

- **Op.** The SparseMoe Op's mathematical contract is unchanged (§2.2), with BF16 activations in and
  a BF16 destination. EXL3 banks are a new weight representation admitted at the Op boundary.
  Unsupported combinations are refused at admission, never run on a silent slow path.
- **Decode.** Each weight's 16-bit state is read from its ring with a funnel shift across 32-bit
  words (§2.4), then mapped by the bank's codebook (§2.5). Positions decode independently, and
  kernels must not serialize the window extraction. The decode is fused into the expert GEMM/GEMV
  and may choose its own intermediate precision. §3.2 qualifies the result.
- **Transforms.** Per selected expert, gate and up apply their own input-side Hadamard and `suh` to
  the 2048-wide input and their own output-side transform over 512 outputs. Down applies its own
  over 512 inputs and 2048 outputs. A route may fold transforms into a reconstruct path (exllamav3
  does so for large batches) when the result is checked against §3.2.
- **Bitrates.** `K` is read per expert from the extent table. Compile-time specializations for
  common uniform rates are allowed, but a mixed bank must have a runtime-`K` path that stays on the
  fused route.
- **Codebooks.** All three are part of the codec. Kernel admission starts with MUL1 (exllamav3
  v1.6.1's default and the only codebook with half-integer rates) and MCG, the two its fused MoE
  supports, and refuses 3INST banks until a qualified route exists.

## 8. Acceptance

1. **Codec**: both oracle implementations reproduce the §3.3 fixture bit-exactly for every admitted
   (`K`, codebook) pair.
2. **Op**: the EXL3 SparseMoe routes pass §3.2 at the §3.3 shapes for uniform and mixed-width banks,
   for each admitted codebook, in both untiered and tiered placements (hot, warm and cache-tail
   residency, including a promotion between rounds).
3. **Import**: a real 35B EXL3 checkpoint at the pinned version imports, loads and serves. Its
   routed banks decode bit-identically to exllamav3's reconstruction on the §6.1 sampled tiles.
4. **Quality per GB**: `ninfer-perplexity` on the fixed corpus (`eval/corpora/perplexity-1m`,
   default protocol) for artifacts that differ only in the routed banks: BF16, NVFP4, and EXL3 at
   two or more routed-bank byte budgets, including one at or below NVFP4's. Report perplexity
   against routed-bank bytes, with the calibration provenance of each EXL3 bank.
5. **Performance**: `ninfer_bench` decode at concurrency 1, 4 and 8 on the tiered 35B path under the
   same device budget, EXL3 against NVFP4 routed banks, plus prefill at 2048 tokens. Report
   throughput and the resident expert count. A slowdown is a reported finding. The pass threshold is
   set by the owner before the measurement.

## 9. Delegation boundary

Delegated, per the strategy's Opus route: the Python and C++ exact codec oracles and a reference
decoder, written against §2–§3 of this contract, together with the generation of the §3.3 fixture.
The fork owns this contract and its pinned reference, the container extension, the tiering change,
both converter routes, the kernels, the acceptance criteria and the baseline validation. A
disagreement between an oracle and the fixture is resolved by correcting the contract or the oracle,
never by loosening the codec criterion.
