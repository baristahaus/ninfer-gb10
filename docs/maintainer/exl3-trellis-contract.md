# EXL3 Trellis Semantic Contract (strategy SP-7)

Status: active reference. This document is the semantic contract for the EXL3 trellis codec in
NInfer's v3 container: the decode mathematics, the oracle specification, the container
extension, the converter recipe, and the kernel contracts. The trellis reference
implementation and its independent naive oracle are delegated work (strategy §"Route to
Opus"): an implementing session executes against this contract, and the fork reviews,
validates against the oracle, and owns every change to what "correct" means. The upstream
references are QTIP (arXiv:2406.11235, Cornell RelaxML) and EXL3 (turboderp-org/exllamav3,
"a streamlined variant of QTIP"); nothing in this contract requires reading their code —
the mathematics below is the fork's own statement of it.

## 1. The semantic model

An EXL3-encoded weight matrix W (logical shape [rows, columns]) carries, per row:

- A **packed trellis sequence** S: a circular bit string. For a row of `T` groups the stored
  sequence is `T · k · V` bits, where `V` is the trellis group size (weights per group), `k`
  is the bits per weight, and `L` is the trellis window depth with `k·V ≤ L`. The sequence
  is tail-biting: the final `L − k·V` bits wrap around to the start, so no initial state is
  stored.
- A **codebook** C shared by the layer (or bank): `2^L` entries of `V` values each. The
  hybrid computed-lookup codes store small codebooks whose entries are dyadic values; the
  computed variants derive entries arithmetically (≤ 4 instructions per weight) instead of
  storing `2^L × V` numbers.
- A **scale** per row (or per row-group): a positive FP16 value, exactly the stored scale.

**Decode (the complete logical formula).** The t-th group of `V` weights (0-indexed) depends
only on the contiguous window of `L` bits of `S` starting at bit `t·k·V` (circular):

```
window_t   = bits [t·k·V, t·k·V + L) of S            (circular bit extraction)
node_t     = the L-bit integer window_t                (the trellis node)
ŵ[t·V : (t+1)·V] = C[node_t] · scale[row]             (element-wise, FP16 scale)
```

The bitshift trellis constraint (Mao and Gray) — node `i` transitions to node `j` iff the
top `L − k·V` bits of `j` equal the bottom `L − k·V` bits of `i` — is what makes the
windows overlap by exactly `L − k·V` bits, so the sequence is a valid trellis path at
`k·V` bits per group and `T·k·V` bits total. The reconstructed weight value is
`ŵ = C[node] · scale`, in the row's **transformed space**: EXL3 quantizes incoherence-
processed weights (a random orthogonal/Hadamard transform applied per layer), so the
container's decoded weights are the transformed weights, and the serving path applies the
matching input transform exactly once per layer. The transform matrices (or their seeds)
are container objects; the transform application is not part of the trellis decode
contract.

**Bitrates.** `k` selects the target bits per weight: EXL3 targets `k·V/L` effective bits
per weight plus the amortized codebook and scale. Mixed per-layer bitrates are a converter
input (a per-layer bitrate map); the codec carries `(L, k, V)` per encoded object, and one
artifact may mix them per tensor.

## 2. The oracle specification

One independent naive FP32/FP64 mathematical oracle, owned by the delegated session and
reviewed by the fork:

- **Input**: the container's stored public inputs — the packed sequence `S` (exact bits),
  the codebook `C` (exact stored values), the scale (exact FP16), and the geometry
  `(rows, columns, L, k, V)`.
- **Evaluation**: for each row, each group: extract the circular window (bit-exact), index
  the codebook, multiply by the scale, and assemble `ŵ` in FP64. No staging casts, no
  fused arithmetic, no workspace dtype, and no reuse of a production kernel's decompositions.
- **Output criterion**: the production kernel's decoded weight at the same (row, column)
  matches the oracle within the codec's tolerance — exact bit equality for the decoded
  FP16/BF16 weight (the decode is a deterministic lookup-scale pipeline; the only legal
  difference is the production route's fused accumulate order inside a matmul, which the
  matmul's own tolerance governs, not the decode's).
- **Shapes**: the oracle runs at the real model shapes (the 35B MoE banks: 256 experts ×
  1024/2048-row banks at the layer's chosen `(L, k, V)`) and at a minimal 1-row shape as
  the converter-side payload verification.

## 3. The container extension (v3)

The v3 container binds tensors as separate objects with logical bindings; EXL3 is a bounded
extension of the existing binding/resource model, not a new container version:

- **Codec**: a new `QuantLayout`/`QType` registration for the trellis layout, with the
  geometry fields `L`, `k`, `V`, and the plane layout: `[packed sequence | codebook |
  scales]` per object, byte-aligned per plane exactly as the existing groupwise banks align
  their codes/high/scales planes.
- **Objects and bindings**: the encoded bank is one object per (layer, bank) — the MoE
  routed banks first (SP-7's sequencing); dense projections and the embedding follow. The
  logical bindings address the transformed weight space; the artifact's provenance records
  the transform seed per layer.
- **Tiering**: the trellis banks carry `rows_per_expert` geometry and participate in the
  expert tier plan, the residency tables, and the adaptive cache unchanged — the tiering
  machinery is codec-agnostic and must not grow EXL3-specific branches.

## 4. The converter recipe (tools/convert/&lt;target&gt;)

Target-private artifact inventories, source recipes, conversion, and converter-side payload
verification live under the converter; the recipe:

1. **Calibration**: on-the-fly Hessians per layer (the QuIP# proxy objective
   `tr((Ŵ−W)H(Ŵ−W)^T)`), from a calibration corpus; the fork's existing frequency-corpus
   discipline supplies the corpus.
2. **Incoherence processing**: the random orthogonal (Hadamard) transform per layer.
3. **Quantization**: BlockLDLQ with the trellis quantizer as the drop-in rounding operator
   — quantizing a `T_x × T_y` block as one trellis sequence (vector dimension
   `T_x·T_y ≫ T_y`), with the fused Viterbi kernel over the bitshift trellis; tail-biting
   via the two-pass rotation approximation (rotate, Viterbi, extract the overlap, re-Viterbi
   with the overlap constraint).
4. **Bitrate selection**: a per-layer bitrate map (converter input), with per-bank overrides
   for the MoE routed banks; the emitted `(L, k, V)` per object follows the map.
5. **Payload verification**: the converter verifies each encoded bank against the naive
   oracle (§2) at the minimal shape before emitting the artifact.

## 5. The kernel contracts (src/ops)

- **Decode-in-matmul**: the MoE banks decode the trellis windows inside the matmul exactly
  as the groupwise banks decode their codes today — the decode is a per-group window
  extraction, codebook resolution (stored or computed), scale multiply, and FMA into the
  accumulator. The production route may fuse the decode into the MMA pipeline and choose
  its own intermediate precision; the oracle (§2) checks the decoded weight, and the
  matmul's output tolerance checks the accumulation.
- **Parallel decoding**: each group's window is independent (they overlap only through the
  shared bit sequence), so the decode parallelizes across groups with no cross-group
  dependencies; kernels must not serialize the window extraction.
- **Shapes first**: the 35B MoE banks (Q4/Q6-class bitrates, the tiered serving path) are
  the first target; the dense-projection kernels follow the MoE bring-up.

## 6. Acceptance (strategy SP-7)

An EXL3-quantized MoE artifact converts, loads, and matches its oracle within tolerance;
quality-per-GB is measured against NVFP4 (the published quality-per-GB methodology records
the comparison workload and the calibration provenance).

## 7. Delegation boundary

The delegated session implements the trellis decode mathematics as a reference
implementation and the independent naive oracle, against §1–§2 of this contract, in the
repository's Python tooling under the oracle discipline above. The fork sovereignly owns the
container extension, the kernel contracts, the converter recipe, the acceptance criteria,
and the baseline validation; nothing that changes what "correct" means is delegated.
