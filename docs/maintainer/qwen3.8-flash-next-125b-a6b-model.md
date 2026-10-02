# Qwen3.8 Flash-Next 125B-A6B model reference

This reference records the exact Text, Vision, MTP, HyperConnection, PLE, QSA, sparse-MoE, and
persistent-state semantics implemented for the registered Qwen3.8 Flash-Next 125B-A6B target. The
artifact representation and conversion contract are defined in
[`qwen3.8-flash-next-125b-a6b-artifact.md`](qwen3.8-flash-next-125b-a6b-artifact.md).

The target instantiates the independent `qwen3_8_flash_next` family runtime. That family owns its
Frontend, prepared-prompt/output types, persistent state, Text/Vision/MTP schedules, workspace, and
CUDA Graph machinery; it does not specialize the `qwen3_5` runtime. Closed mathematical kernels
remain shared Ops where their semantic contracts genuinely coincide.

## Fixed dimensions

| Field | Value |
|---|---:|
| Text hidden size / layers | 2560 / 48 |
| vocabulary rows | 248320 |
| native context | 262144 |
| full-attention / GDN layers | 12 / 36 |
| QSA query heads / KV heads / head width | 24 / 2 / 256 |
| QSA rotary width / scale | 64 / `1/sqrt(256)` |
| GDN key heads / value heads / head width | 16 / 48 / 128 |
| GDN convolution channels / taps | 10240 / 4 |
| routed experts / selected experts | 512 / 10 |
| routed and shared expert width | 640 |
| HyperConnection streams / low-rank width | 4 / 320 |
| MTP layers / maximum draft tokens | 1 / 3 |
| Vision depth / hidden / intermediate | 27 / 1152 / 4304 |
| Vision merger output | 2560 |

Full attention occurs at zero-based layers `3, 7, ..., 47`; every other layer uses Gated
DeltaNet. Every Text layer has a sparse routed expert block and an independently sigmoid-gated
shared expert. The selected routed weights are normalized to sum to one. No expert capacity limit,
token dropping, or stochastic routing applies at inference.

## HyperConnection and layer schedule

The residual state is four BF16 streams of width 2560. Each attention/GDN and MoE sub-block first
forms its learned normalized input mix and injection weights, evaluates the block, and commits the
block result back into the four streams. The fused combine-and-mix implementation preserves the
same materialized BF16 combine boundary before grouped RMSNorm. Up to eight tokens without capture
run the whole mix as one cooperative kernel with one grid barrier: lane-owned CTAs commit, steer
and fold the RMSNorm weight into BF16 activations, Down/injection lane-segment dots run on BF16 MMA
and take the inverse RMS factor after the barrier, and column-owned CTAs rebuild the BF16 low-rank
activation, apply Up on MMA and gate-mix in FP32. Normalized rows and gate logits are not
materialized on that route, and its fixed reduction order is bitwise reproducible. Larger token
counts and capture use the materializing route. The final learned mixer reduces the four streams to
the decoder output.

Layer 1 additionally applies PLE between its token mixer and MoE. PLE selects sixteen 160-element
FP8 rows per token: eight bigram heads and eight trigram heads. Hashes reset at EOS. The selected
2560 values are gathered on the host from the read-only table mapping, transferred for the current
chunk, and consumed by the PLE projections and nine-column persistent causal state. Only selected
rows enter device memory; the complete table is never uploaded.

## QSA and persistent state

QSA projects 24 gated query heads and two K/V heads. Its indexer constructs normalized 128-wide
query/key representations and selects causal four-token key groups before exact attention over the
selected paged K/V positions. Main K/V, raw index keys, and MRoPE positions are persistent cache
state. Main K/V may use BF16 or row-scaled FP8 E4M3; raw index keys remain BF16 and MRoPE positions
remain I32 in both profiles. FP8 K rows apply the shared normalized D256 Hadamard transform before
row quantization, and Q applies the same transform before the dot product. V is row-quantized
without that transform. Prefill uses a tensor-core selected-attention route; decode uses the
bounded split route.

Each GDN layer retains three previous BF16 convolution columns and 48 FP32 recurrent matrices of
shape `[128,128]`. PLE retains nine previous BF16 convolution columns. QSA KV/index state, GDN
state, PLE state, HyperConnection state, and MTP state participate together in prefix snapshots,
speculative replay/fold, commit, rollback, and restore. A generated token is public only after the
target transaction commits it.

## MTP and Vision

The one-layer MTP predictor uses the same QSA, HyperConnection, and MoE mathematics with its own
weights and state. Its expert tensors are BF16. Draft lengths 1 through 3 are supported; ordinary
MTP0 and MTP3 use the same target model and publication rules.

The Vision tower is the 27-layer Qwen multimodal backbone used by the registered Qwen3.6-family
targets, with a checkpoint-specific merger that emits width 2560. Image/video preprocessing,
MRoPE prompt construction, CLI input, and serving protocol translation remain the shared product
routes.

## Numerical boundaries

BF16 weights and activations retain their represented values. Routed main-model experts decode
signed NVFP4 codes with their stored block scales and per-expert input/weight divisors. GDN control
and recurrent state are FP32. PLE table values are FP8 E4M3FN multiplied by the stored BF16 table
scale. Production fusion may choose its reduction and staging precision, but each closed Op is
qualified directly against an independent mathematical oracle at its public output and persistent
state boundaries.

Ordinary decode graphs are enabled by default. Valid-token counts are filled on the device, so
graph replay does not depend on temporary host storage.

Ordinary eager decode and CUDA Graph replay select the same target execution envelope; disabling
CUDA Graphs changes launch machinery without selecting different arithmetic. Prefill and incremental
decode use distinct qualified GEMM/reduction profiles. Fresh full-history prefill is useful for
numerical diagnosis, but identical greedy tokens across those profiles are not a mathematical
oracle: BF16 rounding can change a close argmax. Validate closed Ops against their independent
oracle, and graph replay against eager execution with the same envelope and token history.

## Numerical diagnostics

`ninfer-perplexity --token-scores` exports fixed-history token log-probabilities through the public
Engine scoring route; see [perplexity](../perplexity.md). Cross-engine score differences alone do
not identify an incorrect operator.

For a targeted maintainer investigation, `NINFER_FLASH_NEXT_LOGITS_DIR=<directory>` captures full
BF16 target logits after ordinary decode and MTP verification. Ordinary decode captures logits
after replay, outside the graph. For MTP verification, use `--no-cuda-graph`: synchronous
host copies are prohibited inside capture. Each numbered JSON file identifies the route, input
IDs, positions, active columns, KV rows, vocabulary size, width, and batch; the matching `.bf16`
file stores contiguous vocabulary-major rows. Verification includes tentative columns, so compare
only matching token histories and valid columns, not arbitrary rows at the same position.

`NINFER_FLASH_NEXT_STATE_DIR=<directory>` together with
`NINFER_FLASH_NEXT_STATE_FRONTIER=<execution-token-count>` captures the committed GDN convolution
and recurrent tensors at exactly that frontier, after any speculative rollback. JSON records the
route, lane, physical slot, and ledger. Layer files retain BF16 convolution and FP32 recurrent
values. This is a GDN diagnostic, not a complete continuation snapshot: KV, PLE, and predictor state
are not included. A single dump is approximately 110 MiB; choose a short, specific fixture.
Both diagnostics are disabled when their environment variables are absent, and their timings must
not be used for performance claims.

## Activation controls

The Flash-Next family owns `ActivationControl`, its startup-sized direction arena, bounded capture
arenas and capture record writer. The exact package supplies the family Program normally. Steering
mathematics lives in the fused attention-input kernels of `src/ops/launcher/hyperconnection.cu`; GPU
capture conversion lives in `src/ops/launcher/activation_capture.cu`. No activation
state is shared with another Program or added to the speculative frame layout. Direction and capture
capacity participate in the physical sequence reservation and `activation_capacity_bytes` memory
summary. The direction arena is always allocated so admin activation can use its original addresses;
capture storage is allocated only on a capture server.

Attention hyper-connection steering runs inside the fused combine/grouped-RMSNorm kernel, or the
lane-owned phase of the small-token mix kernel: each (lane, token) block forms the represented BF16 combine of the preceding branch in registers, adjusts
selected residual lanes, writes the steered BF16 state once and normalizes it. Rows, ranks and lane
masks are device data, so inactive blocks run the unsteered arithmetic bit for bit, in prefill,
ordinary decode and MTP verification alike. MLP and MTP hyper-connections are unchanged.
All dot products use the original represented BF16 lane and represented F32 pack vectors; directions
are not removed sequentially. Qualification compares against an independent FP64 formula at width
2560, including nonorthogonal directions, partial lane masks and ranks through the compiled capacity.
The zero-strength path returns before reading or writing lane values. Capture takes the full lane
normalization and feature-wise gates and retains the mixed BF16 boundary. See
[serving](../serving.md#flash-next-activation-capture-and-steering) for the wire, pack and capture formats,
terminal-token semantics. Captures support active steering packs and record the post-steering stream
with the request pack SHA, generation and effective strength.

The native text-only device load binds approximately 125.1B logical parameters (125.7B tensor
entries including auxiliary storage), excluding the 51.2B BF16 PLE elements. Parameter counts refer
to logical shapes, not packed byte or scalar counts. The load-plan guard checks 124B through 126B
device tensor entries and the complete file-backed PLE mapping; the 74B estimate does not describe
this native artifact's logical inventory.
