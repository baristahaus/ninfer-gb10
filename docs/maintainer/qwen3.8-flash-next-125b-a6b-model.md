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
same materialized BF16 combine boundary before grouped RMSNorm. The final learned mixer reduces the
four streams to the decoder output.

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
speculative replay/fold, commit, rollback, and restore. An execution that continues a StateImage
fork reads every persistent component, GDN and PLE alike, from the source checkpoint slot and
writes the destination slot; it never reads the destination's prior content. A generated token is public only after the
target transaction commits it. When an MTP round's row continues in place (not terminal, not
cancelled, no unsettled fork), its GDN and PLE fold is deferred to the next MTP round, from
device-resident row descriptors. That round folds the PLE commit at the head of its graph and copies
the pending GDN record rows to a snapshot; each GDN layer of its verify then folds the row's
committed columns from the snapshot into the slot's state (in place) before recording the round
from it. This is the same arithmetic in the same order as a separate fold. The Program keeps a fold
in the graph only when its row is a lane of that round on the same slot, and otherwise folds
eagerly first. Until then, the sequence's committed state is its slot plus that pending
fold. Every other operation that reads or replaces the state or the records runs the fold on the
stream first: admission, a context transaction, prefill, an active capture, forced tokens, a
non-MTP decode, and finishing.

## MTP and Vision

The one-layer MTP predictor uses the same QSA, HyperConnection, and MoE mathematics with its own
weights and state. Its expert banks are BF16 or NVFP4, as the artifact recipe records them; the
NVFP4 banks take the main layers' routed-expert route. Draft lengths 1 through 3 are supported;
ordinary MTP0 and MTP3 use the same target model and publication rules.

An MTP round is one CUDA Graph: verify with on-device acceptance, the draft steps, and the egress
copy. Its inputs are a device-resident frame. At the end of each round, `mtp_advance_round` writes
the next round's anchors, frontiers, budgets, extents, drafts, RoPE positions, PLE history and
pending folds for every row that continues with its whole licensed output. The host uploads a frame
only when the rows, their order or a field differ from the round's echo: at admission, a fork
settle, a request's end, and the explicit invalidations. The round hashes its own PLE row ids on the
device and publishes them to a pinned mailbox. A Program-owned thread gathers the FP8 rows from the
file-backed table into pinned staging while layer 0 runs. Before layer 1's PLE, the graph waits for
the acknowledgement; a stage that misses its 5 s deadline raises a late flag, which the Program turns
into an execution error. The host gather issues `MADV_RANDOM` once and prefetches the next rows
before copying each, so a round's page faults overlap. Rounds are not pipelined: two rounds in
flight measured slower on GB10, because the serial loop already exposes under 0.1 ms of host time
per round.

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

## Numerical diagnostics

`ninfer-perplexity --token-scores` exports fixed-history token log-probabilities through the public
Engine scoring route; see [perplexity](../perplexity.md). Cross-engine score differences alone do
not identify an incorrect operator.

For a targeted maintainer investigation, `NINFER_FLASH_NEXT_LOGITS_DIR=<directory>` captures full
BF16 target logits after ordinary decode and MTP verification. Use `--no-cuda-graph`: synchronous
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
