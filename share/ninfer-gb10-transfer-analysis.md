# What the GB10 Flash-Next work could offer upstream NInfer

**Status:** analysis for discussion, 2026-10-01; K4 outcome added after GB10 verification.
Nothing here has run on an RTX 5090.
**Source:** fork `baristahaus/ninfer-gb10`, branch `claude/peaceful-cori-e9myku`, diverged from
upstream at `e31bc99b`.
**Audience:** NInfer maintainers, and a peer with an RTX 5090 who can validate before anything is
proposed upstream.

## Summary

Most of the fork's work is specific to Qwen3.8 Flash-Next on GB10. One piece is not: the
**pipelined MTP decode round**. It is staged K1–K4 and lets the host's per-round output processing
and commit overlap the next round on the GPU. Qwen3.5's MTP loop (27B Dense, 35B-A3B MoE) has
the same structure Flash-Next had before this work, so the change ports almost directly.

Whether it is worth porting depends on one number nobody has measured on an RTX 5090:
**decode host time per round**. The published 5090 rounds are short (about 4.7 ms for 35B C1).
Even a ~1 ms host gap would be a large share there. That gap can be read from existing serve logs.

## Outcome on GB10 (read this first)

The pipelining was built and verified exact, and it **did not pay**:
- **Where the host gap was:** with K3 in place the serial loop exposes 0.05 ms of host time per
  ~72 ms round at C4 K=1. Almost all of the 8–9 ms measured earlier was Flash-Next's per-round
  PLE gather from the file-backed table, which K3 moved inside the round.
- **What pipelining did:** two rounds in flight then had nothing to hide. The backup copies it
  needs added ~2 ms of device time per round, a ~2% decode regression (ABBA, untraced).
- **Status:** K4a/K4b were removed from the fork in `1749c466`. K1–K3 remain. The K4 design
  below is kept for reference; its code lives in the fork's history at `f86d582a`.

**Consequence for Qwen3.5.** Qwen3.5 has no PLE, so its host gap may already be small, and the
case for porting K1–K4 is weaker than §3 below suggests. The step-0 measurement in §7 decides it.
Port only if decode host time is still a material share of the round.

## 1. What was built in the fork

| Stage | Commit | What it does |
|---|---|---|
| K1 | `596a63f9` | A continuing MTP row's GDN/PLE commit fold moves from the host's eager commit tail to the head of the next round's CUDA graph, removing one launch group and a second synchronize per round. |
| K2 | `bd8a8ecb` | The round's input frame stays on the device. A new Op, `mtp_advance_round`, writes the next round's frame at the end of each round. The host uploads only when its own frame differs from the device's echo, so a skipped upload never changes what the graph reads. |
| K3 | `b90fd5a4` | Flash-Next only: the PLE table hash moves into the graph, with a host gather thread and a device wait before the first PLE consumer. |
| K4a | `3d3d7a25` | Each MTP lane gets one spare state slot. A round reads one of the lane's two slots and commits into the other, so the pre-round state survives until the commit is final. |
| K4b | `f86d582a` | Two rounds in flight. The Engine launches round N+1 before round N's preview and commit, and the Program keeps the early round whole or discards it whole, so outputs equal the serial loop. Engine contract: `engine-architecture.md` §6.4. |

Measured on GB10, C4 K=1, Flash-Next 125B: before K3 the decode host gap was 8–9 ms of a ~79 ms
round with a warm page cache; after K3 it is 0.05 ms. K1–K4 are verified bitwise identical to the
serial loop on the real test and at serve level on the same configuration. K4b pipelining
measured ~2% slower, and K4a/K4b were removed (`1749c466`).

## 2. Why Qwen3.5 has the same structure

Qwen3.5's MTP path:
- builds the full host input every round (`program/decode.cpp`);
- runs an eager `replay_fold` plus a synchronize in the commit (`program/prefill.cpp`);
- copies the egress to the host after the graph;
- runs serially;
- uses the same `mtp_prepare_next_round` Op.

Flash-Next's runtime was split off from this code. The Engine half of K4 is already
family-neutral: any Program that implements `submit_successor`, `has_successor` and
`collect_successor` gets pipelining.

## 3. Why it could matter more on an RTX 5090

Round times implied by the published MTP3 decode-saturation tables (about 3 tokens per round at
~68% acceptance):

| Model | C1 round | C8 round |
|---|---:|---:|
| 35B-A3B MoE | ~4.7 ms | ~17.6 ms |
| 27B `nvfp4` | ~15 ms | ~21 ms |

On GB10 the host gap is 1.8–3.1 ms at C1 and 8–9 ms at C4, which includes Flash-Next's PLE
fetch. Qwen3.5 has no PLE, and an x86 host is probably faster, so its gap should be smaller in
absolute terms. But it is set against much shorter rounds. At ~1 ms it would be ~20% of a 35B C1
round and ~6% of a 27B C1 round. Pipelining hides up to that amount, less the costs in §5.

## 4. Proposed port for Qwen3.5 (MTP only)

1. **K1:** defer the commit fold to the next graph head. Small gain on its own; a prerequisite.
2. **K2:** device-resident frame; `mtp_advance_round` reused unchanged.
3. **K4a:** two state slots per lane.
4. **K4b:** the Program's submit/collect split, backups of the pending round's records and
   hiddens, egress buffers alternating by round parity, and discard repair.

K3 is not needed.

**Scope limits:**
- DFlash proposes and verifies within one round.
- DFlash2 publishes penalty occurrences only after commit.
- Both need their own analysis. Start with MTP.

**Independent follow-up, K5 (not built):** a fused GDN fold-and-verify kernel saves about one
state read per request per round. By bandwidth estimate on the 5090, that is ~3% at C8 on the 27B
and ~1.5% on the 35B.

## 5. Costs and risks

- **Memory:** one extra GDN state slot per lane.

  | Model | Per lane | At C8 |
  |---|---:|---:|
  | 27B | 151 MB (48 layers × 48 heads × 128² × FP32) | 1.2 GB |
  | 35B | 63 MB | 0.5 GB |

  On 32 GB this competes with the KV and context-cache budget. The memory-free alternative
  applies pending tokens inside the verify kernel, which is K5-level work.
- **Discarded rounds:** a round whose commit finishes or shortens any row (stop token or string,
  thinking-control switch) discards its early successor whole. The remaining rows re-run, one
  wasted round per such event: about 2% of rounds at C4 on GB10, more for short outputs.
- **Draining:** while a request waits for admission, every boundary drains, so the loop is serial
  until admission proceeds.
- **Backup copies:** each early launch copies the pending round's replay records (about 1.3 MB per
  token of record capacity for Flash-Next; smaller for 35B).
- **Exactness:** outputs must equal the serial loop bitwise. That is the acceptance gate, not a
  tolerance.

## 6. Other fork changes and their relevance upstream

| Change | Commit | Relevance |
|---|---|---|
| QSA workspace sized for every call of ≤16 rows (fixes a `bad_alloc` on short chunk tails) | `eb9e87fa` | Bug fix for upstream Flash-Next (RTX PRO 6000) |
| QSA skips index selection when all visible keys fit the budget | `d1efb24f` | Flash-Next speed: on GB10, attention −60% and decode +5.2% at C4 |
| Fused HyperConnection gate mix in the FP8 up projection | `8630b6a1` | Flash-Next; speed-neutral, small favourable perplexity change |
| Dense causal-cache split budgets from the device SM count | `097ad988` | No effect on a 5090; helps other SM counts |
| Admission gate for unsettled StateImage forks, with re-arm | `a51fba6a`, `dc622216` | Repairs a fork regression. The underlying planner/Program mismatch ("selected pressure target could not be sealed", C=8) may exist upstream; reproduce first |
| Blocked-admission counters in the request log | `aff11e8e` | Optional observability |
| FP8-row dense and NVFP4 MTP expert recipes | 7a/7d | Flash-Next artifact formats; not relevant to 27B/35B |

## 7. Validation plan on an RTX 5090

**Step 0: decide whether the port is worth it. No code change.**
1. Run the published 35B-A3B and 27B `nvfp4` MTP3 serve workloads at C1 and C4 with the request
   log enabled.
2. Report `engine_timing.decode.host_exposed_seconds / decode.rounds` (the decode field, not the
   top-level `host_exposed_seconds.total`). Also report `device_wait_exposed_seconds` per round.
3. Rule of thumb:
   - host time ≥ 10% of the round at a concurrency people use → port;
   - under ~3% → don't.

**Step 1: port, then check exactness**
- Port K1, K2, K4a and K4b to Qwen3.5 MTP as separate commits.
- After each commit:
  - Qwen3.5 real-artifact tests and MTP goldens are bitwise identical to the serial build;
  - include runs ending on stop strings and `max_tokens`, which exercise the discard path.

**Step 2: measure**
- Same workloads as step 0, A/B against the serial build. Use one fixed page-cache state and
  alternate arms (ABBA).
- Report:
  - decode tok/s;
  - host ms per round;
  - startup memory (expected +1.2 GB on the 27B, +0.5 GB on the 35B, at C8);
  - how often the early round is discarded.

**Step 3:** share upstream with the measured results.
