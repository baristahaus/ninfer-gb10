# Design: session-scoped retention and the state spill tier (agent-class cache economics)

Status: design, pre-implementation. This document designs one package against the context-cache
authority (`resource-scheduling-and-context-cache.md`): **session-scoped retention weighting**
and a **disk spill tier for state**. Motivation, evidence, and the standing invariants come from
the fork strategy's cache-economics frame (§4, added 2026-10-23). On execution, the touched
authority sections are revised and translated to English in the same change, and this plan is
removed.

## 1. The use-case and the demand

Agent-class serving: input-dominated requests, one long-horizon session producing many requests
that share a large common prefix (system prompt, tool definitions, accumulated scratch state),
and idle gaps between turns. The economics (validated external evidence): unit cost of serving
varies ~7× on KV hit rate alone; the engine's controllable levers are retention (how long a
prefix stays recoverable) and the cost of resuming after pressure (O(tail) recomputation, never
O(context)).

What the engine already does (verified): exact-identity prefix reuse with longest-prefix
selection (partial misses recompute only the suffix), a shared-prefix catalog with fork
semantics, SessionKey candidate narrowing, `LongAnchor` retention class, host state slots and a
host KV arena as the warm tier, and placement changes only at resource boundaries (admission,
capture, finish, inactive release) — never during decode.

What is missing: (a) retention does not weight by session activity — an ongoing session's
anchors compete with closed sessions' prefixes at equal prior; (b) there is no tier below host —
under pressure, anchors are evicted and resumption recomputes O(prefix); (c) the exact-identity
invariant has no disk analog to protect.

## 2. Session-scoped retention weighting

### 2.1 Semantics

A **session record** joins requests that carry the same session key. The ResourceManager already
owns SessionKey for candidate lookup and binding; the extension gives the retention policy a
second, activity-driven input:

- Each session record tracks its last activity as a `publication_order` (the existing monotonic
  per-request order — no new clock) and the set of checkpoints and owners cataloged under its
  prefixes. The record set is bounded: `max_tracked_sessions`, resolved at construction like
  every other fixed capacity; the oldest-inactive record is retired when the bound is hit, and
  retirement only removes the *weighting*, never the state (catalog entries keep their own
  identity rules).
- A session is **active** when a request under its key was admitted or completed within the
  configured activity window; **closed** otherwise. The window is a fixed startup constant,
  not a capacity axis.
- Retention weighting composes with the existing owner prior (§10.1): owners and checkpoints
  cataloged under an active session receive a demand prior that the planner already consumes;
  closed sessions lose the prior. No fixed eviction order is introduced — the planner's
  portfolio-value computation keeps ownership of the decision.
- **Promotion and demotion happen only at the existing boundaries** (admission, capture, finish,
  inactive release). Ordinary decode never runs periodic promotion, preserving the authority's
  placement-change rule verbatim.
- On session close, the demotion *favors spill over eviction*: the retention decision for a
  closed session's unique anchor coverage is "spill to the state tier", not "delete". Deletion
  remains the planner's choice when the spill tier is full or the coverage is worthless (the
  planner keeps portfolio-value judgment; the policy changes the default prior, not the
  mechanism).

### 2.2 Configuration

`ContextCacheOptions` gains one nested struct, resolved at construction:

- `max_tracked_sessions` (bounded catalog; default from `max_concurrency` scaling like the
  other derived axes);
- `session_activity_window_ms` (fixed constant; activity weighting only).

Both are documented in the authority's configuration-semantics section (§11) as fixed resolved
capacities, and the boundedness conditions extend: session-record storage is sized at
construction; target expansion never breaks the bound.

## 3. The state spill tier

### 3.1 Placement semantics

Placement gains a third location for checkpoint StateImages and KV logical pages:
**Device / Host / Spill**. The active-completion guarantee is unchanged — active requests
require Device/Host only. Spill residency counts as *retained coverage*: a checkpoint whose
last full copy lives in the spill tier is not evicted; it is recoverable by page-in.

The core invariant this package exists to protect, stated in the authority's invariant list:
**an admission candidate with exact identity that resides in the spill tier is materialized by
page-in, never by recomputation.** The resource plan for such an admission includes the page-in
transfers; the planner prices them through the existing transfer-cost model
(`ContextCostOptions`), so a spill-resident hit competes honestly against a shorter recomputed
prefix when it is genuinely more expensive. The O(tail) invariant is preserved because prefix
reuse still selects the *longest* recoverable prefix; spill extends how far back "recoverable"
reaches.

### 3.2 Mechanics

- One spill file per engine, path and capacity configured (`spill_capacity_bytes`, resolved at
  construction; the file grows within the bound). The spill page layout **equals the host KV
  page layout** (`plan_host_kv_page_layout` reused verbatim), so page-in is a read from the
  file into the host arena followed by the existing device-copy work — no new physical format,
  no new verification surface beyond file read/write integrity.
- Spill writes occur at the same resource boundaries as every other placement change, batched
  with the boundary's existing transfer plan. A spill-out is a Host→Spill copy that frees host
  capacity in the same target the planner already seals.
- Global occupancy accounting gains the spill axis (§3's capacity model): spill-resident bytes
  are billed once, like host-resident bytes. The active guarantee never waits on spill I/O;
  page-ins are admission-time work, planned and priced like prefill.
- Crash semantics: the spill file is a cache, not a record. Engine restart may drop it
  wholesale; checkpoints that existed only in spill are simply gone, and the next admission
  recomputes them (that is an O(prefix) miss after a restart — acceptable and explicitly not a
  correctness surface, because no published state ever depended on spill residency).

### 3.3 What stays out of scope

- Multi-node cache machinery, shared cache clusters, cross-machine hits — out of the product
  contract (strategy §4).
- Automatic tiering of *active* state — active requests always run Device/Host; the tier
  serves inactive, long-horizon retention only.
- The EXL3 banks route's looped-decode prefill note goes to the EXL3 contract instead: batched
  prefill is a *requirement* for any banks-route revival under agent-class traffic.

## 4. Implementation sites

Verified against the post-SP-7 master (2026-10-31). The 27B route's live context cache is the
old-engine cache; the pre-SP-7 qwen3_5 program files named by the first draft of this table
(`program/storage/context.cpp`, `program/planning/pressure*.cpp`) are dead sources that no build
target compiles and that include headers absent from the tree.

| Piece | Site |
|---|---|
| Spill arena (file-backed, host page layout) | `src/core/spill_kv_store.h/.cpp` (landed, 2026-10-31); transfer work reused |
| Spill transfer directions and pricing | `ContextTransferDirection` + `ContextMachineCostModel` (`src/runtime/contract/resources.h`, `src/runtime/engine/context_cache/context_cost.*`) — landed 2026-10-31; old preset files inherit the h2d row for the spill directions |
| Spill placement in plans | the live Session demotion/reclaim machinery: `src/models/qwen3_5/session/{reclaim,checkpoints}.cpp` (`quote_demotion`, `release_checkpoint` spill-out, `host_bytes_released`), plus `src/runtime/engine/context_cache/resource_manager.h` (`candidates` pricing a spill-resident alternative) |
| Page-in transfers at admission | `ResourceManager::candidates` `SourceChoice.transfers` (cost = transfers + prefill at `resource_manager.h:134`); the binding/restore path consumes the chosen source in `src/runtime/engine/engine_schedule.inl` |
| Session records, activity weighting | `src/runtime/engine/context_cache/` (SessionKey owner) + the retention prior consumed by `resource_manager` ranking |
| Configuration | `include/ninfer/types.h` (`ContextCacheOptions`), CLI/serving option plumbing, documented defaults |
| Authority revision | `resource-scheduling-and-context-cache.md` §5 (placement), §10 (retention), §11 (configuration), §12 (invariants) — touched sections translated to English |

## 5. Verification

- **Numerical/semantic**: a page-in restore is bit-identical to the host copy it replaced —
  exact comparison at the KV-page level (typed KV coverage checks already exist and must pass
  unchanged after a spill round-trip).
- **Behavioral**: a 27B real-artifact route — long shared prefix, pressure the host tier with
  filler traffic until the anchor spills, then resume the session; assert (a) the resumed
  request admits from the spill candidate, (b) its prefill token count equals the tail only,
  (c) output is unaffected by where the prefix lived. Plus the retention-weighting behavior:
  an active session's anchor survives pressure that evicts a closed session's prefix.
- **Boundedness**: construction-time capacity resolution asserts (session catalog, spill bytes,
  occupancy arithmetic representable).
- GPU validation runs after the current 35B acceptance completes (single-instance rule).

## 6. Cost and risk

The package is bounded: no new physical format (spill reuses the host layout), no decode-path
change (all placement at existing boundaries), one new transfer kind in the planner. The main
risks: (a) the planner's portfolio-value computation must treat spill-resident coverage's
page-in cost honestly or the tier degrades into write-only overhead — the cost model already
prices transfers, and the acceptance asserts a spill hit actually wins admission when it
should; (b) capacity accounting must not double-bill a page present in host and spill during a
boundary's transition window — the target-seal semantics (single logical end state per
decision) already covers this if the spill axis enters the same resource revision.

## 7. Handoff to the 5090 lane (2026-10-31)

For the workstation running the 5090/tiering lane. No action is needed until the session-level
wiring lands on master; this section is the pickup list when it does.

- **Spill transfer calibration.** The spill directions currently inherit the `h2d` row of every
  cost table (generic defaults, the compiled 5090 preset, and any preset JSON without explicit
  keys — see the schema note in section 4). When the wiring is live, measure host-file
  read/write bandwidth on the lane box (page-sized reads and writes against a real spill file)
  and add explicit `h2spill`/`spill2h` rows to the 5090 preset. The context-cost preset schema
  accepts them; the calibration bench fills them from its measured `h2d` fit until then.
- **Confirmation run for the MXFP8 A16 linear suite.** `ninfer_linear_mxfp8_a16_test` is green
  on the GB10 workstation with full row comparison (every weight row of all eleven shapes) and
  workspace-envelope checks. One confirmation run on the 5090 closes the same chain the EXL3
  suites got (follow-up plan M4). No fixture changes are expected; the test is self-contained.
- **Flash-Next spill primitive.** The 125B route's `state_image_store.h` does not have the
  spill residency the shared `StateDirectory` gained (`b3d1fb42`). Porting the same primitive
  there is deliberately left to the lane that owns that tree; do it after the convergence
  settles, and reuse the shape (guards mirror the replica rules; byte copies stay with the
  caller).
- **Dead-source decision still open.** `src/models/qwen3_5/program/{program_impl.cpp,
  prefill.cpp, storage/context.cpp, transactions/commit.cpp, grammar_masks.cpp}` include
  headers that do not exist and are in no build target (orphaned by the SP-7 merge `e0a0c214`).
  Delete or restore — the tiering work routes around them either way.
