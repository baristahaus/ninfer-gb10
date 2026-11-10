# Engineering lessons

Transferable method rules this tree learned the hard way. Each entry is one rule, the instance that
taught it, what the mistake cost, and where the evidence is. They are about engineering method, not
about this codebase: a rule that only applies to `src/models` is worth less than one that applies to
the next problem.

The evidence for most of them is in the session's review notes, which are **not part of this
repository** (see [Provenance](#provenance-of-these-lessons)); where a fact is in the tree, the entry
cites the commit or file instead, and those citations are checkable from a fresh clone.

Companion references: [`op-development.md`](op-development.md) for Op qualification, and
[`build-system.md`](build-system.md) for source ownership and target registration.

## 1. "No artifact" is not "not live"

> Ask **two** questions before calling code dead: *does a build target compile it, and does anything
> call it?* **and** *is it the sole occupant of a product code path?*

The first question found the easy cases: `src/serve/serve_options.cpp` (497 lines, in no CMake target
and included only by itself, deleted in `0d3dd160`), `src/models/qwen3_5/program/speculative/mtp.cpp`
(221 lines, no caller once its one function lost its last use, deleted in `3068455f`), and a
1,437-line `program_impl.h` header that nothing included and that still compiled (`d4eecb22`). All
were deleted on that evidence.

The same method, applied to `src/models/qwen4_exp`, gives the **wrong** answer. It has no artifact
anywhere — not on `/nfs`, no model card, and
[`tests/fixtures/artifacts/manifest.json`](../../tests/fixtures/artifacts/manifest.json) records
`"no matching artifact in /nfs/ninfer-gb10"` with both real-artifact tests permanently skipped — and
it is not in the product contract in [`AGENTS.md`](../../AGENTS.md). But it compiles unconditionally
(`src/models/CMakeLists.txt`), its tests compile, and it is the **only** model wired into the
distributed lane: `build_distributed_model<qwen4_exp::ParallelStartup>`
(`src/runtime/engine/startup.cpp`), `run_rank<qwen4_exp::RankPreparation>`
(`src/runtime/engine/worker.cpp`), and the distributed control protocol's payload contract is its
`wire`/`codec`. The live family has no parallel, preparation or wire path at all.

**Cost of the wrong answer:** the planned deletion was ~19.6 k lines, and it would have removed the
only occupant of a code path. The review's own recommendation was already conditional — *delete
after moving the parallel plan* — which is the safe form of the rule.

**The distinction to keep:** "no artifact" is evidence about **shipment**, not about **reachability**.
A family with no artifact can still be the sole implementation behind a lane, the payload contract of
a protocol, or the target of a documented conversion.

## 2. Test at the value the failing case uses

> When a verdict is about a function of a parameter, test it at the value the failing case actually
> uses. When two paths differ in exactly one parameter, vary that parameter and nothing else.

Three consecutive rounds of one investigation (rounds 5–7 of it) reached three wrong verdicts, each
by measuring the convenient value:

| round | verdict | measured at | the failing case uses |
|---|---|---|---|
| 1 | "the fold is innocent" | `commit_columns = 2` | `commit_columns = 1` |
| 2 | "the records are wrong" | the suspect's own records (circular) | a plain forward's state |
| 3 | "the publish writes a transpose" | the declared `[C,T,B]` contract | the record's value at width 1 **and** 2 |

**Cost:** three rounds. The fix was in the records, and round 1's check could not have seen it.

The question that collapses all three, and that is cheap enough to ask before believing any check:
**which value does the failing case use, and have I measured at that value?** A check at one parameter
value generalised to the others is the single most expensive habit in this list.

## 3. Start from the owner, not from the failing call site

> Design by asking which object can **own** the thing being added and **outlive** its borrowers, then
> place the change there. A design written outward from the failure lands the work in the layer that
> raised the error — usually the layer that cannot own anything.

Four instances in one lane, each moving the seam **up** one layer from where the error appeared:

| written from | the ownership question | where it belongs |
|---|---|---|
| the mirror needs a row lease | who owns rank 1's lease? | `HistoryDirectory`, which owns rank 0's — not the mirror wrapper |
| the mirror needs an origin pool | who owns the view over the storage's planes? | the storage, lazily — not the caller |
| rank 1 needs a Program | who owns the second Program's storage? | the Program's peer member, not the executor |
| the consumer rejects many ranges | who can hold a gathered buffer? | `ModelShard`, not `MaterializedArtifact` |

**Cost:** each round spent on the outward-from-failure design was a round that had to be redone; each
was cheaper than landing the wrong version.

*Weaker than it sounds:* the review names four instances, not five. A fifth ("the plan geometry
belongs to the planning layer") is recalled but is not written down anywhere, so it is not claimed
here.

## 4. Build everything, not the target you care about

> A header needs no target to exist. Verify with a full build, or with a check that proves every
> source is registered, not with the two libraries your change touched.

Two separate findings came from this:

- A **source-ownership check** — every `.cpp` under `src/`, `apps/`, `bench/` must be named by some
  CMake target — found a live orphan on its first run: `src/models/qwen3_5/program/speculative/
  mtp.cpp`, 221 lines, no target, no `#include`, no caller (`3068455f`). The same defect class had
  already cost this tree three parsers that no target compiled: `src/serve/serve_options.cpp`
  (497 lines, `0d3dd160`), `apps/cli/options.cpp` (`05e69631`) and
  `bench/inference/ninfer_bench_support.cpp` (`5473dee0`). A two-library build cannot see any of
  them, because a `.cpp` that nothing compiles is simply not built.
- A **full-tree build** (`cmake --build` with no target list) found a sixth broken test target:
  `tests/models/qwen4_exp/test_score_state_real.cpp` uses a `ShardedGroup` member its own stub does
  not define. It was pre-existing and invisible to every verification that built a target list.

**Cost:** four dead translation units and one broken test, each invisible to the
"build the two libraries and stop" habit. `NINFER_ENABLE_DISTRIBUTED` being `OFF` by default is the
same failure mode one level up: anything behind it does not build in the default configuration.

*Precision:* the counts are not one series. Three dead parsers were found and deleted over separate
rounds; the ownership check then found the fourth (the `mtp.cpp` orphan) on its first run and reported
three `.cu` files that no target compiles (`.cu` is deliberately outside that check, because some are
`#include`d into another translation unit); the full build separately found one broken test target.
The rule is the same in every case.

## 5. Inferred from X rather than from Y

> A quantity that looks like it settles the question is often read from the wrong place. The check
> that settles it is almost always cheaper than the work it gates.

Three inferences, three reliability levels:

| inference | read from | what actually settled it |
|---|---|---|
| "the packed GDN gate blocks the 27B" | the gate, not its caller | a `grep` for the caller: the 27B never reaches it |
| "Merged is cheap: `2*24/16 = 3` divides" | the tile count, not the tile boundary | a three-line `nvcc` probe |
| "the 27B's a/b projections are two 48-row parameters" | the shard **table** (a declaration) | still open — needs a bound artifact |

**Cost:** two rounds reached opposite wrong conclusions; the compiler settled the second in three
lines. The ordering to trust: a **bound artifact** beats the **compiler** for a layout question, which
beats **arithmetic**, which beats a **declaration** (a shard table, a config field, a comment).

## 6. A test and the code can share the same wrong assumption

> When a fixture sets up the very quantity under test, it cannot test it.

The shard table's extent check compared the table's row extent against `binding.elements`, which is the
parameter's **total** element count. Every synthetic fixture set `elements` to the extent by hand, so
the check and its tests agreed — and both were wrong. The first real width-2 load failed with
`sharding: text/output_head binds 1271398400 elements but the table declares 248320`
(`149ca926`, fixed in both family tables).

**Cost:** the defect survived the whole test suite and needed a real artifact to expose it. The code
now carries the comment at
`src/models/qwen3_8_flash_next_125b_a6b/impl/shard_rules.cpp` recording why the synthetic tests could
not catch it.

The corollary is stronger than the instance: a test that constructs the input from the same
understanding as the implementation is a tautology with extra steps. Prefer a fixture whose values
come from an independent source — an artifact, an oracle, or a literal the test author derived by hand.

## 7. A contract sentence can be the bug

> A sentence in a header comment is a specification. If it is false for a caller you have, it is a
> bug before any code is wrong.

`allreduce_sum`'s contract said *"both ranks call this together"*. True of one owner; false of two
Programs on two threads. That sentence produced two bugs — a slot desync and a mailbox that would
hang — before any implementation line was wrong.

**Cost:** two bugs, and the fix was to the **contract** (which caller may issue the collective, and
what the other side does meanwhile), not to the kernel.

The transferable habit: when a bug appears in a concurrent or collective path, read the contract
sentence first and ask *for which caller is this true?* The comment is often the specification that
was never reviewed.

## 8. Measure on the box in the state it will run in

> A number from an idle machine is a property of the idle machine.

A transport decision rested on a cross-device copy measured at 26.9 GB/s. With the engine running —
idle, serving nothing, 0 % GPU utilisation — the same copy measured **12× slower**. VRAM occupancy,
BAR1 occupancy, pinned host memory and a foreign process holding the device were all flat; only the
engine's presence mattered.

**Cost:** the conclusion was revised three times in one day (mailbox → staged → mailbox), and a
~1,350-line staged transport would have been ported on the strength of an idle-box number.

The habit: state the machine's *state* alongside the number (idle, serving, loaded, other tenants),
and prefer a measurement taken while the thing you are sizing actually runs. A benchmark that cannot
run in the deployment's state does not settle a deployment question.

## Provenance of these lessons

The detailed evidence — tables, logs, commands, and the reasoning that did not survive — is in the
session's review notes. **They are not part of this repository**: they live in the review workspace
outside this checkout, so they are named here rather than linked. A reader without access can still
check every claim that cites a commit or a file above.

| lesson | note (outside the repository) |
|---|---|
| 1 | `qwen4-decision.md`; the dead-source method in `docs-guard.md`, `guard-consolidate.md` |
| 2 | `mtp-recurrent.md` §0 |
| 3 | `q35-lane.md` §41 |
| 4 | `guard-consolidate.md` §5, §9; `q35-lane.md` §42 |
| 5 | `gdn-merged.md` §7; `gdn-control.md` §5 |
| 6 | no note: the evidence is commit `149ca926` and the code comment, both cited in §6 |
| 7 | `q35-forward.md` |
| 8 | `95-NEXT-STEP.md` §7 (review root) |

Where a lesson's evidence is a commit or a source comment, the citation is in the lesson itself and is
verifiable from a clone; the note only adds the narrative.
