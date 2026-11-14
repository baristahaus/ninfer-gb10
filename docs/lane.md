# The two-GPU lane

The tensor-parallel lane runs one model across two devices. A width-2 round is checked against a
width-1 round over the same artifact, the same prompt and the same token ids, and the two token
lists are compared. This page states what that comparison has measured, how to repeat it, and what
it cannot catch.

## What is verified

One sweep over thirteen prompt lengths, 2026-11-14, 27B nvfp4 v3 (artifact sha256 `044fc2f0…` on
both hosts), reference on one device and sharded over two:

| length | 8 | 10 | 12 | 16 | 17 | 20 | 24 | 28 | 31 | 32 | 40 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| result | exact | exact | exact | exact | exact | tie | exact | exact | tie | tie | exact | tie | exact |

Nine lengths matched token for token, four differed only inside the tie tolerance (positions 6, 18,
1 and 1; margins 0.375, 0.125, 0.375 and 0.125 nats against a 1.0-nat band), and no length was a
confident disagreement. Every length produced both chains, with no give-up and no failed arm. The
sharded arm's collective counts were symmetric with the reference's at every length.

This reproduces the table recorded on 2026-11-12 to the position and the margin, on a different
executor (see *History* below). The checked-in table is `tools/lane/sweep-results.tsv`; the script's
own summary is `tools/lane/sweep-results.txt`.

Length 28 stopped at ten tokens on both arms: the tenth token is the end-of-sequence token, and both
routes chose it. Agreement on where to stop is part of the comparison, not a shorter measurement.

## The acceptance criterion is tie-tolerant

The two routes do not do the same floating-point work: the sharded route reduces partial sums across
two devices in a different association than the single-device route. A token difference is therefore
not by itself a defect. It is a defect only when the reference's own scoring separates the two
candidate tokens by more than the noise band, and the reference host decides that by scoring both
candidates at their first divergence. The band is 1.0 nat; a difference below it is recorded as a
`tie`, above it as a `confident-disagreement`, which is a failure.

The sharded host cannot adjudicate its own disagreement: the scoring route is single-device only.

## Why two hosts

No single host in the lab fits both rounds:

| host | devices | constraint |
|---|---|---|
| hephaestus | 2 x 16 GB | the width-2 round fits; the width-1 reference needs 20.4 GB and does not |
| trueno | 1 x 32 GB | the reference fits; there is no second device for the sharded round |

So each arm runs where it fits and the token lists are compared afterwards. That is weaker than one
process comparing both arms — a defect that depends on host state would not be caught — and the
comparison's scope is stated here rather than hidden.

## The sharded forward must be wired

`kShardedForwardWired` in `src/models/qwen3_5/program/internal.h` selects the sharded forward. It is
`false` in the repository, because the sharded forward is not yet the default path. The **sharded**
host must therefore be built with it `true`; the reference host keeps the default, which is what the
comparison needs.

`sweep-tp2-lane.sh` applies the switch on the sharded host after the sync and prints the resulting
line into the run's own log, so a row's meaning travels with the row. A sharded host built without it
runs the fallback path, and the comparison would then measure the fallback rather than the lane.

## The artifact

Both hosts must carry **the same bytes**, verified by digest before any run: two hosts holding
different quantizations at the same byte count would make every comparison measure the artifacts
rather than the lane. Each host reads its own **local** copy — a network filesystem stalls the
direct-I/O artifact loader in `nfs_direct_wait` in uninterruptible state `D`, which is
indistinguishable from a deadlock. The script refuses an artifact on an `nfs`, `cifs`, `smb`, `fuse`
or `9p` filesystem rather than starting a run that will stall. The policy and the measurements are in
[storage requirements for validation](maintainer/test-storage.md).

## The procedure

```
tools/lane/sweep-tp2-lane.sh [--lengths "8 10 …"] [--work DIR] [--no-build] [--dry-run]
```

For each length it slices the checked-in prompt fixture, runs arm 1 on the reference host and arm 2
on the sharded host, and compares the chains. On a mismatch it re-runs arm 1 on the reference host
with `NINFER_TP2_COMPARE` set to arm 2's chain, so the reference adjudicates the first disagreement.

The match/mismatch pattern across lengths is not monotonic, so one length says almost nothing about
it; the profile is the measurement, not a pass at a point.

Every step is bounded: each arm and the adjudication carry `ARM_TIMEOUT` (1200 s by default), and the
pre-run idle wait is bounded too. A step that exceeds its bound is recorded as its own classification
rather than hanging the profile — a bound that is assumed but not enforced does not exist.

The sweep is resumable: a length already present in `$WORK/sweep.tsv` is skipped, and a row is
appended only once that length's verdict is known. Prompts are slices of `tests/fixtures/tp2_accept`
(`prompt.ids` extended by `golden_plain.ids` beyond 32 ids), so neither host tokenizes anything and
both arms read the same ids.

Run output lives next to the code, never in `/tmp`: `$WORK` defaults to
`/home/barista/ninferno/work/lane/run-<timestamp>` on the driving host and `$REMOTE_WORK` to
`/home/barista/src/wt-tp2.work` on each lane host. On this workstation `/tmp` is a 31 GiB tmpfs, and
filling it stops every subprocess the harness starts, not just the run.

## What is verified and what is not

- **Verified:** the thirteen-length profile above, from one run of the sweep on 2026-11-14, with both
  chains produced at every length, zero give-ups, zero failed arms, and symmetric collective counts
  between the two ranks.
- **Verified:** the two hosts' artifacts were digest-identical before the run, and both hosts read
  them from local storage.
- **Not verified:** the two arms run as separate processes on separate hosts, so a defect that depends
  on host state would not be caught. No host in the lab fits both rounds in one process.
- **Not verified:** lengths above 32 are the captured prompt extended by the reference's own greedy
  continuation, so they are not independent samples.
- **Not in scope:** whether the tie tolerance is the right criterion for subtle defects. It is the
  criterion this lane adjudicates with, and a tighter one is a separate question.

## History

The lane produced no chain at width 2 at any length between 2026-11-13 and 2026-11-14. The cause was
three defects in the local two-rank executor, each visible only after the previous one was removed:
`poll()` returned on a missing rank-0 event without advancing the peer a cross-rank collective needed;
a polled event that the release decision did not deliver was discarded and never produced again; and
the diagnostics that would have named either were defeated by a throttle keyed without the rank. The
fixes are `b71f09c8` and `c6e78803`; the diagnosis is recorded in the issue tracker under
`ninfer-gb10-kil`.
