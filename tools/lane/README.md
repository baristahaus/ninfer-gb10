# Lane tools

Scripts that drive the two-GPU lane from a workstation. The lane itself is documented in
[`docs/lane.md`](../../docs/lane.md); this file covers the scripts.

| Script | Purpose |
|---|---|
| [`run-tp2-arm.sh`](run-tp2-arm.sh) | sync the checkout to a lane host, build its release tree, and run one arm of the two-GPU acceptance test |
| [`sweep-tp2-lane.sh`](sweep-tp2-lane.sh) | run the acceptance comparison across a range of prompt lengths and record a verification profile |

## run-tp2-arm.sh

```
tools/lane/run-tp2-arm.sh --arm 1|2 [--host HOST] [--artifact PATH]
                          [--src DIR] [--dst DIR] [--build DIR] [--dry-run]
```

The script prints the plan before acting and refuses when the host is unreachable, the artifact is
not visible on the host, or the host has no configured build tree. It does not configure a build:
the lane host's toolchain and cache are its own.

`--dry-run` prints the plan and stops before the reachability checks.

The two arms are compared by hand from the token lists they print. See
[`docs/lane.md`](../../docs/lane.md) for why the comparison is split across hosts and what it can
and cannot catch.

## sweep-tp2-lane.sh

```
tools/lane/sweep-tp2-lane.sh [--lengths "8 10 ..."] [--work DIR] [--no-build] [--dry-run]
```

The lane's match/mismatch pattern across prompt lengths is not monotonic, so one length says almost
nothing about it. This script runs the same contract the acceptance test runs — a width-1 reference
round and a width-2 sharded round over the same token ids — once per length, and records a row per
length instead of a single pass.

Per length it slices the checked-in prompt fixture, runs arm 1 on the reference host and arm 2 on the
sharded host, and compares the two chains. On a mismatch it re-runs arm 1 on the reference host with
`NINFER_TP2_COMPARE` set to arm 2's chain, so the reference host adjudicates the first disagreement
against the 1.0-nat tie tolerance; the sharded arm cannot adjudicate its own disagreement because the
scoring route is single-device only. Each row is `length, match, position, margin_nats,
classification`, where the classification is `verified` (exact), `tie` (inside tolerance) or
`confident-disagreement` (a defect).

Both hosts' artifacts are digest-compared before any run: two hosts carrying different quantizations
at the same byte count would make every comparison measure the artifacts rather than the lane.

The sweep is resumable — a length already present in `$WORK/sweep.tsv` is skipped, and a row is only
appended once that length's verdict is known. Prompts are slices of `tests/fixtures/tp2_accept`
(`prompt.ids` extended by `golden_plain.ids` beyond 32 ids), so neither host tokenizes anything and
both arms read the same ids. The test's `NINFER_TP2_PROMPT` names the ids file for one length.

## Recorded sweep

`sweep-results.tsv` is the table from the 13-length sweep (2026-11-12, 27B nvfp4.v3, artifact
sha256 `044fc2f0…` on both hosts), and `sweep-results.txt` is the script's own summary. Nine lengths
were exact, four were tie-only (20, 31, 32, 48) and none was a confident disagreement; the largest
margin was 0.375 nats against the 1.0-nat tolerance. Lengths above 32 are the captured prompt
extended by the reference's own greedy continuation, so they are not independent samples.
