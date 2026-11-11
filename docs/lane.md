# The two-GPU lane

The tensor-parallel lane runs one model across two devices. **It loads and refuses to infer.** A
round with a width above one is refused until the sharded forward has a validated token-for-token
comparison against a single-device round; `NINFER_TP2_UNVALIDATED=1` lifts the refusal for that
comparison only, and must not be set for serving. The refusal is not a missing feature flag: it is
the lane's current state, and the comparison is how it changes.

This page is the procedure for making that comparison. It is the same procedure the lane has been
driven with; the script in `tools/lane/` is that procedure written down.

## Why two hosts

No single host in the lab fits both rounds:

| host | devices | constraint |
|---|---|---|
| hephaestus | 2 x 16 GB | the width-2 round fits; the width-1 reference needs 20.4 GB and does not |
| twofour | 121 GiB unified, one device | the reference fits; there is no second device for the sharded round |

So each arm runs where it fits, and the two token lists are compared afterwards. That is weaker than
one process comparing both arms — a defect that depends on host state would not be caught — and the
acceptance test states that limit rather than hiding it.

## The procedure

1. **Sync.** The lane host holds a plain rsync copy of the checkout, not a git worktree. `build/` is
   excluded so the copy's object files survive.
2. **Build.** `build/rtx5090-release` on the lane host, with that host's own toolchain and cache. The
   script does not configure a build tree; the lane host's is its own.
3. **Run one arm.** `tests/models/qwen3_5/test_engine_tp2_accept_real.cpp` with
   `NINFER_TP2_ACCEPT=1`, `NINFER_TP2_ARM=1` (single-device reference) or `=2` (sharded), and
   `NINFER_TP2_UNVALIDATED=1`. It prints `arm <width> tokens (<n>): …`.
4. **Compare the two lists.** They must agree token for token, in order and in length. That is the
   acceptance criterion; a plausible-looking mismatch is still a failure.

The artifact is the v3 27B:

```
/nfs/ninfer-gb10/qwen3_8_27b_v3_plain/qwen3_8_27b_v3-plain.ninfer
```

Both hosts must see the same NFS path.

## The script

```
tools/lane/run-tp2-arm.sh --arm 1|2 [--host HOST] [--artifact PATH]
                          [--src DIR] [--dst DIR] [--build DIR] [--dry-run]
```

Defaults are the lab's lane host and paths. The script prints everything it is about to do, then
syncs, builds and runs one arm. It refuses before doing anything if:

- the host is not reachable over ssh (`BatchMode`, so a missing key refuses rather than prompting);
- the artifact is not a file on that host;
- the host has no configured build tree at the expected path.

`--dry-run` prints the plan and stops before the reachability checks, so it works on a machine that
cannot reach the lane host.

## What is verified and what is not

- **Verified:** the lane loads a 27B artifact on two devices and refuses to infer; the refusal test
  (`test_engine_tp2_refusal_real.cpp`) asserts that a two-device `Engine` constructs and its first
  `generate` throws. The script's argument validation and dry run are exercised on every machine.
- **Not verified:** a passing token-for-token comparison. None has been recorded; that is why the
  refusal stands. The script's reachability, artifact and build-tree refusals have been exercised
  only on hosts where those conditions hold or the host is unreachable — a refusal that fires for
  the right reason on a reachable host with a missing artifact is the one case a lab run should
  confirm the first time it is available.
- **Not in scope:** the sharded forward's own correctness. The comparison is the measurement, not
  the fix.
