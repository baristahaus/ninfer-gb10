# Lane tools

Scripts that drive the two-GPU lane from a workstation. The lane itself is documented in
[`docs/lane.md`](../../docs/lane.md); this file covers the scripts.

| Script | Purpose |
|---|---|
| [`run-tp2-arm.sh`](run-tp2-arm.sh) | sync the checkout to a lane host, build its release tree, and run one arm of the two-GPU acceptance test |

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
