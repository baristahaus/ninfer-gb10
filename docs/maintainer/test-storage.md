# Storage requirements for validation

**NFS is a distribution and archive medium, not a runtime medium.** An artifact or a fixture that a
test reads must live on local storage of the host that runs the test. Copy it from the archive first,
verify the copy, then run. This page records the policy, the requirements behind it, and the specific
blockers that produced it.

## The policy

- **Zero network-filesystem dependency for any test.** No test reads an artifact, a fixture or a
  checkpoint from NFS, CIFS or any other network mount at run time.
- **Local storage of the running host.** Copy from the archive to a local path, verify the copy
  against the manifest, and point the test at the local path.
- **The archive is still the source of truth.** NFS remains the convenient way to distribute an
  artifact between hosts and to keep it. It is the last hop before local storage, not the storage the
  engine reads from.
- **A refusal beats a stall.** If a test is pointed at a network-filesystem artifact it should say so
  and stop, not hang. The lane script does this; the engine does not yet, which is tracked as a
  robustness gap in the issue tracker.

## Why: the loader requires direct I/O

The artifact loader is built on an io_uring **direct-I/O** read queue (`src/core/io/read_queue.cpp`):
it registers file descriptors and buffers with `io_uring_register_files` / `io_uring_register_buffers`
and validates every range and buffer against 512-byte alignment (`src/core/io/file.cpp`,
`validate_direct_read`). Both artifact open sites (`src/artifact/backing.cpp`,
`src/artifact/materializer.cpp`) open with `io::ReadMode::Direct`.

That is a construction choice, not a default that can be flipped: a buffered escape
(`NINFER_ARTIFACT_BUFFERED_IO=1`) was attempted and reverted (`48697dcc`) because the queue's whole
design assumes direct I/O. Supporting buffered reads means a second materializer path — real work, not
a flag. See issue `9rk`.

The practical consequence: **an artifact on a network filesystem is unusable**, and its failure mode
is the worst kind. On the lab's NFS mount a 118 GB artifact stalls in `nfs_direct_wait` with the
process in uninterruptible state `D`, surviving `SIGKILL`. That is indistinguishable from an engine
deadlock — it cost two rounds of investigation before `/proc/<pid>/wchan` identified it.

## The requirements, measured

| Requirement | Value | Source |
|---|---|---|
| Free local space for the 125B artifact | ~119 GB, 5 container files, each size-verified | `docs/test-artifacts.md` |
| Free local space for the 27B lane artifact | ~20.4 GB | `docs/lane.md` |
| Copy throughput from the archive to local | ~247 MB/s with `dd bs=4M` | measured on twofour |
| Host unified memory for the 125B run | ~72 GB of the GB10's ~121 GB | `docs/test-artifacts.md` |

## Copy procedure

`cp -a` stalls on the same storage. `dd conv=noerror,sync` is worse than useless: it pads short reads,
so the copied file is larger than the original and the container rejects it with
`entry length differs from directory`. The procedure that works:

```sh
dd if=<archive-path> of=<local-path> bs=4M
# then compare the size against the manifest entry, byte for byte
```

A copy that has not been size-verified is not a copy. An artifact whose five files are not all present
and correct is not an artifact.

## The standing checklist

Ask this at the start of any pass that will run a real-artifact test:

1. **Is every artifact this run touches on local storage of the host that runs it?** If not, copy it
   first and verify the size.
2. **Is there room?** ~119 GB for the 125B, ~20.4 GB for the 27B lane artifact, plus the existing
   corpus — and the copy doubles the requirement while it is in flight if the archive path is not the
   one being replaced.
3. **For a cross-host comparison, is it the SAME artifact on both hosts?** Confirm by digest, not by
   path or size. Measured counter-example: the 27B existed on two lane hosts at exactly 21,492,938,224
   bytes with different sha256 values - different quantizations that happen to share a byte count. A
   token-for-token comparison across those hosts would fail as a plausible lane defect. Compare
   `sha256sum` output on both hosts before comparing tokens.
4. **Is any other process holding the pool?** The 125B needs ~72 GB of ~121 GB. A stale process makes
   the run fail with `resident set exceeds integrated-pool budget`, which reads like a budget bug and
   is not.
5. **Two engines cannot be resident at once.** A comparison that needs two configurations runs them
   one after the other, not together.

## Options considered for the future

- **An external USB device** dedicated to validation artifacts. Simple, portable, and independent of
  the lab's NFS server; the cost is throughput on first copy.
- **A local NVMe reorganisation** so that validation artifacts have a dedicated volume with a known
  free-space floor. Faster and more robust than USB; needs a partition or a mount point and a rule
  about what may live there.

Either removes the last network dependency from validation. Neither is required for correctness
today, because the copy-then-verify procedure above already gives a local path.
