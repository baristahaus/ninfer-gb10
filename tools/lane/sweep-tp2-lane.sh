#!/usr/bin/env bash
# Sweep the two-GPU lane's tie-tolerant acceptance criterion across prompt lengths.
#
# One length tells you almost nothing: the lane's match/mismatch pattern is not monotonic, so the
# honest statement is a profile over a range, not a pass at one point. This script runs the same
# contract the acceptance test runs -- a width-1 reference round and a width-2 sharded round over the
# same token ids, adjudicated on the reference host -- once per length, and records a row per length.
#
# The comparison is split across two hosts because no single host fits both arms: the reference needs
# one device with the 27B resident and the sharded lane needs two. Each arm is checked against the
# same token ids, and a disagreement is adjudicated on the REFERENCE host by scoring both candidates
# at their first divergence, because the scoring route is single-device only.
#
# Usage:
#   tools/lane/sweep-tp2-lane.sh [--lengths "8 10 ..."] [--work DIR] [--spec BACKEND] [--dry-run] [--no-build]
#
# Resumable: a length whose row already exists in $WORK/sweep.tsv is skipped, so an interrupted
# sweep is continued by running the same command again. Partial rows are never written; a row is
# appended only once the length's verdict is known.
#
# Outputs are split across the two sides of the sweep, and neither side uses /tmp:
#   $WORK (here, default /home/barista/ninferno/work/lane/run-<id>):
#     sweep.tsv              length, match, position, margin_nats, classification
#     prompts/T<N>.ids       the prompt both arms were given (the local copy)
#     chains/T<N>.arm{1,2}.ids   the two token chains
#     sweep.log              every arm's raw output
#   $REMOTE_WORK (on both lane hosts, default /home/barista/src/wt-tp2.work):
#     prompts/               the same prompts, staged for the arm binaries
#     chains/                the sharded chain staged for the reference host's adjudication
# /tmp is deliberately absent: on a host whose /tmp is a tmpfs it is RAM, not storage, and filling it
# breaks every subprocess on the box -- which is exactly what happened and cost a session.
set -euo pipefail

LENGTHS="8 10 12 16 17 20 24 28 31 32 40 48 64"
# THE WORK DIRECTORY IS PERSISTENT AND PER-RUN. It used to default to /tmp, which is wrong twice
# over: on a host whose /tmp is a tmpfs the directory can vanish with the process that made it, and a
# tsv whose age cannot be told from its contents WILL be misread -- it was, and a day-old table was
# reported as the current run's. The run id is in the directory name so the table's age is in its
# path rather than something a reader has to check with stat.
RUN_ID=$(date +%Y%m%dT%H%M%S)
WORK=${WORK:-/home/barista/ninferno/work/lane/run-$RUN_ID}
# The lane hosts keep their run inputs next to the checkout they run, for the same reason this host
# does: a scratch directory that is adjacent to the code it belongs to can be found, aged and cleaned
# by whoever owns the code, and it is never the shared tmpfs.
REMOTE_WORK=${REMOTE_WORK:-/home/barista/src/wt-tp2.work}
REF_HOST=172.30.30.16
REF_ART=/home/barista/ninfer/models/qwen3_8_27b_nvfp4.v3.ninfer
LANE_HOST=172.30.30.15
LANE_ART=/home/barista/artifacts/27b-v3-nvfp4/qwen3_8_27b_nvfp4.v3.ninfer
DST=/home/barista/src/wt-tp2
BUILD=build/rtx5090-release
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DRY_RUN=0
NO_BUILD=0
# THE PER-ARM BOUND. The harness had NO bound on an arm: the only budget was wait_idle's 1200 s
# precondition wait, so a hanging sharded arm hung the sweep forever -- measured: 11+ minutes on one
# length, 0 rows, 24 GB resident across both cards, 0% utilisation, and it would have sat there
# indefinitely. A bound that is assumed but not enforced is a bound that does not exist.
# Derived from measurement, not guessed: the reference arm takes ~8 min (67 rounds, arm 1 tokens (32)
# on trueno), so 1200 s is 2.5x that and generous enough for the sharded arm to finish if it can.
ARM_TIMEOUT=${ARM_TIMEOUT:-1200}
# THE LANES ARE VALIDATED ONE AT A TIME. NINFER_TP2_SPEC selects a speculative backend, and the
# sharded forward has to reproduce each lane's reference before kShardedForwardWired can be flipped.
# Without this option the sweep could only run the plain lane, which is why the speculative lanes
# stayed unvalidated at width 2.
SPEC=${SPEC:-}
# This workstation's /etc/ssh/ssh_config.d carries a file with wrong ownership, so the default system
# config makes every ssh fail with "Bad owner or permissions" before it reaches the host. The lane
# hosts are addressed by IP, so an empty config is both sufficient and reproducible; SSH_CONFIG can
# point at a real config when a host needs an alias or a non-default identity.
SSH_CONFIG=${SSH_CONFIG:-/dev/null}

usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --lengths) LENGTHS="${2:?--lengths needs a list}"; shift 2 ;;
        --work)    WORK="${2:?--work needs a directory}"; shift 2 ;;
        --no-build) NO_BUILD=1; shift ;;
        --spec)     SPEC="${2:?--spec needs a backend name (mtp, dflash2)}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

say() { printf '%s\n' "$*"; }
die() { printf 'refusing: %s\n' "$*" >&2; exit 1; }
remote() {
    local host="$1"; shift
    ssh -F "$SSH_CONFIG" -o BatchMode=yes -o ConnectTimeout=8 \
        -o StrictHostKeyChecking=accept-new "$host" "$@"
}

[ -d "$SRC" ] || die "source directory $SRC does not exist"

say "== lane sweep"
say "   lengths  $LENGTHS"
say "   work     $WORK"
say "   reference $REF_HOST ($REF_ART)"
say "   sharded   $LANE_HOST ($LANE_ART)"
say "   checkout  $SRC -> $DST"
if [ -n "$SPEC" ]; then say "   spec      $SPEC (both arms run that lane)"; fi

if [ "$DRY_RUN" -eq 1 ]; then
    say "== dry run: nothing synced, built or run"
    exit 0
fi

mkdir -p "$WORK/chains" "$WORK/prompts"
: > "$WORK/sweep.log"
TSV="$WORK/sweep.tsv"
[ -f "$TSV" ] || printf 'length\tmatch\tposition\tmargin_nats\tclassification\n' > "$TSV"

for host in "$REF_HOST" "$LANE_HOST"; do
    remote "$host" true >/dev/null 2>&1 || die "host $host is not reachable over ssh (BatchMode)"
done

# The two artifacts must be the same bytes. Last week they were not: the two hosts carried different
# quantizations at the same byte count, and every comparison made between them was meaningless. A
# byte-count check is not enough; this compares the digest, and refuses on any difference.
say "== artifact digests"
ref_digest=$(remote "$REF_HOST" "sha256sum '$REF_ART'" | cut -d' ' -f1)
lane_digest=$(remote "$LANE_HOST" "sha256sum '$LANE_ART'" | cut -d' ' -f1)
say "   reference $ref_digest"
say "   sharded   $lane_digest"
[ -n "$ref_digest" ] && [ "$ref_digest" = "$lane_digest" ] ||
    die "the two hosts' artifacts are not the same file (reference $ref_digest, sharded $lane_digest). A comparison between them would measure the artifacts, not the lane."

# A network filesystem stalls the direct-I/O loader in uninterruptible state D. Refuse rather than
# hang, for the same reason run-tp2-arm.sh refuses.
for pair in "$REF_HOST:$REF_ART" "$LANE_HOST:$LANE_ART"; do
    host=${pair%%:*}; art=${pair#*:}
    fstype=$(remote "$host" "stat -f -c %T '$art'" 2>/dev/null || true)
    case "$fstype" in
        nfs*|cifs*|smb*|fuse*|9p*) die "artifact on $host is on a $fstype filesystem; it must be on local storage" ;;
    esac
done

if [ "$NO_BUILD" -eq 0 ]; then
    say "== sync both hosts"
    # Sync every host before building any: a build failure on the first host must not leave the second
    # carrying a stale source tree that the sweep would then run against.
    for host in "$REF_HOST" "$LANE_HOST"; do
        remote "$host" "mkdir -p '$DST'"
        rsync -a -e "ssh -F $SSH_CONFIG -o BatchMode=yes" --delete --exclude=.git --exclude=build --exclude=.agents --exclude=.agent-tmp \
              "$SRC/" "barista@$host:$DST/" >> "$WORK/sweep.log" 2>&1
    done
    # THE SHARDED ARM RUNS A PATH THAT MASTER DOES NOT ENABLE BY DEFAULT. kShardedForwardWired is the
    # compile-time switch for the sharded forward; master leaves it false, so a freshly synced lane
    # host would run the width-2 arm on the fallback path and never exercise the code under test. The
    # precondition is applied here, on the sharded host only, and printed into the run's own log, so
    # the meaning of a row is recorded with the row instead of living in someone's memory of a
    # hand-edited file. Idempotent: the pattern matches only the false spelling.
    say "== wire the sharded forward on the sharded host"
    remote "$LANE_HOST" "cd '$DST' && \
        sed -i 's/inline constexpr bool kShardedForwardWired = false;/inline constexpr bool kShardedForwardWired = true;/' \
            src/models/qwen3_5/program/internal.h && \
        grep -n 'kShardedForwardWired' src/models/qwen3_5/program/internal.h" 2>&1 |
        tee -a "$WORK/sweep.log" | sed 's/^/   /'
    say "== build both hosts"
    for host in "$REF_HOST" "$LANE_HOST"; do
        # The presence test is on the generated build file, not on the directory: an interrupted
        # configure leaves the directory behind with no build.ninja, and testing the directory would
        # then skip the configure and fail the build.
        # A non-interactive ssh shell has no CUDA compiler on PATH, so a configure there fails with
        # "Failed to find nvcc" unless CUDACXX names it. -DBUILD_TESTING=ON is what creates the test
        # targets at all; without it the acceptance target does not exist.
        remote "$host" "cd '$DST' && [ -f '$BUILD/build.ninja' ] || \
            CUDACXX=/opt/cuda/bin/nvcc cmake --preset rtx5090-release -B '$BUILD' \
            -DBUILD_TESTING=ON" >> "$WORK/sweep.log" 2>&1
        say "   building on $host"
        remote "$host" "cd '$DST' && cmake --build '$BUILD' -j \$(nproc)" >> "$WORK/sweep.log" 2>&1 ||
            die "build failed on $host; see $WORK/sweep.log"
    done
fi

# The sweep prompt is the checked-in 32-id prompt extended by the reference's own recorded greedy
# continuation, so every length up to 64 is a natural token sequence and neither host tokenizes
# anything. Both arms read the same file, which is the whole point: two hosts that tokenize the same
# text can disagree about what was prefilled.
python3 - "$SRC" "$WORK" $LENGTHS <<'PY'
import pathlib, sys
src, work, lengths = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), [int(x) for x in sys.argv[3:]]
base = (src / "tests/fixtures/tp2_accept/prompt.ids").read_text().split()
extra = (src / "tests/fixtures/tp2_accept/golden_plain.ids").read_text().split()
long_prompt = base + extra
for length in lengths:
    if length > len(long_prompt):
        raise SystemExit(f"length {length} exceeds the {len(long_prompt)} ids available")
    (work / "prompts" / f"T{length}.ids").write_text(" ".join(long_prompt[:length]) + "\n")
print(f"wrote {len(lengths)} prompt files from {len(long_prompt)} available ids")
PY

say "== prompts to both hosts"
for host in "$REF_HOST" "$LANE_HOST"; do
    remote "$host" "mkdir -p ${REMOTE_WORK}/prompts"
    rsync -a -e "ssh -F $SSH_CONFIG -o BatchMode=yes" "$WORK/prompts/" "barista@$host:${REMOTE_WORK}/prompts/" >> "$WORK/sweep.log" 2>&1
done

# The test prints "arm <w> tokens (<n>): <ids>"; that line is the chain, and it is the only thing
# compared. A run that fails or prints nothing is a hard error, not a row: a missing arm must not be
# recorded as a disagreement.
# Another workstream may hold a lane box, and a run that starts while the PREVIOUS process on this
# box has not yet released its memory fails on the device budget with a number that looks like a
# permanent limit but is only a transient. A fixed sleep is not enough: the card can report free
# memory while a compute process is still tearing down. So poll for both conditions - no compute
# processes AND no significant memory in use - and refuse loudly if the box never drains, rather
# than starting a run that will fail for a reason the caller cannot see.
wait_idle() {
    local host="$1" budget="${2:-1200}" waited=0 procs used
    while [ "$waited" -lt "$budget" ]; do
        procs=$(remote "$host" "nvidia-smi --query-compute-apps=pid --format=csv,noheader" 2>/dev/null |
                grep -c . || true)
        used=$(remote "$host" "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" 2>/dev/null |
               awk '{ if ($1+0 > m) m = $1+0 } END { print m+0 }')
        if [ "${procs:-0}" -eq 0 ] && [ "${used:-0}" -lt 500 ]; then return 0; fi
        say "   $host not drained yet (${procs:-?} compute processes, ${used:-?} MiB in use); waiting"
        sleep 20
        waited=$((waited + 20))
    done
    return 1
}

run_arm() {
    local host="$1" artifact="$2" arm="$3" prompt="$4" out="$5"
    # THE REFERENCE ARM IS PINNED TO DEVICE 0. trueno has ONE GPU and the binary's default ordinal
    # is 1, so every width-1 round died at embedding.cu:341 with cudaErrorInvalidDevice: invalid
    # device ordinal -- 13 identical failures that a precondition should have caught once. The
    # SHARDED arm is deliberately NOT pinned: it genuinely needs both devices on hephaestus.
    local visible=""
    if [ "$arm" = "1" ]; then visible="CUDA_VISIBLE_DEVICES=0 "; fi
    # The lane's diagnostics are opt-in and travel with the arm. Setting NINFER_LANE_TRACE in the
    # sweep's environment turns the instrument on for both arms, so the mechanism behind a row can be
    # read from the same run that produced the row instead of from a separate reproduction.
    local trace=""
    if [ -n "${NINFER_LANE_TRACE:-}" ]; then trace="NINFER_LANE_TRACE=$NINFER_LANE_TRACE "; fi
    local spec=""
    if [ -n "$SPEC" ]; then spec="NINFER_TP2_SPEC=$SPEC "; fi
    local command="cd '$DST' && timeout $ARM_TIMEOUT env ${visible}${trace}${spec}NINFER_TP2_ACCEPT=1 NINFER_TP2_CAPTURE=1 NINFER_TP2_ARM=$arm \
        NINFER_TP2_PROMPT='$prompt' NINFER_TEST_ARTIFACT='$artifact' \
        ./'$BUILD'/tests/ninfer_qwen3_5_tp2_accept_real_test"
    # One run per arm: the chain and the log come from the same execution, so a length costs two GPU
    # rounds plus at most one adjudication, not double that.
    local raw
    raw=$(printf '\n== %s arm %s %s\n' "$host" "$arm" "$prompt"; remote "$host" "$command" 2>&1 || true)
    printf '%s\n' "$raw" >> "$WORK/sweep.log"
    printf '%s\n' "$raw" | sed -n "s/^arm $arm tokens ([0-9]*): //p" | tail -1 > "$out"
    if [ ! -s "$out" ] && printf '%s\n' "$raw" | grep -qE "exit (124|137)|Terminated|timed out"; then
        printf '%s\tno\t-\t-\tarm%s-timeout: bound %ss exceeded\n' "$length" "$arm" "$ARM_TIMEOUT" >> "$TSV"
        say "   T=$length  arm $arm EXCEEDED the ${ARM_TIMEOUT}s bound -> recorded as arm-timeout"
        return 1
    fi
    if [ ! -s "$out" ]; then
        # A length whose arm produced no chain is a row, not a gap: the sweep's deliverable is the
        # profile, and "the sharded arm could not run at this length" is part of it. The reason is the
        # arm's own last line, kept verbatim.
        reason=$(printf '%s\n' "$raw" | grep -E "FAIL:|SKIP:|error|Error" | tail -1 | tr '\t' ' ')
        printf '%s\tno\t-\t-\tarm%s-failed: %s\n' "$length" "$arm" "${reason:-no chain printed}" >> "$TSV"
        say "   T=$length  arm $arm produced no chain -> recorded as a failed arm"
        return 1
    fi
}

# A PRECONDITION THAT FAILS 13 TIMES IDENTICALLY SHOULD FAIL ONCE, LOUDLY, BEFORE THE SWEEP STARTS.
# The reference arm binds device 0 (pinned above); assert the host actually exposes it, and name both
# numbers when it does not.
ref_devices=$(remote "$REF_HOST" "nvidia-smi -L | wc -l" 2>/dev/null | tr -d ' ')
if [ "${ref_devices:-0}" -lt 1 ]; then
    die "the reference host exposes ${ref_devices:-0} devices; the width-1 arm needs device 0. Refusing to run a sweep that would fail every length for the same reason."
fi
say "   reference host exposes $ref_devices device(s); width-1 arm pinned to device 0"
say "   per-arm bound: ${ARM_TIMEOUT}s (a hanging arm is recorded as arm-timeout, not as an anonymous failure)"

say "== sweep"
for length in $LENGTHS; do
    if cut -f1 "$TSV" | grep -qx "$length"; then
        say "   T=$length  already recorded, skipped"
        continue
    fi
    prompt="${REMOTE_WORK}/prompts/T$length.ids"
    local_prompt="$WORK/prompts/T$length.ids"
    arm1="$WORK/chains/T$length.arm1.ids"
    arm2="$WORK/chains/T$length.arm2.ids"
    wait_idle "$REF_HOST" || die "T=$length: the reference host never drained (still reports compute processes or held memory after 20 minutes); refusing to start a run that would fail on the device budget for a reason the caller cannot see"
    run_arm "$REF_HOST" "$REF_ART" 1 "$prompt" "$arm1" || continue
    wait_idle "$LANE_HOST" || die "T=$length: the sharded host never drained (still reports compute processes or held memory after 20 minutes); refusing to start a run that would fail on the device budget for a reason the caller cannot see"
    run_arm "$LANE_HOST" "$LANE_ART" 2 "$prompt" "$arm2" || continue

    if cmp -s "$arm1" "$arm2"; then
        printf '%s\texact\t-\t0\tverified\n' "$length" >> "$TSV"
        say "   T=$length  exact match over $(wc -w < "$arm1") tokens"
        continue
    fi

    # The sharded arm cannot adjudicate its own disagreement (the scoring route is single-device
    # only), so the reference host scores both candidates at their first divergence.
    remote "$REF_HOST" "mkdir -p ${REMOTE_WORK}/chains"
    scp -q -F "$SSH_CONFIG" -o BatchMode=yes "$arm2" "barista@$REF_HOST:${REMOTE_WORK}/chains/T$length.arm2.ids"
    # THE ADJUDICATION IS BOUNDED TOO. It runs the same binary as the reference arm, so it gets the
    # same bound: without one it was the last place in the sweep that could sit forever, and a
    # single unbounded step is enough to hang the whole profile.
    adjudication=$(remote "$REF_HOST" "cd '$DST' && timeout $ARM_TIMEOUT env CUDA_VISIBLE_DEVICES=0 \
        NINFER_TP2_ACCEPT=1 NINFER_TP2_ARM=1 \
        NINFER_TP2_PROMPT='$prompt' NINFER_TP2_COMPARE='${REMOTE_WORK}/chains/T$length.arm2.ids' \
        NINFER_TEST_ARTIFACT='$REF_ART' ./'$BUILD'/tests/ninfer_qwen3_5_tp2_accept_real_test" 2>&1 || true)
    printf '\n== adjudication T=%s\n%s\n' "$length" "$adjudication" >> "$WORK/sweep.log"
    if ! printf '%s\n' "$adjudication" | grep -qE "disagreement at position|near-tie on the reference|CONFIDENT disagreement"; then
        printf '%s\tno\t-\t-\tadjudication-inconclusive: bound %ss or no verdict\n' "$length" "$ARM_TIMEOUT" >> "$TSV"
        say "   T=$length  adjudication produced no verdict within the ${ARM_TIMEOUT}s bound -> recorded"
        continue
    fi
    position=$(printf '%s\n' "$adjudication" | sed -n 's/.*disagreement at position \([0-9]*\):.*/\1/p' | head -1)
    margin=$(printf '%s\n' "$adjudication" | sed -n 's/.*margin \([0-9.eE+-]*\) nats.*/\1/p' | head -1)
    if printf '%s\n' "$adjudication" | grep -q "is a near-tie on the reference"; then
        printf '%s\tno\t%s\t%s\ttie\n' "$length" "${position:-?}" "${margin:-?}" >> "$TSV"
        say "   T=$length  mismatch at ${position:-?}, margin ${margin:-?} nats -> tie"
    elif printf '%s\n' "$adjudication" | grep -q "CONFIDENT disagreement"; then
        printf '%s\tno\t%s\t%s\tconfident-disagreement\n' "$length" "${position:-?}" "${margin:-?}" >> "$TSV"
        say "   T=$length  mismatch at ${position:-?}, margin ${margin:-?} nats -> CONFIDENT DISAGREEMENT"
    else
        die "T=$length: the adjudication printed neither a tie nor a confident disagreement; see $WORK/sweep.log"
    fi
done

say "== profile"
column -t -s$'\t' "$TSV"
say ""
say "   exact:      $(awk -F'\t' 'NR>1 && $2=="exact" {print $1}' "$TSV" | tr '\n' ' ')"
say "   tie-only:   $(awk -F'\t' 'NR>1 && $5=="tie" {print $1}' "$TSV" | tr '\n' ' ')"
say "   defective:  $(awk -F'\t' 'NR>1 && $5=="confident-disagreement" {print $1}' "$TSV" | tr '\n' ' ')"
say "   table at $TSV"
