#!/usr/bin/env bash
# Run one arm of the two-GPU acceptance test on a remote lane host.
#
# The two-GPU lane is validated by comparing a width-1 round against a width-2 round token for
# token (tests/models/qwen3_5/test_engine_tp2_accept_real.cpp). No single host in the lab fits
# both: the sharded lane needs two devices with the 27B resident, and the reference needs one
# device with 20.4 GB. So the comparison is run as two arms on two hosts and the token lists are
# compared by hand or by a later step. This script runs one arm.
#
# It syncs this checkout to the lane host, builds the host's release tree there, and runs the arm
# with the three NINFER_TP2_* variables and the artifact. It refuses before doing anything if the
# host is unreachable, if the artifact is not visible on the host, or if the host has no configured
# build tree -- the last is deliberate: this script does not configure a build, because the lane
# host's cache and toolchain are its own.
#
# Usage:
#   tools/lane/run-tp2-arm.sh --arm 1|2 [--host HOST] [--artifact PATH]
#                             [--src DIR] [--dst DIR] [--build DIR] [--dry-run]
#
# Defaults match the lab's lane host and are the values this procedure has been driven with.
set -euo pipefail

ARM=2
HOST=172.30.30.15
ARTIFACT=/nfs/ninfer-gb10/qwen3_8_27b_v3_plain/qwen3_8_27b_v3-plain.ninfer
SRC=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DST=/home/barista/src/wt-tp2
BUILD=build/rtx5090-release
DRY_RUN=0

usage() {
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --arm)      ARM="${2:?--arm needs 1 or 2}"; shift 2 ;;
        --host)     HOST="${2:?--host needs a host}"; shift 2 ;;
        --artifact) ARTIFACT="${2:?--artifact needs a path}"; shift 2 ;;
        --src)      SRC="${2:?--src needs a directory}"; shift 2 ;;
        --dst)      DST="${2:?--dst needs a directory}"; shift 2 ;;
        --build)    BUILD="${2:?--build needs a directory}"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -h|--help)  usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 2 ;;
    esac
done

case "$ARM" in
    1|2) ;;
    *) echo "--arm must be 1 (single-device reference) or 2 (sharded); got '$ARM'" >&2; exit 2 ;;
esac

say() { printf '%s\n' "$*"; }
die() { printf 'refusing: %s\n' "$*" >&2; exit 1; }

remote() { ssh -o BatchMode=yes -o ConnectTimeout=8 "$HOST" "$@"; }

[ -d "$SRC" ] || die "source directory $SRC does not exist"

say "== lane arm $ARM"
say "   host     $HOST"
say "   source   $SRC -> $DST"
say "   build    $DST/$BUILD"
say "   artifact $ARTIFACT"
say "   variables NINFER_TP2_ACCEPT=1 NINFER_TP2_ARM=$ARM NINFER_TP2_UNVALIDATED=1"

# The dry run stops here: it is the mode that works on a machine that cannot reach the lane host,
# so it must not depend on the host being reachable.
if [ "$DRY_RUN" -eq 1 ]; then
    say "== dry run: nothing synced, built or run"
    exit 0
fi

if ! remote true >/dev/null 2>&1; then
    die "host $HOST is not reachable over ssh (BatchMode). Check the address and the key."
fi
say "   host reachable"

if ! remote test -f "$ARTIFACT" >/dev/null 2>&1; then
    die "artifact $ARTIFACT is not a file on $HOST. The lane host must see the same NFS path."
fi
say "   artifact present on the host"

if ! remote test -d "$DST/$BUILD" >/dev/null 2>&1; then
    die "$HOST:$DST/$BUILD is not a configured build tree. Configure it on the host first; this script does not configure builds."
fi
say "   build tree present"

say "== sync $SRC -> $HOST:$DST"
rsync -a --delete --exclude=.git --exclude=build --exclude=.agents --exclude=.agent-tmp \
      "$SRC/" "barista@$HOST:$DST/"

say "== build $BUILD on $HOST"
remote "cd $DST && cmake --build $BUILD -j 16 2>&1 | tail -5"

say "== arm $ARM"
remote "cd $DST && NINFER_TP2_ACCEPT=1 NINFER_TP2_ARM=$ARM NINFER_TP2_UNVALIDATED=1 \
        NINFER_TEST_ARTIFACT=$ARTIFACT ./$BUILD/tests/ninfer_qwen3_5_tp2_accept_real_test 2>&1 | tail -40"
