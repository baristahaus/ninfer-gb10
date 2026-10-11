#!/usr/bin/env bash
# Flash-Next MoE / HyperConnection op tests at 20-32 verify rows (ninfer-gb10-dnf),
# 2026-10-11. Post-reboot, one GPU job at a time: build, then the two tests
# sequentially. Question: is the MoE op wrong at 24 rows (NaN expert ids), or do
# bad rows arrive from earlier in the verify chain?
set -u
cd /home/apollo11/ninfer-gb10
OUT=profiles/bench/gb10/op-tests-2026-10-11
mkdir -p "$OUT"
clog() { echo "[op-tests $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$OUT/run.log"; }

clog "start; HEAD $(git rev-parse --short HEAD)"
free_g=$(free -g | awk '/^Mem:/{print $7}')
clog "free memory ${free_g}G available"
clog "building (incremental)"
cmake --build build -j >"$OUT/build.log" 2>&1 || { clog "BUILD FAILED: see build.log"; exit 1; }
clog "build OK"

test -x build/tests/ninfer_flash_next_moe_test || { clog "moe test binary missing"; exit 1; }
test -x build/tests/ninfer_hyperconnection_test || { clog "hyperconnection test binary missing"; exit 1; }

build/tests/ninfer_flash_next_moe_test >"$OUT/moe_test.log" 2>&1
rc1=$?
clog "moe test rc=$rc1"

build/tests/ninfer_hyperconnection_test >"$OUT/hc_test.log" 2>&1
rc2=$?
clog "hyperconnection test rc=$rc2"

clog "done: moe rc=$rc1 hc rc=$rc2"
exit $(( (rc1 != 0 || rc2 != 0) ? 1 : 0 ))
