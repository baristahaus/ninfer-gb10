#!/usr/bin/env python3
"""QSA stage time from an nsys sqlite export plus the serve request log.

Usage: pr36fix_c4_stage.py TRACE_SQLITE REQUEST_JSONL

Stage = GPU time of the QSA op kernels (attention: selected_attention_split/batched_fp8 and
reduce_selected_attention_splits; selection/indexing: the index-selection family in
flash_next_qsa.cu). Steps = decode rounds from the request log (engine_timing.decode.rounds
over the measured requests). Prints the stage total, per-step value, and the kernel
composition so a route change shows up as a composition change.
"""
import json
import sqlite3
import sys

STAGE_KERNELS = {
    "selected_attention_split_fp8_kernel",
    "selected_attention_batched_fp8_kernel",
    "reduce_selected_attention_splits_kernel",
    "select_top_groups_kernel",
    "score_groups_batched_kernel",
    "score_groups_mma_kernel",
    "compress_index_groups_kernel",
    "expand_indices_batched_kernel",
    "dense_indices_batched_kernel",
    "order_groups_like_persistent_topk_kernel",
    "hierarchical_top_groups_kernel",
    "prepare_index_query_kernel",
}


def rounds_total(path, skip=0):
    rounds = 0
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("event") != "request_done":
                continue
            if n < skip:
                n += 1
                continue
            n += 1
            rounds += rec.get("engine_timing", {}).get("decode", {}).get("rounds", 0)
    return rounds, n


def main():
    db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
    cur = db.cursor()
    skip = 5 if len(sys.argv) > 3 and sys.argv[3] == "measured-only" else 0
    rows = {}
    for name, c, ms in cur.execute(
        """SELECT s.value, COUNT(*), SUM(k.end - k.start) / 1e6
           FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName = s.id
           WHERE s.value LIKE '%attention%' OR s.value LIKE '%top_groups%'
              OR s.value LIKE '%index%' OR s.value LIKE '%groups%' OR s.value LIKE '%hadamard%'
           GROUP BY s.value"""
    ):
        if any(part in name for part in STAGE_KERNELS):
            rows[name.split("::")[-1][:60]] = (ms, c)
    total = sum(ms for ms, _ in rows.values())
    launches = sum(c for _, c in rows.values())
    rounds, nreq = rounds_total(sys.argv[2], skip)
    print(f"requests counted: {nreq} (skip={skip}); decode rounds: {rounds}")
    print(f"QSA stage total: {total:.1f} ms over {launches} launches")
    if rounds:
        print(f"per decode step (round): {total / rounds:.3f} ms")
    for name, (ms, c) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
        per = (ms / c) if c else 0
        print(f"  {ms:9.1f} ms  n={c:<7d} {per:8.4f} ms/launch  {name}")


if __name__ == "__main__":
    main()
