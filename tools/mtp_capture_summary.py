#!/usr/bin/env python3
"""Summarize an MTP round capture file written by --mtp-capture.

Reads the header, then streams fixed-size row records. Reports acceptance by draft
position, acceptance by context-position bucket, extent distribution, and (with
--artifact) how often rejected drafts fall outside the first N rows of the artifact's
optimized proposal-head shortlist — the signal that sizes --proposal-head-rows.

Usage:
  python3 tools/mtp_capture_summary.py CAPTURE [--artifact ART.ninfer] [--rows N]
                                              [--bucket 8192]
"""

import argparse
import json
import os
import struct
import sys

MAGIC = b"NMTPCAP\x01"
HEADER = struct.Struct("<8sIIIIII")
RECORD_PREFIX = struct.Struct("<QIIIII")


def fail(message):
    print(f"error: {message}", file=sys.stderr)
    return 1


def read_shortlist(artifact, count=None):
    """Return the artifact's optimized proposal head token id list (ascending rows)."""
    with open(artifact, "rb") as handle:
        lead = handle.read(32)
        json_bytes = struct.unpack("<Q", lead[8:16])[0]
        catalog = json.loads(handle.read(json_bytes))
        base_dir = os.path.dirname(os.path.abspath(artifact))
        entry_payload = ((32 + json_bytes) + 4095) // 4096 * 4096
        # Map absolute object offsets to their volume file: volume i covers the payload
        # range starting at the sum of the previous volumes' payload bytes.
        spans, low = [], 0
        for index, volume in enumerate(catalog["files"]):
            payload = volume["payload_bytes"]
            path = artifact if index == 0 else os.path.join(base_dir, volume["path"])
            spans.append((low, low + payload, path, entry_payload if index == 0 else 4096))
            low += payload
        obj = next(o for o in catalog["objects"]
                   if o["id"] == "ninfer.optimized_proposal_head.token_ids")
        for span_low, span_high, path, payload_start in spans:
            if span_low <= obj["offset"] < span_high:
                physical = payload_start + (obj["offset"] - span_low)
                break
        else:
            raise ValueError("token_ids object not found in any artifact volume")
        with open(path, "rb") as volume_file:
            volume_file.seek(physical)
            payload = volume_file.read(obj["bytes"])
    ids = struct.unpack(f"<{obj['bytes'] // 4}i", payload)
    if count is not None:
        ids = ids[:count]
    return ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("capture")
    parser.add_argument("--artifact", help=".ninfer artifact for shortlist membership")
    parser.add_argument("--rows", type=int, default=0,
                        help="proposal-head prefix size to test (default: full)")
    parser.add_argument("--bucket", type=int, default=8192,
                        help="base_frontier bucket width (default: 8192)")
    args = parser.parse_args(argv)

    with open(args.capture, "rb") as handle:
        header = handle.read(HEADER.size)
        if len(header) < HEADER.size:
            return fail("file too small for a capture header")
        magic, version, header_size, k, width, token_domain, ranks = HEADER.unpack(header)
        if magic[:7] != MAGIC[:7]:
            return fail("not an MTP capture file")
        if version not in (1, 2):
            return fail(f"unsupported capture version {version}")
        if version == 1:
            ranks = 0
        record_bytes = 28 + (2 * k + 1) * 4 + (k + 1) * (4 + ranks * 8) if ranks else \
            28 + (2 * k + 1) * 4
        handle.seek(0)
        stream = handle.read()

    if args.artifact:
        shortlist = read_shortlist(args.artifact)
        prefix = set(shortlist[: args.rows]) if args.rows else set(shortlist)
        prefix_rows = args.rows or len(shortlist)
    else:
        prefix = None

    rounds = set()
    rows = 0
    fallbacks = 0
    drafted = 0
    accepted = 0
    by_position = [0] * k
    extent_hist = {}
    bucket_rows = {}
    bucket_accepted = {}
    rejected_total = 0
    rejected_outside = 0
    accepted_inside = 0
    accepted_outside = 0
    chosen_sum = chosen_n = 0.0
    chosen_n = 0
    near_miss = near_miss_rank_sum = 0
    near_miss_logprob_sum = 0.0
    rejected_scored = 0

    offset = header_size
    while offset + record_bytes <= len(stream):
        round_index, lane, base_frontier, extent, committed_count, _pad = (
            RECORD_PREFIX.unpack_from(stream, offset))
        offset += RECORD_PREFIX.size
        drafts = struct.unpack_from(f"<{k}i", stream, offset)
        offset += 4 * k
        committed = struct.unpack_from(f"<{k + 1}i", stream, offset)
        offset += 4 * (k + 1)
        scores = None
        if ranks:
            scores = []
            for _column in range(k + 1):
                (chosen,) = struct.unpack_from("<f", stream, offset)
                offset += 4
                pairs = struct.unpack_from("<" + "if" * ranks, stream, offset)
                offset += ranks * 8
                scores.append((chosen, [(pairs[2 * i], pairs[2 * i + 1])
                                        for i in range(ranks)]))

        rows += 1
        rounds.add(round_index)
        bucket = base_frontier // max(1, args.bucket)
        bucket_rows[bucket] = bucket_rows.get(bucket, 0) + 1
        if extent == 0:
            fallbacks += 1
            continue
        extent_hist[extent] = extent_hist.get(extent, 0) + 1
        drafted += extent
        accepted += committed_count - 1
        for position in range(committed_count - 1):
            by_position[position] += 1
        bucket_accepted[bucket] = bucket_accepted.get(bucket, 0) + committed_count - 1
        if scores is not None:
            for position in range(committed_count - 1):
                chosen_sum += scores[position + 1][0]
                chosen_n += 1
            for position in range(extent):
                if position < committed_count - 1:
                    continue
                column = scores[position + 1][1]
                for rank, (top_id, top_logprob) in enumerate(column):
                    if top_id == drafts[position]:
                        near_miss += 1
                        near_miss_rank_sum += rank
                        near_miss_logprob_sum += top_logprob
                        break
                rejected_scored += 1
        if prefix is not None:
            for position in range(extent):
                token = drafts[position]
                inside = token in prefix
                hit = position < committed_count - 1
                if hit:
                    if inside:
                        accepted_inside += 1
                    else:
                        accepted_outside += 1
                if not hit:
                    rejected_total += 1
                    if not inside:
                        rejected_outside += 1

    if rows == 0:
        return fail("capture contains no records")

    print(f"file            {args.capture}")
    print(f"draft window k  {k}   token domain {token_domain}")
    print(f"rounds          {len(rounds)}   rows {rows}")
    print(f"fallback steps  {fallbacks}   drafted {drafted}   accepted {accepted}")
    if drafted:
        print(f"acceptance      {accepted / drafted:.3f}")
    if ranks:
        print(f"target score report (top-{ranks} alternatives per verify column):")
        if chosen_n:
            print(f"  mean committed-token logprob: {chosen_sum / chosen_n:.3f} over {chosen_n}")
        if rejected_scored:
            print(f"  rejected drafts inside the target top-{ranks}: {near_miss}/{rejected_scored} "
                  f"({100.0 * near_miss / rejected_scored:.1f}%) - near misses an accept "
                  "predictor could catch")
            if near_miss:
                print(f"    mean rank {near_miss_rank_sum / near_miss:.2f}, "
                      f"mean logprob {near_miss_logprob_sum / near_miss:.3f}")
    print("acceptance by draft position:")
    for position, hits in enumerate(by_position):
        print(f"  pos {position}: {hits}/{rows - fallbacks}")
    print("extent histogram:", dict(sorted(extent_hist.items())))
    print("acceptance by base_frontier bucket:")
    for bucket in sorted(bucket_rows):
        fall = 0
        label = f"{bucket * args.bucket:>9}-{(bucket + 1) * args.bucket:>9}"
        acc = bucket_accepted.get(bucket, 0)
        rate = acc / (bucket_rows[bucket] * k) if bucket_rows[bucket] else 0.0
        print(f"  {label}: rows {bucket_rows[bucket]:>6}  accept/pos {rate:.3f}")
    if prefix is not None:
        print(f"shortlist prefix rows {prefix_rows} of {len(shortlist)}")
        if rejected_total:
            print(f"rejected drafts outside prefix: {rejected_outside}/{rejected_total} "
                  f"({rejected_outside / rejected_total:.1%})")
        print(f"accepted drafts inside prefix:  {accepted_inside}")
        if accepted_inside + accepted_outside:
            print(f"accepted drafts outside prefix: {accepted_outside} "
                  f"({accepted_outside / (accepted_inside + accepted_outside):.1%} of accepted) "
                  "- direct truncation acceptance loss")
    return 0


if __name__ == "__main__":
    sys.exit(main())
