#!/usr/bin/env python3
"""Per-launch-shape kernel durations from an nsys SQLite export.

Groups every kernel whose short name matches --name by (name, grid, block, dynamic shared
memory) and prints launch count and duration percentiles. One kernel template serves many
projection shapes (fp8_a16_sliced_k_mma_kernel), and the grid separates them: grid.x is
rows / 16 for the FP8 sliced-K tiles. Decode shapes show launch counts in multiples of the
captured rounds.

    tools/gb10/kernel_shapes.py profiles/.../trace.sqlite --name fp8_a16_sliced_k_mma
"""

import argparse
import re
import sqlite3
import statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sqlite")
    parser.add_argument("--name", default=".", help="regex on the kernel short name")
    parser.add_argument("--min-count", type=int, default=50)
    args = parser.parse_args()
    pattern = re.compile(args.name)
    db = sqlite3.connect(args.sqlite)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    groups: dict[tuple, list[float]] = {}
    for short, start, end, gx, gy, gz, bx, smem in db.execute(
            "SELECT shortName, start, end, gridX, gridY, gridZ, blockX, dynamicSharedMemory "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        name = strings.get(short, str(short))
        if not pattern.search(name):
            continue
        groups.setdefault((name, (gx, gy, gz), bx, smem), []).append((end - start) / 1e3)
    print(f"{'kernel':40} {'grid':>16} {'block':>5} {'smem':>6} {'count':>7} "
          f"{'p10 us':>8} {'median':>8} {'p90 us':>8} {'total ms':>9}")
    for (name, grid, block, smem), durations in sorted(
            groups.items(), key=lambda item: -sum(item[1])):
        if len(durations) < args.min_count:
            continue
        durations.sort()
        p10 = durations[len(durations) // 10]
        p90 = durations[(len(durations) * 9) // 10]
        grid_text = "x".join(str(g) for g in grid)
        print(f"{name[:40]:40} {grid_text:>16} {block:>5} {smem:>6} {len(durations):>7} "
              f"{p10:>8.2f} {statistics.median(durations):>8.2f} {p90:>8.2f} "
              f"{sum(durations) / 1e3:>9.2f}")


if __name__ == "__main__":
    main()
