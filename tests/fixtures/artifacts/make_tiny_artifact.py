#!/usr/bin/env python3
"""Build the small deterministic `.ninfer` fixture the artifact tests can read from a clean checkout.

    tests/fixtures/artifacts/make_tiny_artifact.py [OUT]

Default OUT is tests/fixtures/artifacts/tiny.ninfer next to this script. The artifact holds one
BF16 tensor and one binding: it is a framing/codec fixture, not a model. It cannot stand in for a
real model artifact — the real-artifact tests need the artifacts named in manifest.json.

The output is byte-deterministic: the writer's artifact identity, normally a fresh UUID, is fixed
here, so the committed digest in the manifest stays valid across regenerations. Everything else is
already deterministic (the directory encoding and the payload bytes are functions of the inputs).
"""

from __future__ import annotations

import struct

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.artifact import writer as writer_module  # noqa: E402
from tools.artifact.schema import TensorSpec  # noqa: E402

# One fixed identity for every regeneration of this fixture. The writer reads `uuid4().bytes`.
class _FixedIdentity:
    bytes = bytes(range(16))


writer_module.uuid4 = _FixedIdentity

ROWS, COLUMNS = 8, 16


def payload() -> bytes:
    """A deterministic BF16 pattern: every value is (row * 16 + column) / 64 as BF16 bits."""
    words = []
    for row in range(ROWS):
        for column in range(COLUMNS):
            value = (row * COLUMNS + column) / 64.0
            fp32 = struct.pack("<f", value)
            # BF16 keeps the high 16 bits of the FP32 encoding.
            words.append(fp32[2:4])
    return b"".join(words)


def build(out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    with writer_module.ArtifactWriter(
        out,
        [TensorSpec("tiny_matrix", (ROWS, COLUMNS), "bf16", "contiguous_le_v1")],
        components={"text": {"config": {"fixture": "tiny"}}},
        bindings={"whole": {"object": "tiny_matrix"}},
        metadata={"fixture": "tests/fixtures/artifacts/tiny.ninfer"},
    ) as writer:
        writer.write_object("tiny_matrix", payload())
    return out


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else Path(__file__).with_name("tiny.ninfer")
    path = build(out)
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
