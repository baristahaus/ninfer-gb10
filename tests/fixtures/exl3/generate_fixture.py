"""Generate the EXL3 codec fixture from the pinned exllamav3 (contract section 3.3).

Run on a CUDA GPU with exllamav3 v1.6.1 at commit 3a513ac9 installed exactly as described in
`tests/fixtures/exl3/README.md`. The script refuses any other exllamav3 build. It decodes with
exllamav3's own runtime class (`LinearEXL3`), so the emitted FP16 values are what exllamav3
reconstructs, independent of NInfer's oracles and of the contract text.

Inputs are uniformly random trellis words, which are valid EXL3 tensors: any bit string is a
tail-biting ring. Output (`--output`, JSON) holds:

- `tiles`: for every K in 1..8 and {1.5, 2.5, 3.5} and each codebook it admits (MUL1 only for
  half-integer K), TILES_PER_RATE random 16x16 tiles of a random 128x128 linear, each with its
  stored trellis elements and exllamav3's reconstructed W_inner;
- `coverage`: per codebook, one K = 8 linear whose rings are built so that every 16-bit state
  occurs at least once, pinning the codebook's complete 65,536-entry table (an addition to the
  contract's random tiles: random tiles cannot reach every state);
- `linear`: one 128x128 linear (K = 4, MUL1) with random suh/svh, its reconstructed W_inner and
  exllamav3's `get_weight_tensor()` result, the FP16 composition of contract section 2.1.

Arrays are lowercase hex of their little-endian bytes: trellis as int16 [k/16, n/16, 16 K],
W_inner and weight as FP16 [k, n] row-major, suh/svh as FP16 vectors.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

PINNED_VERSION = "1.6.1"
PINNED_COMMIT = "3a513ac9f91aae777f317ca2341b8e445b62c64e"
FORMAT = "ninfer_exl3_fixture"
FORMAT_VERSION = 1
RATES_K2 = (2, 4, 6, 8, 10, 12, 14, 16, 3, 5, 7)  # 2K
CODEBOOKS = ("3inst", "mcg", "mul1")
TILES_PER_RATE = 4
TILE_LINEAR = 128


def check_pin() -> str:
    import exllamav3
    from exllamav3.version import __version__

    if __version__ != PINNED_VERSION:
        sys.exit(f"exllamav3 {__version__} is installed; the fixture pins {PINNED_VERSION}")
    source = Path(exllamav3.__file__).resolve().parent.parent
    try:
        head = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sys.exit(f"exllamav3 at {source} is not the pinned git checkout (install it editable)")
    if head != PINNED_COMMIT:
        sys.exit(f"exllamav3 checkout is at {head}; the fixture pins {PINNED_COMMIT}")
    dirty = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        sys.exit(f"exllamav3 checkout at {source} has local modifications")
    return head


def hex_bytes(tensor: torch.Tensor) -> str:
    return tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes().hex()


def markers(codebook: str) -> dict:
    from exllamav3.modules.quant.exl3_lib.quantize import codebook_markers

    return {key: value.cuda() for key, value in codebook_markers({codebook: True}).items()} \
        if codebook != "3inst" else {}


def make_linear(trellis: torch.Tensor, codebook: str, suh: torch.Tensor | None = None,
                svh: torch.Tensor | None = None):
    from exllamav3.modules.quant import LinearEXL3

    k = trellis.shape[0] * 16
    n = trellis.shape[1] * 16
    if suh is None:
        suh = torch.ones(k, dtype=torch.half, device="cuda")
    if svh is None:
        svh = torch.ones(n, dtype=torch.half, device="cuda")
    return LinearEXL3(None, k, n, suh=suh, svh=svh, trellis=trellis, **markers(codebook))


def as_int16(values: list[int]) -> list[int]:
    """uint16 bit patterns as the int16 values that store them."""

    return [value - (1 << 16) if value >= 1 << 15 else value for value in values]


def random_trellis(generator: torch.Generator, rows: int, cols: int, k2: int) -> torch.Tensor:
    shape = (rows, cols, 8 * k2)
    return torch.randint(-(1 << 15), 1 << 15, shape, generator=generator, dtype=torch.int16).cuda()


def tiles_section(generator: torch.Generator) -> list[dict]:
    out = []
    for k2 in RATES_K2:
        codebooks = CODEBOOKS if k2 % 2 == 0 else ("mul1",)
        for codebook in codebooks:
            tiles = TILE_LINEAR // 16
            trellis = random_trellis(generator, tiles, tiles, k2)
            inner = make_linear(trellis, codebook).get_inner_weight_tensor()
            picks = torch.randint(0, tiles * tiles, (TILES_PER_RATE,), generator=generator)
            for pick in picks.tolist():
                a, b = divmod(pick, tiles)
                out.append({
                    "k2": k2,
                    "codebook": codebook,
                    "tile": [a, b],
                    "trellis": hex_bytes(trellis[a, b]),
                    "inner": hex_bytes(inner[16 * a:16 * a + 16, 16 * b:16 * b + 16]),
                })
    return out


def de_bruijn_pairs(symbols: int) -> list[int]:
    """A cyclic sequence over `symbols` values in which every ordered pair occurs exactly once."""

    sequence, a = [], [0] * (2 * symbols)

    def search(t: int, p: int) -> None:
        if t > 2:
            if 2 % p == 0:
                sequence.extend(a[1:p + 1])
            return
        a[t] = a[t - p]
        search(t + 1, p)
        for value in range(a[t - p] + 1, symbols):
            a[t] = value
            search(t + 1, t)

    search(1, 1)
    return sequence


def coverage_rings() -> list[list[int]]:
    """256-byte rings (K = 8) whose consecutive byte pairs, the 16-bit states, cover all 65,536.

    At K = 8 a ring of 256 bytes yields the states (byte[i-1] << 8) | byte[i], cyclically, per
    the contract's section 2.4. The de Bruijn sequence is cut into rings; the pairs lost at the
    cuts are covered by extra rings."""

    sequence = de_bruijn_pairs(256)
    rings = [sequence[i:i + 256] for i in range(0, len(sequence), 256)]
    seen = set()
    for ring in rings:
        seen.update((ring[i - 1] << 8) | ring[i] for i in range(256))
    missing = sorted(set(range(1 << 16)) - seen)
    flat = [byte for state in missing for byte in (state >> 8, state & 0xFF)]
    while flat:
        ring, flat = flat[:256], flat[256:]
        ring += [0] * (256 - len(ring))
        rings.append(ring)
        seen.update((ring[i - 1] << 8) | ring[i] for i in range(256))
    if len(seen) != 1 << 16:
        raise AssertionError("coverage rings miss states")
    return rings


def ring_elements(ring: list[int]) -> list[int]:
    """Stored int16 elements of a ring of bytes: ring byte 4m+j is byte j (most significant
    first) of little-endian uint32 word m (contract section 2.4)."""

    elements = []
    for m in range(0, len(ring), 4):
        word = (ring[m] << 24) | (ring[m + 1] << 16) | (ring[m + 2] << 8) | ring[m + 3]
        elements += [word & 0xFFFF, word >> 16]
    return elements


def coverage_section() -> list[dict]:
    rings = coverage_rings()
    tiles_n = TILE_LINEAR // 16
    rows = -(-len(rings) // tiles_n)
    rows = -(-rows // 8) * 8  # k a multiple of 128, as in every EXL3 tensor
    rings += [[0] * 256] * (rows * tiles_n - len(rings))
    elements = torch.tensor([as_int16(ring_elements(ring)) for ring in rings], dtype=torch.int16)
    trellis = elements.reshape(rows, tiles_n, 128).cuda()
    out = []
    for codebook in CODEBOOKS:
        inner = make_linear(trellis, codebook).get_inner_weight_tensor()
        out.append({
            "k2": 16,
            "codebook": codebook,
            "k": rows * 16,
            "n": TILE_LINEAR,
            "trellis": hex_bytes(trellis),
            "inner": hex_bytes(inner),
        })
    return out


def linear_section(generator: torch.Generator) -> dict:
    k2, codebook = 8, "mul1"
    tiles = TILE_LINEAR // 16
    trellis = random_trellis(generator, tiles, tiles, k2)
    signs_u = torch.randint(0, 2, (TILE_LINEAR,), generator=generator) * 2 - 1
    signs_v = torch.randint(0, 2, (TILE_LINEAR,), generator=generator) * 2 - 1
    scale_u = torch.rand(TILE_LINEAR, generator=generator) * 0.75 + 0.25
    scale_v = torch.rand(TILE_LINEAR, generator=generator) * 0.75 + 0.25
    suh = (signs_u * scale_u).to(torch.half).cuda()
    svh = (signs_v * scale_v).to(torch.half).cuda()
    linear = make_linear(trellis, codebook, suh, svh)
    return {
        "k2": k2,
        "codebook": codebook,
        "k": TILE_LINEAR,
        "n": TILE_LINEAR,
        "trellis": hex_bytes(trellis),
        "suh": hex_bytes(suh),
        "svh": hex_bytes(svh),
        "inner": hex_bytes(linear.get_inner_weight_tensor()),
        "weight": hex_bytes(linear.get_weight_tensor()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("the fixture needs a CUDA device")
    commit = check_pin()
    torch.cuda.set_device(0)
    generator = torch.Generator().manual_seed(args.seed)
    with torch.inference_mode():
        fixture = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "exllamav3": {"version": PINNED_VERSION, "commit": commit},
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
                "capability": "sm_%d%d" % torch.cuda.get_device_capability(0),
            },
            "seed": args.seed,
            "tiles": tiles_section(generator),
            "coverage": coverage_section(),
            "linear": linear_section(generator),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {args.output}: {len(fixture['tiles'])} tiles, "
          f"{len(fixture['coverage'])} coverage linears, 1 composed linear")


if __name__ == "__main__":
    main()
