# EXL3 codec fixture

`generate_fixture.py` produces `exllamav3-1.6.1.json`, the external ground truth of the EXL3
contract (`docs/maintainer/exl3-trellis-contract.md` §3.3). It runs exllamav3 v1.6.1 at commit
`3a513ac9f91aae777f317ca2341b8e445b62c64e` on a CUDA GPU and refuses any other exllamav3 build.
The fixture is generated once and committed here. Until it exists, the tests that consume it skip.

## Prerequisites

- An NVIDIA GPU with a driver that runs CUDA 13 builds. The commands below target the RTX 5090
  (`sm_120`).
- A CUDA 13 toolkit with `nvcc` (the box's CUDA 13.4.92 at `/usr/local/cuda`), used to compile
  exllamav3's extension against the CUDA 13.0 build of torch (same major version).
- Python 3.11, `git`, network access to `github.com`, `download.pytorch.org` and PyPI.

## Install (once)

Run from any directory. The environment is separate from NInfer's maintainer environment.

```bash
python3.11 -m venv ~/venvs/exl3-fixture
source ~/venvs/exl3-fixture/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130

git clone https://github.com/turboderp-org/exllamav3 ~/src/exllamav3
git -C ~/src/exllamav3 checkout 3a513ac9f91aae777f317ca2341b8e445b62c64e
python -m pip install -r ~/src/exllamav3/requirements.txt

export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="12.0"
python -m pip install --no-build-isolation -e ~/src/exllamav3

python -c "import torch, exllamav3.ext; print(torch.__version__, torch.cuda.get_device_name(0))"
```

The editable install (`-e`) is required: the script reads the checkout's `git HEAD` to prove the
pin and rejects a checkout with modified tracked files. `--no-build-isolation` compiles the
extension once against the installed torch instead of compiling it at first import. torch 2.11.0
is the CUDA 13.0 build named in exllamav3 v1.6.1's own sources. Any torch >= 2.6 whose CUDA build
supports the GPU produces the same reconstruction, because the reconstruction is exllamav3's own
CUDA code.

## Run

From the NInfer repository root, inside the activated environment:

```bash
python tests/fixtures/exl3/generate_fixture.py --output tests/fixtures/exl3/exllamav3-1.6.1.json
```

It takes seconds and writes about 2 MB. The default `--seed` (20261014) is the one to commit.

## Check

With the fixture in place, in NInfer's maintainer environment:

```bash
python3 -m pytest tests/convert/quantization/test_exl3.py -q
cmake --build build -j --target ninfer_exl3_codec_test
ctest --test-dir build -R ninfer_exl3_codec_test --output-on-failure
```

Both oracles must reproduce every fixture tile and coverage linear bit-exactly. The composed
`get_weight_tensor()` result, which exllamav3 evaluates in FP16, is compared with the binary64
composition of contract §2.1 within an FP16 tolerance.

## Contents

| Key | Content |
|---|---|
| `tiles` | for each K in 1–8 and {1.5, 2.5, 3.5} and each codebook it admits (MUL1 only for half-integer K), 4 random tiles of a random 128×128 linear: `k2` (= 2K), `codebook`, `tile` [a, b], `trellis` (16 K int16), `inner` (256 FP16, row-major r, c) |
| `coverage` | per codebook, one K = 8 linear (`k` × 128) whose rings contain every 16-bit state: `trellis` int16 [k/16, 8, 128], `inner` FP16 [k, 128] |
| `linear` | one 128×128 linear (K = 4, MUL1) with random `suh`/`svh`: `trellis`, `suh`, `svh`, `inner`, and `weight` = exllamav3 `get_weight_tensor()` FP16 [128, 128] |

Arrays are lowercase hex of their little-endian bytes. `exllamav3` records the pin, and
`environment` records the torch, CUDA and device that produced the file.
