# Local GB10 run config (machine-local; gitignored).
# SP-6 bring-up: the Flash-Next NVFP4 artifact (explicit path, per the SP-6 handoff).
# The FP8 MTP artifact (previous ART) returns here for the fp8mtp work:
# /home/apollo11/ninfer-gb10/out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer
ART=/home/apollo11/models/qwen3_8_flash_next_v3_fork/qwen3_8_flash_next_125b_a6b_nvfp4.ninfer
PORT=8000
KV_DTYPE=fp8
DRAFT_TOKENS=2
SERVE_ARGS=(
  --max-context 262144
  --kv-capacity auto
  --max-concurrency 2
  --kv-dtype fp8
  --spec mtp --draft-tokens 2 --lm-head-draft
  --vision
)
PYTHON=python3
RUN_SERVING=1
QWEN35_ART=/home/apollo11/models/qwen3_8_27b_v3-plain.ninfer
TOKENIZER=/home/apollo11/models/tokenizers/qwen3_8_flash_next
# I8 (PLE int6 side-check): BF16_SOURCE and FP8_SOURCE not on disk, left unset.
# (Disk has ample space as of 2026-10-14; the blocker is only the missing source checkpoints.)
