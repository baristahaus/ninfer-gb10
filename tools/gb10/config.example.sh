# GB10 run configuration. Copy to tools/gb10/config.local.sh (ignored by git) and edit.
# Every step script sources config.local.sh; nothing else needs to be set.

# Explicit Flash-Next artifact. Never a glob or a "latest" name.
ART=/absolute/path/to/qwen3_8_flash_next_125b_a6b_nvfp4.ninfer

# Port for the temporary server started by steps 1 and 2. It must be free.
PORT=18087

# KV cache dtype and MTP draft tokens used by the benchmarks. Match what you serve.
KV_DTYPE=fp8
DRAFT_TOKENS=2

# Server flags for steps 1 and 2 (the scripts add the artifact path and --port).
# Replace with the flags you normally serve with.
SERVE_ARGS=(
  --max-context 73728
  --max-concurrency 2
  --kv-dtype fp8
  --spec mtp --draft-tokens 2 --lm-head-draft
  --preserve-thinking
)

# Python 3.11+ with the standard library is enough for every step except the optional
# serving matrix, which also needs `transformers` to build its prompt fixture.
PYTHON=python3

# Optional serving matrix in step 2 (long: roughly an hour on GB10). Set RUN_SERVING=1 and
# point TOKENIZER at the local Flash-Next checkpoint or tokenizer directory to enable it.
RUN_SERVING=0
TOKENIZER=
