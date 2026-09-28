#!/usr/bin/env bash
# Step 7a run order, item 1: build the current tree and run the named FP8 oracle
# tests, the full ctest suite, and the dense-FP8 converter pytest.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DIR=$(step_dir "step7a-build-test")
log "commit: $(git rev-parse --short HEAD)"
require_untraced_build

log "build (incremental)"
cmake --build build -j 2>&1 | tee "$DIR/build.log"
log "build done"

for t in ninfer_linear_fp8_a16_test ninfer_hyperconnection_test \
         ninfer_flash_next_qsa_test ninfer_argmax_test; do
    require_binary "build/tests/$t"
    log "oracle test: $t"
    "./build/tests/$t" 2>&1 | tee "$DIR/$t.log"
done
# Converter test suite needs pytest + torch in the project venv (.venv is
# gitignored); CPU torch is sufficient (the fp8_row path is CPU, tests are small).
if [[ ! -x .venv/bin/python ]]; then
    log "creating .venv (pytest + cpu torch)"
    uv venv .venv
    uv pip install --python .venv/bin/python pytest numpy safetensors
    uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
fi
log "converter pytest"
.venv/bin/python -m pytest tests/convert/test_flash_next_dense_fp8.py -q 2>&1 | tee "$DIR/pytest.log"
log "full ctest"
ctest --test-dir build --output-on-failure 2>&1 | tee "$DIR/ctest.log"

log "converter pytest"
"$PYTHON" -m pytest tests/convert/test_flash_next_dense_fp8.py -q 2>&1 | tee "$DIR/pytest.log"

log "all step 7a item-1 checks done: $DIR"
