// The two-GPU lane's acceptance test: a width-2 round must produce the same tokens as a width-1
// round, token for token, from the same prompt and the same options.
//
// Not approximately. Every defect this lane has found -- the extent units, the column-slice order,
// the contiguity checks, the fused parents, the bind-by-object-handle -- produced numbers that
// looked plausible. A token comparison is the only check that catches the last one.
//
// Opt-in by design: while the refusal stands (a width-2 round has never been validated), running
// this test would fail for a known and intended reason, so it needs NINFER_TP2_ACCEPT=1. With the
// flag set, a refusal is a failure -- that is the point of the flag.
//
// Needs two GPUs and a 27B artifact; skips otherwise.

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

std::vector<ninfer::TokenId> run_round(const char* artifact, int width, int device,
                                       const std::vector<ninfer::TokenId>& prompt_tokens) {
    ninfer::EngineOptions options;
    options.artifact_path  = artifact;
    options.use_cuda_graph = false; // width > 1 refuses capture (caf91488)
    options.max_context    = 2048;
    if (width > 1) {
        options.devices = {0, 1};
    } else {
        options.device = device;
    }
    ninfer::Engine engine(options);
    auto prompt = engine.prepare_tokens(prompt_tokens);
    return engine.generate(std::move(prompt), {}).generated_token_ids;
}

} // namespace

int main() {
    if (std::getenv("NINFER_TP2_ACCEPT") == nullptr) {
        std::cerr << "SKIP: the two-GPU acceptance comparison is opt-in (set NINFER_TP2_ACCEPT=1)"
                  << '\n';
        return 77;
    }
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    const bool have_artifact = artifact != nullptr && *artifact != '\0';
    int devices                 = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&devices);
    const bool have_two_devices = count_err == cudaSuccess && devices >= 2;
    if (!have_artifact || !have_two_devices) {
        std::cerr << "SKIP: the two-GPU acceptance case needs two CUDA devices and a 27B artifact"
                  << " (devices: "
                  << (count_err == cudaSuccess ? std::to_string(devices)
                                               : std::string("unavailable"))
                  << ", NINFER_TEST_ARTIFACT: " << (have_artifact ? "set" : "not set") << ")\n";
        return 77;
    }

    const std::vector<ninfer::TokenId> prompt{9707, 11, 1879, 13, 5765, 374};
    std::vector<ninfer::TokenId> single;
    try {
        single = run_round(artifact, 1, 0, prompt);
    } catch (const std::exception& error) {
        return fail(std::string("the single-device reference round threw: ") + error.what());
    }
    if (single.empty()) { return fail("the single-device reference round produced no tokens"); }

    std::vector<ninfer::TokenId> sharded;
    try {
        sharded = run_round(artifact, 2, 0, prompt);
    } catch (const std::exception& error) {
        // With the flag set, a refusal is exactly what this test exists to prevent shipping.
        return fail(std::string("the width-2 round did not run: ") + error.what());
    }

    if (sharded.size() != single.size()) {
        return fail("the rounds produced different token counts: width 2 " +
                    std::to_string(sharded.size()) + " vs width 1 " +
                    std::to_string(single.size()));
    }
    for (std::size_t i = 0; i < single.size(); ++i) {
        if (sharded[i] != single[i]) {
            return fail("token " + std::to_string(i) + " differs: width 2 " +
                        std::to_string(sharded[i]) + " vs width 1 " + std::to_string(single[i]));
        }
    }
    std::cout << "ok: " << single.size() << " tokens identical at width 1 and width 2\n";
    return 0;
}
