// The two-GPU lane loads but cannot infer yet, and this test pins that property rather than the
// message: a sharded-lane Engine must construct (the load milestone) and its first round must
// refuse, while a single-device Engine must not carry the refusal at all.
//
// Why the refusal exists: both ranks now take the sharded call sites, but no width-2 round has ever
// been compared against a width-1 round, and a sharded projection that is subtly wrong produces
// plausible numbers rather than an error. A silent wrong answer is the one failure mode this lane
// must not have, so the round entry throws until the comparison passes.
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

bool mentions_refusal(const std::exception& error) {
    const std::string what = error.what();
    return what.find("sharded forward has no validated round") != std::string::npos;
}

} // namespace

int main() {
    const char* artifact   = std::getenv("NINFER_TEST_ARTIFACT");
    const bool have_artifact = artifact != nullptr && *artifact != '\0';
    // Both prerequisites are checked before either is reported. Checking only the artifact first
    // told a one-device host that the artifact was missing, which sent a reader after the wrong
    // cause; the device count is the binding constraint on a single-GPU node.
    int devices                 = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&devices);
    const bool have_two_devices = count_err == cudaSuccess && devices >= 2;
    if (!have_artifact || !have_two_devices) {
        std::cerr << "SKIP: the two-GPU refusal case needs two CUDA devices and a 27B artifact"
                  << " (devices: "
                  << (count_err == cudaSuccess ? std::to_string(devices)
                                               : std::string("unavailable"))
                  << ", NINFER_TEST_ARTIFACT: " << (have_artifact ? "set" : "not set") << ")\n";
        return 77;
    }
    const std::vector<ninfer::TokenId> tokens{9707, 11, 1879, 13, 5765, 374};

    // The sharded lane must load. This is the milestone the two-GPU run is for: both ranks
    // materialize, both Programs are built, every mirror attaches.
    {
        ninfer::EngineOptions options;
        options.artifact_path  = artifact;
        options.devices        = {0, 1};
        options.use_cuda_graph = false; // width > 1 refuses capture (caf91488)
        options.max_context    = 2048;
        ninfer::Engine engine(options);
        // ... and must refuse the first round rather than infer wrongly.
        try {
            auto prompt = engine.prepare_tokens(tokens);
            (void)engine.generate(std::move(prompt), {});
            return fail("the sharded lane inferred instead of refusing");
        } catch (const std::exception& error) {
            if (!mentions_refusal(error)) {
                return fail(std::string("the sharded lane threw the wrong error: ") + error.what());
            }
        }
    }

    // The single-device lane must not carry the refusal: one short round has to run.
    {
        ninfer::EngineOptions options;
        options.artifact_path  = artifact;
        options.device         = 0;
        options.use_cuda_graph = false;
        options.max_context    = 2048;
        ninfer::Engine engine(options);
        auto prompt = engine.prepare_tokens(tokens);
        try {
            const auto result = engine.generate(std::move(prompt), {});
            if (result.generated_token_ids.empty()) {
                return fail("the single-device round produced no tokens");
            }
        } catch (const std::exception& error) {
            return fail(std::string("the single-device lane threw: ") + error.what());
        }
    }
    std::cout << "ok\n";
    return 0;
}
