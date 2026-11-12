// The two-GPU lane's acceptance test: a width-2 round must produce the same tokens as a width-1
// round, token for token, from the same prompt and the same options -- and both must reproduce the
// chain recorded on the reference host.
//
// Not approximately. Every defect this lane has found -- the extent units, the column-slice order,
// the contiguity checks, the fused parents, the bind-by-object-handle -- produced numbers that
// looked plausible. A token comparison is the only check that catches the last one.
//
// WHY THE PROMPT IS A FIXTURE. The chain this test compares is only as good an oracle as the prompt
// that produces it. The 6-token prompt this test used first, {9707, 11, 1879, 13, 5765, 374}, makes
// the reference lane repeat its own tail forever -- `11 1879 13 5765 374` for the whole 32-token
// budget, a period-5 cycle over five distinct tokens -- so any implementation that reproduces the
// loop passes, including wrong ones. The prompt and both goldens therefore live in
// tests/fixtures/tp2_accept/ as TOKEN IDS: real, in-distribution ids from the committed corpus
// slice, so neither arm tokenizes anything and the two hosts cannot disagree about the prompt.
//
// THE GOLDEN IS THE REFERENCE SEMANTICS. tests/fixtures/tp2_accept/golden_plain.ids is the width-1
// greedy chain captured on the reference host (trueno, RTX 5090), and golden_mtp.ids is the same
// for the MTP lane (NINFER_TP2_SPEC=mtp). An arm run must reproduce the golden for its lane; the
// two-arm comparison additionally requires width 2 to equal width 1. Regenerate with
// NINFER_TP2_CAPTURE=1 and write the printed chain into the fixture.
//
// Opt-in by design: while the refusal stands (a width-2 round has never been validated), running
// this test would fail for a known and intended reason, so it needs NINFER_TP2_ACCEPT=1. With the
// flag set, a refusal is a failure -- that is the point of the flag.
//
// Needs two GPUs and a 27B artifact; skips otherwise.

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

// The exact round both arms run, so a manual cross-machine run can hold the same contract.
// Greedy is stated rather than relied on: temperature 0 with top_k 1 makes the choice
// deterministic on any host, and a pinned output budget makes the comparison bounded and
// independent of where either round stops.
constexpr std::uint32_t kOutputTokens = 32;

// ---- fixtures ----------------------------------------------------------------------------------

std::string fixture_path(const char* name) {
    return std::string(NINFER_SOURCE_DIR) + "/tests/fixtures/tp2_accept/" + name;
}

// Whitespace-separated token ids, the same format for the prompt and for a golden chain.
std::vector<ninfer::TokenId> read_ids(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot read fixture " + path +
                                 "; run with NINFER_TP2_CAPTURE=1 to record it");
    }
    std::vector<ninfer::TokenId> ids;
    long long id = 0;
    while (file >> id) {
        if (id < 0 || id > 0x7fffffffLL) {
            throw std::runtime_error("fixture token id out of range in " + path);
        }
        ids.push_back(static_cast<ninfer::TokenId>(id));
    }
    if (ids.empty()) { throw std::runtime_error("fixture is empty: " + path); }
    return ids;
}

std::string chain_text(const std::vector<ninfer::TokenId>& tokens) {
    std::string text;
    for (const auto token : tokens) {
        if (!text.empty()) { text += ' '; }
        text += std::to_string(token);
    }
    return text;
}

// The chain a lane must reproduce, named after the speculative backend so the MTP lane has its own.
const char* golden_name(bool mtp) { return mtp ? "golden_mtp.ids" : "golden_plain.ids"; }

std::vector<ninfer::TokenId> run_round(const char* artifact, int width, int device,
                                       const std::vector<ninfer::TokenId>& prompt_tokens,
                                       bool mtp) {
    ninfer::EngineOptions options;
    options.artifact_path  = artifact;
    options.use_cuda_graph = false; // width > 1 refuses capture (caf91488)
    // The round this test validates is the plain text lane. Prefix reuse/context cache and every
    // speculative backend are refused at width > 1 by the option gate (2529e77c) because no
    // two-rank round has been compared for them, so the test states the configuration it exercises
    // instead of inheriting defaults that the gate would refuse. Each of those combinations gets
    // its own row in that table, and its own comparison, when it is validated.
    options.context_cache.enabled = false;
    if (mtp) {
        options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
    } else {
        options.speculative.backend = ninfer::SpeculativeBackend::None;
    }
    options.max_context           = 2048;
    // The default capacity is 8192 tokens, outside the usable range for a 2048 context; without this
    // the engine refuses at option validation before either round runs.
    options.kv_capacity    = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    if (width > 1) {
        options.devices = {0, 1};
    } else {
        options.device = device;
    }
    ninfer::Engine engine(options);
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = kOutputTokens;
    request.execution.sampling.temperature     = 0.0F;
    request.execution.sampling.top_k           = 1;
    auto prompt = engine.prepare_tokens(prompt_tokens);
    return engine.generate(std::move(prompt), request).generated_token_ids;
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
    const bool have_devices     = count_err == cudaSuccess && devices >= 1;

    // Single-arm mode, for the lab this test cannot run in: nowhere has one host where the 27B fits
    // at BOTH widths (hephaestus has 2x16 GB and width 1 needs 20.4 GB; trueno has one 32 GB 5090
    // where width 2 cannot run; twofour has 121 GiB unified but one device). `NINFER_TP2_ARM=1` runs
    // only the reference and prints its tokens; `=2` runs only the sharded arm. Each arm is checked
    // against the recorded golden here, which is what makes the cross-machine comparison meaningful:
    // both hosts compare against the same chain instead of against each other's output.
    //
    // THE DEVICE REQUIREMENT FOLLOWS THE ARM. The reference arm runs on ONE device by definition, so
    // requiring two would skip it on exactly the host the arm mode exists for; the sharded arm and
    // the two-arm comparison both need two.
    const char* arm_env = std::getenv("NINFER_TP2_ARM");
    const int arm_width = arm_env != nullptr ? std::atoi(arm_env) : 0;
    const int needed    = arm_env != nullptr && arm_width <= 1 ? 1 : 2;
    if (!have_artifact || !have_devices || devices < needed) {
        std::cerr << "SKIP: this arm needs " << needed << " CUDA device"
                  << (needed == 1 ? "" : "s") << " and a 27B artifact (devices: "
                  << (count_err == cudaSuccess ? std::to_string(devices)
                                               : std::string("unavailable"))
                  << ", NINFER_TEST_ARTIFACT: " << (have_artifact ? "set" : "not set") << ")\n";
        return 77;
    }

    // NINFER_TP2_SPEC selects a speculative backend so one binary can run each option-gate
    // comparison. Unset means the plain lane, which is the configuration this test validated first.
    const char* spec = std::getenv("NINFER_TP2_SPEC");
    const bool mtp   = spec != nullptr && std::string_view(spec) == "mtp";
    const bool capture = std::getenv("NINFER_TP2_CAPTURE") != nullptr;

    // The prompt is read up front; a golden is read only AFTER the round has produced tokens, so a
    // lane that cannot run at all reports that instead of being masked by a missing fixture.
    std::vector<ninfer::TokenId> prompt;
    try {
        prompt = read_ids(fixture_path("prompt.ids"));
    } catch (const std::exception& error) { return fail(error.what()); }

    if (const char* arm = std::getenv("NINFER_TP2_ARM")) {
        const int width = std::atoi(arm);
        std::vector<ninfer::TokenId> tokens;
        try {
            tokens = run_round(artifact, width, 0, prompt, mtp);
        } catch (const std::exception& error) {
            return fail(std::string("width ") + arm + " round threw: " + error.what());
        }
        std::cout << "arm " << width << " tokens (" << tokens.size() << "): "
                  << chain_text(tokens) << '\n';
        if (capture) {
            std::cout << "capture: write the chain above into " << golden_name(mtp) << '\n';
            return 0;
        }
        std::vector<ninfer::TokenId> golden;
        try {
            golden = read_ids(fixture_path(golden_name(mtp)));
        } catch (const std::exception& error) { return fail(error.what()); }
        if (tokens != golden) {
            return fail(std::string("arm ") + arm + " does not reproduce the golden\n  golden: " +
                        chain_text(golden) + "\n  actual: " + chain_text(tokens) +
                        "\n  (a mismatch on a host other than the reference is a result to report, "
                        "not necessarily a defect; regenerate with NINFER_TP2_CAPTURE=1 only after "
                        "the reference lane itself changed)");
        }
        std::cout << "ok: arm " << width << " reproduces the golden (" << golden.size()
                  << " tokens)\n";
        return 0;
    }

    std::vector<ninfer::TokenId> single;
    try {
        single = run_round(artifact, 1, 0, prompt, mtp);
    } catch (const std::exception& error) {
        return fail(std::string("the single-device reference round threw: ") + error.what());
    }
    if (single.empty()) { return fail("the single-device reference round produced no tokens"); }

    std::vector<ninfer::TokenId> sharded;
    try {
        sharded = run_round(artifact, 2, 0, prompt, mtp);
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
    if (!capture) {
        std::vector<ninfer::TokenId> golden;
        try {
            golden = read_ids(fixture_path(golden_name(mtp)));
        } catch (const std::exception& error) { return fail(error.what()); }
        if (single != golden) {
            return fail("the reference lane does not reproduce the golden\n  golden: " +
                        chain_text(golden) + "\n  actual: " + chain_text(single));
        }
    }
    std::cout << "ok: " << single.size() << " tokens identical at width 1 and width 2"
              << (capture ? " (captured, golden not compared)" : ", and equal to the golden") << '\n';
    return 0;
}
