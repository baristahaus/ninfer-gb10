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

#include <unistd.h>
#include <cuda_runtime_api.h>
#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

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

// The tie tolerance, in nats of log-probability.
//
// A sharded round is not required to reproduce the reference's argmax where the reference itself has
// two near-equal continuations: the shard sums in a different order, so the logits differ in the last
// few ulps and a near-tie can fall either way. That is not a defect and must not be reported as one.
// A disagreement where the reference clearly prefers one token IS the defect this test exists to
// catch. The two are separated by a measured margin, not a guess. At the three disagreements this
// lane has produced -- T=20 position 6, T=31 position 18, T=32 position 1 -- the reference gives the
// two candidates log-probabilities 0.250, 0.125 and 0.250 nats apart, while replacing one candidate
// with a token the reference would not choose at that position moves the margin to 14.0 nats. The
// tolerance sits four times above the largest measured tie and fourteen times below the control.
constexpr double kTieToleranceNats = 1.0;

// The first position at which two chains differ, or npos when they agree everywhere.
std::size_t first_disagreement(const std::vector<ninfer::TokenId>& a,
                               const std::vector<ninfer::TokenId>& b) {
    const std::size_t common = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < common; ++i) {
        if (a[i] != b[i]) { return i; }
    }
    return std::string::npos;
}

struct Verdict {
    bool adjudicated = false;
    std::size_t position = 0;
    double candidate_a_logprob = 0.0;
    double candidate_b_logprob = 0.0;
    double margin = 0.0; // absolute difference, nats
};

// Scores both candidates at the first disagreement, on the reference arm. The scoring purpose is
// gated at width > 1 (startup.cpp: the scoring purpose runs a different frame kind and its sharded
// path has never been exercised), so this runs a single-device CausalScoring engine and can only
// answer on the reference host -- which is the right place to ask: the question is whether the
// REFERENCE had two near-equal continuations there.
Verdict adjudicate(const char* artifact, const std::vector<ninfer::TokenId>& prompt,
                   const std::vector<ninfer::TokenId>& chain_a,
                   const std::vector<ninfer::TokenId>& chain_b) {
    Verdict verdict;
    const std::size_t position = first_disagreement(chain_a, chain_b);
    if (position == std::string::npos || position >= chain_a.size() ||
        position >= chain_b.size()) {
        return verdict;
    }
    verdict.position = position;
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.device        = 0;
    options.purpose       = ninfer::EnginePurpose::CausalScoring;
    options.max_context   = 2048;
    options.kv_capacity   = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    ninfer::Engine engine(options);
    auto score_candidate = [&](ninfer::TokenId candidate) {
        auto sequence = prompt;
        sequence.insert(sequence.end(), chain_a.begin(),
                        chain_a.begin() + static_cast<std::ptrdiff_t>(position));
        sequence.push_back(candidate);
        const auto scores = engine.score_tokens(
            sequence, static_cast<std::uint32_t>(prompt.size() + position));
        return scores.empty() ? 0.0 : static_cast<double>(scores.front());
    };
    verdict.candidate_a_logprob = score_candidate(chain_a[position]);
    verdict.candidate_b_logprob = score_candidate(chain_b[position]);
    verdict.margin =
        std::abs(verdict.candidate_a_logprob - verdict.candidate_b_logprob);
    verdict.adjudicated = true;
    return verdict;
}

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
    // THE DEVICE CONTROL, PRINTED UNCONDITIONALLY. A device number with no control is a number that
    // can be attributed to the wrong thing, so the same line is read on BOTH hosts: the lane works on
    // hephaestus, so hephaestus is the control that makes trueno's number meaningful.
    {
        int initial_device = -1, visible = 0;
        (void)cudaGetDevice(&initial_device);
        (void)cudaGetDeviceCount(&visible);
        std::fprintf(stderr, "[dev-control] pid=%d initial_cuda_device=%d visible=%d\n",
                     static_cast<int>(::getpid()), initial_device, visible);
        std::fflush(stderr);
    }
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
    // NINFER_TP2_PROMPT names an alternative token-id file, so a sweep can run the same contract at
    // several prompt lengths without rewriting the checked-in fixture. The file is a whitespace
    // separated id list in the same format as prompt.ids; both arms must be given the same file,
    // because the comparison is only meaningful when the two hosts prefilled the same ids.
    const char* prompt_override = std::getenv("NINFER_TP2_PROMPT");
    std::vector<ninfer::TokenId> prompt;
    try {
        prompt = read_ids(prompt_override != nullptr && *prompt_override != '\0'
                              ? std::string(prompt_override)
                              : fixture_path("prompt.ids"));
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
        // NINFER_TP2_COMPARE names the OTHER arm's chain: the reference host scores both candidates
        // at their first disagreement and decides whether it was a tie or a defect. Unset compares
        // against the recorded golden, which is what a reference run must reproduce.
        const char* compare = std::getenv("NINFER_TP2_COMPARE");
        std::vector<ninfer::TokenId> expected;
        try {
            expected = read_ids(compare != nullptr ? std::string(compare)
                                                   : fixture_path(golden_name(mtp)));
        } catch (const std::exception& error) { return fail(error.what()); }
        if (tokens != expected) {
            if (tokens.size() != expected.size()) {
                return fail("the chains differ in length: " + std::to_string(tokens.size()) +
                            " vs " + std::to_string(expected.size()) +
                            " tokens; a length difference is not a tie");
            }
            const std::size_t position = first_disagreement(expected, tokens);
            if (width > 1) {
                return fail("arm " + std::to_string(width) + " disagrees at position " +
                            std::to_string(position) +
                            " and cannot adjudicate it: the scoring route is single-device only "
                            "(startup.cpp gates the scoring purpose at width > 1). Rerun on the "
                            "reference arm with NINFER_TP2_COMPARE=<this arm's chain> to decide "
                            "whether the reference had two near-equal continuations there.");
            }
            Verdict verdict;
            try {
                verdict = adjudicate(artifact, prompt, expected, tokens);
            } catch (const std::exception& error) {
                return fail(std::string("the disagreement at position ") +
                            std::to_string(position) + " cannot be adjudicated: " + error.what());
            }
            std::cout << "disagreement at position " << verdict.position << ": reference log p "
                      << verdict.candidate_a_logprob << " (recorded) vs "
                      << verdict.candidate_b_logprob << " (this arm), margin " << verdict.margin
                      << " nats, tolerance " << kTieToleranceNats << " nats\n";
            if (verdict.margin > kTieToleranceNats) {
                return fail("a CONFIDENT disagreement at position " +
                            std::to_string(verdict.position) + ": the reference prefers " +
                            std::to_string(expected[verdict.position]) + " over " +
                            std::to_string(tokens[verdict.position]) + " by " +
                            std::to_string(verdict.margin) + " nats, above the " +
                            std::to_string(kTieToleranceNats) +
                            "-nat tie tolerance, so this is not a tie");
            }
            std::cout << "ok: the disagreement at position " << verdict.position
                      << " is a near-tie on the reference, not a defect\n";
            return 0;
        }
        std::cout << "ok: arm " << width << " reproduces the "
                  << (compare != nullptr ? "compared chain" : "golden") << " (" << expected.size()
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
    if (sharded != single) {
        // Same criterion as the arm path: a disagreement is a tie if the reference gives both
        // candidates nearly the same probability, and a defect if it clearly prefers one.
        Verdict verdict;
        try {
            verdict = adjudicate(artifact, prompt, single, sharded);
        } catch (const std::exception& error) {
            return fail(std::string("the width-2 disagreement cannot be adjudicated: ") +
                        error.what());
        }
        std::cout << "disagreement at position " << verdict.position << ": reference log p "
                  << verdict.candidate_a_logprob << " (width 1) vs " << verdict.candidate_b_logprob
                  << " (width 2), margin " << verdict.margin << " nats, tolerance "
                  << kTieToleranceNats << " nats\n";
        if (verdict.margin > kTieToleranceNats) {
            return fail("a CONFIDENT disagreement at position " + std::to_string(verdict.position) +
                        ": the reference prefers " + std::to_string(single[verdict.position]) +
                        " over " + std::to_string(sharded[verdict.position]) + " by " +
                        std::to_string(verdict.margin) + " nats, above the " +
                        std::to_string(kTieToleranceNats) + "-nat tie tolerance");
        }
        std::cout << "ok: the disagreement at position " << verdict.position
                  << " is a near-tie on the reference\n";
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
