// The speculative-invariant contract (plan W1.3): greedy decoding must produce the
// identical token sequence with and without the speculative schedules, and the run
// must demonstrably exercise rejections (accepted drafts strictly below drafted).
// Exercises both MTP and DFlash2 over the loaded artifact, single stream and batched,
// so the decomposed record path and the fold/commit machinery are covered at the
// observable committed-output level. Optional artifact via NINFER_TEST_ARTIFACT; the
// test skips when it is absent so ordinary suites stay GPU-free.

#include "ninfer/engine.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

struct EngineRun {
    std::vector<ninfer::TokenId> tokens;
    std::uint64_t drafted  = 0;
    std::uint64_t accepted = 0;
    std::uint64_t rounds   = 0;
};

EngineRun greedy_run(const char* artifact, const std::string& prompt_text, bool speculative,
                     ninfer::SpeculativeBackend backend, unsigned draft_tokens,
                     unsigned outputs) {
    ninfer::EngineOptions options;
    options.artifact_path   = artifact;
    options.max_context     = 2304;
    options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(2304 * 2);
    options.prefill_chunk   = 2304;
    options.max_concurrency = 2;
    options.use_cuda_graph  = true;
    if (speculative) {
        options.speculative.backend      = backend;
        options.speculative.draft_tokens = draft_tokens;
        options.speculative.proposal_head = ninfer::ProposalHead::Full;
    }
    ninfer::Engine engine(options);

    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = outputs;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.sampling.seed           = 0;
    request.execution.allow_prefix_reuse      = false;
    request.stop.include_model_defaults       = false;

    const auto tokens = engine.tokenize_text(prompt_text);
    const auto result = engine.generate(engine.prepare_tokens(tokens), request);
    require(result.generated_token_ids.size() == outputs,
            "the invariant run did not honor the output budget");
    require(result.finish_reason == ninfer::FinishReason::OutputLimit,
            "the invariant run finished early");

    EngineRun run;
    run.tokens   = result.generated_token_ids;
    run.drafted  = result.stats.speculative.drafted_tokens;
    run.accepted = result.stats.speculative.accepted_tokens;
    run.rounds   = result.stats.speculative.rounds;
    if (speculative) {
        require(run.rounds != 0, "the speculative run recorded no verification rounds");
        require(run.accepted != 0, "the speculative run accepted no draft token");
        require(run.accepted < run.drafted,
                "the speculative run showed no rejection; the invariant is untested");
    }
    return run;
}

void compare(const EngineRun& reference, const EngineRun& speculative, const char* label) {
    if (reference.tokens != speculative.tokens) {
        throw std::runtime_error(std::string(label) +
                                 ": greedy tokens diverged from the non-speculative run");
    }
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        const std::string prompt =
            "Complete this sentence with exactly one short factual clause: "
            "The primary gas in Earth's atmosphere is";

        // The non-speculative reference for every comparison below.
        const auto reference = greedy_run(artifact, prompt, false, ninfer::SpeculativeBackend::Mtp,
                                          0, 32);

        // MTP: single stream. The draft and verify rounds must reproduce the reference
        // exactly, with at least one rejection proving the fold/commit path ran.
        const auto mtp = greedy_run(artifact, prompt, true, ninfer::SpeculativeBackend::Mtp, 3, 32);
        compare(reference, mtp, "MTP single stream");

        // DFlash2: the wide-window record path (the decomposed EXL3 route at T=8).
        const auto dflash2 = greedy_run(artifact, prompt, true,
                                        ninfer::SpeculativeBackend::DFlash2, 7, 32);
        compare(reference, dflash2, "DFlash2 single stream");

        // Batched: two concurrent rows through the compact record round (B=2). Each
        // row's committed tokens must match the reference — the record round must not
        // let one row's draft contaminate another's state.
        {
            ninfer::EngineOptions options;
            options.artifact_path   = artifact;
            options.max_context     = 2304;
            options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(2304 * 2);
            options.prefill_chunk   = 2304;
            options.max_concurrency = 2;
            options.use_cuda_graph  = true;
            options.speculative.backend       = ninfer::SpeculativeBackend::DFlash2;
            options.speculative.draft_tokens  = 7;
            options.speculative.proposal_head = ninfer::ProposalHead::Full;
            ninfer::Engine engine(options);

            ninfer::RequestOptions request;
            request.execution.requested_output_tokens = 32;
            request.execution.sampling.temperature    = 0.0F;
            request.execution.sampling.seed           = 0;
            request.stop.include_model_defaults       = false;

            const auto tokens = engine.tokenize_text(prompt);
            std::vector<ninfer::GenerationHandle> handles;
            for (unsigned row = 0; row < 2; ++row) {
                handles.push_back(engine.submit(engine.prepare_tokens(tokens), request));
            }
            for (auto& handle : handles) {
                const auto result = handle.wait();
                require(result.generated_token_ids.size() == 32,
                        "the batched invariant run did not honor the budget");
                require(result.generated_token_ids == reference.tokens,
                        "batched speculative tokens diverged from the reference");
                require(result.stats.speculative.accepted_tokens <
                            result.stats.speculative.drafted_tokens,
                        "the batched run showed no rejection");
            }
        }
        std::cout << "ok: speculative invariant holds for MTP and DFlash2\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fail: " << error.what() << "\n";
        return 1;
    }
}
