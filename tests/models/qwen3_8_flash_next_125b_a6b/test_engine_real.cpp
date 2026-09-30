#include "artifact/reader.h"

#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                    = artifact;
    options.max_context                      = 512;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(1024);
    options.prefill_chunk                    = 256;
    options.speculative.backend              = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens         = 3;
    options.speculative.proposal_head        = ninfer::ProposalHead::Full;
    options.enable_vision                    = true;
    options.use_cuda_graph                   = true;
    // Maintainer override: the built-in logit/state capture hooks reject graph capture.
    if (const char* no_graphs = std::getenv("NINFER_TEST_NO_GRAPHS");
        no_graphs != nullptr && *no_graphs != '\0') {
        options.use_cuda_graph = false;
    }
    options.max_concurrency                  = 2;
    options.max_pending_requests             = 2;
    options.context_cache.device_state_slots = 4;
    options.context_cache.max_private_continuations = 2;
    options.context_cache.max_shared_prefixes       = 1;
    return options;
}

const std::vector<ninfer::TokenId>& canonical_prompt() {
    static const std::vector<ninfer::TokenId> prompt{
        248045, 846, 198,  814, 20139,  303, 2250,   2716,  22157, 3069,   279, 12515,  7701, 6105,
        2261,   279, 1834, 13,  248046, 198, 248045, 74455, 198,   248068, 271, 248069, 271};
    return prompt;
}

// Greedy output for the canonical non-thinking chat template. The BF16-dense and FP8-dense
// (fp8_projections, fp8_mtp) artifacts and the MTP verify and plain decode paths currently share
// one greedy prefix; each path and recipe is still checked against it exactly. Exactness pins
// regressions, not quality, which the step 7 perplexity gate measures. A near-tie greedy choice
// is route-sensitive: until the within-budget QSA route, the fp8 MTP verify path met an exact
// BF16 logit tie at generated index 2 (5435 and 27891 both at logprob -1.06157) and recorded its
// own golden through the tie-break; the dense QSA route's reduction order resolves it to the
// plain-decode choice by 0.375 nats with perplexity unchanged (3.998162 against 3.998118, floor
// 1.3e-4). A route change that moves such a choice re-records the golden, validated by the gate;
// if a recipe or path diverges again, it gets its own golden here.

bool fp8_recipe(const std::string& recipe) {
    return recipe == "qwen3_8_flash_next_125b_a6b_nvfp4_fp8_projections-v3" ||
           recipe == "qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp-v3";
}

// Cross-path boundaries. The reuse checks compare a continuation served from captured
// state against the same continuation computed cold. The two are distinct evaluations
// for every recipe: the captured recurrent state comes from the first request's decode
// and MTP verify kernels, the cold state from prefill kernels, and they differ by
// rounding. Exact output equality therefore holds only where the fixture has no near-tie
// greedy choice after the boundary, and the checks stay exact because an exact match is
// what detects captured-state or lane corruption. When a route change moves a near-tie
// onto a boundary (the T=2..4 FP8 Tensor Core re-route did: "wavelength" vs "blue
// wavelengths"), the recipe's boundary is moved, as a golden is re-recorded; it is never
// relaxed to a length check.
struct CrossPathFixture {
    ninfer::TokenId separator;         // appended after the first output before reuse
    std::ptrdiff_t resumed_prefix;     // golden tokens a concurrent resumed request carries
};

// One boundary serves every recipe until a recipe's fixture needs its own; the failure
// output prints both paths' tokens to choose it from. The T=2..4 Tensor Core re-route
// landed a near-tie greedy choice on the FP8 recipe at the default separator (198, after
// the golden's "...but shorter"): cold [11855 89661] ("blue wavelengths") vs reused
// [86 33705] ("wavelength") — a ULP-scale boundary-state perturbation flipping a
// borderline argmax. The FP8 boundary moves to 11855 ("blue"), the model's own
// continuation at the old boundary, where the next choice is confident; the checks stay
// exact.
CrossPathFixture cross_path_fixture(const std::string& recipe) {
    if (fp8_recipe(recipe)) { return {11855, 4}; }
    return {198, 4};
}

void print_tokens(const char* label, const std::vector<ninfer::TokenId>& tokens) {
    std::cerr << label << ':';
    for (const auto token : tokens) { std::cerr << ' ' << token; }
    std::cerr << '\n';
}
const std::vector<ninfer::TokenId>& canonical_output() {
    static const std::vector<ninfer::TokenId> golden{
        29108, 4009, 27891, 8964, 579, 16078, 321, 1100, 9872, 303, 660, 17425};
    return golden;
}

std::string artifact_recipe(const char* artifact) {
    const ninfer::artifact::Reader reader(artifact);
    const auto& provenance = reader.directory().provenance;
    if (provenance.contains("recipe") && provenance.at("recipe").is_string()) {
        return provenance.at("recipe").get<std::string>();
    }
    return {};
}

ninfer::RequestOptions greedy_options(std::uint32_t outputs, bool reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

std::vector<std::uint8_t> gradient_ppm() {
    std::vector<std::uint8_t> ppm;
    const std::string header = "P6\n64 64\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int index = 0; index < 64 * 64; ++index) {
        ppm.push_back(static_cast<std::uint8_t>(index & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 3) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7) & 0xff));
    }
    return ppm;
}

int exercise_mtp_and_prefix(ninfer::Engine& engine,
                            const std::vector<ninfer::TokenId>& expected_prefix,
                            const CrossPathFixture& fixture) {
    // Per-recipe greedy golden for the canonical non-thinking chat template (see
    // canonical_output). Checking semantic text would require duplicating the tokenizer in
    // this C++ integration test, so protect the exact token prefix instead. This catches
    // numerically plausible but language-corrupt model execution, including routed-MoE
    // row-layout regressions.
    const auto& prompt                   = canonical_prompt();
    const ninfer::GenerationResult first = engine.generate(
        engine.prepare_tokens(prompt), greedy_options(expected_prefix.size(), true));
    if (first.generated_token_ids != expected_prefix ||
        first.speculative.backend != ninfer::SpeculativeBackend::Mtp ||
        first.speculative.rounds == 0) {
        std::cerr << "Flash-Next greedy text prefix is corrupt or did not complete through MTP\n";
        print_tokens("mtp", first.generated_token_ids);
        print_tokens("expected", expected_prefix);
        std::cerr << "(backend mtp=" << (first.speculative.backend == ninfer::SpeculativeBackend::Mtp)
                  << ", rounds " << first.speculative.rounds << ")\n";
        return 1;
    }

    std::vector<ninfer::TokenId> continuation = prompt;
    continuation.insert(continuation.end(), first.generated_token_ids.begin(),
                        first.generated_token_ids.end());
    continuation.push_back(fixture.separator);
    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare_tokens(continuation), greedy_options(2, true));
    const ninfer::GenerationResult cold =
        engine.generate(engine.prepare_tokens(continuation), greedy_options(2, false));
    const std::uint32_t expected_reuse =
        static_cast<std::uint32_t>(prompt.size() + first.generated_token_ids.size() - 1);
    if (reused.reused_prompt_tokens != expected_reuse || reused.generated_token_ids.size() != 2 ||
        cold.reused_prompt_tokens != 0 || cold.generated_token_ids != reused.generated_token_ids) {
        std::cerr << "Flash-Next prefix reuse is incorrect: reused=" << reused.reused_prompt_tokens
                  << " expected=" << expected_reuse << '\n';
        print_tokens("reused", reused.generated_token_ids);
        print_tokens("cold", cold.generated_token_ids);
        return 1;
    }

    if (first.generated_token_ids[0] == first.generated_token_ids[1]) {
        std::cerr << "Flash-Next partial-terminal fixture repeats its first token\n";
        return 1;
    }
    ninfer::RequestOptions stop_options = greedy_options(6, true);
    stop_options.stop.token_ids.push_back(first.generated_token_ids[1]);
    const ninfer::GenerationResult stopped =
        engine.generate(engine.prepare_tokens(prompt), stop_options);
    if (stopped.finish_reason != ninfer::FinishReason::StopToken ||
        stopped.generated_token_ids.size() != 2 ||
        stopped.generated_token_ids[0] != first.generated_token_ids[0] ||
        stopped.generated_token_ids[1] != first.generated_token_ids[1]) {
        std::cerr << "Flash-Next custom stop did not terminate inside the MTP round\n";
        return 1;
    }

    std::vector<ninfer::TokenId> stopped_continuation = prompt;
    stopped_continuation.insert(stopped_continuation.end(), stopped.generated_token_ids.begin(),
                                stopped.generated_token_ids.end());
    stopped_continuation.push_back(fixture.separator);
    const ninfer::GenerationResult stopped_reuse =
        engine.generate(engine.prepare_tokens(stopped_continuation), greedy_options(1, true));
    const ninfer::GenerationResult stopped_cold =
        engine.generate(engine.prepare_tokens(stopped_continuation), greedy_options(1, false));
    const std::uint32_t expected_stopped_reuse =
        static_cast<std::uint32_t>(prompt.size() + stopped.generated_token_ids.size() - 1);
    if (stopped_reuse.reused_prompt_tokens != expected_stopped_reuse ||
        stopped_cold.reused_prompt_tokens != 0 ||
        stopped_reuse.generated_token_ids.size() != 1 ||
        stopped_cold.generated_token_ids != stopped_reuse.generated_token_ids) {
        std::cerr << "Flash-Next partial MTP terminal reused " << stopped_reuse.reused_prompt_tokens
                  << ", expected " << expected_stopped_reuse << '\n';
        print_tokens("reused", stopped_reuse.generated_token_ids);
        print_tokens("cold", stopped_cold.generated_token_ids);
        return 1;
    }
    return 0;
}

int exercise_ordinary_greedy(const char* artifact,
                             const std::vector<ninfer::TokenId>& expected_prefix) {
    ninfer::EngineOptions options    = engine_options(artifact);
    options.speculative.backend      = ninfer::SpeculativeBackend::None;
    options.speculative.draft_tokens = 0;
    options.enable_vision = false;
    ninfer::Engine engine(std::move(options));
    const ninfer::GenerationResult result =
        engine.generate(engine.prepare_tokens(canonical_prompt()),
                        greedy_options(expected_prefix.size(), false));
    if (result.speculative.backend != ninfer::SpeculativeBackend::None ||
        result.generated_token_ids != expected_prefix) {
        std::cerr << "Flash-Next ordinary greedy output disagrees with the artifact's "
                     "ordinary-path golden\n";
        print_tokens("ordinary", result.generated_token_ids);
        print_tokens("expected", expected_prefix);
        return 1;
    }
    return 0;
}

int exercise_concurrent_state(ninfer::Engine& engine,
                              const std::vector<ninfer::TokenId>& expected_prefix,
                              const CrossPathFixture& fixture) {
    // Different frontiers exercise local prefill rows and shared decode rows. Repeat in
    // reversed admission order to reuse both physical lanes and recurrent state slots. Each
    // order must decode both rows together: in the reversed order the resumed request forks a
    // cached StateImage, and the root queued behind it is admitted once that fork settles, not
    // after the resumed request completes.
    auto continuation = canonical_prompt();
    continuation.insert(continuation.end(), expected_prefix.begin(),
                        expected_prefix.begin() + fixture.resumed_prefix);
    const auto cold =
        engine.generate(engine.prepare_tokens(continuation), greedy_options(8, false));
    for (bool reverse : {false, true}) {
        const auto before = engine.runtime_stats();
        auto first =
            engine.submit(engine.prepare_tokens(reverse ? continuation : canonical_prompt()),
                          greedy_options(reverse ? 8 : 12, false));
        auto second =
            engine.submit(engine.prepare_tokens(reverse ? canonical_prompt() : continuation),
                          greedy_options(reverse ? 12 : 8, false));
        const auto a        = first.wait();
        const auto b        = second.wait();
        const auto& root    = reverse ? b : a;
        const auto& resumed = reverse ? a : b;
        if (root.generated_token_ids != expected_prefix ||
            resumed.generated_token_ids != cold.generated_token_ids) {
            std::cerr << "Flash-Next concurrent lane reuse changed the canonical fixture"
                      << (reverse ? " (reversed admission)" : "") << '\n';
            print_tokens("root", root.generated_token_ids);
            print_tokens("resumed", resumed.generated_token_ids);
            print_tokens("cold", cold.generated_token_ids);
            return 1;
        }
        const auto after = engine.runtime_stats();
        if (after.decode_row_rounds - before.decode_row_rounds <=
            after.decode_rounds - before.decode_rounds) {
            std::cerr << "Flash-Next concurrent fixture never executed a two-row decode"
                      << (reverse ? " (reversed admission)" : "") << '\n';
            return 1;
        }
    }
    return 0;
}

int exercise_vision(ninfer::Engine& engine) {
    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = gradient_ppm();
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "inline.ppm";

    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(std::move(image));
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "What is visible?", .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;

    const ninfer::GenerationResult result =
        engine.generate(engine.prepare(std::move(input)), greedy_options(1, false));
    if (!result.prompt.has_media || result.generated_token_ids.size() != 1 ||
        result.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "Flash-Next Vision did not complete through the public Engine\n";
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN38_FLASH_NEXT_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') { return 77; }
    const std::string recipe      = artifact_recipe(artifact);
    const auto& expected_prefix   = canonical_output();
    const CrossPathFixture fixture = cross_path_fixture(recipe);
    try {
        for (const auto head : {ninfer::ProposalHead::Full, ninfer::ProposalHead::Optimized}) {
            auto options = engine_options(artifact);
            options.speculative.proposal_head = head;
            ninfer::Engine engine(options);
            const ninfer::LoadSummary load = engine.load_summary();
            if (load.architecture != "Qwen3_8FlashNextForCausalLM" ||
                load.host_to_device_bytes == 0) {
                std::cerr << "Flash-Next Engine construction has an invalid load summary\n";
                return 1;
            }
            if (exercise_mtp_and_prefix(engine, expected_prefix, fixture) != 0) { return 1; }
            if (exercise_concurrent_state(engine, expected_prefix, fixture) != 0) { return 1; }
            if (exercise_vision(engine) != 0) { return 1; }
        }
        if (exercise_ordinary_greedy(artifact, expected_prefix) != 0) { return 1; }
        std::cout << "OK Qwen3.8 Flash Next real Engine\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Qwen3.8 Flash Next real Engine: " << error.what() << '\n';
        return 1;
    }
}
