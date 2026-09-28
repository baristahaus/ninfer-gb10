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

// Greedy outputs for the canonical non-thinking chat template, keyed on the artifact's
// declared dense-precision recipe (provenance.recipe) and the execution path. A
// different-precision artifact has its own greedy prefix by construction; at low
// precision (FP8 dense) the MTP verify path and the plain decode path are distinct
// numerical evaluations of the same model, so each is pinned exactly (on the BF16
// dense artifact the two paths agree, one golden). A new dense-precision recipe must
// add its goldens here.
const std::vector<ninfer::TokenId>& canonical_output(const std::string& recipe,
                                                     bool mtp_path) {
    static const std::vector<ninfer::TokenId> nvfp4{  // BF16 dense: the source checkpoint.
        29108, 4009, 27891, 8964, 579, 16078, 321, 1100, 9872, 303, 660, 17425};
    static const std::vector<ninfer::TokenId> fp8_mtp{  // fp8_row_maxabs dense, MTP path.
        29108, 4009, 5435, 660, 7736, 314, 279, 9155, 19142, 11, 864, 43000};
    static const std::vector<ninfer::TokenId> fp8_ordinary{  // same artifact, plain decode.
        29108, 4009, 5435, 660, 7736, 11, 694, 22602, 6105, 89661, 43000, 777};
    if (recipe == "qwen3_8_flash_next_125b_a6b_nvfp4_fp8_dense-v3") {
        return mtp_path ? fp8_mtp : fp8_ordinary;
    }
    return nvfp4;
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
                            const std::vector<ninfer::TokenId>& expected_prefix) {
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
        return 1;
    }

    std::vector<ninfer::TokenId> continuation = prompt;
    continuation.insert(continuation.end(), first.generated_token_ids.begin(),
                        first.generated_token_ids.end());
    continuation.push_back(198);
    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare_tokens(continuation), greedy_options(2, true));
    const ninfer::GenerationResult cold =
        engine.generate(engine.prepare_tokens(continuation), greedy_options(2, false));
    const std::uint32_t expected_reuse =
        static_cast<std::uint32_t>(prompt.size() + first.generated_token_ids.size() - 1);
    if (reused.reused_prompt_tokens != expected_reuse || reused.generated_token_ids.size() != 2 ||
        cold.reused_prompt_tokens != 0 || cold.generated_token_ids != reused.generated_token_ids) {
        std::cerr << "Flash-Next prefix reuse is incorrect: reused=" << reused.reused_prompt_tokens
                  << " expected=" << expected_reuse << " cold/reused output match="
                  << (cold.generated_token_ids == reused.generated_token_ids) << '\n';
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
    stopped_continuation.push_back(198);
    const ninfer::GenerationResult stopped_reuse =
        engine.generate(engine.prepare_tokens(stopped_continuation), greedy_options(1, true));
    const ninfer::GenerationResult stopped_cold =
        engine.generate(engine.prepare_tokens(stopped_continuation), greedy_options(1, false));
    const std::uint32_t expected_stopped_reuse =
        static_cast<std::uint32_t>(prompt.size() + stopped.generated_token_ids.size() - 1);
    if (stopped_reuse.reused_prompt_tokens != expected_stopped_reuse ||
        stopped_cold.reused_prompt_tokens != 0 ||
        stopped_cold.generated_token_ids != stopped_reuse.generated_token_ids) {
        std::cerr << "Flash-Next partial MTP terminal reused " << stopped_reuse.reused_prompt_tokens
                  << ", expected " << expected_stopped_reuse << ", cold/reused output match="
                  << (stopped_cold.generated_token_ids == stopped_reuse.generated_token_ids)
                  << '\n';
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
        std::cerr << "ordinary:";
        for (const auto token : result.generated_token_ids) { std::cerr << ' ' << token; }
        std::cerr << "\nexpected:";
        for (const auto token : expected_prefix) { std::cerr << ' ' << token; }
        std::cerr << '\n';
        return 1;
    }
    return 0;
}

int exercise_concurrent_state(ninfer::Engine& engine,
                              const std::vector<ninfer::TokenId>& expected_prefix) {
    // Different frontiers exercise local prefill rows and shared decode rows. Repeat in
    // reversed admission order to reuse both physical lanes and recurrent state slots.
    auto continuation = canonical_prompt();
    continuation.insert(continuation.end(), expected_prefix.begin(),
                        expected_prefix.begin() + 4);
    const auto cold =
        engine.generate(engine.prepare_tokens(continuation), greedy_options(8, false));
    const auto before = engine.runtime_stats();
    for (bool reverse : {false, true}) {
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
            std::cerr << "Flash-Next concurrent lane reuse changed the canonical fixture\n";
            return 1;
        }
    }
    const auto after = engine.runtime_stats();
    if (after.decode_row_rounds - before.decode_row_rounds <=
        after.decode_rounds - before.decode_rounds) {
        std::cerr << "Flash-Next concurrent fixture never executed a two-row decode\n";
        return 1;
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
    const auto& expected_prefix   = canonical_output(recipe, true);
    const auto& ordinary_prefix   = canonical_output(recipe, false);
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
            if (exercise_mtp_and_prefix(engine, expected_prefix) != 0) { return 1; }
            if (exercise_concurrent_state(engine, expected_prefix) != 0) { return 1; }
            if (exercise_vision(engine) != 0) { return 1; }
        }
        if (exercise_ordinary_greedy(artifact, ordinary_prefix) != 0) { return 1; }
        std::cout << "OK Qwen3.8 Flash Next real Engine\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Qwen3.8 Flash Next real Engine: " << error.what() << '\n';
        return 1;
    }
}
