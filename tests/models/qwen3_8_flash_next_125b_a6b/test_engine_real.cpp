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

const std::vector<ninfer::TokenId>& canonical_output() {
    static const std::vector<ninfer::TokenId> output{29108, 4009, 27891, 8964, 579, 16078,
                                                     321,   1100, 9872,  303,  660, 17425};
    return output;
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

int exercise_mtp_and_prefix(ninfer::Engine& engine) {
    // Greedy output from the source checkpoint for the canonical non-thinking chat template.
    // Checking semantic text would require duplicating the tokenizer in this C++ integration
    // test, so protect the exact token prefix instead. This catches numerically plausible but
    // language-corrupt model execution, including routed-MoE row-layout regressions.
    const auto& prompt                   = canonical_prompt();
    const auto& expected_prefix          = canonical_output();
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

int exercise_ordinary_greedy(const char* artifact) {
    ninfer::EngineOptions options    = engine_options(artifact);
    options.speculative.backend      = ninfer::SpeculativeBackend::None;
    options.speculative.draft_tokens = 0;
    options.enable_vision = false;
    ninfer::Engine engine(std::move(options));
    const ninfer::GenerationResult result =
        engine.generate(engine.prepare_tokens(canonical_prompt()),
                        greedy_options(canonical_output().size(), false));
    if (result.speculative.backend != ninfer::SpeculativeBackend::None ||
        result.generated_token_ids != canonical_output()) {
        std::cerr << "Flash-Next ordinary greedy output disagrees with the checkpoint/MTP path\n";
        std::cerr << "ordinary:";
        for (const auto token : result.generated_token_ids) { std::cerr << ' ' << token; }
        std::cerr << "\nexpected:";
        for (const auto token : canonical_output()) { std::cerr << ' ' << token; }
        std::cerr << '\n';
        return 1;
    }
    return 0;
}

int exercise_concurrent_state(ninfer::Engine& engine) {
    // Different frontiers exercise local prefill rows and shared decode rows. Repeat in
    // reversed admission order to reuse both physical lanes and recurrent state slots.
    auto continuation = canonical_prompt();
    continuation.insert(continuation.end(), canonical_output().begin(),
                        canonical_output().begin() + 4);
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
        if (root.generated_token_ids != canonical_output() ||
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

// Compare graph replay with the same eager arithmetic schedules on the supplied artifact.
// Projection and selected-slot state mathematics have independent FP64 Op qualification.
int exercise_graph_equivalence(const char* artifact) {
    std::vector<std::vector<ninfer::TokenId>> reference;
    std::vector<std::vector<ninfer::TokenId>> continuation_reference;
    std::vector<std::vector<ninfer::TokenId>> batch_reference;
    for (const bool graphs : {false, true}) {
        auto options                     = engine_options(artifact);
        options.speculative.backend      = ninfer::SpeculativeBackend::None;
        options.speculative.draft_tokens = 0;
        options.enable_vision            = false;
        options.use_cuda_graph           = graphs;
        options.max_context              = 1024;
        options.kv_capacity              = ninfer::KvCapacityPolicy::explicit_capacity(2048);
        ninfer::Engine engine(options);
        if ((engine.memory_summary().cuda_graph_allowance_bytes != 0) != graphs) {
            throw std::runtime_error("ordinary graphs were silently disabled");
        }
        std::size_t case_index = 0;
        for (const std::size_t length : {27U, 124U, 508U}) {
            auto prompt = canonical_prompt();
            prompt.resize(length, 198);
            const auto first =
                engine.generate(engine.prepare_tokens(prompt), greedy_options(16, false));
            if (first.generated_token_ids.size() != 16) {
                throw std::runtime_error("graph fixture did not execute all requested steps");
            }
            if (!graphs) {
                reference.push_back(first.generated_token_ids);
            } else if (reference.at(case_index) != first.generated_token_ids) {
                throw std::runtime_error("graph/eager output mismatch at prompt length " +
                                         std::to_string(length));
            }
            for (int repeat = 0; repeat < 2; ++repeat) {
                const auto repeated =
                    engine.generate(engine.prepare_tokens(prompt), greedy_options(16, false));
                if (repeated.generated_token_ids != first.generated_token_ids) {
                    throw std::runtime_error("ordinary repeated output mismatch");
                }
            }
            engine.generate(engine.prepare_tokens(prompt), greedy_options(16, true));
            auto history = prompt;
            history.insert(history.end(), first.generated_token_ids.begin(),
                           first.generated_token_ids.end());
            history.push_back(198);
            const auto reused =
                engine.generate(engine.prepare_tokens(history), greedy_options(8, true));
            const auto cold =
                engine.generate(engine.prepare_tokens(history), greedy_options(8, false));
            if (reused.reused_prompt_tokens == 0) {
                throw std::runtime_error("ordinary continuation fixture never reused a prefix");
            }
            // Prefill and incremental schedules have their own qualified floating-point profiles.
            // Verify that graph replay preserves each schedule, including cached state, rather
            // than imposing bitwise equality between different arithmetic schedules.
            if (!graphs) {
                continuation_reference.push_back(reused.generated_token_ids);
                continuation_reference.push_back(cold.generated_token_ids);
            } else if (continuation_reference.at(case_index * 2) != reused.generated_token_ids ||
                       continuation_reference.at(case_index * 2 + 1) != cold.generated_token_ids) {
                throw std::runtime_error("graph/eager continuation output mismatch");
            }
            ++case_index;
        }
        auto second_prompt = canonical_prompt();
        second_prompt.insert(second_prompt.end(), reference[0].begin(), reference[0].begin() + 4);
        const auto before = engine.runtime_stats();
        for (bool reverse : {false, true}) {
            auto a =
                engine.submit(engine.prepare_tokens(reverse ? second_prompt : canonical_prompt()),
                              greedy_options(12, false));
            auto b =
                engine.submit(engine.prepare_tokens(reverse ? canonical_prompt() : second_prompt),
                              greedy_options(12, false));
            const auto first    = a.wait();
            const auto second   = b.wait();
            const auto& root    = reverse ? second : first;
            const auto& resumed = reverse ? first : second;
            // Batch size selects projection/reduction routes. Compare the same compact-batch
            // schedule across launch modes; selected-slot mathematics is independently checked
            // by the GDN FP64 oracle, rather than treating single-row output as an oracle.
            if (!graphs) {
                batch_reference.push_back(root.generated_token_ids);
                batch_reference.push_back(resumed.generated_token_ids);
            } else {
                const std::size_t index = reverse ? 2 : 0;
                if (root.generated_token_ids != batch_reference.at(index) ||
                    resumed.generated_token_ids != batch_reference.at(index + 1)) {
                    throw std::runtime_error("ordinary graph/eager compact batch mismatch");
                }
            }
        }
        const auto after = engine.runtime_stats();
        if (after.decode_row_rounds - before.decode_row_rounds <=
            after.decode_rounds - before.decode_rounds) {
            throw std::runtime_error("graph fixture never executed a two-row batch");
        }
        std::cout << "OK " << (graphs ? "graph" : "eager")
                  << " equivalence: repeats, continuations, batch/lane "
                     "reuse, 128/512 profile boundaries; decode_rounds="
                  << after.decode_rounds << '\n';
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_QWEN38_FLASH_NEXT_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') { return 77; }
    try {
        if (argc == 2 && std::string(argv[1]) == "--graph-equivalence") {
            return exercise_graph_equivalence(artifact);
        }
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
            if (exercise_mtp_and_prefix(engine) != 0) { return 1; }
            if (exercise_concurrent_state(engine) != 0) { return 1; }
            if (exercise_vision(engine) != 0) { return 1; }
        }
        if (exercise_ordinary_greedy(artifact) != 0) { return 1; }
        std::cout << "OK Qwen3.8 Flash Next real Engine\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Qwen3.8 Flash Next real Engine: " << error.what() << '\n';
        return 1;
    }
}
