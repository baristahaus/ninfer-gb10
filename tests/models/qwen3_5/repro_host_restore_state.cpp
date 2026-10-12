// Standalone repro for ninfer-gb10-dn9 (Engine bug #8): after host demotion, the restored
// request reuses the KV pages from Host but uploads zero state images (state_h2d_count stays
// flat). Groupwise (Full head) passes the identical scenario; only the nvfp4 artifact with the
// optimized proposal head fails.
//
// This program mirrors exercise_host_restore (test_engine_prefix_real.cpp) but is focused on the
// state-transfer check. It:
//   1. Establishes a retained MTP checkpoint on a live session.
//   2. Fires an uncached pressure request that demotes the checkpoint to Host (state_d2h up).
//   3. Restores the same continuation (state_h2d must go up for the state images to come back).
//
// Exit 0 = the state images were materialized from Host (correct); exit 1 = state_h2d did not
// increase (bug: the optimized head's state images never reached Host / were not uploaded).
//
// Build against the fork's libraries and run with NINFER_TEST_ARTIFACT set to the 27B artifact:
//   NINFER_TEST_ARTIFACT=<groupwise|nvfp4 .ninfer> ./repro_host_restore_state
//
// Expected on groupwise (Full head): ok (state_h2d increases).
// Expected on nvfp4 (optimized head): BUG (state_h2d stays flat, state=0 in the failure line).

#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

ninfer::EngineOptions host_restore_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 512;
    options.kv_capacity                          = ninfer::KvCapacityPolicy::explicit_capacity(512);
    options.prefill_chunk                        = 256;
    options.speculative.backend                  = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens             = 3;
    options.speculative.proposal_head            = ninfer::ProposalHead::Optimized;
    options.max_concurrency                      = 1;
    options.max_pending_requests                 = 1;
    options.context_cache.device_state_slots     = 1;
    options.context_cache.host_state_slots       = 2;
    options.context_cache.host_kv_capacity_bytes = 256ULL << 20;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return options;
}

ninfer::RequestOptions make_options(std::uint32_t outputs, bool reuse) {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = outputs;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = reuse;
    request.stop.include_model_defaults       = false;
    return request;
}

ninfer::PromptInput retained_input() {
    std::string text;
    text.reserve(6U * 300U);
    for (std::uint32_t index = 0; index < 300; ++index) { text += "alpha "; }
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking   = false;
    input.context_cache.session_key = "repro-host-restore";
    input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
    return input;
}

void print_stats(const char* label, const ninfer::RuntimeStats& s) {
    std::cout << label
              << " state_d2h=" << s.state_d2h_count
              << " state_h2d=" << s.state_h2d_count
              << " main_kv_d2h=" << s.main_kv_d2h_pages
              << " main_kv_h2d=" << s.main_kv_h2d_pages
              << " backend_kv_d2h=" << s.backend_kv_d2h_pages
              << " backend_kv_h2d=" << s.backend_kv_h2d_pages
              << " degraded=" << s.pressure_private_owners_degraded
              << " evicted=" << s.pressure_private_owners_evicted << '\n';
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cerr << "NINFER_TEST_ARTIFACT must name the 27B .ninfer artifact\n";
        return 2;
    }

    ninfer::Engine engine(host_restore_engine_options(artifact));

    // 1. Establish a retained MTP checkpoint on a live session.
    const ninfer::GenerationResult retained =
        engine.generate(engine.prepare(retained_input()), make_options(5, /*reuse=*/true));
    if (retained.prompt.prompt_tokens <= 256 || retained.generated_token_ids.size() != 5) {
        std::cerr << "source request did not complete: prompt=" << retained.prompt.prompt_tokens
                  << " outputs=" << retained.generated_token_ids.size() << '\n';
        return 1;
    }

    // 2. Build the continuation (source + assistant + followup).
    ninfer::PromptInput continuation = retained_input();
    ninfer::ChatMessage assistant;
    assistant.role              = ninfer::ChatRole::Assistant;
    assistant.reasoning_content = retained.reasoning;
    assistant.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = retained.content, .media = {}});
    continuation.messages.push_back(std::move(assistant));
    ninfer::ChatMessage followup;
    followup.role = ninfer::ChatRole::User;
    followup.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "Continue briefly.", .media = {}});
    continuation.messages.push_back(std::move(followup));

    // 3. Pressure: demote the checkpoint to Host.
    const ninfer::RuntimeStats before_pressure = engine.runtime_stats();
    const ninfer::GenerationResult pressure_result =
        engine.generate(engine.prepare(continuation), make_options(2, /*reuse=*/false));
    const ninfer::RuntimeStats after_pressure = engine.runtime_stats();
    print_stats("after_pressure", after_pressure);
    if (pressure_result.generated_token_ids.size() != 2 ||
        after_pressure.state_d2h_count <= before_pressure.state_d2h_count ||
        after_pressure.main_kv_d2h_pages <= before_pressure.main_kv_d2h_pages) {
        std::cerr << "Host pressure did not demote the complete MTP checkpoint\n";
        return 1;
    }

    // 4. Restore: the state images must come back from Host.
    const ninfer::GenerationResult restored =
        engine.generate(engine.prepare(continuation), make_options(2, /*reuse=*/true));
    const ninfer::RuntimeStats after_restore = engine.runtime_stats();
    print_stats("after_restore ", after_restore);

    if (restored.generated_token_ids.size() != 2 ||
        restored.prefix_reuse_path != ninfer::PrefixReusePath::PrivateTurnClosure ||
        restored.reused_prompt_tokens == 0) {
        std::cerr << "restore did not reuse the demoted checkpoint: path="
                  << static_cast<int>(restored.prefix_reuse_path)
                  << " reused=" << restored.reused_prompt_tokens << '\n';
        return 1;
    }
    if (after_restore.state_h2d_count <= after_pressure.state_h2d_count) {
        std::cerr << "BUG: state images were not materialized from Host on restore "
                    "(state_h2d stayed flat): path="
                  << static_cast<int>(restored.prefix_reuse_path)
                  << " reused=" << restored.reused_prompt_tokens
                  << " outputs=" << restored.generated_token_ids.size()
                  << " state_h2d=" << after_restore.state_h2d_count
                  << " main_kv_h2d=" << after_restore.main_kv_h2d_pages
                  << " backend_kv_h2d=" << after_restore.backend_kv_h2d_pages
                  << " degraded=" << after_restore.pressure_private_owners_degraded
                  << " evicted=" << after_restore.pressure_private_owners_evicted << '\n';
        return 1;
    }
    std::cout << "ok: the state images were materialized from Host on restore\n";
    return 0;
}
