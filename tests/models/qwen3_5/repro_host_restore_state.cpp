// Focused validator for the host-restore State semantics (originally filed as ninfer-gb10-dn9,
// Engine bug #8: "restore uploads KV pages from Host but zero state images").
//
// Root cause found on sm_120a: not an Engine defect. The pressure planner demotes the typed
// rewrite checkpoint first ("a typed rewrite is the less destructive state relief while the
// endpoint remains usable", pressure.cpp) and the endpoint keeps riding the active-concurrency
// Device slot (Capacity_DeviceState = C + H, resource-scheduling-and-context-cache.md §5.1).
// The original scenario restored a continuation follow-up that reproduces the assistant turn, so
// the identity match extended past the turn boundary to the full cached endpoint: a
// PrivateEndpoint reuse that binds the still-Device-resident endpoint zero-copy. Zero state
// uploads is the correct outcome for that shape; the old expectation (PrivateTurnClosure +
// state_h2d up) was checking the wrong follow-up.
//
// This program validates the actual contract:
//   1. Establish a retained MTP checkpoint on a live session.
//   2. Fire an uncached pressure request that demotes the checkpoint to Host
//      (the typed rewrite State image plus the KV pages; state_d2h/main/backend kv_d2h up).
//   3. Regeneration restore: a follow-up whose only new content is a replacement user turn, so
//      the identity match stops at the turn boundary and the reuse binds the demoted rewrite
//      checkpoint from Host. The State image and the KV pages must both come back
//      (state_h2d/main/backend kv_h2d up). This is the Host-to-Device State restore path.
//   4. Negative control: a second regeneration with a different replacement turn probes the same
//      turn boundary and binds the rewrite image that step 3 restored to Device. No further
//      State or KV upload may happen (state_h2d and kv_h2d flat): a Device-resident source
//      binds zero-copy, never redundantly.
//
// Exit 0 = the Host State restore path works and is not over-applied; exit 1 = a check failed.
//
// Build against the fork's libraries and run with NINFER_TEST_ARTIFACT set to the 27B artifact:
//   NINFER_TEST_ARTIFACT=<groupwise|nvfp4 .ninfer> ./repro_host_restore_state
//
// The proposal head is selected from the artifact (Optimized when the text component binds a
// proposal object, Full otherwise), so both artifacts run: groupwise exercises the Full-head
// control path, nvfp4 the optimized-head path.

#include "ninfer/engine.h"

#include "artifact/reader.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

// Select the strongest proposal head the artifact carries (mirrors the prefix real test).
ninfer::ProposalHead artifact_proposal_head(const char* artifact) {
    ninfer::artifact::Reader reader(artifact);
    return reader.directory().component("text").proposal ? ninfer::ProposalHead::Optimized
                                                         : ninfer::ProposalHead::Full;
}

ninfer::EngineOptions host_restore_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 512;
    options.kv_capacity                          = ninfer::KvCapacityPolicy::explicit_capacity(512);
    options.prefill_chunk                        = 256;
    options.speculative.backend                  = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens             = 3;
    options.speculative.proposal_head            = artifact_proposal_head(artifact);
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
    for (std::uint32_t index = 0; index < 300U; ++index) { text += "alpha "; }
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
    std::cout << label << " state_d2h=" << s.state_d2h_count << " state_h2d=" << s.state_h2d_count
              << " main_kv_d2h=" << s.main_kv_d2h_pages << " main_kv_h2d=" << s.main_kv_h2d_pages
              << " backend_kv_d2h=" << s.backend_kv_d2h_pages
              << " backend_kv_h2d=" << s.backend_kv_h2d_pages
              << " degraded=" << s.pressure_private_owners_degraded
              << " evicted=" << s.pressure_private_owners_evicted << '\n';
}

void print_restore(const char* label, const ninfer::GenerationResult& result,
                   const ninfer::RuntimeStats& stats) {
    std::cout << label << " path=" << static_cast<int>(result.prefix_reuse_path)
              << " reused=" << result.reused_prompt_tokens
              << " outputs=" << result.generated_token_ids.size() << '\n';
    print_stats(label, stats);
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "SKIP: supply NINFER_TEST_ARTIFACT\n";
        return 77;
    }

    try {
        ninfer::Engine engine(host_restore_engine_options(artifact));

        // 1. Establish a retained MTP checkpoint on a live session.
        const ninfer::GenerationResult retained =
            engine.generate(engine.prepare(retained_input()), make_options(5, /*reuse=*/true));
        if (retained.prompt.prompt_tokens <= 256 || retained.generated_token_ids.size() != 5) {
            std::cerr << "source request did not complete: prompt=" << retained.prompt.prompt_tokens
                      << " outputs=" << retained.generated_token_ids.size() << '\n';
            return 1;
        }

        // 2. Pressure: demote the checkpoint (typed rewrite State image + KV pages) to Host.
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

        const ninfer::RuntimeStats before_pressure = engine.runtime_stats();
        const ninfer::GenerationResult pressure_result =
            engine.generate(engine.prepare(continuation), make_options(2, /*reuse=*/false));
        const ninfer::RuntimeStats after_pressure = engine.runtime_stats();
        print_stats("after_pressure", after_pressure);
        if (pressure_result.generated_token_ids.size() != 2 ||
            after_pressure.state_d2h_count <= before_pressure.state_d2h_count ||
            after_pressure.main_kv_d2h_pages <= before_pressure.main_kv_d2h_pages ||
            after_pressure.backend_kv_d2h_pages <= before_pressure.backend_kv_d2h_pages) {
            std::cerr << "Host pressure did not demote the complete MTP checkpoint\n";
            return 1;
        }

        // 3. Regeneration restore: only the user turn is replaced, so the identity match stops
        //    at the turn boundary and the reuse binds the demoted typed rewrite checkpoint.
        ninfer::PromptInput regeneration = retained_input();
        ninfer::ChatMessage replacement;
        replacement.role = ninfer::ChatRole::User;
        replacement.parts.push_back(
            ninfer::MessagePart{.kind  = ninfer::MessagePartKind::Text,
                                .text  = "Answer the question differently this time.",
                                .media = {}});
        regeneration.messages.push_back(std::move(replacement));

        const ninfer::GenerationResult restored =
            engine.generate(engine.prepare(regeneration), make_options(2, /*reuse=*/true));
        const ninfer::RuntimeStats after_restore = engine.runtime_stats();
        print_restore("after_restore ", restored, after_restore);

        if (restored.generated_token_ids.size() != 2 ||
            restored.prefix_reuse_path != ninfer::PrefixReusePath::PrivateTurnClosure ||
            restored.reused_prompt_tokens == 0 ||
            restored.reused_prompt_tokens >= retained.prompt.prompt_tokens) {
            std::cerr << "regeneration did not restore the turn-boundary checkpoint: path="
                      << static_cast<int>(restored.prefix_reuse_path)
                      << " reused=" << restored.reused_prompt_tokens
                      << " prompt=" << retained.prompt.prompt_tokens << '\n';
            return 1;
        }
        if (after_restore.state_h2d_count <= after_pressure.state_h2d_count ||
            after_restore.main_kv_h2d_pages <= after_pressure.main_kv_h2d_pages ||
            after_restore.backend_kv_h2d_pages <= after_pressure.backend_kv_h2d_pages) {
            std::cerr << "BUG: the demoted checkpoint was not materialized from Host on the "
                         "turn-boundary restore: path="
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

        // 4. Negative control: the same regeneration prompt probes the same frontiers with the
        //    same digests, so it reuses the first regeneration's committed checkpoints. Those
        //    are Device-resident (committed by the live request), so no further State or KV
        //    upload may happen: a Device-resident source binds zero-copy.
        ninfer::PromptInput control_prompt = retained_input();
        ninfer::ChatMessage replacement_again;
        replacement_again.role = ninfer::ChatRole::User;
        replacement_again.parts.push_back(
            ninfer::MessagePart{.kind  = ninfer::MessagePartKind::Text,
                                .text  = "Answer the question differently this time.",
                                .media = {}});
        control_prompt.messages.push_back(std::move(replacement_again));

        const ninfer::GenerationResult control =
            engine.generate(engine.prepare(control_prompt), make_options(2, /*reuse=*/true));
        const ninfer::RuntimeStats after_control = engine.runtime_stats();
        print_restore("after_control ", control, after_control);
        if (control.generated_token_ids.size() != 2 || control.reused_prompt_tokens == 0 ||
            after_control.state_h2d_count != after_restore.state_h2d_count ||
            after_control.main_kv_h2d_pages != after_restore.main_kv_h2d_pages ||
            after_control.backend_kv_h2d_pages != after_restore.backend_kv_h2d_pages) {
            std::cerr << "negative control failed (redundant or missing Device-resident "
                         "handling): path="
                      << static_cast<int>(control.prefix_reuse_path)
                      << " reused=" << control.reused_prompt_tokens
                      << " outputs=" << control.generated_token_ids.size()
                      << " state_h2d=" << after_control.state_h2d_count << " (expected "
                      << after_restore.state_h2d_count << ")"
                      << " main_kv_h2d=" << after_control.main_kv_h2d_pages << " (expected "
                      << after_restore.main_kv_h2d_pages << ")"
                      << " backend_kv_h2d=" << after_control.backend_kv_h2d_pages << " (expected "
                      << after_restore.backend_kv_h2d_pages << ")\n";
            return 1;
        }
        std::cout << "ok: the demoted State image was restored from Host on the turn-boundary "
                     "reuse, and no redundant State upload happened for the Device-resident "
                     "source\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
