// Standalone repro for ninfer-gb10-hdt (Engine bug #7): a vision prompt computes a different
// prompt state when allow_prefix_reuse=true vs false, with no cache hit, no MTP, and a fresh
// encode on the first request of a fresh engine.
//
// The flag alone must not change the computation. This program builds two fresh engines (one per
// flag value) so no reuse can occur, runs the same vision prompt at temperature 0, and compares
// the greedy token sequences. Exit 0 = identical (correct); exit 1 = divergence (bug).
//
// Build against the fork's libraries and run with NINFER_TEST_ARTIFACT set to the 27B artifact:
//   NINFER_TEST_ARTIFACT=<groupwise|nvfp4 .ninfer> ./repro_vision_reuse_divergence
//
// Expected on the groupwise (Q4/Q5) artifact: divergence (the margins flip the greedy token).
// Expected on the nvfp4 artifact: identical (the perturbation is present but below the margin).
// The exact observed token sequences (groupwise) were:
//   allow_prefix_reuse=false -> 27775 383 279 2099 3766
//   allow_prefix_reuse=true  -> 760 2099 18126 264 5215

#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<std::uint8_t> gradient_ppm(int width = 64, int height = 64) {
    std::vector<std::uint8_t> ppm;
    const std::string header =
        "P6\n" + std::to_string(width) + ' ' + std::to_string(height) + "\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int index = 0; index < width * height; ++index) {
        ppm.push_back(static_cast<std::uint8_t>(index & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 3) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7) & 0xff));
    }
    return ppm;
}

ninfer::PromptInput vision_input() {
    ninfer::MessagePart image;
    image.kind             = ninfer::MessagePartKind::Media;
    image.media.kind       = ninfer::MediaKind::Image;
    image.media.bytes      = gradient_ppm();
    image.media.media_type = "image/x-portable-pixmap";
    image.media.source_name = "inline.ppm";

    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(std::move(image));
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "What is visible?", .media = {}});

    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking   = false;
    input.context_cache.session_key = "repro-vision-reuse";
    input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
    return input;
}

std::vector<ninfer::TokenId> generate(const char* artifact, bool reuse) {
    ninfer::EngineOptions options;
    options.artifact_path              = artifact;
    options.max_context                = 4096;
    options.kv_capacity                = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk              = 1024;
    options.speculative.backend        = ninfer::SpeculativeBackend::None;
    options.enable_vision              = true;
    options.max_concurrency            = 1;
    options.max_pending_requests       = 1;
    options.context_cache.device_state_slots = 4;

    ninfer::Engine engine(options);
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 5;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = reuse;
    request.stop.include_model_defaults       = false;

    const ninfer::GenerationResult result =
        engine.generate(engine.prepare(vision_input()), request);
    return result.generated_token_ids;
}

void print(const char* label, const std::vector<ninfer::TokenId>& tokens) {
    std::cout << label;
    for (const auto token : tokens) { std::cout << ' ' << token; }
    std::cout << '\n';
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cerr << "NINFER_TEST_ARTIFACT must name the 27B .ninfer artifact\n";
        return 2;
    }
    // Two fresh engines: the second has no knowledge of the first, so reused_prompt_tokens is 0
    // on both and the only difference between the two runs is the allow_prefix_reuse flag.
    const std::vector<ninfer::TokenId> no_reuse   = generate(artifact, /*reuse=*/false);
    const std::vector<ninfer::TokenId> with_reuse = generate(artifact, /*reuse=*/true);
    print("allow_prefix_reuse=false:", no_reuse);
    print("allow_prefix_reuse=true: ", with_reuse);
    if (no_reuse == with_reuse) {
        std::cout << "ok: the flag did not change the computation\n";
        return 0;
    }
    std::cerr << "BUG: allow_prefix_reuse changed the computation with no cache hit and a fresh "
                "encode (reused_prompt_tokens=0 on both runs)\n";
    return 1;
}
