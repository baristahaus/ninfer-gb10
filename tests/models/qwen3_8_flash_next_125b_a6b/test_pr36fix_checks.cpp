// PR #36 fix verification driver (GB10, 2026-09-30). Engine-level checks for the capacity fix
// (eb9e87fa) and its base (ed6525fa), run by tools/gb10/pr36fix_verify.sh.
//
// Golden-free checks (prefix reuse, partial MTP terminal, concurrent lane consistency, the
// two-row decode stat) are strict: a failure is a real regression. The two golden comparisons
// (the MTP greedy prefix and the concurrent root) are PRINT-ONLY on both trees: on the fix
// branch the fp8_mtp MTP golden is stale by design until the route change is re-recorded per
// the fixture policy, and on the base it is the recorded golden. The near-tie check prints the
// top logprob entries at generated-token index 2 (the 27891 vs 5435 position).

#include "artifact/reader.h"

#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using ninfer::TokenId;

ninfer::EngineOptions engine_options(const char* artifact, bool with_logprobs) {
    ninfer::EngineOptions options;
    options.artifact_path    = artifact;
    options.max_context      = 512;
    options.kv_capacity      = ninfer::KvCapacityPolicy::explicit_capacity(1024);
    options.prefill_chunk    = 256;
    options.speculative.backend   = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Full;
    options.enable_vision    = true;
    options.use_cuda_graph   = true;
    options.max_concurrency  = 2;
    options.max_pending_requests = 2;
    options.context_cache.device_state_slots = 4;
    options.context_cache.max_private_continuations = 2;
    options.context_cache.max_shared_prefixes = 1;
    options.token_logprobs = with_logprobs;
    return options;
}

const std::vector<TokenId>& canonical_prompt() {
    static const std::vector<TokenId> prompt{
        248045, 846, 198,  814, 20139, 303, 2250,   2716, 22157, 3069,   279, 12515,  7701, 6105,
        2261,   279, 1834, 13,  248046, 198, 248045, 74455, 198,   248068, 271, 248069, 271};
    return prompt;
}

// The two recorded fp8 goldens (from test_engine_real.cpp, unchanged by the PR).
const std::vector<TokenId>& fp8_mtp_golden() {
    static const std::vector<TokenId> golden{
        29108, 4009, 5435, 660, 7736, 314, 279, 9155, 19142, 11, 864, 43000};
    return golden;
}
const std::vector<TokenId>& fp8_ordinary_golden() {
    static const std::vector<TokenId> golden{
        29108, 4009, 27891, 8964, 579, 16078, 321, 1100, 9872, 303, 660, 17425};
    return golden;
}

// The fp8 cross-path fixture (separator 11855, resumed prefix 4), from test_engine_real.cpp.
struct CrossPathFixture {
    TokenId separator;
    std::ptrdiff_t resumed_prefix;
};

ninfer::RequestOptions greedy_options(std::uint32_t outputs, bool reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

void print_tokens(const char* label, const std::vector<TokenId>& tokens) {
    std::cerr << label << ':';
    for (const auto token : tokens) { std::cerr << ' ' << token; }
    std::cerr << '\n';
}

std::string artifact_recipe(const char* artifact) {
    const ninfer::artifact::Reader reader(artifact);
    const auto& provenance = reader.directory().provenance;
    if (provenance.contains("recipe") && provenance.at("recipe").is_string()) {
        return provenance.at("recipe").get<std::string>();
    }
    return {};
}

// Check 2a: the MTP greedy line. Print-only (the fp8_mtp golden is stale on the fix branch).
void check_mtp_line(ninfer::Engine& engine, std::vector<TokenId>& line_out) {
    const auto& prompt = canonical_prompt();
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(12, true));
    line_out = first.generated_token_ids;
    std::cerr << "mtp:";
    for (const auto token : line_out) { std::cerr << ' ' << token; }
    std::cerr << '\n';
    const bool is_mtp   = first.speculative.backend == ninfer::SpeculativeBackend::Mtp;
    std::cerr << "mtp backend=" << (is_mtp ? "yes" : "no") << " rounds=" << first.speculative.rounds
              << " matches_fp8_mtp_golden="
              << (line_out == fp8_mtp_golden() ? "yes" : "no")
              << " matches_fp8_ordinary_golden="
              << (line_out == fp8_ordinary_golden() ? "yes" : "no") << '\n';
}

// Check 2b, part 1: prefix reuse, captured state vs cold (test_engine_real.cpp lines 169-186).
int check_prefix_reuse(ninfer::Engine& engine, const std::vector<TokenId>& first) {
    std::vector<TokenId> continuation = canonical_prompt();
    continuation.insert(continuation.end(), first.begin(), first.end());
    continuation.push_back(11855);  // the fp8 fixture separator
    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare_tokens(continuation), greedy_options(2, true));
    const ninfer::GenerationResult cold =
        engine.generate(engine.prepare_tokens(continuation), greedy_options(2, false));
    const std::uint32_t expected_reuse =
        static_cast<std::uint32_t>(canonical_prompt().size() + first.size() - 1);
    if (reused.reused_prompt_tokens != expected_reuse || reused.generated_token_ids.size() != 2 ||
        cold.reused_prompt_tokens != 0 || cold.generated_token_ids != reused.generated_token_ids) {
        std::cerr << "FAIL prefix reuse: reused=" << reused.reused_prompt_tokens
                  << " expected=" << expected_reuse << '\n';
        print_tokens("reused", reused.generated_token_ids);
        print_tokens("cold", cold.generated_token_ids);
        return 1;
    }
    std::cerr << "PASS prefix reuse (reused=" << reused.reused_prompt_tokens << ", tokens:";
    print_tokens("reuse", reused.generated_token_ids);
    return 0;
}

// Check 2b, part 2: custom stop inside the MTP round, then the partial-terminal reuse
// (test_engine_real.cpp lines 188-223).
int check_stop_partial_terminal(ninfer::Engine& engine, const std::vector<TokenId>& first) {
    if (first[0] == first[1]) {
        std::cerr << "FAIL partial-terminal fixture repeats its first token\n";
        return 1;
    }
    ninfer::RequestOptions stop_options = greedy_options(6, true);
    stop_options.stop.token_ids.push_back(first[1]);
    const ninfer::GenerationResult stopped =
        engine.generate(engine.prepare_tokens(canonical_prompt()), stop_options);
    if (stopped.finish_reason != ninfer::FinishReason::StopToken ||
        stopped.generated_token_ids.size() != 2 || stopped.generated_token_ids[0] != first[0] ||
        stopped.generated_token_ids[1] != first[1]) {
        std::cerr << "FAIL custom stop did not terminate inside the MTP round\n";
        return 1;
    }
    std::vector<TokenId> stopped_continuation = canonical_prompt();
    stopped_continuation.insert(stopped_continuation.end(), stopped.generated_token_ids.begin(),
                                stopped.generated_token_ids.end());
    stopped_continuation.push_back(11855);
    const ninfer::GenerationResult stopped_reuse =
        engine.generate(engine.prepare_tokens(stopped_continuation), greedy_options(1, true));
    const ninfer::GenerationResult stopped_cold =
        engine.generate(engine.prepare_tokens(stopped_continuation), greedy_options(1, false));
    const std::uint32_t expected_stopped_reuse =
        static_cast<std::uint32_t>(canonical_prompt().size() + stopped.generated_token_ids.size() - 1);
    if (stopped_reuse.reused_prompt_tokens != expected_stopped_reuse ||
        stopped_cold.reused_prompt_tokens != 0 || stopped_reuse.generated_token_ids.size() != 1 ||
        stopped_cold.generated_token_ids != stopped_reuse.generated_token_ids) {
        std::cerr << "FAIL partial MTP terminal reused " << stopped_reuse.reused_prompt_tokens
                  << ", expected " << expected_stopped_reuse << '\n';
        print_tokens("reused", stopped_reuse.generated_token_ids);
        print_tokens("cold", stopped_cold.generated_token_ids);
        return 1;
    }
    std::cerr << "PASS custom stop + partial MTP terminal (reused="
              << stopped_reuse.reused_prompt_tokens << ")\n";
    return 0;
}

// Check 2b, part 3: concurrent lanes. The root-lane golden comparison is print-only (stale
// golden on the fix branch); the resumed-lane vs cold-lane consistency and the two-row decode
// stat are strict (test_engine_real.cpp exercise_concurrent_state).
int check_concurrent_state(ninfer::Engine& engine, const std::vector<TokenId>& expected_prefix,
                           const CrossPathFixture& fixture,
                           const std::vector<TokenId>& mtp_line) {
    auto continuation = canonical_prompt();
    continuation.insert(continuation.end(), expected_prefix.begin(),
                        expected_prefix.begin() + fixture.resumed_prefix);
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
        std::cerr << "concurrent root (" << (reverse ? "reversed" : "normal") << "):";
        print_tokens("root", root.generated_token_ids);
        if (root.generated_token_ids != expected_prefix) {
            std::cerr << "NOTE root differs from the recorded golden (expected pre re-record)"
                      << (reverse ? " (reversed admission)" : "") << '\n';
        }
        if (resumed.generated_token_ids != cold.generated_token_ids) {
            std::cerr << "FAIL concurrent lane reuse changed the canonical fixture"
                      << (reverse ? " (reversed admission)" : "") << '\n';
            print_tokens("resumed", resumed.generated_token_ids);
            print_tokens("cold", cold.generated_token_ids);
            return 1;
        }
    }
    const auto after = engine.runtime_stats();
    if (after.decode_row_rounds - before.decode_row_rounds <=
        after.decode_rounds - before.decode_rounds) {
        std::cerr << "FAIL concurrent fixture never executed a two-row decode\n";
        return 1;
    }
    std::cerr << "PASS concurrent lane consistency (mtp line reused as root: "
              << (mtp_line == expected_prefix ? "root equals the MTP greedy line"
                                              : "root differs from the MTP greedy line")
              << ")\n";
    return 0;
}

// Check 3: the top logprob entries at generated-token index 2 (the 27891 vs 5435 position),
// from a fresh engine loaded with token_logprobs. Prints the top list and the top-1/top-2 gap.
int check_near_tie(const char* artifact, std::vector<TokenId>& line_out) {
    ninfer::Engine engine(engine_options(artifact, true));
    const auto& prompt = canonical_prompt();
    const ninfer::GenerationResult result =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(12, true));
    line_out = result.generated_token_ids;
    if (result.token_logprobs.size() < 3) {
        std::cerr << "FAIL token_logprobs reports missing (got "
                  << result.token_logprobs.size() << " of 12)\n";
        return 1;
    }
    const auto& report = result.token_logprobs[2];
    std::cerr << "near-tie index 2: token " << report.token << " logprob " << report.logprob;
    std::cerr << " top:";
    for (const auto& entry : report.top) { std::cerr << " " << entry.token << '=' << entry.logprob; }
    std::cerr << '\n';
    if (report.top.size() >= 2) {
        const double gap = static_cast<double>(report.top[0].logprob) -
                           static_cast<double>(report.top[1].logprob);
        std::cerr << "top-1/top-2 gap: " << gap << " nats\n";
    } else {
        std::cerr << "NOTE fewer than two top entries reported\n";
    }
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN38_FLASH_NEXT_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') { return 77; }
    try {
        std::cerr << "pr36fix checks on " << artifact << " (recipe " << artifact_recipe(artifact)
                  << ")\n";
        int failures = 0;
        {
            ninfer::Engine engine(engine_options(artifact, false));
            std::vector<TokenId> line;
            check_mtp_line(engine, line);
            failures += check_prefix_reuse(engine, line);
            failures += check_stop_partial_terminal(engine, line);
            failures += check_concurrent_state(engine, fp8_mtp_golden(), {11855, 4}, line);
        }
        {
            std::vector<TokenId> line;
            failures += check_near_tie(artifact, line);
            print_tokens("near-tie mtp line", line);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " pr36fix checks (" << failures
                  << " failures)\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "pr36fix checks: " << error.what() << '\n';
        return 1;
    }
}
