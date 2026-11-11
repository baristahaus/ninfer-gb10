#include "artifact/reader.h"

#include "ninfer/engine.h"
#include "core/device.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

// Opt-in prefix-reuse state dump. The engine's maintainer capture
// (src/models/qwen3_8_flash_next/impl/runtime/diagnostics.h) writes, for one commit, every layer's
// convolution and recurrent state of the committed slot at one execution frontier. This driver
// turns it on and compares the reused run's dump against the cold run's, bitwise.
//
// It reads the SAME slot the restore writes: the hook dumps
// `physical_slot(sequence.state.read)` at the commit, and once a Fork has settled the read handle
// IS the restore's destination. A dump that read any other slot would prove nothing.
//
// COVERAGE: every component the pool owns — the GDN state (conv BF16 and recurrent FP32, every
// layer), continuation_hidden, mtp_hidden, ple and, for a masked-draft Program, the local cyclic
// K/V per layer. A component the pool does not carry is absent from both runs' dumps and is
// skipped, so the comparison never invents a difference from a component that does not exist.
constexpr const char* kPrefixStateDumpEnv = "NINFER_PREFIX_STATE_DUMP";

[[nodiscard]] std::optional<std::filesystem::path> prefix_state_dump_dir() {
    const char* directory = std::getenv(kPrefixStateDumpEnv);
    if (directory == nullptr || *directory == '\0') { return std::nullopt; }
    return std::filesystem::path(directory);
}

// The hook latches its environment at first use, so these must be set before the first generate.
// The frontier is the end of the continuation's prefill: the state the first decode round reads,
// and the one frontier both runs commit at (their post-decode frontiers differ, because the tokens
// diverge — that is the defect). The continuation is prompt + generated + separator and the golden
// fixes the generated count, so the frontier is known before anything runs.
void configure_prefix_state_dump(const std::vector<ninfer::TokenId>& prompt,
                                 const std::vector<ninfer::TokenId>& generated) {
    const auto directory = prefix_state_dump_dir();
    if (!directory) { return; }
    std::error_code error;
    std::filesystem::create_directories(*directory, error);
    const std::uint32_t frontier =
        static_cast<std::uint32_t>(prompt.size() + generated.size() + 1);
    const char* override_frontier = std::getenv("NINFER_PREFIX_STATE_DUMP_FRONTIER");
    const std::string value = override_frontier != nullptr && *override_frontier != '\0'
                                  ? std::string(override_frontier)
                                  : std::to_string(frontier);
    ::setenv("NINFER_FLASH_NEXT_STATE_DIR", directory->string().c_str(), 1);
    ::setenv("NINFER_FLASH_NEXT_STATE_FRONTIER", value.c_str(), 1);
}

[[nodiscard]] std::set<std::string> dump_stems(const std::filesystem::path& directory) {
    std::set<std::string> stems;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(directory, error)) {
        if (entry.path().extension() == ".json") { stems.insert(entry.path().stem().string()); }
    }
    return stems;
}

[[nodiscard]] std::set<std::string>
new_dump_stems(const std::filesystem::path& directory, const std::set<std::string>& before) {
    std::set<std::string> created;
    for (const auto& stem : dump_stems(directory)) {
        if (before.count(stem) == 0) { created.insert(stem); }
    }
    return created;
}

[[nodiscard]] std::vector<char> read_binary(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

struct DumpDifference {
    bool differs      = false;
    std::size_t total = 0;
    std::size_t count = 0;
    std::size_t first_element = 0;
    char first_reused[24] = {};
    char first_cold[24]   = {};
    double max_delta      = 0.0;
    std::size_t max_element = 0;
    char max_reused[24]     = {};
    char max_cold[24]       = {};
    std::string report; // sizes differ, or empty
};

// The element as a float, for the magnitude summary: BF16 is the high half of an FP32, so its bits
// shifted up are exactly its value.
[[nodiscard]] double element_value(const char* bytes, std::size_t element_bytes) {
    if (element_bytes == 2) {
        const auto raw = static_cast<std::uint16_t>(
            static_cast<unsigned char>(bytes[0]) |
            (static_cast<std::uint16_t>(static_cast<unsigned char>(bytes[1])) << 8));
        const std::uint32_t wide = static_cast<std::uint32_t>(raw) << 16;
        float value = 0.0F;
        std::memcpy(&value, &wide, sizeof(value));
        return static_cast<double>(value);
    }
    float value = 0.0F;
    std::memcpy(&value, bytes, sizeof(value));
    return static_cast<double>(value);
}

// One component file, compared element by element. It reports the FIRST difference — which may be a
// benign one-ulp kernel-path artifact — and the LARGEST, because a single ulp at element 2882 must
// not stand in for the whole comparison: the defect this instrument hunts is a state that is wrong
// in a way that moves a token, and that shows up in the magnitude, not in who differs first.
[[nodiscard]] DumpDifference compare_dump_file(const std::filesystem::path& directory,
                                               const std::string& reused_stem,
                                               const std::string& cold_stem,
                                               const std::string& suffix,
                                               std::size_t element_bytes) {
    const std::vector<char> reused = read_binary(directory / (reused_stem + suffix));
    const std::vector<char> cold   = read_binary(directory / (cold_stem + suffix));
    if (reused.size() != cold.size()) {
        return {.differs = true,
                .report  = suffix + " sizes differ: reused " + std::to_string(reused.size()) +
                          " bytes, cold " + std::to_string(cold.size()) + " bytes"};
    }
    DumpDifference result;
    result.total = reused.size() / element_bytes;
    for (std::size_t element = 0; element < result.total; ++element) {
        const std::size_t offset = element * element_bytes;
        if (std::memcmp(reused.data() + offset, cold.data() + offset, element_bytes) == 0) {
            continue;
        }
        const double reused_value = element_value(reused.data() + offset, element_bytes);
        const double cold_value   = element_value(cold.data() + offset, element_bytes);
        const double delta        = std::abs(reused_value - cold_value);
        if (result.count == 0) {
            result.first_element = element;
            std::snprintf(result.first_reused, sizeof(result.first_reused), "%.6g", reused_value);
            std::snprintf(result.first_cold, sizeof(result.first_cold), "%.6g", cold_value);
        }
        ++result.count;
        if (delta > result.max_delta || result.count == 1) {
            result.max_delta   = delta;
            result.max_element = element;
            std::snprintf(result.max_reused, sizeof(result.max_reused), "%.6g", reused_value);
            std::snprintf(result.max_cold, sizeof(result.max_cold), "%.6g", cold_value);
        }
    }
    result.differs = result.count != 0;
    return result;
}

void compare_prefix_state_dumps(const std::filesystem::path& directory,
                                const std::set<std::string>& reused_stems,
                                const std::set<std::string>& cold_stems) {
    if (reused_stems.empty() || cold_stems.empty()) {
        std::cerr << "prefix state dump: no dump at the requested frontier (reused "
                  << reused_stems.size() << ", cold " << cold_stems.size()
                  << "); the commit did not reach it\n";
        return;
    }
    const std::string reused_stem = *reused_stems.begin();
    const std::string cold_stem   = *cold_stems.begin();
    nlohmann::json reused_meta;
    nlohmann::json cold_meta;
    {
        std::ifstream file(directory / (reused_stem + ".json"));
        file >> reused_meta;
    }
    {
        std::ifstream file(directory / (cold_stem + ".json"));
        file >> cold_meta;
    }
    const auto reused_slot     = reused_meta.value("physical_slot", -1);
    const auto cold_slot       = cold_meta.value("physical_slot", -1);
    const auto reused_frontier = reused_meta.value("frontier", -1U);
    const auto cold_frontier   = cold_meta.value("frontier", -1U);
    const auto layers          = reused_meta.value("layers", 0U);
    const auto conv_channels   = reused_meta.value("conv_channels", 0);
    const auto conv_width      = reused_meta.value("conv_width", 0);
    std::cerr << "prefix state dump: reused slot " << reused_slot << " frontier " << reused_frontier
              << ", cold slot " << cold_slot << " frontier " << cold_frontier << ", layers "
              << layers << ", conv " << conv_channels << "x" << conv_width << '\n';
    if (reused_frontier != cold_frontier) {
        std::cerr << "prefix state dump: the two runs did not commit at the same frontier; the "
                     "comparison would prove nothing\n";
        return;
    }
    // Two different sequences normally occupy different physical slots, and the comparison is by
    // content at one frontier, so a mismatch is reported rather than silently compared — but a
    // caller who knows the runs are distinct sequences can say so explicitly.
    if (reused_slot != cold_slot) {
        const char* policy = std::getenv("NINFER_PREFIX_STATE_DUMP_SLOT_POLICY");
        const bool allow   = policy != nullptr && std::strcmp(policy, "any") == 0;
        std::cerr << "prefix state dump: the runs committed different physical slots (reused "
                  << reused_slot << ", cold " << cold_slot << ")";
        if (!allow) {
            std::cerr << "; refusing to compare (set NINFER_PREFIX_STATE_DUMP_SLOT_POLICY=any to "
                         "compare by content at the same frontier)\n";
            return;
        }
        std::cerr << "; comparing by content at the same frontier\n";
    }
    std::size_t differing_components = 0;
    std::size_t differing_elements   = 0;
    std::size_t compared_elements    = 0;
    std::uint32_t differing_layers   = 0;
    const auto report = [&](const std::string& label, const DumpDifference& difference) {
        if (!difference.differs) { return; }
        ++differing_components;
        differing_elements += difference.count;
        compared_elements += difference.total;
        std::cerr << "prefix state dump: " << label << ": " << difference.count << " of "
                  << difference.total << " elements differ; first element "
                  << difference.first_element << " reused " << difference.first_reused << " cold "
                  << difference.first_cold << "; max |delta| " << difference.max_delta
                  << " at element " << difference.max_element << " (reused "
                  << difference.max_reused << ", cold " << difference.max_cold << ")";
        if (conv_width > 0 && label.rfind("conv layer", 0) == 0) {
            std::cerr << "; that element is channel " << difference.max_element / conv_width
                      << ", tap " << difference.max_element % conv_width;
        }
        std::cerr << '\n';
    };
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const DumpDifference conv = compare_dump_file(
            directory, reused_stem, cold_stem, ".conv." + std::to_string(layer) + ".bf16", 2);
        if (conv.differs) { ++differing_layers; }
        report("conv layer " + std::to_string(layer), conv);
        const DumpDifference recurrent = compare_dump_file(
            directory, reused_stem, cold_stem, ".recurrent." + std::to_string(layer) + ".fp32", 4);
        if (recurrent.differs) { ++differing_layers; }
        report("recurrent layer " + std::to_string(layer), recurrent);
    }
    const std::vector<std::string> reused_components =
        reused_meta.value("components", std::vector<std::string>{});
    const std::vector<std::string> cold_components =
        cold_meta.value("components", std::vector<std::string>{});
    const auto has_component = [](const std::vector<std::string>& list, const char* name) {
        return std::find(list.begin(), list.end(), name) != list.end();
    };
    const auto compare_single = [&](const char* component, const std::string& suffix) {
        if (!has_component(reused_components, component) ||
            !has_component(cold_components, component)) {
            std::cerr << "prefix state dump: " << component << " is absent from a run; skipped\n";
            return;
        }
        report(component, compare_dump_file(directory, reused_stem, cold_stem, suffix, 2));
    };
    compare_single("continuation_hidden", ".continuation_hidden.bf16");
    compare_single("mtp_hidden", ".mtp_hidden.bf16");
    compare_single("ple", ".ple.bf16");
    const auto dflash_layers = reused_meta.value("dflash_local_layers", 0U);
    for (std::uint32_t layer = 0; layer < dflash_layers; ++layer) {
        const std::string suffix = std::to_string(layer) + ".bf16";
        compare_single("dflash_local_k", ".dflash_local_k." + suffix);
        compare_single("dflash_local_v", ".dflash_local_v." + suffix);
    }
    std::cerr << "prefix state dump: " << differing_components << " component/layer files differ, "
              << differing_layers << " layer components, " << differing_elements << " of "
              << compared_elements << " compared elements\n";
    if (differing_components == 0) {
        std::cerr << "prefix state dump: every component of the committed slot is bitwise identical\n";
    }
}


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
    // A prefix-state dump run performs synchronous copies at its commits, so it must not capture
    // graphs; the dump is opt-in, so this changes nothing when it is unset.
    if (prefix_state_dump_dir()) { options.use_cuda_graph = false; }
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

// The MXFP8 QAD recipe's MTP verify route resolves the documented 27891/5435 near-tie at
// generated index 2 the other way (the MXFP8 T=2 MMA reduction order against the FP8-source
// weights); its verify stream then follows its own confident continuation.
bool qad_recipe(const std::string& recipe) {
    return recipe == "qwen3_8_flash_next_125b_a6b_qad_native-v1";
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
    // Re-recorded 2026-10: the FP8 route's exact tie at generated index 2 (5435 and 27891 both
    // at logprob -1.06157 in the recorded environment) no longer reproduces - the vendor GEMM
    // schedule drift moved that near-tie onto a confident 5435 choice on both the master
    // generation (4c2fc2f9) and this port, which agree token-for-token through 24 outputs. The
    // golden follows the test's own re-record protocol; the step 7 perplexity gate remains the
    // semantic quality check.
    //
    // The golden is per lane: sm_120a resolves that same documented near-tie to 5435, and the GB10
    // (sm_121a) resolves it to 27891 with a different continuation from index 2 on. Recorded
    // 2026-11-10 on twofour against the fp8_mtp artifact; each chain is authoritative for its own
    // compute capability only, and a new lane records its own.
    static const std::vector<ninfer::TokenId> golden_sm120a{
        29108, 4009, 5435, 660, 7736, 314, 279, 9155, 19142, 11, 694, 22602};
    static const std::vector<ninfer::TokenId> golden_sm121a{
        29108, 4009, 27891, 8964, 579, 16078, 321, 1100, 9872, 303, 660, 17425};
    static const int capability = ninfer::DeviceContext(0).compute_capability();
    return capability == 121 ? golden_sm121a : golden_sm120a;
}

// The QAD recipe's MTP verify golden: the 27891/5435 near-tie resolves to 5435 under the
// MXFP8 verify route (see qad_recipe); from there the stream follows the model's confident
// continuation. The plain decode path still checks the canonical golden exactly.
const std::vector<ninfer::TokenId>& qad_mtp_output() {
    static const std::vector<ninfer::TokenId> golden{
        29108, 4009, 5435, 660, 7736, 314, 279, 9155, 19142, 11, 694, 948};
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
    const auto dump_directory = prefix_state_dump_dir();
    std::set<std::string> dump_before_reused;
    if (dump_directory) { dump_before_reused = dump_stems(*dump_directory); }
    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare_tokens(continuation), greedy_options(2, true));
    std::set<std::string> reused_dump_stems;
    std::set<std::string> dump_before_cold;
    if (dump_directory) {
        reused_dump_stems = new_dump_stems(*dump_directory, dump_before_reused);
        dump_before_cold  = dump_stems(*dump_directory);
    }
    const ninfer::GenerationResult cold =
        engine.generate(engine.prepare_tokens(continuation), greedy_options(2, false));
    if (dump_directory) {
        compare_prefix_state_dumps(*dump_directory, reused_dump_stems,
                                   new_dump_stems(*dump_directory, dump_before_cold));
    }
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

    // The two-token comparison above catches a divergence at the first decode round only. A longer
    // comparison is the assertion that actually matters for prefix reuse: the reused round must
    // produce the same token stream as the cold round for the same prompt, token for token, over
    // enough rounds that a state restored slightly wrong drifts visibly. The hit count is not
    // asserted here — earlier runs in this test have captured further checkpoints, so the reused
    // length is allowed to grow — but the reused run must reuse something, or the comparison would
    // silently degrade into cold-versus-cold.
    const ninfer::GenerationResult reused_long =
        engine.generate(engine.prepare_tokens(continuation), greedy_options(16, true));
    const ninfer::GenerationResult cold_long =
        engine.generate(engine.prepare_tokens(continuation), greedy_options(16, false));
    if (reused_long.reused_prompt_tokens == 0 || cold_long.reused_prompt_tokens != 0 ||
        reused_long.generated_token_ids != cold_long.generated_token_ids) {
        std::cerr << "Flash-Next long prefix reuse diverges: reused="
                  << reused_long.reused_prompt_tokens << " cold=" << cold_long.reused_prompt_tokens
                  << '\n';
        print_tokens("reused", reused_long.generated_token_ids);
        print_tokens("cold", cold_long.generated_token_ids);
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
    // reversed admission order to reuse both physical lanes and recurrent state slots.
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
        if (root.generated_token_ids != expected_prefix ||
            resumed.generated_token_ids != cold.generated_token_ids) {
            std::cerr << "Flash-Next concurrent lane reuse changed the canonical fixture"
                      << (reverse ? " (reversed admission)" : "") << '\n';
            print_tokens("root", root.generated_token_ids);
            print_tokens("resumed", resumed.generated_token_ids);
            print_tokens("cold", cold.generated_token_ids);
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

// A non-thinking chat prompt whose user turn repeats one 15-token sentence.
std::vector<ninfer::TokenId> repeated_sentence_prompt(int repeats) {
    std::vector<ninfer::TokenId> prompt{248045, 846, 198};
    const std::vector<ninfer::TokenId> sentence{814,   20139, 303,  2250, 2716, 22157, 3069, 279,
                                                12515, 7701,  6105, 2261, 279,  1834,  13};
    for (int repeat = 0; repeat < repeats; ++repeat) {
        prompt.insert(prompt.end(), sentence.begin(), sentence.end());
    }
    prompt.insert(prompt.end(), {248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271});
    return prompt;
}

int exercise_prefill_yield(ninfer::Engine& engine) {
    // A prompt of two 256-token chunks parks after its first chunk for a short request and then
    // resumes. Neither request decodes beside the other (the short one generates one token at its
    // prefill's end), so both must match their solo greedy outputs exactly; a mismatch means the
    // interleaved prefill disturbed the parked sequence's state.
    const auto long_prompt = repeated_sentence_prompt(26);
    const auto solo_long =
        engine.generate(engine.prepare_tokens(long_prompt), greedy_options(8, false));
    const auto solo_short =
        engine.generate(engine.prepare_tokens(canonical_prompt()), greedy_options(1, false));
    const auto before = engine.runtime_stats();
    auto long_handle  = engine.submit(engine.prepare_tokens(long_prompt), greedy_options(8, false));
    auto short_handle =
        engine.submit(engine.prepare_tokens(canonical_prompt()), greedy_options(1, false));
    const auto yielded_long  = long_handle.wait();
    const auto yielded_short = short_handle.wait();
    const auto after         = engine.runtime_stats();
    if (after.admission_prefill_yields == before.admission_prefill_yields ||
        yielded_long.generated_token_ids != solo_long.generated_token_ids ||
        yielded_short.generated_token_ids != solo_short.generated_token_ids) {
        std::cerr << "Flash-Next prefill yield changed an output or never parked the long prefill"
                  << " (yields " << after.admission_prefill_yields - before.admission_prefill_yields
                  << ")\n";
        print_tokens("yielded long", yielded_long.generated_token_ids);
        print_tokens("solo long", solo_long.generated_token_ids);
        print_tokens("yielded short", yielded_short.generated_token_ids);
        print_tokens("solo short", solo_short.generated_token_ids);
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

// Graph preparation at the serving shape: eight lanes, K=3 (32-row verify graphs) and a 73728-token
// context, whose graph profiles include the sparse-QSA envelopes that a 512-token context never
// reaches. This configuration crashed at startup (ninfer-gb10-dnf); eight concurrent requests
// then replay the widest graphs. Lanes of one batch are not compared with each other: batch
// composition changes kernel shapes and with them the rounding, so greedy outputs may diverge.
int exercise_serving_shape(const char* artifact) {
    ninfer::EngineOptions options                   = engine_options(artifact);
    options.max_context                             = 73728;
    options.kv_capacity                             = ninfer::KvCapacityPolicy::automatic();
    options.prefill_chunk                           = 4096;
    options.max_concurrency                         = ninfer::kMaximumConcurrency;
    options.max_pending_requests                    = ninfer::kMaximumConcurrency;
    options.context_cache.max_private_continuations = ninfer::kMaximumConcurrency;
    options.enable_vision                           = false;
    ninfer::Engine engine(std::move(options));
    std::vector<ninfer::GenerationHandle> handles;
    for (std::uint32_t lane = 0; lane < ninfer::kMaximumConcurrency; ++lane) {
        handles.push_back(
            engine.submit(engine.prepare_tokens(canonical_prompt()), greedy_options(12, false)));
    }
    for (auto& handle : handles) {
        const ninfer::GenerationResult result = handle.wait();
        if (result.generated_token_ids.size() != 12 ||
            result.speculative.backend != ninfer::SpeculativeBackend::Mtp) {
            std::cerr << "Flash-Next serving-shape request did not complete through MTP\n";
            print_tokens("lane", result.generated_token_ids);
            return 1;
        }
    }
    return 0;
}

// Multi-turn shared-prefix reuse under pressure, exercising the demote-then-restore compound's
// execution path end to end; see the comment inside the exercise for the geometry and for the
// relief-ordering caveat that decides when the own-source compound is selected.
int exercise_own_source_demote_restore(const char* artifact) {
    // Multi-turn shared-prefix reuse under pressure. The pressured engine runs two lanes with no
    // spare Device StateImage slots: the first turn's shared-prefix capture takes the second
    // slot, so the follow-up (submitted at once, diverging after the shared boundary so the first
    // turn's private endpoint can never serve it) is admitted while the first turn still runs.
    // A shared source is retained, never consumed, so the follow-up's materialization must fork
    // it, and its fork restores the source's replicas through the Host-to-Device fallback. The
    // Host quota is set explicitly because a DGX Spark host quota resolves to zero, which leaves
    // no demotion destination at all; with it, the shared checkpoint state moves to Host and the
    // shared prefix stays catalogued and reusable.
    // Per the documented relief ordering, ordinary (non-reused) owners are relieved before reused
    // ones, so the own-source demote-then-restore compound is selected exactly when no ordinary
    // third-party relief exists; the forced own-source-only geometry is tracked separately.
    const auto options_for = [](const char* artifact, std::uint32_t device_state_slots) {
        ninfer::EngineOptions options                = engine_options(artifact);
        options.max_context                          = 8192;
        options.kv_capacity                          = ninfer::KvCapacityPolicy::explicit_capacity(8192);
        options.max_concurrency                      = 2;
        options.max_pending_requests                 = 2;
        options.enable_vision                        = false;
        // Two lanes and no spare slot: the two lane states plus the shared capture contend for
        // two Device StateImage slots, so the follow-up's fork admission always plans under
        // pressure, independent of capture timing.
        options.context_cache.device_state_slots     = device_state_slots;
        options.context_cache.host_state_slots       = 4;
        options.context_cache.host_kv_capacity_bytes = 1ULL << 30;
        // Defer the follow-up's long prefill so its admission is planned while the first turn
        // still runs, deterministic regardless of prefill speed.
        options.long_prefill_wait_ms                 = 200;
        return options;
    };

    const std::string sentence = "The quick brown fox jumps over the lazy dog. ";
    std::string body;
    for (int repeat = 0; repeat < 120; ++repeat) { body += sentence; }
    // The follow-up's second message differs, so it matches the shared marker boundary exactly
    // and can never reuse the first turn's private endpoint.
    std::string divergent;
    for (int repeat = 0; repeat < 120; ++repeat) {
        divergent += "Pack my box with five dozen liquor jugs. ";
    }
    const auto text_message = [](std::string text, ninfer::ChatRole role) {
        ninfer::ChatMessage message;
        message.role = role;
        message.parts.push_back(
            ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text,
                                .text = std::move(text), .media = {}});
        return message;
    };
    const auto shared_marker = [] {
        ninfer::PromptCacheMarker marker;
        marker.after_message_count = 1;
        return marker;
    };
    // Two user messages put the shared marker boundary mid-prompt: the shared capture publishes
    // while the first turn is still prefilling.
    const auto turn_input = [&](bool with_question) {
        ninfer::PromptInput input;
        input.messages.push_back(text_message(body, ninfer::ChatRole::User));
        input.messages.push_back(text_message(with_question ? divergent : body,
                                              ninfer::ChatRole::User));
        if (with_question) {
            input.messages.push_back(text_message(
                "Answer with one short sentence about the text above.",
                ninfer::ChatRole::User));
        }
        input.options.enable_thinking           = false;
        input.context_cache.markers.push_back(shared_marker());
        return input;
    };

    ninfer::GenerationResult first;
    ninfer::GenerationResult second;
    {
        ninfer::Engine engine(options_for(artifact, 0));
        auto first_handle =
            engine.submit(engine.prepare(turn_input(false)), greedy_options(64, true));
        auto second_handle =
            engine.submit(engine.prepare(turn_input(true)), greedy_options(8, true));
        first  = first_handle.wait();
        second = second_handle.wait();
        if (first.generated_token_ids.empty() || first.content.empty()) {
            std::cerr << "Flash-Next compound fixture conversation turn did not generate\n";
            return 1;
        }
        const auto after_follow_up = engine.runtime_stats();
        if (second.prefix_reuse_path != ninfer::PrefixReusePath::SharedStablePrefix ||
            second.reused_prompt_tokens < 1024 ||
            after_follow_up.state_d2h_count == 0 ||
            after_follow_up.host_state_occupied_slots == 0 ||
            after_follow_up.pressure_shared_owners_evicted != 0) {
            std::cerr << "Flash-Next follow-up under pressure did not keep its shared source "
                         "across Host (reused "
                      << second.reused_prompt_tokens << ", path "
                      << static_cast<int>(second.prefix_reuse_path) << ", d2h "
                      << after_follow_up.state_d2h_count << ", host states "
                      << after_follow_up.host_state_occupied_slots << ", shared evicted "
                      << after_follow_up.pressure_shared_owners_evicted << ")\n";
            return 1;
        }
        std::cout << "compound follow-up: reused " << second.reused_prompt_tokens << ", path "
                  << static_cast<int>(second.prefix_reuse_path) << ", d2h "
                  << after_follow_up.state_d2h_count << ", private degraded "
                  << after_follow_up.pressure_private_owners_degraded << ", shared degraded "
                  << after_follow_up.pressure_shared_owners_degraded << ", searches "
                  << after_follow_up.pressure_searches << ", device states "
                  << after_follow_up.device_state_occupied_slots << ", host states "
                  << after_follow_up.host_state_occupied_slots << '\n';
    }

    // The pressured admission must produce the same continuation as the standard fork path
    // without pressure: an engine with spare Device state slots forks the same shared source on
    // Device, and the greedy outputs must match token for token.
    ninfer::Engine roomy(options_for(artifact, 4));
    auto roomy_first = roomy.submit(roomy.prepare(turn_input(false)), greedy_options(64, true));
    auto roomy_second_handle =
        roomy.submit(roomy.prepare(turn_input(true)), greedy_options(8, true));
    const auto roomy_first_result = roomy_first.wait();
    const auto roomy_second       = roomy_second_handle.wait();
    if (roomy_first_result.generated_token_ids != first.generated_token_ids ||
        roomy_second.generated_token_ids != second.generated_token_ids) {
        std::cerr << "Flash-Next pressured shared-source follow-up differs from the unpressured "
                     "fork (reused " << roomy_second.reused_prompt_tokens << ", path "
                  << static_cast<int>(roomy_second.prefix_reuse_path) << ")\n";
        print_tokens("pressured", second.generated_token_ids);
        print_tokens("roomy", roomy_second.generated_token_ids);
        return 1;
    }
    return 0;
}

// Forced own-source pressure geometry, in two deterministic phases.
//
// Two engine facts (recorded while building this exercise against the demote-then-restore
// planner) keep the full conjunction - the shared source demoted, no ordinary owner relieved,
// and the follow-up forking - out of reach of one admission:
//   1. Every completed turn catalogues an endpoint whose Device StateImages relieve more
//      cheaply than the candidate's own shared source (relieving the own source also strips
//      the restore credit from the assessed fork work). The endpoint disappears from the
//      victim domain only while a later turn of the same conversation is consuming it.
//   2. With the endpoint consumed and the shared source the only victim, the search still
//      selects the root candidate with the same single-victim demote: the composed fork whose
//      own pressure demotes its source never beats the root plan. The demote-then-restore
//      therefore executes across admissions - a source demoted to Host by one admission is
//      restored by a later fork planned around a Host-resident source.
// Phase 1 pins the forced victim domain: the middle turn consumes the capture endpoint, so
// the follow-up's admission has exactly one pressure victim - the shared source - and the
// selected plan demotes it to Host, never evicts it, drops no checkpoint and relieves no
// private owner. Phase 2 pins the restore execution: with the shared source already
// Host-resident, the follow-up forks it through the Host-to-Device fallback and matches the
// unpressured fork token for token.
int exercise_forced_own_source_compound(const char* artifact) {
    const std::string sentence = "The quick brown fox jumps over the lazy dog. ";
    std::string body;
    for (int repeat = 0; repeat < 120; ++repeat) { body += sentence; }
    std::string divergent;
    for (int repeat = 0; repeat < 120; ++repeat) {
        divergent += "Pack my box with five dozen liquor jugs. ";
    }
    const auto text_message = [](std::string text, ninfer::ChatRole role) {
        ninfer::ChatMessage message;
        message.role = role;
        message.parts.push_back(
            ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text,
                                .text = std::move(text), .media = {}});
        return message;
    };
    const auto shared_marker = [] {
        ninfer::PromptCacheMarker marker;
        marker.after_message_count = 1;
        return marker;
    };
    const auto capture_turn = [&]() {
        ninfer::PromptInput input;
        input.messages.push_back(text_message(body, ninfer::ChatRole::User));
        input.messages.push_back(text_message(body, ninfer::ChatRole::User));
        input.options.enable_thinking = false;
        input.context_cache.markers.push_back(shared_marker());
        return input;
    };
    const auto follow_up = [&]() {
        ninfer::PromptInput input;
        input.messages.push_back(text_message(body, ninfer::ChatRole::User));
        input.messages.push_back(text_message(divergent, ninfer::ChatRole::User));
        input.messages.push_back(text_message(
            "Answer with one short sentence about the text above.", ninfer::ChatRole::User));
        input.options.enable_thinking = false;
        input.context_cache.markers.push_back(shared_marker());
        return input;
    };
    // Continues the capture turn's own conversation through its assistant response, so the
    // admission consumes the capture endpoint in place instead of forking a new StateImage and
    // instead of touching the shared source.
    const auto consuming_middle_turn = [&](const std::string& capture_content) {
        ninfer::PromptInput input;
        input.messages.push_back(text_message(body, ninfer::ChatRole::User));
        input.messages.push_back(text_message(body, ninfer::ChatRole::User));
        ninfer::ChatMessage response;
        response.role = ninfer::ChatRole::Assistant;
        response.parts.push_back(
            ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text,
                                .text = capture_content, .media = {}});
        input.messages.push_back(std::move(response));
        input.messages.push_back(text_message("Summarize in five words.", ninfer::ChatRole::User));
        input.options.enable_thinking = false;
        return input;
    };

    // Phase 1: the forced victim domain. The middle turn consumes the capture endpoint while
    // it runs, so the follow-up's admission - planned 200 ms after its submission, well inside
    // the middle turn's multi-second run - sees exactly one pressure victim: the shared source.
    ninfer::GenerationResult capture;
    ninfer::GenerationResult middle_result;
    ninfer::GenerationResult follow_up_result;
    {
        ninfer::EngineOptions options                    = engine_options(artifact);
        options.max_context                              = 8192;
        options.kv_capacity                              = ninfer::KvCapacityPolicy::explicit_capacity(8192);
        options.max_concurrency                          = 2;
        options.max_pending_requests                     = 2;
        // Three Device state slots in total: the capture endpoint, its turn-closure image and
        // the shared source fill the pool exactly; the consuming middle turn fits without
        // pressure, and the follow-up's admission is left with a one-slot deficit.
        options.context_cache.device_state_slots         = 1;
        options.enable_vision                            = false;
        options.context_cache.host_state_slots           = 4;
        options.context_cache.host_kv_capacity_bytes     = 2ULL << 30;
        options.context_cache.max_private_continuations  = 4;
        options.context_cache.max_shared_prefixes        = 4;
        // Release the follow-up's long prefill 200 ms after submission so its admission is
        // planned while the middle turn still runs and still consumes the capture endpoint.
        options.long_prefill_wait_ms                     = 200;
        ninfer::Engine engine(std::move(options));
        capture = engine.submit(engine.prepare(capture_turn()), greedy_options(64, true)).wait();
        if (capture.generated_token_ids.empty() || capture.content.empty()) {
            std::cerr << "Flash-Next forced-compound capture turn did not generate\n";
            return 1;
        }
        auto middle_handle =
            engine.submit(engine.prepare(consuming_middle_turn(capture.content)),
                          greedy_options(64, true));
        // Read the middle turn's admission bookkeeping before the follow-up is submitted: the
        // follow-up's own admission pressure lands while the middle turn still runs and would
        // otherwise pollute this snapshot.
        const auto after_middle = engine.runtime_stats();
        auto follow_up_handle =
            engine.submit(engine.prepare(follow_up()), greedy_options(8, true));
        middle_result = middle_handle.wait();
        if (middle_result.prefix_reuse_path != ninfer::PrefixReusePath::PrivateEndpoint ||
            middle_result.reused_prompt_tokens < 1024) {
            std::cerr << "Flash-Next forced-compound middle turn did not consume the capture "
                         "endpoint (reused " << middle_result.reused_prompt_tokens << ", path "
                      << static_cast<int>(middle_result.prefix_reuse_path) << ")\n";
            return 1;
        }
        follow_up_result = follow_up_handle.wait();
        if (follow_up_result.content.empty()) {
            std::cerr << "Flash-Next forced-compound follow-up did not generate\n";
            return 1;
        }
        const auto after = engine.runtime_stats();
        if (after.pressure_shared_owners_degraded == 0 ||
            after.pressure_shared_owners_evicted != 0 ||
            after.pressure_checkpoints_dropped != 0 ||
            after.pressure_private_owners_degraded != 0 ||
            after.pressure_private_owners_evicted != 0) {
            std::cerr << "Flash-Next forced-compound victim domain was not exactly the shared "
                         "source (shared degraded " << after.pressure_shared_owners_degraded
                      << ", shared evicted " << after.pressure_shared_owners_evicted
                      << ", private degraded " << after.pressure_private_owners_degraded
                      << ", private evicted " << after.pressure_private_owners_evicted
                      << ", dropped " << after.pressure_checkpoints_dropped << ")\n";
            return 1;
        }
        std::cout << "forced compound: middle consumed " << middle_result.reused_prompt_tokens
                  << ", shared degraded " << after.pressure_shared_owners_degraded
                  << ", private relieved 0, dropped 0\n";
    }

    // Phase 2: the restore execution. One Device state slot in total leaves the shared source
    // on Host after the capture (it never wins a Device slot from the running lane), so the
    // follow-up's fork is planned around a Host-resident source and restores it through the
    // Host-to-Device fallback. The capture endpoint is the only Device checkpoint owner and
    // the engine must relieve it; the plan demotes (retains) it rather than evicting it, and
    // no checkpoint is destroyed.
    {
        const auto options_for = [&](std::uint32_t device_state_slots) {
            ninfer::EngineOptions options                    = engine_options(artifact);
            options.max_context                              = 8192;
            options.kv_capacity                              = ninfer::KvCapacityPolicy::explicit_capacity(8192);
            options.max_concurrency                          = 1;
            options.max_pending_requests                     = 2;
            options.context_cache.device_state_slots         = device_state_slots;
            options.enable_vision                            = false;
            options.context_cache.host_state_slots           = 4;
            options.context_cache.host_kv_capacity_bytes     = 2ULL << 30;
            options.context_cache.max_private_continuations  = 4;
            options.context_cache.max_shared_prefixes        = 4;
            return options;
        };
        ninfer::GenerationResult restore_capture;
        ninfer::GenerationResult restore_follow_up;
        {
            ninfer::Engine engine(options_for(0));
            restore_capture =
                engine.submit(engine.prepare(capture_turn()), greedy_options(64, true)).wait();
            if (restore_capture.generated_token_ids.empty()) {
                std::cerr << "Flash-Next forced-compound restore capture did not generate\n";
                return 1;
            }
            const auto after_capture = engine.runtime_stats();
            if (after_capture.host_state_occupied_slots == 0 ||
                after_capture.pressure_shared_owners_degraded != 0 ||
                after_capture.pressure_shared_owners_evicted != 0) {
                std::cerr << "Flash-Next forced-compound restore capture did not park the "
                             "shared source on Host (host states "
                          << after_capture.host_state_occupied_slots << ")\n";
                return 1;
            }
            restore_follow_up =
                engine.submit(engine.prepare(follow_up()), greedy_options(8, true)).wait();
            if (restore_follow_up.prefix_reuse_path !=
                    ninfer::PrefixReusePath::SharedStablePrefix ||
                restore_follow_up.reused_prompt_tokens < 1024) {
                std::cerr << "Flash-Next forced-compound restore follow-up did not fork the "
                             "shared source (reused "
                          << restore_follow_up.reused_prompt_tokens << ", path "
                          << static_cast<int>(restore_follow_up.prefix_reuse_path) << ")\n";
                return 1;
            }
            const auto after = engine.runtime_stats();
            if (after.state_h2d_count == 0 ||
                after.pressure_shared_owners_degraded != 0 ||
                after.pressure_shared_owners_evicted != 0 ||
                after.pressure_checkpoints_dropped != 0 ||
                after.pressure_private_owners_evicted != 0) {
                std::cerr << "Flash-Next forced-compound restore follow-up did not restore the "
                             "shared source through Host (h2d " << after.state_h2d_count
                          << ", shared degraded " << after.pressure_shared_owners_degraded
                          << ", shared evicted " << after.pressure_shared_owners_evicted
                          << ", private evicted " << after.pressure_private_owners_evicted
                          << ", dropped " << after.pressure_checkpoints_dropped << ")\n";
                return 1;
            }
            std::cout << "forced compound: restore fork reused "
                      << restore_follow_up.reused_prompt_tokens << ", h2d "
                      << after.state_h2d_count << ", private demoted "
                      << after.pressure_private_owners_degraded << ", dropped 0\n";
        }

        // The restored fork must match an unpressured fork token for token.
        ninfer::Engine roomy(options_for(4));
        auto roomy_capture = roomy.submit(roomy.prepare(capture_turn()), greedy_options(64, true));
        auto roomy_follow_up = roomy.submit(roomy.prepare(follow_up()), greedy_options(8, true));
        const auto roomy_capture_result   = roomy_capture.wait();
        const auto roomy_follow_up_result = roomy_follow_up.wait();
        if (roomy_capture_result.generated_token_ids != restore_capture.generated_token_ids ||
            roomy_follow_up_result.generated_token_ids !=
                restore_follow_up.generated_token_ids) {
            std::cerr << "Flash-Next forced-compound restored fork differs from the unpressured "
                         "fork\n";
            print_tokens("pressured", restore_follow_up.generated_token_ids);
            print_tokens("roomy", roomy_follow_up_result.generated_token_ids);
            return 1;
        }
    }
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN38_FLASH_NEXT_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cerr << "SKIP: NINFER_QWEN38_FLASH_NEXT_WEIGHTS is not set; the 125B real-engine test did not run. Set it to the 125B .ninfer file\n";
        return 77;
    }
    const std::string recipe       = artifact_recipe(artifact);
    const bool qad                 = qad_recipe(recipe);
    const auto& expected_prefix    = qad ? qad_mtp_output() : canonical_output();
    configure_prefix_state_dump(canonical_prompt(), expected_prefix);
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
            if (exercise_prefill_yield(engine) != 0) { return 1; }
            if (exercise_vision(engine) != 0) { return 1; }
            // The Flash-Next lane has no pause, preemption, or replay machinery (OOM backoff
            // is admission-level), so these counters are structurally zero for the whole
            // battery; a nonzero value means a lifecycle path landed without its accounting
            // contract (docs/serving.md documents the consumer-facing zeros).
            const ninfer::RuntimeStats lane = engine.runtime_stats();
            if (lane.preemptions != 0 || lane.snapshot_restores != 0 ||
                lane.replay_restores != 0 || lane.replayed_tokens != 0 ||
                lane.paused_requests != 0 || lane.replaying_requests != 0) {
                std::cerr << "Flash-Next lane reports nonzero pause/preempt/replay counters:"
                          << " preemptions=" << lane.preemptions
                          << " snapshot_restores=" << lane.snapshot_restores
                          << " replay_restores=" << lane.replay_restores
                          << " replayed_tokens=" << lane.replayed_tokens
                          << " paused=" << lane.paused_requests
                          << " replaying=" << lane.replaying_requests << '\n';
                return 1;
            }
            // The speculative exercises ran with draft_tokens 3; the Engine-cumulative
            // per-position vectors must be grown to that window by now, never beyond it.
            if (lane.speculative_attempted_per_position.size() != 3 ||
                lane.speculative_accepted_per_position.size() != 3) {
                std::cerr << "Flash-Next cumulative per-position vectors are not draft-window"
                          << " sized: attempted="
                          << lane.speculative_attempted_per_position.size()
                          << " accepted=" << lane.speculative_accepted_per_position.size() << '\n';
                return 1;
            }
        }
        if (exercise_ordinary_greedy(artifact, canonical_output()) != 0) { return 1; }
        if (exercise_serving_shape(artifact) != 0) { return 1; }
        if (exercise_own_source_demote_restore(artifact) != 0) { return 1; }
        if (exercise_forced_own_source_compound(artifact) != 0) { return 1; }
        std::cout << "OK Qwen3.8 Flash Next real Engine\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Qwen3.8 Flash Next real Engine: " << error.what() << '\n';
        return 1;
    }
}
