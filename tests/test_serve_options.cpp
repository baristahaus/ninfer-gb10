#include "product/config/options.h"
#include "serve/translate.h"
#include "serve/openai_chat.h"
#include "product/logging/configuration.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer::serve;
using ninfer::product::ServeOptions;

ServeOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    for (auto& argument : arguments) argv.push_back(argument.data());
    return ninfer::product::parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Function>
void rejects(Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid server configuration was accepted");
}

void test_resources_and_units() {
    const auto defaults = parse({"ninfer-serve", "model.ninfer"});
    const auto& engine  = defaults.launch.engine;
    require(engine.context_cache.enabled && !engine.context_cache.extra_state_slots &&
                !engine.context_cache.host_capacity_bytes,
            "server resource defaults were resolved before model preparation");
    require(defaults.launch.generation.max_tokens == 8192 &&
                !defaults.launch.generation.sampling.seed &&
                !defaults.launch.generation.enable_thinking &&
                defaults.launch.logging.stats_interval_ms == 5000,
            "server defaults changed");
    require(!defaults.model_id_override, "public model alias is unexpectedly set by default");
    for (auto [quantity, bytes] : std::vector<std::pair<std::string, std::uint64_t>>{
             {"0B", 0},
             {"1.5 MiB", 1572864},
             {"0.00000095367431640625MiB", 1},
             {"18446744073709551615B", UINT64_MAX},
             {"17592186044415.99999904632568359375MiB", UINT64_MAX}}) {
        const auto parsed =
            parse({"ninfer-serve", "model.ninfer", "--host-cache-capacity", quantity});
        require(parsed.launch.engine.context_cache.host_capacity_bytes == bytes,
                "capacity lost exact bytes");
    }
    const auto controls = parse(
        {"ninfer-serve", "--pending-timeout", "1.25s", "model.ninfer", "--stats-interval", "0ms",
         "--media-cache-capacity", "0B", "--max-request-size", "1.5MiB", "--extra-state-slots", "0",
         "--max-concurrency", "4", "--media-preprocess-threads", "auto", "--kv-capacity", "auto"});
    require(controls.launch.engine.pending_timeout_ms == 1250 &&
                controls.launch.logging.stats_interval_ms == 0 &&
                controls.max_request_bytes == 1572864 &&
                controls.launch.engine.context_cache.extra_state_slots == 0 &&
                controls.launch.engine.media_cache_bytes == 0 &&
                controls.launch.engine.kv_capacity.mode == ninfer::KvCapacityMode::Automatic,
            "unit, zero or automatic controls changed");
    for (const auto& suffix :
         std::vector<std::vector<std::string>>{{"--host-cache-capacity", "18446744073709551616B"},
                                               {"--host-cache-capacity", "0.1B"},
                                               {"--host-cache-capacity", "-1MiB"},
                                               {"--pending-timeout", "0ms"},
                                               {"--pending-timeout", "0.1ms"},
                                               {"--pending-timeout", "4294967296ms"},
                                               {"--media-live-capacity", "0B"},
                                               {"--max-concurrency", "9"},
                                               {"--port", "65536"},
                                               {"--top-k", "21"},
                                               {"--media-preprocess-threads", "0"}}) {
        std::vector<std::string> args{"ninfer-serve", "model.ninfer"};
        args.insert(args.end(), suffix.begin(), suffix.end());
        rejects([&] { parse(args); });
    }
}

void test_anchor_options() {
    // --turn-anchors: off by default (the pre-change behaviour), a count when given, and
    // `default` restores off. The bound --max-long-anchors-per-continuation is the sibling knob.
    const auto defaults = parse({"ninfer-serve", "model.ninfer"});
    require(defaults.launch.engine.context_cache.turn_anchors == 0,
            "turn anchors are on by default");
    require(!defaults.launch.engine.context_cache.max_long_anchors_per_continuation,
            "the long-anchor bound is set before model preparation");

    const auto configured = parse({"ninfer-serve", "model.ninfer", "--turn-anchors", "3",
                                   "--max-long-anchors-per-continuation", "4"});
    require(configured.launch.engine.context_cache.turn_anchors == 3,
            "--turn-anchors did not reach the engine options");
    require(configured.launch.engine.context_cache.max_long_anchors_per_continuation == 4,
            "--max-long-anchors-per-continuation did not reach the engine options");

    const auto zero = parse({"ninfer-serve", "model.ninfer", "--turn-anchors", "0"});
    require(zero.launch.engine.context_cache.turn_anchors == 0, "--turn-anchors 0 is not off");

    const auto unset = parse({"ninfer-serve", "model.ninfer", "--turn-anchors", "default"});
    require(unset.launch.engine.context_cache.turn_anchors == 0,
            "--turn-anchors default is not off");

    rejects([] { (void)parse({"ninfer-serve", "model.ninfer", "--turn-anchors", "-1"}); });
    rejects([] { (void)parse({"ninfer-serve", "model.ninfer", "--turn-anchors"}); });
    rejects([] {
        (void)parse({"ninfer-serve", "model.ninfer", "--max-long-anchors-per-continuation",
                     "-1"});
    });
}

void test_recovered_serve_flags() {
    // These flags were documented but unreachable: the historical parser that declared them was
    // never compiled. They are wired into the live schema now, and this pins the wiring.
    const auto capacities = parse({"ninfer-serve", "model.ninfer", "--host-state-slots", "6",
                                   "--host-kv-mib", "4096", "--max-private-continuations", "4",
                                   "--max-shared-prefixes", "8", "--extra-state-slots", "2"});
    const auto& cache = capacities.launch.engine.context_cache;
    require(cache.host_state_slots == 6, "--host-state-slots did not reach the engine options");
    require(cache.host_kv_capacity_bytes == (4096ULL << 20U),
            "--host-kv-mib did not convert MiB to bytes");
    require(cache.max_private_continuations == 4,
            "--max-private-continuations did not reach the engine options");
    require(cache.max_shared_prefixes == 8, "--max-shared-prefixes did not reach the engine options");
    require(cache.extra_state_slots == 2, "--extra-state-slots did not reach the engine options");

    // The startup configuration report must carry the same values: a field with no report case
    // renders as an empty row.
    const auto view = ninfer::product::configuration_view(capacities);
    const auto reported = [&](std::string_view key) -> std::optional<std::uint64_t> {
        for (const auto& item : view.items) {
            if (item.key != key) { continue; }
            if (const auto* value = std::get_if<std::uint64_t>(&item.value)) { return *value; }
            return std::nullopt;
        }
        return std::nullopt;
    };
    require(reported("context_cache.host_state_slots") == std::uint64_t{6},
            "the startup report lost host_state_slots");
    require(reported("context_cache.host_kv_mib") == std::uint64_t{4096},
            "the startup report lost host_kv_mib");
    require(reported("context_cache.max_private_continuations") == std::uint64_t{4},
            "the startup report lost max_private_continuations");
    require(reported("context_cache.max_shared_prefixes") == std::uint64_t{8},
            "the startup report lost max_shared_prefixes");

    const auto scheduling = parse({"ninfer-serve", "model.ninfer", "--prefill-decode-share", "125",
                                   "--long-prefill-wait-ms", "500"});
    require(scheduling.launch.engine.prefill_decode_share_percent == 125,
            "--prefill-decode-share did not reach the engine options");
    require(scheduling.launch.engine.long_prefill_wait_ms == 500,
            "--long-prefill-wait-ms did not reach the engine options");

    const auto logprobs = parse({"ninfer-serve", "model.ninfer", "--token-logprobs"});
    require(logprobs.launch.engine.token_logprobs, "--token-logprobs did not reach the engine");
    const auto no_logprobs = parse({"ninfer-serve", "model.ninfer", "--no-token-logprobs"});
    require(!no_logprobs.launch.engine.token_logprobs, "--no-token-logprobs did not clear the flag");

    // Ranges and the `default` keyword.
    const auto unset = parse({"ninfer-serve", "model.ninfer", "--host-state-slots", "default",
                              "--host-kv-mib", "default", "--max-shared-prefixes", "default"});
    require(!unset.launch.engine.context_cache.host_state_slots &&
                !unset.launch.engine.context_cache.host_kv_capacity_bytes &&
                !unset.launch.engine.context_cache.max_shared_prefixes,
            "`default` did not clear a capacity");
    rejects([] { (void)parse({"ninfer-serve", "model.ninfer", "--prefill-decode-share", "1001"}); });
    rejects([] { (void)parse({"ninfer-serve", "model.ninfer", "--host-state-slots", "-1"}); });
}

void test_request_defaults() {
    const auto options =
        parse({"ninfer-serve", "model.ninfer", "--temperature", "0", "--top-p", "0.9", "--seed",
               "0", "--thinking-budget", "37", "--no-thinking", "--preserve-thinking"});
    GenerationRequest request;
    request.max_tokens = 1;
    auto semantics     = resolve_prompt_semantics(request, options);
    auto execution     = to_request_options(request, options, semantics, true);
    require(execution.execution.sampling.temperature == 0 &&
                execution.execution.sampling.seed == 0 && !execution.execution.thinking.budget &&
                execution.execution.allow_prefix_reuse,
            "startup defaults did not reach the request");
    request.sampling.temperature = 1.1;
    request.sampling.seed        = 42;
    request.enable_thinking      = true;
    request.preserve_thinking    = false;
    semantics                    = resolve_prompt_semantics(request, options);
    execution                    = to_request_options(request, options, semantics, false);
    require(execution.execution.sampling.temperature == 1.1F &&
                execution.execution.sampling.seed == 42 &&
                execution.execution.thinking.budget == 37 &&
                !execution.execution.allow_prefix_reuse && semantics.preserve_thinking == false,
            "explicit request fields did not override startup defaults");
    request.thinking_budget = 9;
    require(to_request_options(request, options, semantics, true).execution.thinking.budget == 9,
            "request thinking budget did not override the default");
    const auto large  = parse({"ninfer-serve", "model.ninfer", "--max-tokens", "4294967295"});
    const auto parsed = parse_chat_completion_request(
        RequestJson{{"model", "test"},
                    {"messages", RequestJson::array({{{"role", "user"}, {"content", "x"}}})}},
        {.default_max_tokens = large.launch.generation.max_tokens});
    require(parsed.generation.max_tokens == UINT32_MAX,
            "server output default overflowed at the protocol boundary");
}

void test_provenance() {
    for (auto args : std::vector<std::vector<std::string>>{
             {"ninfer-serve", "model.ninfer", "--api-key", "private-value"},
             {"ninfer-serve", "--api-key=private-value", "model.ninfer"}}) {
        const auto options = parse(args);
        require(options.api_key == "private-value",
                "authentication value was not retained for the server");
        const auto view = ninfer::product::configuration_view(options);
        for (const auto& argument : view.argv)
            require(argument.find("private-value") == std::string::npos, "argv leaked the API key");
        const auto rendered = ninfer::product::format_configuration(view);
        require(rendered.find("private-value") == std::string::npos &&
                    rendered.find("configured") != std::string::npos,
                "startup display leaked or lost authentication state");
    }
}

} // namespace

int main() {
    try {
        test_resources_and_units();
        test_anchor_options();
        test_recovered_serve_flags();
        test_request_defaults();
        test_provenance();
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
