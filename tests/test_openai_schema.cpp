#include "serve/generation_service.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"
#include "serve/translate.h"
#include "serve/json_output.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Json = ninfer::serve::RequestJson;
using namespace ninfer::serve;
using ninfer::ChatRole;

int check(bool condition, const std::string& label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

template <typename Function>
ApiError api_error(Function&& function) {
    try {
        function();
    } catch (const ApiException& exception) { return exception.error(); }
    return ApiError{.status = 0, .message = "no exception"};
}

template <typename Function>
bool throws_logic(Function&& function) {
    try {
        function();
    } catch (const std::logic_error&) { return true; }
    return false;
}

RequestLimits limits() { return RequestLimits{.default_max_tokens = 512}; }

Json base_request() {
    return Json{{"model", "qwen"},
                {"messages", Json::array({Json{{"role", "user"}, {"content", "hello"}}})}};
}

OpenAIChatRequest parse(Json body) { return parse_chat_completion_request(body, limits()); }

ResolvedPromptSemantics semantics(const GenerationRequest& request) {
    ServeOptions server;
    return resolve_prompt_semantics(request, server);
}

ninfer::PromptInput prompt(const GenerationRequest& request) {
    return to_prompt_input(request, semantics(request), {});
}

ninfer::RequestOptions options(const GenerationRequest& request) {
    ServeOptions server;
    return to_request_options(request, server, semantics(request), true);
}

Json parse_sse(const std::string& event) {
    constexpr std::string_view prefix = "data: ";
    if (!event.starts_with(prefix) || !event.ends_with("\n\n")) {
        throw std::runtime_error("invalid SSE framing");
    }
    return Json::parse(event.substr(prefix.size(), event.size() - prefix.size() - 2));
}

int test_request_envelope_and_sampling() {
    int failures                  = 0;
    Json body                     = base_request();
    body["stream"]                = true;
    body["stream_options"]        = Json{{"include_usage", true}, {"include_obfuscation", false}};
    body["max_completion_tokens"] = 48;
    body["max_tokens"]            = 9;
    body["temperature"]           = 0.7;
    body["top_p"]                 = 0.8;
    body["presence_penalty"]      = 0.3;
    body["frequency_penalty"]     = -0.2;
    body["seed"]                  = -1;
    body["top_k"]                 = 17;
    body["min_p"]                 = 0.05;
    body["timings_per_token"]     = true;
    body["return_progress"]       = true;

    const OpenAIChatRequest request = parse(body);
    failures += check(request.model == "qwen", "model remains in OpenAI envelope");
    failures += check(request.stream && request.include_usage, "stream metadata parsed");
    failures += check(request.timings_per_token && request.return_progress,
                      "llama.cpp response observations remain in the protocol envelope");
    failures += check(request.output_tokens_explicit && request.generation.max_tokens == 48,
                      "max_completion_tokens wins and explicitness stays in envelope");
    failures += check(request.generation.sampling.seed == std::numeric_limits<std::uint64_t>::max(),
                      "signed seed maps modulo 2^64");
    failures +=
        check(request.generation.sampling.top_k == 17 && request.generation.sampling.min_p == 0.05,
              "compatible sampler extensions parsed");
    const ninfer::RequestOptions translated = options(request.generation);
    failures +=
        check(translated.execution.sampling.top_k == 17, "top_k reaches Engine request options");
    failures +=
        check(translated.execution.sampling.min_p && *translated.execution.sampling.min_p == 0.05F,
              "min_p reaches Engine request options");
    failures +=
        check(translated.execution.sampling.seed == std::numeric_limits<std::uint64_t>::max(),
              "signed seed reaches Engine request options");

    const OpenAIChatRequest defaults = parse(base_request());
    failures +=
        check(!defaults.stream && !defaults.include_usage && !defaults.output_tokens_explicit &&
                  !defaults.timings_per_token && !defaults.return_progress &&
                  defaults.generation.max_tokens == limits().default_max_tokens,
              "protocol defaults remain outside GenerationRequest");

    Json malformed              = base_request();
    malformed["stream_options"] = true;
    failures += check(api_error([&] { (void)parse(malformed); }).param == "stream_options",
                      "malformed stream_options rejected");
    malformed                      = base_request();
    malformed["timings_per_token"] = "yes";
    failures += check(api_error([&] { (void)parse(malformed); }).param == "timings_per_token",
                      "non-boolean timings_per_token rejected");
    malformed                    = base_request();
    malformed["return_progress"] = 1;
    failures += check(api_error([&] { (void)parse(malformed); }).param == "return_progress",
                      "non-boolean return_progress rejected");
    return failures;
}

int test_standard_field_policy() {
    int failures  = 0;
    auto rejected = [&](const char* key, Json value, const char* code) {
        Json body            = base_request();
        body[key]            = std::move(value);
        const ApiError error = api_error([&] { (void)parse(body); });
        failures += check(error.param == key && error.code == code,
                          std::string(key) + " non-neutral value rejected");
    };

    rejected("n", 2, "n_not_supported");
    rejected("logit_bias", Json{{"12", 1}}, "logit_bias_not_supported");
    rejected("logprobs", true, "logprobs_not_supported");
    rejected("top_logprobs", 2, "logprobs_not_supported");
    {
        // response_format json_object is accepted and recorded (prompt-guided JSON).
        Json body              = base_request();
        body["response_format"] = Json{{"type", "json_object"}};
        const OpenAIChatRequest json_req = parse(body);
        failures += check(
            json_req.generation.response_format.kind == ResponseFormatKind::JsonObject,
            "json_object response_format accepted and recorded");
    }
    {
        // response_format json_schema accepted; the schema is captured for the
        // folded instruction.
        Json body         = base_request();
        Json schema_spec  = Json::object();
        schema_spec["schema"] = Json{{"type", "object"}};
        body["response_format"] = Json{{"type", "json_schema"}, {"json_schema", schema_spec}};
        const OpenAIChatRequest schema_req = parse(body);
        failures += check(
            schema_req.generation.response_format.kind == ResponseFormatKind::JsonSchema &&
                !schema_req.generation.response_format.schema_json.empty(),
            "json_schema response_format accepted with schema");
    }
    {
        // malformed json_schema (missing schema object) is still rejected.
        Json body = base_request();
        body["response_format"] = Json{{"type", "json_schema"}, {"json_schema", Json::object()}};
        const ApiError error = api_error([&] { (void)parse(body); });
        failures += check(error.param == "response_format" &&
                          error.code == "response_format_not_supported",
                          "json_schema without schema object rejected");
    }
    rejected("modalities", Json::array({"text", "audio"}), "modality_not_supported");
    rejected("web_search_options", Json::object(), "web_search_not_supported");
    rejected("moderation", Json::object(), "moderation_not_supported");
    rejected("verbosity", "high", "verbosity_not_supported");
    rejected("store", true, "store_not_supported");
    rejected("functions", Json::array({Json{{"name", "legacy"}}}), "legacy_tools_not_supported");

    Json neutral                      = base_request();
    neutral["n"]                      = 1;
    neutral["logit_bias"]             = Json{{"12", 0}, {"13", 0.0}};
    neutral["logprobs"]               = false;
    neutral["top_logprobs"]           = 0;
    neutral["response_format"]        = Json{{"type", "text"}};
    neutral["modalities"]             = Json::array({"text"});
    neutral["audio"]                  = Json{{"voice", "alloy"}};
    neutral["prediction"]             = Json{{"type", "content"}, {"content", "expected"}};
    neutral["verbosity"]              = "medium";
    neutral["store"]                  = false;
    neutral["functions"]              = Json::array();
    neutral["function_call"]          = "auto";
    neutral["metadata"]               = Json{{"trace", "client"}};
    neutral["user"]                   = "user-1";
    neutral["safety_identifier"]      = "safe-1";
    neutral["prompt_cache_key"]       = "cache-1";
    neutral["prompt_cache_options"]   = Json{{"retention", "24h"}};
    neutral["prompt_cache_retention"] = "24h";
    neutral["service_tier"]           = "priority";
    neutral["future_unknown_field"]   = Json{{"value", 1}};
    failures += check(parse(neutral).generation.messages.size() == 1,
                      "neutral controls and advisory hints are accepted");

    Json zero_limit                     = base_request();
    zero_limit["max_completion_tokens"] = 0;
    const OpenAIChatRequest zero        = parse(zero_limit);
    failures += check(zero.output_tokens_explicit && zero.generation.max_tokens == 0,
                      "an explicit zero output limit reaches Engine's no-generation path");
    return failures;
}

int test_constrained_decoding_extensions() {
    int failures                                           = 0;
    const std::vector<std::pair<const char*, Json>> active = {
        {"grammar", "root ::= \"yes\" | \"no\""},
        {"structured_outputs", Json{{"json", Json{{"type", "object"}}}}},
        {"guided_json", Json{{"type", "object"}}},
        {"guided_regex", "[a-z]+"},
        {"guided_choice", Json::array({"yes", "no"})},
        {"guided_grammar", "root ::= \"yes\" | \"no\""},
    };
    for (const auto& [field, value] : active) {
        Json body            = base_request();
        body[field]          = value;
        const ApiError error = api_error([&] { (void)parse(body); });
        failures +=
            check(error.param == field && error.code == "constrained_decoding_not_supported" &&
                      error.message.find(field) != std::string::npos,
                  std::string(field) + " constrained decoding is explicitly rejected");
    }

    Json neutral                  = base_request();
    neutral["grammar"]            = "";
    neutral["structured_outputs"] = nullptr;
    neutral["guided_json"]        = nullptr;
    neutral["guided_regex"]       = nullptr;
    neutral["guided_choice"]      = nullptr;
    neutral["guided_grammar"]     = nullptr;
    failures += check(parse(neutral).generation.messages.size() == 1,
                      "neutral constrained-decoding extension values are accepted");
    return failures;
}

Json function_tool(std::string name = "weather", bool strict = false) {
    return Json{{"type", "function"},
                {"function", Json{{"name", std::move(name)},
                                  {"description", "Get weather"},
                                  {"parameters", Json{{"type", "object"}}},
                                  {"strict", strict}}}};
}

int test_tools() {
    int failures                      = 0;
    Json body                         = base_request();
    body["tools"]                     = Json::array({function_tool()});
    const OpenAIChatRequest automatic = parse(body);
    failures += check(automatic.generation.uses_tools(), "function tools default to auto");
    failures += check(prompt(automatic.generation).options.tool_jsons.size() == 1,
                      "auto tools reach PromptInput");

    body["tools"][0]["future_item_field"]                 = "ignored";
    body["tools"][0]["function"]["future_function_field"] = "ignored";
    const std::string normalized_definition = prompt(parse(body).generation).options.tool_jsons[0];
    failures += check(normalized_definition.find("future_item_field") == std::string::npos &&
                          normalized_definition.find("future_function_field") == std::string::npos,
                      "unknown tool fields do not silently alter the model prompt");

    body["tool_choice"]          = "none";
    body["parallel_tool_calls"]  = false;
    const OpenAIChatRequest none = parse(body);
    failures +=
        check(!none.generation.uses_tools() && prompt(none.generation).options.tool_jsons.empty(),
              "tool_choice none makes parallel_tool_calls neutral and removes executable tools");

    body["tool_choice"] = "required";
    body.erase("parallel_tool_calls");
    const OpenAIChatRequest required = parse(body);
    failures += check(required.generation.tool_choice.mode == ToolChoiceMode::Required &&
                          required.generation.uses_tools(),
                      "required tool choice is accepted and keeps tools enabled");
    {
        Json no_tools = base_request();
        no_tools["tool_choice"] = "required";
        const ApiError no_tools_error = api_error([&] { (void)parse(no_tools); });
        failures += check(no_tools_error.code == "tool_choice_not_supported",
                          "required tool choice without tools rejected");
    }
    body["tool_choice"] = Json{{"type", "function"}, {"function", Json{{"name", "weather"}}}};
    failures += check(api_error([&] { (void)parse(body); }).code == "tool_choice_not_supported",
                      "named tool choice rejected");

    body          = base_request();
    body["tools"] = Json::array({function_tool(), function_tool("search")});
    body["tool_choice"] =
        Json{{"type", "allowed_tools"},
             {"allowed_tools",
              Json{{"mode", "auto"},
                   {"tools", Json::array({Json{{"type", "function"}, {"name", "search"}}})}}}};
    const GenerationRequest allowed = parse(body).generation;
    failures += check(allowed.tools.size() == 1 && allowed.tools[0].name == "search" &&
                          prompt(allowed).options.tool_jsons.size() == 1,
                      "allowed_tools auto narrows the executable function set");

    body["tool_choice"] =
        Json{{"type", "allowed_tools"},
             {"mode", "auto"},
             {"tools", Json::array({Json{{"type", "function"}, {"name", "weather"}}})}};
    const GenerationRequest direct_allowed = parse(body).generation;
    failures += check(direct_allowed.tools.size() == 1 && direct_allowed.tools[0].name == "weather",
                      "direct allowed_tools compatibility shape is accepted");
    body["tool_choice"]["mode"]     = "required";
    const ApiError required_allowed = api_error([&] { (void)parse(body); });
    failures +=
        check(required_allowed.code == "tool_choice_not_supported" &&
                  required_allowed.message.find("at least one tool call") != std::string::npos,
              "required allowed_tools reports the unenforceable guarantee");
    body["tool_choice"]["mode"]             = "auto";
    body["tool_choice"]["tools"][0]["name"] = "missing";
    failures += check(api_error([&] { (void)parse(body); }).param == "tool_choice",
                      "allowed_tools rejects names absent from the declared tool set");

    body          = base_request();
    body["tools"] = Json::array({function_tool("weather", true)});
    failures += check(api_error([&] { (void)parse(body); }).code == "strict_tools_not_supported",
                      "strict tools rejected");
    body["tools"] = Json::array({Json{{"type", "custom"}, {"name", "shell"}}});
    failures += check(api_error([&] { (void)parse(body); }).code == "tool_type_not_supported",
                      "custom tools rejected");

    body                        = base_request();
    body["tools"]               = Json::array({function_tool()});
    body["parallel_tool_calls"] = false;
    failures +=
        check(api_error([&] { (void)parse(body); }).code == "parallel_tool_calls_not_supported",
              "parallel_tool_calls=false rejected when tools exist");
    body.erase("tools");
    failures += check(parse(body).generation.tools.empty(),
                      "parallel_tool_calls=false is neutral without tools");
    body["tool_choice"] = "auto";
    failures +=
        check(parse(body).generation.tools.empty(), "tool_choice auto is neutral without tools");

    Json history = base_request();
    history["messages"] =
        Json::array({Json{{"role", "user"}, {"content", "weather?"}},
                     Json{{"role", "assistant"},
                          {"content", nullptr},
                          {"tool_calls",
                           Json::array({Json{{"id", "call_1"},
                                             {"type", "function"},
                                             {"function", Json{{"name", "weather"},
                                                               {"arguments", "not-json-yet"}}}}})}},
                     Json{{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", "sunny"}}});
    failures += check(parse(history).generation.has_tool_history(),
                      "tool-call history follows wire types without inventing JSON validation");

    Json mixed_assistant        = base_request();
    mixed_assistant["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "inspect"}},
         Json{{"role", "assistant"},
              {"content", "I will inspect it"},
              {"tool_calls",
               Json::array({Json{
                   {"id", "call_2"},
                   {"type", "function"},
                   {"function", Json{{"name", "inspect"}, {"arguments", R"({"path":"a"})"}}}}})}}});
    const GenerationRequest mixed_request  = parse(mixed_assistant).generation;
    const ninfer::PromptInput mixed_prompt = prompt(mixed_request);
    failures += check(mixed_request.messages[1].cache_boundary_after &&
                          !mixed_request.messages[1].content[0].cache_boundary_after &&
                          !mixed_prompt.context_cache.markers.empty() &&
                          mixed_prompt.context_cache.markers.back().location ==
                              ninfer::PromptCacheMarkerLocation::MessageBoundary &&
                          mixed_prompt.context_cache.markers.back().after_message_count == 2,
                      "automatic caching stops after a complete assistant text/tool-call turn");
    failures += check(mixed_prompt.context_cache.allow_engine_automatic_shared_prefixes,
                      "default automatic caching lets the Engine add its stable-layer candidates");
    Json explicit_only                    = mixed_assistant;
    explicit_only["prompt_cache_options"] = Json{{"mode", "explicit"}};
    failures += check(
        !prompt(parse(explicit_only).generation).context_cache.allow_engine_automatic_shared_prefixes,
        "explicit prompt caching keeps Engine candidates out");

    const Json ordered = Json::parse(
        R"({"model":"qwen","messages":[{"role":"user","content":"probe"}],"tools":[{"type":"function","function":{"name":"probe","parameters":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"integer"}}}}}]})");
    const ninfer::PromptInput ordered_prompt = prompt(parse(ordered).generation);
    failures += check(
        ordered_prompt.options.tool_jsons.size() == 1 &&
            ordered_prompt.options.tool_jsons.front() ==
                R"({"type":"function","function":{"name":"probe","parameters":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"integer"}}},"strict":false}})",
        "OpenAI Chat changed tool-schema member order before PromptInput");
    return failures;
}

int test_messages_and_media() {
    int failures                   = 0;
    Json body                      = base_request();
    body["messages"][0]["content"] = Json::array(
        {Json{{"type", "text"}, {"text", "alpha"}}, Json{{"type", "text"}, {"text", "beta"}}});
    const ninfer::PromptInput translated = prompt(parse(body).generation);
    failures += check(translated.messages[0].parts.size() == 2 &&
                          translated.messages[0].parts[0].text == "alpha" &&
                          translated.messages[0].parts[1].text == "beta",
                      "adjacent text parts preserve exact text without inserted newline");

    body                           = base_request();
    body["messages"][0]["content"] = Json::array(
        {Json{{"type", "image_url"},
              {"image_url", Json{{"url", "https://example.test/a.png"}, {"detail", "auto"}}}},
         Json{{"type", "video_url"}, {"video_url", "https://example.test/a.mp4"}}});
    const GenerationRequest media = parse(body).generation;
    failures += check(media.media_item_count() == 2 &&
                          media.messages[0].content[0].kind == ContentKind::Image &&
                          media.messages[0].content[1].kind == ContentKind::Video,
                      "image and video compatibility inputs normalize to Engine media");

    body["messages"][0]["content"][0]["image_url"]["detail"] = "high";
    failures += check(api_error([&] { (void)parse(body); }).code == "image_detail_not_supported",
                      "explicit image preprocessing detail rejected");

    auto content_rejected = [&](const char* role, const char* type) {
        Json invalid                   = base_request();
        invalid["messages"][0]["role"] = role;
        invalid["messages"][0]["content"] =
            Json::array({Json{{"type", type}, {type, "https://example.test/x"}}});
        return api_error([&] { (void)parse(invalid); }).code == "modality_not_supported";
    };
    failures +=
        check(content_rejected("assistant", "image_url"), "assistant media history rejected");
    failures += check(content_rejected("system", "image_url"),
                      "system media rejected at protocol boundary");

    body["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "capture it"}},
         Json{{"role", "assistant"},
              {"content", nullptr},
              {"tool_calls",
               Json::array({Json{{"id", "call_capture"},
                                 {"type", "function"},
                                 {"function", Json{{"name", "capture"}, {"arguments", "{}"}}}}})}},
         Json{{"role", "tool"},
              {"tool_call_id", "call_capture"},
              {"content",
               Json::array({Json{{"type", "text"}, {"text", "captured"}},
                            Json{{"type", "image_url"},
                                 {"image_url", Json{{"url", "https://example.test/capture.png"},
                                                    {"detail", "auto"}}}}})}}});
    const GenerationRequest tool_image = parse(body).generation;
    failures += check(tool_image.messages.back().role == ninfer::ChatRole::Tool &&
                          tool_image.messages.back().tool_call_id == "call_capture" &&
                          tool_image.messages.back().content.size() == 2 &&
                          tool_image.messages.back().content[0].kind == ContentKind::Text &&
                          tool_image.messages.back().content[1].kind == ContentKind::Image,
                      "tool result text and image parts normalize to one tool turn");

    body["messages"].back()["content"] = Json::array(
        {Json{{"type", "video_url"}, {"video_url", "https://example.test/capture.mp4"}}});
    failures += check(api_error([&] { (void)parse(body); }).code == "modality_not_supported",
                      "tool result video remains outside the Chat compatibility extension");

    body = base_request();
    body["messages"][0]["content"] =
        Json::array({Json{{"type", "input_audio"}, {"input_audio", Json::object()}}});
    failures += check(api_error([&] { (void)parse(body); }).code == "modality_not_supported",
                      "input audio rejected");
    body["messages"][0]["content"] =
        Json::array({Json{{"type", "file"}, {"file", Json::object()}}});
    failures += check(api_error([&] { (void)parse(body); }).code == "modality_not_supported",
                      "file input rejected");

    body                        = base_request();
    body["messages"][0]["name"] = "speaker";
    failures += check(api_error([&] { (void)parse(body); }).code == "message_name_not_supported",
                      "message name rejected");

    body = base_request();
    body["messages"].push_back(Json{
        {"role", "assistant"},
        {"content", nullptr},
        {"tool_calls",
         Json::array({Json{{"id", "call_1"},
                           {"type", "function"},
                           {"function", Json{{"name", "get_status"}, {"arguments", "{}"}}}}})}});
    body["messages"].push_back(Json{
        {"role", "tool"}, {"name", "get_status"}, {"tool_call_id", "call_1"}, {"content", "ok"}});
    const GenerationRequest named_tool_history = parse(body).generation;
    const ChatTurn& named_tool                 = named_tool_history.messages.back();
    failures += check(named_tool.role == ninfer::ChatRole::Tool &&
                          named_tool.tool_call_id == "call_1" && !named_tool.tool_result_name &&
                          named_tool.content.size() == 1 && named_tool.content[0].text == "ok",
                      "tool message name is an ignored compatibility extension");

    body["messages"].back()["name"] = Json::array();
    failures +=
        check(api_error([&] { (void)parse(body); }).message == "message name must be a string",
              "tool message name remains type checked");

    body                           = base_request();
    body["messages"][0]["name"]    = "";
    body["messages"][0]["content"] = Json::array();
    failures += check(parse(body).generation.messages[0].content.empty(),
                      "empty names and empty content arrays remain neutral");

    body = base_request();
    body["messages"].push_back(
        Json{{"role", "assistant"},
             {"content", Json::array({Json{{"type", "refusal"}, {"refusal", "part"}}})},
             {"refusal", "top-level"}});
    const GenerationRequest refusal_history = parse(body).generation;
    const ChatTurn& refusal                 = refusal_history.messages.back();
    failures += check(refusal.content.size() == 2 && refusal.content[0].text == "part" &&
                          refusal.content[1].text == "top-level",
                      "assistant refusal history is preserved as assistant text");

    body = base_request();
    body["messages"].push_back(Json{{"role", "assistant"}});
    failures += check(parse(body).generation.messages.back().content.empty(),
                      "an empty assistant history turn is representable");

    body["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "run it"}},
         Json{{"role", "assistant"},
              {"content", nullptr},
              {"function_call", Json{{"name", "legacy"}, {"arguments", R"({"value":1})"}}}},
         Json{{"role", "function"}, {"name", "legacy"}, {"content", "done"}}});
    const GenerationRequest legacy = parse(body).generation;
    failures += check(legacy.messages[1].tool_calls.size() == 1 &&
                          legacy.messages[1].tool_calls[0].name == "legacy" &&
                          legacy.messages[2].role == ninfer::ChatRole::Tool,
                      "legacy function-call history lowers to Engine tool history");

    body["messages"] = Json::array(
        {Json{{"role", "assistant"},
              {"content", nullptr},
              {"tool_calls",
               Json::array({Json{{"id", ""},
                                 {"type", "function"},
                                 {"function", Json{{"name", "weather"}, {"arguments", "{}"}}}}})}},
         Json{{"role", "tool"}, {"tool_call_id", ""}, {"content", "done"}}});
    failures += check(parse(body).generation.has_tool_history(),
                      "string tool-call identifiers may be empty without changing history");
    return failures;
}

int test_reasoning_and_extensions() {
    int failures = 0;
    Json body    = base_request();
    body["messages"].push_back(Json{{"role", "assistant"},
                                    {"content", "answer"},
                                    {"reasoning_content", "thought"},
                                    {"reasoning", "thought"}});
    failures += check(parse(body).generation.messages.back().reasoning_content == "thought",
                      "assistant reasoning aliases normalize");
    body["messages"].back()["reasoning"] = "different";
    failures += check(api_error([&] { (void)parse(body); }).code == "conflicting_template_option",
                      "conflicting assistant reasoning aliases rejected");
    body["messages"].back()["reasoning_content"] = "";
    failures += check(parse(body).generation.messages.back().reasoning_content == "different",
                      "an empty reasoning alias does not conflict with a meaningful alias");
    body = base_request();
    body["messages"].push_back(Json{
        {"role", "assistant"}, {"content", nullptr}, {"reasoning_content", "unfinished thought"}});
    failures +=
        check(parse(body).generation.messages.back().reasoning_content == "unfinished thought",
              "reasoning-only assistant history is preserved");

    body                         = base_request();
    body["enable_thinking"]      = true;
    body["preserve_thinking"]    = false;
    body["chat_template_kwargs"] = Json{{"enable_thinking", true}, {"preserve_thinking", false}};
    const GenerationRequest normalized = parse(body).generation;
    failures += check(normalized.enable_thinking == true && normalized.preserve_thinking == false,
                      "Qwen/vLLM template aliases normalize");
    body["chat_template_kwargs"]["enable_thinking"] = false;
    failures += check(api_error([&] { (void)parse(body); }).code == "conflicting_template_option",
                      "conflicting thinking aliases rejected");
    body                         = base_request();
    body["chat_template_kwargs"] = Json{{"future", 1}};
    failures +=
        check(Json::parse(parse(body).generation.chat_template_kwargs_json).at("future") == 1,
              "custom template keyword did not survive protocol parsing");
    body["chat_template_kwargs"] = Json{{"future", nullptr}};
    failures += check(parse(body).generation.messages.size() == 1,
                      "null unknown template option is neutral");

    body                        = base_request();
    body["repetition_penalty"]  = 1.0;
    body["mm_processor_kwargs"] = Json{{"max_pixels", nullptr}};
    failures +=
        check(parse(body).generation.messages.size() == 1, "neutral ecosystem defaults accepted");
    body["repetition_penalty"] = 1.1;
    failures +=
        check(api_error([&] { (void)parse(body); }).code == "repetition_penalty_not_supported",
              "non-neutral repetition penalty rejected");
    body                        = base_request();
    body["mm_processor_kwargs"] = Json{{"max_pixels", 100}};
    failures +=
        check(api_error([&] { (void)parse(body); }).code == "mm_processor_kwargs_not_supported",
              "non-empty media processor kwargs rejected");
    return failures;
}

int test_stops_and_ranges() {
    int failures                            = 0;
    Json body                               = base_request();
    body["stop"]                            = Json::array({"A", "B"});
    const ninfer::RequestOptions translated = options(parse(body).generation);
    failures += check(translated.stop.strings.size() == 4,
                      "each stop string applies to Content and Reasoning");
    failures += check(translated.stop.strings[0].channel == ninfer::OutputChannel::Content &&
                          translated.stop.strings[1].channel == ninfer::OutputChannel::Reasoning,
                      "stop channel ordering is explicit");

    body["stop"] = Json::array({"1", "2", "3", "4", "5"});
    failures += check(api_error([&] { (void)parse(body); }).param == "stop",
                      "more than four stop strings rejected");
    body["stop"] = "";
    failures +=
        check(api_error([&] { (void)parse(body); }).param == "stop", "empty stop string rejected");

    body = base_request();
    failures += check(options(parse(body).generation).stop.include_model_defaults,
                      "an omitted ignore_eos keeps the checkpoint's own stop tokens");
    body["ignore_eos"] = false;
    failures += check(options(parse(body).generation).stop.include_model_defaults,
                      "ignore_eos false keeps the checkpoint's own stop tokens");
    body["ignore_eos"] = true;
    failures += check(parse(body).generation.ignore_eos &&
                          !options(parse(body).generation).stop.include_model_defaults,
                      "ignore_eos suppresses the checkpoint's own stop tokens");
    body["stop"] = Json::array({"A"});
    failures += check(options(parse(body).generation).stop.strings.size() == 2 &&
                          !options(parse(body).generation).stop.include_model_defaults,
                      "ignore_eos leaves caller stop strings in place");
    body.erase("stop");
    body["ignore_eos"] = "true";
    failures += check(api_error([&] { (void)parse(body); }).param == "ignore_eos",
                      "a non-boolean ignore_eos is rejected");

    body                                  = base_request();
    body["top_k"]                         = 21;
    const GenerationRequest invalid_top_k = parse(body).generation;
    failures += check(api_error([&] { (void)options(invalid_top_k); }).param == "top_k",
                      "Engine translator owns sampler value range");
    body["top_k"]                         = 5;
    body["min_p"]                         = 1.1;
    const GenerationRequest invalid_min_p = parse(body).generation;
    failures += check(api_error([&] { (void)options(invalid_min_p); }).param == "min_p",
                      "min_p range enforced by common Engine translator");
    return failures;
}

GenerationOutcome sample_outcome() {
    GenerationOutcome outcome;
    outcome.text                                = "answer";
    outcome.reasoning                           = "thought";
    outcome.prompt_tokens                       = 20;
    outcome.completion_tokens                   = 7;
    outcome.reasoning_tokens                    = 3;
    outcome.finish_reason                       = ninfer::FinishReason::StopToken;
    outcome.metrics.prefix_cache_hit_tokens     = 12;
    outcome.metrics.prompt_wall_seconds         = 0.04;
    outcome.metrics.generation_wall_seconds     = 0.03;
    outcome.metrics.speculative_draft_tokens    = 9;
    outcome.metrics.speculative_accepted_tokens = 6;
    return outcome;
}

OpenAIChatResponseIdentity identity() {
    return OpenAIChatResponseIdentity{.id = "chatcmpl-test", .model = "qwen", .created = 42};
}

int test_aggregate_response() {
    int failures              = 0;
    GenerationOutcome outcome = sample_outcome();
    Json response = Json::parse(make_chat_completion_response(identity(), outcome, 0));
    failures += check(response["choices"][0]["message"]["content"] == "answer" &&
                          response["choices"][0]["message"]["reasoning_content"] == "thought" &&
                          response["choices"][0]["message"]["refusal"].is_null(),
                      "aggregate response separates reasoning and content");
    failures += check(response["choices"][0]["logprobs"].is_null(),
                      "aggregate choice carries nullable logprobs");
    failures += check(response["usage"]["prompt_tokens_details"]["cached_tokens"] == 12 &&
                          response["usage"]["completion_tokens_details"]["reasoning_tokens"] == 3,
                      "aggregate usage exposes cache hits and reasoning tokens");
    failures += check(
        response["timings"]["cache_n"] == 12 && response["timings"]["prompt_n"] == 8 &&
            response["timings"]["prompt_ms"] == 40.0 &&
            response["timings"]["prompt_per_second"] == 200.0 &&
            response["timings"]["predicted_n"] == 7 &&
            response["timings"]["predicted_ms"] == 30.0 &&
            response["timings"]["predicted_per_second"] == 200.0 &&
            response["timings"]["draft_n"] == 9 && response["timings"]["draft_n_accepted"] == 6,
        "aggregate timings use exact cache and N-1 generation intervals");

    outcome.text.clear();
    outcome.tool_calls.push_back(ninfer::GeneratedToolCall{
        .name = "Edit",
        .arguments_json =
            R"({"file_path":"/tmp/probe.cpp","old_string":"old","new_string":"new"})"});
    response         = Json::parse(make_chat_completion_response(identity(), outcome, 0));
    const Json& call = response["choices"][0]["message"]["tool_calls"][0];
    failures += check(response["choices"][0]["finish_reason"] == "tool_calls" &&
                          response["choices"][0]["message"]["content"].is_null(),
                      "aggregate tool call has OpenAI terminal shape");
    failures += check(
        call["id"].get<std::string>().starts_with("call_") && call["function"]["name"] == "Edit" &&
            !Json::parse(call["function"]["arguments"].get<std::string>()).contains("replace_all"),
        "OpenAI adapter owns wire tool-call identifiers");
    return failures;
}

int test_aggregate_logprobs() {
    int failures              = 0;
    GenerationOutcome outcome = sample_outcome();
    // Reports name vocabulary ids, and an alternative is a different id with its own text. Position in
    // the generated sequence must never be what selects a piece.
    outcome.token_logprobs = {
        ninfer::GeneratedTokenLogprob{
            .token = 151644, .logprob = -0.25F, .top = {{151644, -0.25F}, {198, -1.5F}}},
        ninfer::GeneratedTokenLogprob{.token = 362, .injected = true},
        ninfer::GeneratedTokenLogprob{
            .token = 5215, .logprob = -0.5F, .top = {{264, -1.0F}}},
    };
    outcome.token_pieces = {{151644, "Hello"},
                            {198, "\n"},
                            {362, " there"},
                            {5215, " world"},
                            // A piece that ends inside a multi-byte character: `token` is lossy, the
                            // byte list is not.
                            {264, "\xF0\x9F\x98"}};
    const Json logprobs =
        Json::parse(make_chat_completion_response(identity(), outcome, 2))["choices"][0]["logprobs"];
    failures += check(logprobs["content"].size() == 3 && logprobs["refusal"].is_null(),
                      "logprobs reports one entry per generated token");
    failures += check(logprobs["content"][0]["token"] == "Hello" &&
                          logprobs["content"][0]["logprob"] == -0.25,
                      "the chosen token reports its own piece");
    // The regression this guards: alternatives are vocabulary ids. Id 198 is a tenth of a plausible
    // position index and 151644 is far beyond any of them, so a lookup by position yields either
    // nothing or the neighbouring token's text.
    failures += check(logprobs["content"][0]["top_logprobs"][0]["token"] == "Hello" &&
                          logprobs["content"][0]["top_logprobs"][1]["token"] == "\n" &&
                          logprobs["content"][0]["top_logprobs"][1]["bytes"] == Json::array({10}),
                      "each alternative resolves to the piece of the id it names");
    failures += check(logprobs["content"][1]["injected"] == true &&
                          logprobs["content"][1]["logprob"] == 0 &&
                          logprobs["content"][1]["token"] == " there",
                      "an injected control span is marked and keeps its position");
    failures += check(logprobs["content"][2]["top_logprobs"][0]["bytes"].size() == 3 &&
                          logprobs["content"][2]["top_logprobs"][0]["token"].get<std::string>() ==
                              "\xEF\xBF\xBD",
                      "a split character stays exact in bytes and lossy in text");
    return failures;
}

int test_stream_response() {
    int failures = 0;
    OpenAIChatStream stream(identity(), true);
    Json role = parse_sse(stream.start());
    failures += check(role["choices"][0]["delta"]["role"] == "assistant" &&
                          role["choices"][0]["logprobs"].is_null() && role["usage"].is_null(),
                      "stream starts with role, nullable logprobs, and null usage");
    Json reasoning = parse_sse(stream.reasoning_delta("thought"));
    Json content   = parse_sse(stream.content_delta("ans"));
    failures += check(reasoning["choices"][0]["delta"]["reasoning_content"] == "thought" &&
                          content["choices"][0]["delta"]["content"] == "ans",
                      "stream separates reasoning and content deltas");

    GenerationOutcome outcome             = sample_outcome();
    const std::vector<std::string> events = stream.finish(outcome);
    failures +=
        check(events.size() == 4, "finish emits buffered suffix, terminal, usage, and done");
    failures += check(parse_sse(events[0])["choices"][0]["delta"]["content"] == "wer",
                      "terminal content suffix is emitted exactly once");
    failures += check(parse_sse(events[1])["choices"][0]["finish_reason"] == "stop",
                      "stream terminal finish reason emitted");
    const Json usage = parse_sse(events[2]);
    failures += check(usage["choices"].empty() &&
                          usage["usage"]["prompt_tokens_details"]["cached_tokens"] == 12 &&
                          usage["usage"]["completion_tokens_details"]["reasoning_tokens"] == 3 &&
                          usage["timings"]["predicted_n"] == 7,
                      "dedicated stream usage carries token accounting and terminal timings");
    failures += check(events.back() == "data: [DONE]\n\n", "stream ends with DONE sentinel");

    // JSON response_format content is withheld while streaming and cleaned at the end; the
    // terminal suffix then carries the cleaned value as one content chunk.
    OpenAIChatStream json_stream(identity(), false);
    (void)json_stream.start();
    (void)json_stream.reasoning_delta("thought");
    GenerationOutcome json_outcome = sample_outcome();
    json_outcome.text = ninfer::serve::json_output::extract("```json\n{\"a\":1}\n```\n");
    const std::vector<std::string> json_events = json_stream.finish(json_outcome);
    failures += check(json_events.size() >= 2 &&
                          parse_sse(json_events[0])["choices"][0]["delta"]["content"] == "{\"a\":1}",
                      "withheld JSON content streams once as the cleaned value");

    OpenAIChatStream mismatch(identity(), false);
    (void)mismatch.start();
    (void)mismatch.content_delta("different");
    failures += check(throws_logic([&] { (void)mismatch.finish(outcome); }),
                      "stream encoder rejects terminal/content divergence");

    OpenAIChatStream tool_stream(identity(), false);
    (void)tool_stream.start();
    const Json tool_progress = parse_sse(tool_stream.tool_call_progress(1536));
    failures += check(tool_progress["choices"][0]["delta"].empty() &&
                          tool_progress["tool_call_progress"]["bytes"] == 1536,
                      "stream reports withheld tool-call progress without exposing markup");
    GenerationOutcome tool_outcome;
    tool_outcome.tool_calls.push_back(ninfer::GeneratedToolCall{
        .name = "Edit", .arguments_json = R"({"file_path":"/tmp/probe.cpp"})"});
    tool_outcome.finish_reason                 = ninfer::FinishReason::StopToken;
    const std::vector<std::string> tool_events = tool_stream.finish(tool_outcome);
    const Json tool_delta                      = parse_sse(tool_events[0]);
    failures += check(
        tool_delta["choices"][0]["delta"]["tool_calls"][0]["id"].get<std::string>().starts_with(
            "call_") &&
            tool_delta["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "Edit" &&
            parse_sse(tool_events[1])["choices"][0]["finish_reason"] == "tool_calls",
        "stream encoder owns stable OpenAI tool-call shape");
    return failures;
}

int test_stream_observations() {
    int failures = 0;
    OpenAIChatStream stream(identity(), true, true, true);
    const Json role = parse_sse(stream.start());
    failures += check(!role.contains("timings") && !role.contains("prompt_progress"),
                      "transport role chunk precedes Engine observations");

    stream.note_start(
        ninfer::GenerationStart{.prompt = {.prompt_tokens = 32}, .reused_prompt_tokens = 12});
    const Json initial = parse_sse(stream.initial_prompt_progress());
    failures +=
        check(initial["choices"][0]["delta"].empty() && initial["prompt_progress"]["total"] == 32 &&
                  initial["prompt_progress"]["cache"] == 12 &&
                  initial["prompt_progress"]["processed"] == 12 &&
                  initial["prompt_progress"]["time_ms"] == 0,
              "initial prompt progress begins at the admitted cache frontier");

    const Json middle = parse_sse(stream.prompt_progress(ninfer::PromptProgress{
        .total_prompt_tokens     = 32,
        .reused_prompt_tokens    = 12,
        .processed_prompt_tokens = 20,
        .elapsed_ns              = 57000000,
    }));
    failures += check(middle["prompt_progress"]["processed"] == 20 &&
                          middle["prompt_progress"]["time_ms"] == 57,
                      "prompt progress exposes a cumulative completed frontier");
    const Json complete = parse_sse(stream.prompt_progress(ninfer::PromptProgress{
        .total_prompt_tokens     = 32,
        .reused_prompt_tokens    = 12,
        .processed_prompt_tokens = 32,
        .elapsed_ns              = 100000000,
    }));
    failures +=
        check(complete["prompt_progress"]["processed"] == complete["prompt_progress"]["total"],
              "final prompt progress reaches the complete prompt");

    stream.note_timing(ninfer::GenerationTimingObservation{
        .generated_tokens = 1, .prompt_elapsed_ns = 110000000, .generation_elapsed_ns = 0});
    stream.note_timing(ninfer::GenerationTimingObservation{
        .generated_tokens      = 3,
        .prompt_elapsed_ns     = 110000000,
        .generation_elapsed_ns = 20000000,
    });
    const Json content = parse_sse(stream.content_delta("answer"));
    failures +=
        check(content["timings"]["prompt_n"] == 20 && content["timings"]["predicted_n"] == 3 &&
                  content["timings"]["predicted_per_second"] == 100.0,
              "visible output uses the latest independent commit observation");

    GenerationOutcome outcome = sample_outcome();
    outcome.reasoning.clear();
    const std::vector<std::string> terminal = stream.finish(outcome);
    failures += check(parse_sse(terminal[1])["timings"]["predicted_n"] == 7,
                      "terminal usage replaces live timing with exact final accounting");
    return failures;
}

int test_common_objects() {
    int failures      = 0;
    const Json models = Json::parse(make_models_list("qwen", 7, 240000));
    failures +=
        check(models["data"][0]["id"] == "qwen" && models["data"][0]["max_model_len"] == 240000,
              "models list advertises the configured context limit");
    const Json model = Json::parse(make_model_object("qwen", 7, 240000));
    failures += check(model["max_model_len"] == 240000,
                      "model lookup advertises the configured context limit");
    const Json error = Json::parse(make_error_body(
        ApiError{.status = 400, .message = "bad", .param = "messages", .code = "invalid"}));
    failures += check(error["error"]["param"] == "messages" && error["error"]["code"] == "invalid",
                      "OpenAI common error shape remains stable");
    return failures;
}

int test_output_directives() {
    int failures = 0;
    auto text_turn = [](ChatRole role, std::string text) {
        ChatTurn turn;
        turn.role = role;
        ContentPart part;
        part.kind     = ContentKind::Text;
        part.text      = std::move(text);
        part.type_raw  = "text";
        turn.content.push_back(std::move(part));
        return turn;
    };
    auto acquire = [](const ContentPart&) { return ninfer::OwnedMedia{}; };

    // No directive: system turns stay in place; no extra part is appended.
    {
        GenerationRequest request;
        request.messages = {text_turn(ChatRole::System, "You are a terse bot."),
                            text_turn(ChatRole::User, "hi")};
        const ninfer::PromptInput input = to_prompt_input(request, semantics(request), acquire);
        failures += check(input.messages.size() == 2 &&
                              input.messages[0].role == ChatRole::System &&
                              input.messages[0].parts.size() == 1 &&
                              input.messages[1].role == ChatRole::User &&
                              input.messages[1].parts.size() == 1,
                          "no directive leaves the prompt untouched");
    }

    // JSON schema: the directive is a trailing text part of the last user
    // message; the system turn stays in place with its original text.
    {
        GenerationRequest request;
        request.messages = {text_turn(ChatRole::System, "You are a terse bot."),
                            text_turn(ChatRole::User, "hi")};
        request.response_format.kind      = ResponseFormatKind::JsonSchema;
        request.response_format.schema_json = R"({"type":"object"})";
        const ninfer::PromptInput input = to_prompt_input(request, semantics(request), acquire);
        failures += check(input.messages.size() == 2 &&
                              input.messages[0].role == ChatRole::System &&
                              input.messages[0].parts.size() == 1 &&
                              input.messages[0].parts.front().text == "You are a terse bot.",
                          "system turn stays in place, unmodified, under a directive");
        failures += check(input.messages[1].parts.size() == 2 &&
                              input.messages[1].parts.front().text == "hi",
                          "the user text leads its message");
        const std::string& tail = input.messages[1].parts.back().text;
        failures += check(tail.find("single valid JSON object") != std::string::npos &&
                              tail.find(R"({"type":"object"})") != std::string::npos,
                          "directive is a trailing part of the last user message");
        failures += check(tail.find("You are a terse bot.") == std::string::npos,
                          "system text is no longer absorbed into the directive");
    }

    // tool_choice required: the tool-call directive is the trailing part and the
    // conflicting system guidance keeps its original position.
    {
        GenerationRequest request;
        request.messages = {text_turn(ChatRole::System, "Answer directly when you can."),
                            text_turn(ChatRole::User, "hi")};
        ToolDefinition tool;
        tool.name              = "weather";
        tool.input_schema_json = R"({"type":"object"})";
        request.tools.push_back(tool);
        request.tool_choice.mode = ToolChoiceMode::Required;
        const ninfer::PromptInput input = to_prompt_input(request, semantics(request), acquire);
        failures += check(input.messages.size() == 2 &&
                              input.messages[0].parts.front().text ==
                                  "Answer directly when you can.",
                          "conflicting system turn is kept, not folded");
        const std::string& tail = input.messages[1].parts.back().text;
        failures += check(tail.find("You MUST call a tool") != std::string::npos &&
                              tail.find("Answer directly when you can.") == std::string::npos,
                          "tool-call directive trails the last user message");
        failures += check(!prompt(request).options.tool_jsons.empty(),
                          "required tool choice keeps the tools enabled");
    }

    // Both directives: the JSON instruction leads, the tool-call directive last,
    // in one trailing part.
    {
        GenerationRequest request;
        request.messages = {text_turn(ChatRole::User, "hi")};
        request.response_format.kind      = ResponseFormatKind::JsonSchema;
        request.response_format.schema_json = R"({"type":"object"})";
        ToolDefinition tool;
        tool.name              = "weather";
        tool.input_schema_json = R"({"type":"object"})";
        request.tools.push_back(tool);
        request.tool_choice.mode = ToolChoiceMode::Required;
        const ninfer::PromptInput input = to_prompt_input(request, semantics(request), acquire);
        const std::string& tail = input.messages[0].parts.back().text;
        failures += check(tail.find("single valid JSON object") != std::string::npos &&
                              tail.find("You MUST call a tool") != std::string::npos &&
                              tail.find("You MUST call a tool") > tail.find("JSON Schema"),
                          "combined directive: JSON first, tool-call last");
    }

    // Agentic tail: the conversation ends on a tool result, so the directive
    // anchors to the trailing tool turn, not the earlier user turn.
    {
        GenerationRequest request;
        ChatTurn assistant;
        assistant.role = ChatRole::Assistant;
        ToolCall call;
        call.id             = "call_1";
        call.name           = "weather";
        call.arguments_json = R"({"q":"sf"})";
        assistant.tool_calls.push_back(call);
        ChatTurn tool;
        tool.role          = ChatRole::Tool;
        tool.tool_call_id  = "call_1";
        tool.tool_result_name = "weather";
        ContentPart result;
        result.kind     = ContentKind::Text;
        result.text      = "64 and sunny";
        result.type_raw  = "text";
        tool.content.push_back(std::move(result));
        request.messages = {text_turn(ChatRole::User, "weather in SF?"), std::move(assistant),
                            std::move(tool)};
        request.response_format.kind = ResponseFormatKind::JsonObject;
        const ninfer::PromptInput input = to_prompt_input(request, semantics(request), acquire);
        failures += check(input.messages.size() == 3 &&
                              input.messages[0].parts.size() == 1 &&
                              input.messages[1].parts.empty() &&
                              input.messages[2].parts.size() == 2,
                          "directive anchors to the trailing tool turn");
        const std::string& tail = input.messages[2].parts.back().text;
        failures += check(tail.find("single valid JSON object") != std::string::npos,
                          "directive is the trailing part of the tool result turn");
    }

    // Client system/developer turns keep their cache markers at their actual
    // positions: a boundary on the first system text part is the leading
    // instruction marker, and a boundary on a kept turn points at its input
    // position (no fold shifts positions).
    {
        GenerationRequest request;
        ChatTurn system;
        system.role = ChatRole::System;
        ContentPart system_text;
        system_text.kind     = ContentKind::Text;
        system_text.text      = "system text";
        system_text.type_raw  = "text";
        system_text.cache_boundary_after = CacheBoundary{};
        ChatTurn user;
        user.role = ChatRole::User;
        ContentPart user_text;
        user_text.kind     = ContentKind::Text;
        user_text.text      = "hi";
        user_text.type_raw  = "text";
        user_text.cache_boundary_after = CacheBoundary{};
        user.content = {std::move(user_text)};
        system.content = {std::move(system_text)};
        request.messages = {std::move(system), std::move(user)};
        request.response_format.kind = ResponseFormatKind::JsonObject;
        bool threw = false;
        ninfer::PromptInput input;
        try { input = to_prompt_input(request, semantics(request), acquire); }
        catch (...) { threw = true; }
        failures += check(!threw && input.messages.size() == 2 &&
                              input.context_cache.markers.size() == 2,
                          "directive request keeps the prompt and both boundaries");
        const auto leading = std::find_if(
            input.context_cache.markers.begin(), input.context_cache.markers.end(),
            [](const ninfer::PromptCacheMarker& marker) {
                return marker.location == ninfer::PromptCacheMarkerLocation::LeadingInstructionBoundary;
            });
        failures += check(
            leading != input.context_cache.markers.end() &&
                leading->leading_instruction_bytes == 11U,
            "first system text part keeps the leading-instruction marker");
        const auto part_marker = std::find_if(
            input.context_cache.markers.begin(), input.context_cache.markers.end(),
            [](const ninfer::PromptCacheMarker& marker) {
                return marker.location == ninfer::PromptCacheMarkerLocation::MessagePartBoundary;
            });
        failures += check(part_marker != input.context_cache.markers.end() &&
                              part_marker->after_message_count == 2 &&
                              part_marker->after_message_part_count == 1,
                          "kept-turn part boundary marker points at its actual input position");
    }
    return failures;
}

int test_json_output_extract() {
    int failures = 0;
    auto check = [&failures](bool ok, const char* what) {
        if (!ok) {
            std::cerr << "  FAIL " << what << "\n";
            return 1;
        }
        return 0;
    };
    using extract_t = std::string (*)(const std::string&);
    const extract_t extract = &ninfer::serve::json_output::extract;

    // A complete object is returned verbatim; values containing the marker
    // substring "think" and a newline are never damaged.
    {
        const std::string in = "{\"summary\":\"I was thinking\nabout the plot\"}";
        failures += check(extract(in) == in, "complete object with 'think' in a value is untouched");
    }
    // A leaked Qwen thinking block before the object is dropped.
    {
        const std::string in = "\nThe user wants JSON.\n{\"capital\":\"Paris\"}\ntail";
        failures += check(extract(in) == "{\"capital\":\"Paris\"}", "qwen thinking leak is dropped");
    }
    // The DeepSeek-style pair is stripped too.
    {
        const std::string in = "a\n<end_think>\nb {\"a\":1} tail";
        failures += check(extract(in) == "{\"a\":1}", "deepseek thinking pair is dropped");
    }
    // An unclosed fragment is left in place, not erased to the end.
    {
        const std::string in = "preamble think unterminated {\"a\":1} tail";
        const std::string out = extract(in);
        failures += check(out == "{\"a\":1}", "unclosed thinking fragment is not erased");
    }
    // Markdown-fenced JSON.
    {
        const std::string in = "```json\n{\"a\":1}\n```";
        failures += check(extract(in) == "{\"a\":1}", "fenced json is unwrapped");
    }
    // Fenced JSON preceded by a thinking leak: both are handled.
    {
        const std::string in = "\nleak\n```json\n{\"a\":1}\n```";
        failures += check(extract(in) == "{\"a\":1}", "leak and fence are both stripped");
    }
    // Prose before and after the object: the outermost object wins.
    {
        const std::string in = "Sure, here you go: {\"a\":{\"b\":2}} hope that helps";
        failures += check(extract(in) == "{\"a\":{\"b\":2}}", "prose on both sides is trimmed");
    }
    // No JSON object at all: unchanged.
    {
        const std::string in = "no json here at all";
        failures += check(extract(in) == in, "non-json text passes through");
    }
    // Surrounding whitespace is not part of the value.
    {
        failures += check(extract("\n\n{\"a\":1}\n") == "{\"a\":1}", "surrounding whitespace is trimmed");
    }
    // A fence inside a string value never damages a complete object.
    {
        const std::string in = "{\"code\":\"use ```x``` here\"}";
        failures += check(extract(in) == in, "fence inside a string value is untouched");
    }
    // A closed thinking block containing braces is dropped before the value is located.
    {
        const std::string in = "<think>maybe {\"a\":0}</think>\n{\"a\":1}";
        failures += check(extract(in) == "{\"a\":1}", "closed thinking block is dropped");
    }
    // Top-level arrays are JSON values too.
    {
        failures += check(extract("```json\n[1,{\"b\":2}]\n```") == "[1,{\"b\":2}]",
                          "fenced top-level array is unwrapped");
        failures += check(extract("Result: [{\"a\":1},{\"a\":2}] done") == "[{\"a\":1},{\"a\":2}]",
                          "top-level array is not cut to its first object");
    }
    // Truncated JSON cannot be recovered and is returned unchanged.
    {
        const std::string in = "{\"a\":[1,2";
        failures += check(extract(in) == in, "unrecoverable json is unchanged");
    }
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_request_envelope_and_sampling();
    failures += test_standard_field_policy();
    failures += test_constrained_decoding_extensions();
    failures += test_tools();
    failures += test_output_directives();
    failures += test_json_output_extract();
    failures += test_messages_and_media();
    failures += test_reasoning_and_extensions();
    failures += test_stops_and_ranges();
    failures += test_aggregate_response();
    failures += test_aggregate_logprobs();
    failures += test_stream_response();
    failures += test_stream_observations();
    failures += test_common_objects();
    if (failures == 0) { std::cout << "OpenAI Chat protocol tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
