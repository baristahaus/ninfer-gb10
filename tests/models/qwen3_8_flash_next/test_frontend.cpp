#include "artifact/reader.h"
#include <ninfer/models/qwen3_8_flash_next/frontend.h>
#include <ninfer/models/qwen3_8_flash_next/frontend_resources.h>

#include <nlohmann/json.hpp>

#include <array>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>

namespace {

namespace flash = ninfer::models::qwen3_8_flash_next;

std::string channel_text(const flash::PublishedOutput& output, ninfer::OutputChannel channel) {
    std::string text;
    for (const auto& delta : output) {
        if (delta.channel == channel) { text += delta.text; }
    }
    return text;
}

void require(bool ok, const char* what) {
    if (!ok) { throw std::runtime_error(what); }
}

flash::PreparedPrompt user_prompt(const flash::Frontend& frontend, bool thinking,
                                  std::vector<std::string> tools = {}) {
    ninfer::PromptInput input;
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = "x"});
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = thinking;
    input.options.tool_jsons      = std::move(tools);
    return frontend.prepare(std::move(input));
}

// A quoted </think> stays reasoning; only a close followed by format whitespace opens content.
void check_quoted_reasoning_close(const flash::Frontend& frontend) {
    const auto prompt = user_prompt(frontend, true);
    auto session      = frontend.make_output_session(prompt, {});
    const auto quoted = frontend.tokenize_text("discussing </think>'; then hidden\n");
    (void)session.preview_model(quoted, 256, ninfer::FinishReason::OutputLimit);
    const auto quoted_output = session.commit_preview();
    require(channel_text(quoted_output, ninfer::OutputChannel::Reasoning) ==
                    "discussing </think>'; then hidden\n" &&
                channel_text(quoted_output, ninfer::OutputChannel::Content).empty(),
            "a quoted reasoning close marker closed the reasoning channel");

    const auto close = frontend.tokenize_text("</think>\n\nreal answer");
    (void)session.preview_model(close, 256, ninfer::FinishReason::OutputLimit);
    const auto close_output = session.commit_preview();
    require(channel_text(close_output, ninfer::OutputChannel::Reasoning).empty() &&
                channel_text(close_output, ninfer::OutputChannel::Content) == "real answer",
            "the real reasoning close did not open the content channel");
}

// A quoted malformed <tool_call> before the real call stays content, and a repeated parameter
// keeps its last value instead of demoting the call to text.
void check_tool_call_recovery(const flash::Frontend& frontend) {
    const auto prompt = user_prompt(
        frontend, false,
        {R"({"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}}}}})"});
    auto session = frontend.make_output_session(prompt, {}, {.tool_name_max_length = 64});
    const std::string quoted =
        "<tool_call>\\n<function=shell>\\n<function=command>\\nbroken\\n</parameter>\\n"
        "</function>\\n</tool_call>";
    const std::string generated = "The failure looked like " + quoted +
                                  "\nThen the real call:\n"
                                  "<tool_call>\n<function=bash>\n<parameter=command>\nfirst\n"
                                  "</parameter>\n<parameter=command>\necho ok\n</parameter>\n"
                                  "</function>\n</tool_call>";
    const auto tokens = frontend.tokenize_text(generated);
    (void)session.preview_model(tokens, static_cast<std::uint32_t>(tokens.size()),
                                ninfer::FinishReason::OutputLimit);
    const auto output = session.commit_preview();
    const auto calls  = session.take_tool_calls();
    require(calls.size() == 1 && calls.front().name == "bash",
            "a quoted marker or repeated parameter demoted the real tool call");
    require(nlohmann::json::parse(calls.front().arguments_json).at("command") == "echo ok",
            "a repeated tool-call parameter did not keep its last value");
    require(session.tool_call_parse_diagnostics().duplicate_parameters_repaired == 1,
            "the repeated tool-call parameter repair was not counted");
    require(channel_text(output, ninfer::OutputChannel::Content) ==
                "The failure looked like " + quoted + "\nThen the real call:",
            "the quoted tool-call marker was not preserved as content");
}

} // namespace

int main() {
    const char* path = std::getenv("NINFER_QWEN38_FLASH_NEXT_WEIGHTS");
    if (!path || !*path) { return 77; }
    try {
        ninfer::artifact::Reader reader(path);
        const auto& resources = reader.directory().component("text").resources;
        auto read             = [&](const char* role) {
            const auto bytes = reader.read_object(resources.at(role));
            return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        };
        const flash::FrontendResources frontend_resources{
            .tokenizer_json         = read("tokenizer.json"),
            .tokenizer_config_json  = read("tokenizer_config.json"),
            .chat_template_jinja    = read("chat_template.jinja"),
            .generation_config_json = read("generation_config.json"),
        };
        auto frontend = flash::make_frontend(frontend_resources, {.vision_enabled = false});
        const std::array<std::optional<bool>, 3> settings{std::nullopt, false, true};
        for (auto thinking : settings) {
            ninfer::PromptInput input;
            ninfer::ChatMessage message;
            message.role = ninfer::ChatRole::User;
            message.parts.push_back(
                {.kind = ninfer::MessagePartKind::Text, .text = "Reply READY."});
            input.messages.push_back(std::move(message));
            input.options.enable_thinking = thinking;
            const auto prompt             = frontend.prepare(std::move(input));
            const bool expected_reasoning = thinking.value_or(true);
            if (prompt.summary().starts_in_reasoning != expected_reasoning) {
                throw std::runtime_error("thinking option selected the wrong output channel");
            }
            auto output       = frontend.make_output_session(prompt, {});
            const auto tokens = frontend.tokenize_text("READY 1");
            (void)output.preview_model(tokens, tokens.size(), ninfer::FinishReason::OutputLimit);
            const auto deltas = output.commit_preview();
            std::string text;
            for (const auto& delta : deltas) {
                const auto expected = expected_reasoning ? ninfer::OutputChannel::Reasoning
                                                         : ninfer::OutputChannel::Content;
                if (!delta.text.empty() && delta.channel != expected) {
                    throw std::runtime_error("output was published in the wrong channel");
                }
                text += delta.text;
            }
            if (text != "READY 1") { throw std::runtime_error("output text was not preserved"); }
        }
        std::cout << "OK Flash-Next default and explicit thinking output channels\n";
        check_quoted_reasoning_close(frontend);
        std::cout << "OK Flash-Next quoted reasoning close stays in reasoning\n";
        check_tool_call_recovery(frontend);
        std::cout << "OK Flash-Next tool-call recovery after quoted marker and repeated parameter\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
