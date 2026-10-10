#include "artifact/reader.h"
#include <ninfer/models/qwen3_8_flash_next/frontend.h>
#include <ninfer/models/qwen3_8_flash_next/frontend_resources.h>

#include <array>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>

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
        namespace flash = ninfer::models::qwen3_8_flash_next;
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
        {
            // OpenAI reasoning_effort "none": the server disables thinking and keeps the effort.
            ninfer::PromptInput input;
            ninfer::ChatMessage message;
            message.role = ninfer::ChatRole::User;
            message.parts.push_back(
                {.kind = ninfer::MessagePartKind::Text, .text = "Reply READY."});
            input.messages.push_back(std::move(message));
            input.options.enable_thinking  = false;
            input.options.reasoning_effort = ninfer::ReasoningEffort::None;
            const auto prompt              = frontend.prepare(std::move(input));
            if (prompt.summary().starts_in_reasoning) {
                throw std::runtime_error("reasoning effort none started in reasoning");
            }
        }
        std::cout << "OK Flash-Next default and explicit thinking output channels\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
