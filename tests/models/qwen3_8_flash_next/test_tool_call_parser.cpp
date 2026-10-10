#include "models/qwen3_8_flash_next/impl/frontend/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using Json   = nlohmann::json;
namespace fi = ninfer::models::qwen3_8_flash_next::frontend_internal;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

std::string bash_call(const std::string& command, const std::string& note) {
    return "<tool_call>\n<function=bash>\n<parameter=command>\n" + command +
           "\n</parameter>\n<parameter=note>\n" + note +
           "\n</parameter>\n</function>\n</tool_call>";
}

} // namespace

int main() {
    Json properties{{"command", Json{{"type", "string"}}}, {"note", Json{{"type", "string"}}}};
    Json function{{"name", "bash"},
                  {"parameters", Json{{"type", "object"}, {"properties", properties}}}};
    const std::vector<std::string> tools = {
        Json{{"type", "function"}, {"function", function}}.dump()};
    const auto contract = fi::build_tool_call_output_contract(
        std::span<const std::string>(tools.data(), tools.size()), true);

    int failures = 0;
    // A close ends a value only where the call continues with the next parameter or the function
    // close; any other standalone close is value text.
    const std::string quoted_close = "echo '</parameter>'";
    const std::string prose_close  = "End each value with </parameter> on its own line.";
    const auto parsed =
        fi::parse_qwen_tool_call_output(bash_call(quoted_close, prose_close), 64, *contract);
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "standalone parameter close inside a value rejected the call");
    if (parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.at("command") == quoted_close && args.at("note") == prose_close,
                          "standalone parameter close changed the value bytes");
    }

    // Balanced nested markup is value text as before; an unmatched nested open stays content.
    const std::string nested = "<parameter=inner>value</parameter>";
    const auto balanced = fi::parse_qwen_tool_call_output(bash_call(nested, "x"), 64, *contract);
    failures +=
        check(balanced.tool_calls.size() == 1 &&
                  Json::parse(balanced.tool_calls.front().arguments_json).at("command") == nested,
              "balanced nested parameter markup was not preserved");
    const std::string unmatched = bash_call("echo '<parameter=open>'", "x");
    const auto rejected         = fi::parse_qwen_tool_call_output(unmatched, 64, *contract);
    failures += check(!rejected.is_tool_call_response && rejected.content == unmatched,
                      "unbalanced nested parameter open was silently repaired");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
