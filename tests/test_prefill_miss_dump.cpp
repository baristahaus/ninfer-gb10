// serve/prefill_miss_dump: which prefix-reuse misses qualify for a dump, which do not, what the
// dump contains, and that a failed dump never throws. The detokenizer is injected, so this runs
// without an Engine or a GPU. Ported case coverage from capy-flash's prefill-miss behaviour.
#include "serve/prefill_miss_dump.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using ninfer::TokenId;
using ninfer::serve::common_prefix_length;
using ninfer::serve::PrefillMissDump;
using ninfer::serve::PrefillMissDumpOptions;
using ninfer::serve::PrefillMissReport;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << "\n";
    return 1;
}

std::vector<TokenId> sequence(std::size_t count, TokenId first) {
    std::vector<TokenId> tokens(count);
    for (std::size_t index = 0; index < count; ++index) {
        tokens[index] = first + static_cast<TokenId>(index);
    }
    return tokens;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

// The detokenizer stands in for the Engine's: one token is one character offset from 'a'.
std::string fake_detokenize(std::span<const TokenId> tokens) {
    std::string text;
    for (const TokenId token : tokens) {
        text.push_back(static_cast<char>('a' + token % 26));
    }
    return text;
}

PrefillMissDumpOptions options_for(const std::filesystem::path& directory,
                                   std::uint32_t min_tokens = 10,
                                   double min_match         = 0.5,
                                   std::size_t history      = 16) {
    PrefillMissDumpOptions options;
    options.directory  = directory;
    options.min_tokens = min_tokens;
    options.min_match  = min_match;
    options.history    = history;
    return options;
}

} // namespace

int main() {
    int failures = 0;

    // Longest common prefix, including the empty and the equal-sequence cases.
    {
        const auto a = sequence(4, 0);
        const auto b = sequence(4, 0);
        failures += check(common_prefix_length(a, b) == 4, "identical sequences lost their prefix");
        failures += check(common_prefix_length({}, b) == 0, "an empty sequence had a prefix");
        std::vector<TokenId> diverging{0, 1, 9, 3};
        failures += check(common_prefix_length(a, diverging) == 2, "the prefix length was wrong");
    }

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        ("ninfer_prefill_miss_test_" + std::to_string(static_cast<long>(::getpid())));
    std::filesystem::remove_all(root);

    // Append-only: the client added 50 tokens after a fully reused 100-token prefix. Nothing was
    // recomputed and the earlier context has only 2 tokens after the shared prefix, so no dump.
    {
        PrefillMissDump dump(options_for(root / "append"), fake_detokenize);
        const auto old_prompt = sequence(100, 0);
        const auto old_output = sequence(2, 0);
        const auto new_prompt = sequence(150, 0);
        failures += check(!dump.observe(old_prompt, old_output, 0, "first").has_value(),
                          "the first request of a process was dumped");
        failures += check(!dump.observe(new_prompt, {}, 100, "append-only").has_value(),
                          "an append-only request was dumped");
    }

    // Server-side reading: nothing was reused, but 80 tokens identical to the closest context were
    // recomputed, so the resume point was earlier than the shared prefix allowed.
    {
        PrefillMissDump dump(options_for(root / "server"), fake_detokenize);
        (void)dump.observe(sequence(100, 0), sequence(2, 0), 0, "old");
        auto new_prompt = sequence(80, 0);
        const auto tail = sequence(20, 500);
        new_prompt.insert(new_prompt.end(), tail.begin(), tail.end());
        const auto report = dump.observe(new_prompt, {}, 0, "server-miss");
        failures += check(report.has_value(), "a server-side reuse miss was not dumped");
        if (report) {
            failures += check(report->common_prefix == 80 && report->reused_tokens == 0 &&
                                  report->prompt_tokens == 100,
                              "the report did not describe the shared prefix and the reuse");
            const std::filesystem::path directory(report->directory);
            for (const char* name : {"new_prompt.txt", "old_context.txt", "new_prompt.tokens.json",
                                     "old_context.tokens.json", "summary.json"}) {
                failures += check(std::filesystem::exists(directory / name),
                                  "the dump is missing a file");
            }
            const auto summary =
                nlohmann::json::parse(read_file(directory / "summary.json"));
            failures += check(summary.at("divergence").at("token_index") == 80,
                              "summary.json lost the divergence index");
            failures += check(summary.at("recomputed_shared_tokens") == 80,
                              "summary.json lost the recomputed shared tokens");
            // Both readings can hold at once: the earlier context also has tokens past the
            // divergence (its committed output), so the client-side reading is reported too.
            const auto& reading = summary.at("reading");
            failures += check(reading.size() == 2 &&
                                  reading.at(0).get<std::string>().rfind("client:", 0) == 0 &&
                                  reading.at(1).get<std::string>().rfind("server:", 0) == 0,
                              "summary.json did not report both readings");
            failures += check(read_file(directory / "new_prompt.txt") == fake_detokenize(new_prompt),
                              "new_prompt.txt is not the detokenized prompt");
        }
    }

    // Client-side reading: the whole 60-token shared prefix was reused, so nothing was recomputed;
    // the earlier context has 90 tokens past the divergence, so the client changed history.
    {
        PrefillMissDump dump(options_for(root / "client"), fake_detokenize);
        (void)dump.observe(sequence(100, 0), sequence(50, 200), 0, "old");
        auto new_prompt = sequence(60, 0);
        const auto tail = sequence(40, 700);
        new_prompt.insert(new_prompt.end(), tail.begin(), tail.end());
        const auto report = dump.observe(new_prompt, {}, 60, "client-miss");
        failures += check(report.has_value(), "a client-side rewrite was not dumped");
        if (report) {
            const auto summary = nlohmann::json::parse(
                read_file(std::filesystem::path(report->directory) / "summary.json"));
            failures += check(summary.at("reading").size() == 1 &&
                                  summary.at("reading").at(0).get<std::string>().rfind(
                                      "client:", 0) == 0,
                              "summary.json did not report the client-side reading");
        }
    }

    // Below min_tokens there is nothing to diagnose; below min_match the contexts are unrelated.
    {
        PrefillMissDump dump(options_for(root / "below"), fake_detokenize);
        (void)dump.observe(sequence(100, 0), {}, 0, "old");
        auto short_prompt = sequence(100, 0);
        short_prompt.resize(105);
        failures += check(!dump.observe(short_prompt, {}, 100, "small-prefill").has_value(),
                          "a request below min_tokens was dumped");
        PrefillMissDump unrelated(options_for(root / "unrelated"), fake_detokenize);
        (void)unrelated.observe(sequence(100, 0), {}, 0, "old");
        auto unrelated_prompt = sequence(20, 0);
        const auto tail       = sequence(80, 900);
        unrelated_prompt.insert(unrelated_prompt.end(), tail.begin(), tail.end());
        failures += check(!unrelated.observe(unrelated_prompt, {}, 0, "unrelated").has_value(),
                          "a request below min_match was dumped");
    }

    // Only the last `history` contexts are considered.
    {
        PrefillMissDump dump(options_for(root / "history", 10, 0.5, 1), fake_detokenize);
        (void)dump.observe(sequence(100, 0), {}, 0, "old");
        (void)dump.observe(sequence(100, 900), {}, 0, "evicts-the-first");
        failures += check(!dump.observe(sequence(100, 0), {}, 0, "old-again").has_value(),
                          "an evicted context was still compared against");
    }

    // A systematic miss cannot fill the disk: at most 64 dumps per process.
    {
        PrefillMissDump dump(options_for(root / "cap"), fake_detokenize);
        (void)dump.observe(sequence(100, 0), {}, 0, "seed");
        std::size_t dumped = 0;
        for (int index = 0; index < 70; ++index) {
            if (dump.observe(sequence(100, 0), {}, 0, "repeat")) { ++dumped; }
        }
        failures += check(dumped == 64, "the dump cap is not 64");
    }

    // A detokenizer that throws loses the dump but not the process, and the context is remembered.
    {
        PrefillMissDump dump(options_for(root / "throwing"),
                             [](std::span<const TokenId>) -> std::string {
                                 throw std::runtime_error("detokenizer failed");
                             });
        (void)dump.observe(sequence(100, 0), {}, 0, "old");
        const auto report = dump.observe(sequence(100, 0), {}, 0, "throws");
        failures += check(report.has_value() && report->directory.empty(),
                          "a failed dump was not reported as an empty directory");
        const auto again = dump.observe(sequence(100, 0), {}, 0, "throws-again");
        failures += check(again.has_value(), "the context was lost after a failed dump");
    }

    std::filesystem::remove_all(root);
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
