#include "models/qwen3_8_flash_next/impl/prompt_lookup.h"

#include <cstdint>
#include <iostream>
#include <vector>

namespace q38 = ninfer::models::qwen3_8_flash_next;

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

std::vector<std::int32_t> tokens(const q38::PromptLookup::Proposal& proposal) {
    return {proposal.tokens.begin(), proposal.tokens.begin() + proposal.count};
}

} // namespace

int main() {
    int failures = 0;
    using Lookup = q38::PromptLookup;

    {
        // The suffix 7 8 9 occurred once, followed by 10 11 12 13.
        const std::vector<std::int32_t> history{1, 7, 8, 9, 10, 11, 12, 13, 2, 3, 7, 8, 9};
        Lookup lookup;
        lookup.sync(history);
        const auto proposal = lookup.propose(history, 3);
        failures += check(tokens(proposal) == std::vector<std::int32_t>{10, 11, 12} &&
                              proposal.match == 3,
                          "a three-token match did not propose its continuation");
        failures += check(tokens(lookup.propose(history, 8)) ==
                              std::vector<std::int32_t>{10, 11, 12, 13, 2, 3, 7, 8},
                          "a long proposal did not run on to the current suffix");
    }
    {
        // The newer occurrence matches 3 tokens, the older one 5: the longer match wins.
        const std::vector<std::int32_t> history{5, 6, 7, 8, 9, 40, 1, 7, 8, 9, 50,
                                                2, 5, 6, 7, 8, 9};
        Lookup lookup;
        lookup.sync(history);
        const auto proposal = lookup.propose(history, 2);
        failures += check(tokens(proposal) == std::vector<std::int32_t>{40, 1} &&
                              proposal.match == 5,
                          "a newer short match won over an older long one");
    }
    {
        const std::vector<std::int32_t> history{1, 2, 3, 4, 5, 6, 7, 8};
        Lookup lookup;
        lookup.sync(history);
        failures += check(lookup.propose(history, 3).count == 0,
                          "a history without a repeated suffix proposed drafts");
    }
    {
        // Periodic text: the continuation reads into the current suffix.
        const std::vector<std::int32_t> history{1, 2, 3, 1, 2, 3};
        Lookup lookup;
        lookup.sync(history);
        failures += check(tokens(lookup.propose(history, 3)) == std::vector<std::int32_t>{1, 2, 3},
                          "a periodic history did not continue its period");
    }
    {
        // Indexing a ledger in rounds equals indexing it at once.
        std::vector<std::int32_t> history;
        Lookup incremental;
        for (int round = 0; round < 40; ++round) {
            for (int i = 0; i < 3; ++i) { history.push_back((round * 7 + i * 3) % 11); }
            incremental.sync(history);
        }
        Lookup whole;
        whole.sync(history);
        const auto a = incremental.propose(history, 5);
        const auto b = whole.propose(history, 5);
        failures += check(tokens(a) == tokens(b) && a.match == b.match && a.count > 0,
                          "incremental indexing proposed differently from a whole index");
    }
    {
        // Policy: a short match against the MTP prior is not worth it, a longer one is.
        Lookup lookup;
        Lookup::Proposal short_match;
        short_match.count = 3;
        short_match.match = 3;
        Lookup::Proposal long_match = short_match;
        long_match.match            = 8;
        failures += check(!lookup.prefer(short_match, 3) && lookup.prefer(long_match, 3),
                          "the lookup priors did not separate short and long matches");
        failures += check(lookup.prefer(short_match, 0),
                          "a lookup proposal lost to a round without MTP drafts");
        for (int round = 0; round < 40; ++round) { lookup.observe_lookup(8, 3, 0); }
        failures += check(!lookup.prefer(long_match, 3),
                          "rejected lookup drafts kept the lookup preferred");
        Lookup confident;
        for (int round = 0; round < 200; ++round) { confident.observe_mtp(3, 3); }
        failures += check(!confident.prefer(long_match, 3) && confident.mtp_expected(3) > 2.9,
                          "an MTP layer accepting every draft lost to a prior lookup rate");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
