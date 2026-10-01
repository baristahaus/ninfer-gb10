#include "ninfer/ops/flash_next_ple_stage.h"
#include "core/device.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr int kHeads             = ops::kFlashNextPleStageHeads;
constexpr std::int32_t kEosToken = 248044;

using Ids = std::array<std::uint32_t, kHeads>;

// Independent whole-sequence reference: the Flash-Next PLE n-gram hash with its segment rule
// (the n-gram context restarts after every end-of-sequence token).
std::vector<Ids> reference_ids(const std::vector<std::int32_t>& sequence) {
    constexpr std::array<std::uint64_t, 3> multipliers = {23703573157769ULL, 20109073645365ULL,
                                                          8052911324071ULL};
    constexpr std::array<std::uint32_t, kHeads> sizes  = {
        20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
        20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
    std::vector<Ids> out(sequence.size());
    std::size_t segment = 0;
    for (std::size_t p = 0; p < sequence.size(); ++p) {
        const std::int32_t s1 = segment >= 1 ? sequence[p - 1] : kEosToken;
        const std::int32_t s2 = segment >= 2 ? sequence[p - 2] : kEosToken;
        const auto mul        = [](std::int32_t token, std::uint64_t m) {
            return static_cast<std::uint64_t>(static_cast<std::int64_t>(token)) * m;
        };
        const std::uint64_t bigram  = mul(sequence[p], multipliers[0]) ^ mul(s1, multipliers[1]);
        const std::uint64_t trigram = bigram ^ mul(s2, multipliers[2]);
        std::uint32_t offset        = 0;
        for (int h = 0; h < kHeads; ++h) {
            const auto value = static_cast<std::int64_t>(h < 8 ? bigram : trigram);
            std::int64_t rem = value % static_cast<std::int64_t>(sizes[h]);
            if (rem < 0) { rem += sizes[h]; }
            out[p][h] = offset + static_cast<std::uint32_t>(rem);
            offset += sizes[h];
        }
        segment = sequence[p] == kEosToken ? 0 : segment + 1;
    }
    return out;
}

struct PinnedMailbox {
    ops::FlashNextPleStageMailbox* box = nullptr;

    PinnedMailbox() {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&box), sizeof(*box)));
        *box = {};
    }

    ~PinnedMailbox() { cudaFreeHost(box); }
};

std::uint32_t load(std::uint32_t& value) {
    return std::atomic_ref<std::uint32_t>(value).load(std::memory_order_acquire);
}

// Rows of `width` tokens with two history tokens each (negative: none); every row's IDs must
// equal the reference over that row's whole known sequence. Answers the handshake from a host
// thread, as the Program's gather stage does.
int run_publish(const std::string& label, int width, const std::vector<std::int32_t>& tokens,
                const std::vector<std::int32_t>& history,
                const std::vector<Ids>* fixed_expected = nullptr) {
    const int batch = static_cast<int>(history.size() / 2);
    std::vector<std::uint32_t> expected;
    for (int b = 0; b < batch; ++b) {
        std::vector<std::int32_t> sequence;
        const std::int32_t h1 = history[static_cast<std::size_t>(2 * b)];
        const std::int32_t h2 = history[static_cast<std::size_t>(2 * b + 1)];
        if (h1 >= 0 && h2 >= 0) { sequence.push_back(h2); }
        if (h1 >= 0) { sequence.push_back(h1); }
        const std::size_t known = sequence.size();
        for (int j = 0; j < width; ++j) {
            sequence.push_back(tokens[static_cast<std::size_t>(b * width + j)]);
        }
        const std::vector<Ids> ids = reference_ids(sequence);
        for (int j = 0; j < width; ++j) {
            const Ids& row = ids[known + static_cast<std::size_t>(j)];
            expected.insert(expected.end(), row.begin(), row.end());
        }
    }
    int failures = 0;
    if (fixed_expected != nullptr) {
        std::vector<std::uint32_t> fixed;
        for (const Ids& row : *fixed_expected) {
            fixed.insert(fixed.end(), row.begin(), row.end());
        }
        failures += verify_exact((label + " reference vs recorded IDs").c_str(), expected, fixed);
    }

    PinnedMailbox mailbox;
    DeviceBuffer d_tokens  = to_device(tokens);
    DeviceBuffer d_history = to_device(history);
    Tensor t_tokens(d_tokens.p, DType::I32, {width, batch});
    Tensor t_history(d_history.p, DType::I32, {2, batch});

    std::vector<std::uint32_t> published;
    std::int32_t published_tokens = 0;
    std::thread host([&] {
        while (load(mailbox.box->request_sequence) != 1U) { std::this_thread::yield(); }
        published_tokens = mailbox.box->tokens;
        published.assign(mailbox.box->ids,
                         mailbox.box->ids + static_cast<std::size_t>(width * batch * kHeads));
        std::atomic_ref<std::uint32_t>(mailbox.box->answer_sequence)
            .store(1U, std::memory_order_release);
    });
    ops::flash_next_ple_publish_ids(t_tokens, t_history, mailbox.box, nullptr);
    ops::flash_next_ple_wait_staged(mailbox.box, 10'000'000'000ULL, nullptr);
    cuda_synchronize();
    host.join();

    failures += verify_exact((label + " IDs").c_str(), published, expected);
    if (published_tokens != width * batch || load(mailbox.box->late) != 0U) {
        std::cerr << label << ": token count or late flag is wrong\n";
        ++failures;
    }

    // An unanswered request must time out and raise the late flag instead of hanging.
    ops::flash_next_ple_publish_ids(t_tokens, t_history, mailbox.box, nullptr);
    ops::flash_next_ple_wait_staged(mailbox.box, 2'000'000ULL, nullptr);
    cuda_synchronize();
    if (load(mailbox.box->request_sequence) != 2U || load(mailbox.box->late) != 1U) {
        std::cerr << label << ": unanswered request did not time out\n";
        ++failures;
    }
    return failures;
}

int run() {
    int failures = 0;
    // The recorded file-backed reference IDs (test_ple_table) for one segment-crossing row.
    const std::vector<Ids> recorded = {
        {16121432, 28938500, 59087997, 73487090, 81148277, 104500129, 120276032, 149373875,
         176283436, 184305849, 216528839, 231080079, 257961536, 266068568, 289043455, 305959965},
        {15220011, 32170723, 40646129, 76511807, 98682440, 106072663, 127158041, 141938523,
         170367010, 180520041, 210849650, 234005644, 252857364, 272068622, 291885712, 311569748},
        {14605717, 24410875, 49313567, 72177428, 86060820, 104022010, 130963819, 146886202,
         161523077, 195939126, 219424565, 220811782, 248020141, 275228530, 284297999, 309645223},
        {1040936, 20493796, 54629949, 79359381, 85142054, 114677089, 129861487, 151753631,
         160770146, 187379937, 207108329, 229509886, 243927029, 279402641, 291463027, 313528485},
        {16786187, 37399507, 51447157, 75303773, 99642929, 108554057, 122668943, 142885423,
         178075680, 189995935, 213432942, 234309713, 243139806, 273195768, 283486804, 316984683},
        {8807125, 30450454, 51272170, 64422588, 81408710, 113737508, 122230591, 146888097,
         162512814, 189149098, 215375741, 226808975, 243902281, 261355125, 280586077, 317339594},
    };
    failures +=
        run_publish("PLE publish recorded", 6, {1, 2, 3, kEosToken, 4, 5}, {-1, -1}, &recorded);

    // Eight rows of six columns (the 64-token mailbox holds the MTP maximum, 8 x 6): no
    // history, one history token, two, an end-of-sequence token in either history slot, and
    // end-of-sequence tokens among the columns.
    std::vector<std::int32_t> tokens;
    for (int b = 0; b < 8; ++b) {
        for (int j = 0; j < 6; ++j) { tokens.push_back(1000 + 9973 * b + 131 * j); }
    }
    tokens[2 * 6 + 0]                       = kEosToken;
    tokens[5 * 6 + 3]                       = kEosToken;
    tokens[7 * 6 + 5]                       = kEosToken;
    const std::vector<std::int32_t> history = {-1,  -1,        501, -1,  502, 503, kEosToken, 504,
                                               505, kEosToken, 506, 507, 508, 509, 0,         1};
    failures += run_publish("PLE publish batch", 6, tokens, history);
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) { return 77; }
    try {
        const int failures = run();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Flash-Next PLE stage\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Flash-Next PLE stage: " << error.what() << '\n';
        return 1;
    }
}
