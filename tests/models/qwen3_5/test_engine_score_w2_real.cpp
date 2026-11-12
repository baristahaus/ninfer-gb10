// Causal scoring at tensor-parallel width 2, compared against a RECORDED width-1 reference.
//
// The two-GPU lane's acceptance criterion judges a token disagreement by the reference's top-1
// margin, so the sharded arm has to be able to score: otherwise a sharded implementation that is
// confidently wrong in a way the reference cannot see passes. This is the comparison that promotes
// the --causal-scoring row of the width>1 option gate.
//
// No host in this lab holds both the 27B and two 32 GB-class devices, so the reference cannot be
// co-resident with the sharded arm. The test therefore has two modes:
//
//   NINFER_SCORE_MODE=capture   width 1 on one device: scores the fixed sequence and writes the
//                               per-position log-probabilities to a fixture, with the artifact
//                               digest, the token ids and the kv profile recorded beside them.
//   NINFER_SCORE_MODE=compare   width 2 on two devices: reads that fixture, refuses if the artifact
//                               digest or the token ids differ, scores the same sequence, and
//                               reports max/mean/p99 |delta log p| plus the adjudication property.
//
// A reference that cannot be silently invalidated is the point of the digest: two hosts once
// carried different 27B quantizations at identical byte counts, and a comparison across them would
// have read as a lane defect.
#include "ninfer/engine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

// 4x the largest route difference measured on the 125B (0.0281 nats), rounded up: a sharded scoring
// path that is right lands inside it, one that is wrong does not.
constexpr double kToleranceNats = 0.12;
constexpr std::uint32_t kMaxContext = 2048;
constexpr std::size_t kTokenCount   = 256;
constexpr std::size_t kAlteredIndex = 128;

int failures = 0;

void fail(const std::string& message) {
    std::cerr << message << '\n';
    ++failures;
}

// --- artifact digest: streaming SHA-256 over the whole file --------------------------------
struct Sha256 {
    std::array<std::uint32_t, 8> state{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                       0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::array<std::uint8_t, 64> block{};
    std::size_t filled  = 0;
    std::uint64_t bytes = 0;

    static std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void compress() {
        static constexpr std::array<std::uint32_t, 64> k{
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2};
        std::array<std::uint32_t, 64> w{};
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<std::uint32_t>(block[i * 4]) << 24) |
                   (static_cast<std::uint32_t>(block[i * 4 + 1]) << 16) |
                   (static_cast<std::uint32_t>(block[i * 4 + 2]) << 8) |
                   static_cast<std::uint32_t>(block[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i]                   = w[i - 16] + s0 + w[i - 7] + s1;
        }
        auto a = state[0], b = state[1], c = state[2], d = state[3];
        auto e = state[4], f = state[5], g = state[6], h = state[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ (~e & g);
            const std::uint32_t t1 = h + s1 + ch + k[i] + w[i];
            const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2 = s0 + mj;
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }

    void update(const std::uint8_t* data, std::size_t size) {
        bytes += size;
        for (std::size_t i = 0; i < size; ++i) {
            block[filled++] = data[i];
            if (filled == block.size()) { compress(); filled = 0; }
        }
    }

    std::string finish() {
        const std::uint64_t bits = bytes * 8;
        std::array<std::uint8_t, 8> length{};
        for (int i = 0; i < 8; ++i) length[i] = static_cast<std::uint8_t>(bits >> (56 - 8 * i));
        const std::uint8_t pad = 0x80;
        update(&pad, 1);
        const std::uint8_t zero = 0;
        while (filled != 56) update(&zero, 1);
        update(length.data(), length.size());
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (const auto word : state) out << std::setw(8) << word;
        return out.str();
    }
};

std::string artifact_sha256(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { return {}; }
    Sha256 hash;
    std::vector<std::uint8_t> buffer(8u << 20);
    while (file) {
        file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        const auto read = static_cast<std::size_t>(file.gcount());
        if (read) hash.update(buffer.data(), read);
    }
    return hash.finish();
}

// --- the fixture ---------------------------------------------------------------------------
struct Reference {
    std::string sequence;
    std::string artifact_sha256;
    std::uint64_t artifact_bytes = 0;
    std::string kv_dtype;
    std::uint32_t max_context = 0;
    std::uint32_t first_target = 0;
    std::vector<ninfer::TokenId> tokens;
    std::vector<float> logprobs;
    std::size_t altered_index = 0;
    ninfer::TokenId altered_token = 0;
    std::vector<float> altered_logprobs;
};

bool write_reference(const std::string& path, const Reference& reference) {
    std::ofstream out(path, std::ios::trunc);
    if (!out) { return false; }
    out << "score_w2_reference 1\n";
    out << "sequence " << reference.sequence << '\n';
    out << "artifact_sha256 " << reference.artifact_sha256 << '\n';
    out << "artifact_bytes " << reference.artifact_bytes << '\n';
    out << "kv_dtype " << reference.kv_dtype << '\n';
    out << "max_context " << reference.max_context << '\n';
    out << "first_target " << reference.first_target << '\n';
    out << "tokens " << reference.tokens.size() << '\n';
    for (const auto token : reference.tokens) out << token << ' ';
    out << '\n';
    out << "logprobs " << reference.logprobs.size() << '\n';
    out << std::setprecision(9);
    for (const auto value : reference.logprobs) out << value << ' ';
    out << '\n';
    out << "altered " << reference.altered_index << ' ' << reference.altered_token << ' '
        << reference.altered_logprobs.size() << '\n';
    for (const auto value : reference.altered_logprobs) out << value << ' ';
    out << '\n';
    return static_cast<bool>(out);
}

// The reason a reference was rejected is reported, not just the fact: "could not read the
// reference" sent one debugging round down a format path when the real fault was a size invariant.
bool read_reference(const std::string& path, Reference& reference, std::string& error) {
    std::ifstream in(path);
    if (!in) {
        error = "cannot open " + path;
        return false;
    }
    std::string magic;
    int version = 0;
    in >> magic >> version;
    if (magic != "score_w2_reference" || version != 1) {
        error = path + " is not a score_w2_reference v1 file";
        return false;
    }
    std::string key;
    std::size_t count = 0;
    while (in >> key) {
        if (key == "sequence") in >> reference.sequence;
        else if (key == "artifact_sha256") in >> reference.artifact_sha256;
        else if (key == "artifact_bytes") in >> reference.artifact_bytes;
        else if (key == "kv_dtype") in >> reference.kv_dtype;
        else if (key == "max_context") in >> reference.max_context;
        else if (key == "first_target") in >> reference.first_target;
        else if (key == "tokens") {
            in >> count;
            reference.tokens.resize(count);
            for (auto& token : reference.tokens) in >> token;
        } else if (key == "logprobs") {
            in >> count;
            reference.logprobs.resize(count);
            for (auto& value : reference.logprobs) in >> value;
        } else if (key == "altered") {
            in >> reference.altered_index >> reference.altered_token >> count;
            reference.altered_logprobs.resize(count);
            for (auto& value : reference.altered_logprobs) in >> value;
        } else {
            error = "unknown key in " + path + ": " + key;
            return false;
        }
    }
    // first_target = 1 means the first token has no prediction, so N tokens produce N - first_target
    // positions. Equality between the two counts is impossible for any correct capture.
    if (reference.tokens.empty() || reference.first_target == 0 ||
        reference.first_target >= reference.tokens.size()) {
        error = "first_target is outside the token sequence";
        return false;
    }
    if (reference.logprobs.size() + reference.first_target != reference.tokens.size()) {
        error = "logprob count " + std::to_string(reference.logprobs.size()) + " + first_target " +
                std::to_string(reference.first_target) + " != token count " +
                std::to_string(reference.tokens.size());
        return false;
    }
    if (reference.altered_logprobs.size() != reference.logprobs.size()) {
        error = "altered logprob count " + std::to_string(reference.altered_logprobs.size()) +
                " != logprob count " + std::to_string(reference.logprobs.size());
        return false;
    }
    return true;
}

// --- engines ---------------------------------------------------------------------------------
ninfer::EngineOptions options_for(const char* artifact, int width) {
    ninfer::EngineOptions options;
    options.artifact_path         = artifact;
    options.purpose               = ninfer::EnginePurpose::CausalScoring;
    options.use_cuda_graph        = false; // width > 1 refuses capture
    options.context_cache.enabled = false; // unvalidated at width > 1
    options.max_context           = kMaxContext;
    // bf16 KV: a quantized pool is its own unvalidated width-2 row and would refuse before scoring
    // ran, so both arms use the configuration the lane is actually validated for.
    options.kv_cache              = ninfer::KvCacheStorage::BFloat16;
    options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(kMaxContext);
    if (width > 1) {
        options.devices = {0, 1};
    } else {
        options.device = 0;
    }
    return options;
}

// The fixed sequence is tokenized by the arm's own engine. It used to be tokenized by a width-1
// engine, which on a 16 GB card is a full 20.02 GiB model and refused before the width-2 arm was
// ever constructed -- a test-side failure that read like a lane defect. One engine per arm.
// The default sequence repeats one paragraph, so a token at position p recurs at p+31: a per-target
// defect and a per-offset-within-the-period defect produce the SAME wrong set, which is a confound.
// `distinct` takes 256 ids that never recur, so wrongness can only follow the target or the offset,
// not both. `control` takes the distinct sequence and repeats ONE target at two different offsets,
// which is the direct control for the confound.
std::vector<ninfer::TokenId> sequence_for(ninfer::Engine& engine, const std::string& kind) {
    const std::string first =
        "The quick brown fox jumps over the lazy dog. NInfer scores every target token from the "
        "preceding hidden state and compares two routes with a tolerance.\n";
    const std::string second =
        "A copper kettle whistles on the stove while winter rain drums against the window; the "
        "orchard gate swings, and a grey heron waits beside the frozen pond.\n";
    const std::string third =
        "Bricklayers repaired the viaduct in March, hauling limestone by narrowboat; engineers "
        "measured each arch with brass calipers and recorded the sag in a leather ledger.\n";
    const std::string fourth =
        "She translated the pharmacist's instructions from Portuguese, folded the prescription "
        "twice, and slipped it beneath the porcelain teapot on the kitchen dresser.\n";
    const std::string fifth =
        "Quantum tunnelling puzzles undergraduates, yet the mathematics is elementary: a wave "
        "decays exponentially through a barrier, and the transmitted amplitude falls with width.\n";
    std::vector<ninfer::TokenId> tokens;
    if (kind == "periodic") {
        std::string text;
        while (tokens.size() < kTokenCount) {
            text += first;
            tokens = engine.tokenize_text(text);
        }
        tokens.resize(kTokenCount);
        return tokens;
    }
    // Distinct: walk a long, varied text and keep only ids not seen yet. The pool is the whole
    // varied corpus, not two repeated paragraphs: two English paragraphs do not contain 256 distinct
    // ids, because common words recur and every repeat is dropped.
    std::string text;
    for (int pass = 0; pass < 3; ++pass) text += first + second + third + fourth + fifth;
    const auto all   = engine.tokenize_text(text);
    std::vector<bool> seen(250000, false);
    for (const auto token : all) {
        if (tokens.size() == kTokenCount) break;
        if (token < 0 || static_cast<std::size_t>(token) >= seen.size()) continue;
        if (seen[token]) continue;
        seen[token] = true;
        tokens.push_back(token);
    }
    // The measurement needs targets that never recur, not a particular count. A shorter distinct
    // sequence is just as decisive; a pool too small to reach the floor is refused rather than
    // silently shortened into a weaker measurement.
    constexpr std::size_t kMinimumDistinct = 96;
    if (tokens.size() < kMinimumDistinct) {
        throw std::runtime_error("the distinct sequence has only " + std::to_string(tokens.size()) +
                                 " unique ids; at least " + std::to_string(kMinimumDistinct) +
                                 " are needed");
    }
    if (kind == "control") {
        // One target at two deliberately different offsets: the control for the confound.
        tokens[64]  = tokens[0];
        tokens[192] = tokens[0];
    }
    return tokens;
}

// A refusal at construction is a result, not an abort: the message names the width and the reason.
std::unique_ptr<ninfer::Engine> build(const char* artifact, int width) {
    std::cerr << "constructing width=" << width << " CausalScoring engine\n";
    try {
        return std::make_unique<ninfer::Engine>(options_for(artifact, width));
    } catch (const std::exception& error) {
        fail(std::string("engine construction refused at width ") + std::to_string(width) + ": " +
             error.what());
        return nullptr;
    }
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) { return 0.0; }
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(fraction * static_cast<double>(values.size() - 1));
    return values[std::min(index, values.size() - 1)];
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    const char* mode     = std::getenv("NINFER_SCORE_MODE");
    const char* fixture  = std::getenv("NINFER_SCORE_FIXTURE");
    int devices          = 0;
    const cudaError_t status = cudaGetDeviceCount(&devices);
    const int device_count   = status == cudaSuccess ? devices : 0;
    const bool capture       = mode != nullptr && std::string(mode) == "capture";
    const bool compare       = mode != nullptr && std::string(mode) == "compare";

    if (artifact == nullptr || *artifact == '\0' || fixture == nullptr || *fixture == '\0' ||
        (!capture && !compare)) {
        std::cerr << "SKIP: set NINFER_TEST_ARTIFACT, NINFER_SCORE_FIXTURE and NINFER_SCORE_MODE="
                     "capture (width 1, one device) or compare (width 2, two devices)\n";
        return 77;
    }
    if (capture && device_count < 1) {
        std::cerr << "SKIP: capture mode needs one CUDA device; found " << device_count << '\n';
        return 77;
    }
    if (compare && device_count < 2) {
        std::cerr << "SKIP: compare mode needs two CUDA devices; found " << device_count << '\n';
        return 77;
    }
    if (compare && std::getenv("NINFER_TP2_UNVALIDATED") == nullptr) {
        std::cerr << "SKIP: the width-2 arm runs under the lane's inference escape; set "
                     "NINFER_TP2_UNVALIDATED=1 to opt in\n";
        return 77;
    }

    const std::string digest = artifact_sha256(artifact);
    if (digest.empty()) {
        std::cerr << "artifact digest could not be computed\n";
        return 1;
    }
    auto engine = build(artifact, capture ? 1 : 2);
    if (!engine) { return 1; }
    const char* sequence_env = std::getenv("NINFER_SCORE_SEQUENCE");
    const std::string kind =
        sequence_env != nullptr && *sequence_env ? sequence_env : "periodic";
    std::vector<ninfer::TokenId> tokens;
    try {
        tokens = sequence_for(*engine, kind);
        // The count sweep: NINFER_SCORE_COUNT truncates the sequence so the same artifact and the
        // same fixture machinery measure one column count at a time.
        if (const char* count_env = std::getenv("NINFER_SCORE_COUNT");
            count_env != nullptr && *count_env) {
            const auto wanted = static_cast<std::size_t>(std::strtoul(count_env, nullptr, 10));
            if (wanted < 2 || wanted > tokens.size()) {
                fail("NINFER_SCORE_COUNT must be in [2, the sequence length]");
                return 1;
            }
            tokens.resize(wanted);
        }
    } catch (const std::exception& error) {
        fail(std::string("the sequence could not be built: ") + error.what());
        return 1;
    }

    if (capture) {
        Reference reference;
        reference.sequence        = kind;
        reference.artifact_sha256 = digest;
        {
            std::ifstream probe(artifact, std::ios::binary | std::ios::ate);
            reference.artifact_bytes = static_cast<std::uint64_t>(probe.tellg());
        }
        reference.kv_dtype     = "bf16";
        reference.max_context  = kMaxContext;
        reference.first_target = 1;
        reference.tokens       = tokens;
        reference.logprobs     = engine->score_tokens(tokens, 1);
        reference.altered_index = tokens.size() / 2;
        auto altered            = tokens;
        reference.altered_token =
            static_cast<ninfer::TokenId>((tokens[reference.altered_index] + 977) % 150000);
        altered[reference.altered_index] = reference.altered_token;
        reference.altered_logprobs = engine->score_tokens(altered, 1);
        if (!write_reference(fixture, reference)) {
            std::cerr << "could not write the reference to " << fixture << '\n';
            return 1;
        }
        std::cout << "captured " << reference.logprobs.size() << " positions to " << fixture
                  << " (artifact sha256 " << digest.substr(0, 16) << "...)\n";
        return 0;
    }

    Reference reference;
    std::string reference_error;
    if (!read_reference(fixture, reference, reference_error)) {
        std::cerr << "REFUSED: the reference could not be read: " << reference_error << '\n';
        return 1;
    }
    // A fixture captured before the sequence selector existed records no sequence; it is the
    // periodic one by construction, so an empty name means periodic rather than unusable.
    if (reference.sequence.empty()) reference.sequence = "periodic";
    if (reference.sequence != kind) {
        std::cerr << "REFUSED: the reference was captured for sequence '" << reference.sequence
                  << "' but this run asked for '" << kind << "'\n";
        return 1;
    }
    if (reference.artifact_sha256 != digest) {
        std::cerr << "REFUSED: the reference was captured against artifact sha256 "
                  << reference.artifact_sha256 << " but this box has " << digest
                  << ". A comparison across different quantizations reads as a lane defect, so it "
                     "is refused rather than reported\n";
        return 1;
    }
    if (reference.tokens != tokens) {
        std::cerr << "REFUSED: the reference's token sequence differs from this box's tokenization\n";
        return 1;
    }

    const std::vector<float> sharded = engine->score_tokens(tokens, 1);
    if (sharded.size() != reference.logprobs.size()) {
        fail("the sharded arm returned a different score count");
        return 1;
    }
    std::vector<double> deltas;
    deltas.reserve(sharded.size());
    double max_delta = 0.0;
    double mean_delta = 0.0;
    for (std::size_t i = 0; i < sharded.size(); ++i) {
        if (!std::isfinite(sharded[i]) || !std::isfinite(reference.logprobs[i])) {
            fail("a score was not finite");
            return 1;
        }
        const double delta = std::abs(static_cast<double>(sharded[i]) - reference.logprobs[i]);
        deltas.push_back(delta);
        max_delta = std::max(max_delta, delta);
        mean_delta += delta;
    }
    mean_delta /= static_cast<double>(deltas.size());
    const std::size_t over_0p01 = static_cast<std::size_t>(
        std::count_if(deltas.begin(), deltas.end(), [](double d) { return d > 0.01; }));
    const std::size_t over_0p05 = static_cast<std::size_t>(
        std::count_if(deltas.begin(), deltas.end(), [](double d) { return d > 0.05; }));
    // The distribution is the lead the fix needs: a global offset or a wrong normalisation corrupts
    // every position, so a SUBSET taking a wrong path points at something positional. Every position
    // is written to a report beside the fixture, and the worst are printed, so one run answers
    // "which positions are wrong" without a second run.
    {
        const std::string report_path = std::string(fixture) + ".positions";
        std::ofstream report(report_path, std::ios::trunc);
        std::vector<std::size_t> order(deltas.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&](std::size_t a, std::size_t b) { return deltas[a] > deltas[b]; });
        if (report) {
            report << "pos target reference sharded delta\n";
            report << std::setprecision(9);
            for (std::size_t i = 0; i < deltas.size(); ++i) {
                report << i << ' ' << tokens[reference.first_target + i] << ' '
                       << reference.logprobs[i] << ' ' << sharded[i] << ' ' << deltas[i] << '\n';
            }
        }
        std::cout << "worst positions (pos target reference sharded delta):\n";
        for (std::size_t k = 0; k < std::min<std::size_t>(20, order.size()); ++k) {
            const std::size_t i = order[k];
            std::cout << "  " << i << ' ' << tokens[reference.first_target + i] << ' '
                      << reference.logprobs[i] << ' ' << sharded[i] << ' ' << deltas[i] << '\n';
        }
        std::cout << "position report: " << report_path << '\n';
    }
    std::cout << "positions=" << deltas.size() << " max|dlogp|=" << max_delta
              << " mean|dlogp|=" << mean_delta << " p99|dlogp|=" << percentile(deltas, 0.99)
              << " over0.01=" << over_0p01 << " over0.05=" << over_0p05 << '\n';
    if (max_delta > kToleranceNats) {
        fail("width-2 scoring differs from the recorded width-1 reference beyond the tolerance of " +
             std::to_string(kToleranceNats) + " nats");
    }

    // The property adjudication needs: both routes must find the real next token clearly better than
    // a deliberately altered one, and agree on how much worse.
    auto altered = tokens;
    altered[reference.altered_index] = reference.altered_token;
    const std::vector<float> sharded_altered = engine->score_tokens(altered, 1);
    const double reference_margin =
        static_cast<double>(reference.logprobs[reference.altered_index]) -
        reference.altered_logprobs[reference.altered_index];
    const double sharded_margin = static_cast<double>(sharded[reference.altered_index]) -
                                  sharded_altered[reference.altered_index];
    std::cout << "margin reference=" << reference_margin << " sharded=" << sharded_margin << '\n';
    if (reference_margin <= 0.0 || sharded_margin <= 0.0) {
        fail("an unambiguous next token was not preferred over the altered one");
    }
    if (std::abs(reference_margin - sharded_margin) > kToleranceNats) {
        fail("the two routes disagree on how much worse the altered token is");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
