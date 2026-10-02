#include "models/qwen3_8_flash_next/impl/activation_control.h"
#include "core/device.h"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen3_8_flash_next {
namespace {
using Json                        = nlohmann::json;
constexpr std::size_t kDirections = 48ULL * ops::kSteeringMaxRank * 4 * 2560;

void require(bool condition, const std::string& check) {
    if (!condition) throw std::invalid_argument("activation check failed: " + check);
}
} // namespace

SteeringPack read_steering_pack(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    require(bool(file), "pack_open " + path.string());
    file.seekg(0, std::ios::end);
    const auto bytes = file.tellg();
    file.seekg(0);
    std::uint64_t length = 0;
    file.read(reinterpret_cast<char*>(&length), 8);
    require(bool(file) && length <= 16 * 1024 * 1024 && bytes >= 8 &&
                length <= static_cast<std::uint64_t>(bytes) - 8,
            "pack_header_length");
    std::string text(length, '\0');
    file.read(text.data(), text.size());
    const auto header = Json::parse(text);
    SteeringPack out;
    out.metadata  = Json::parse(header.at("__metadata__").at("ninfer.steering").get<std::string>());
    const auto& m = out.metadata;
    require(m.at("schema") == "ninfer.steering/1", "schema_major (supported: ninfer.steering/1)");
    require(m.at("max_rank").is_number_integer() && m.at("max_rank").get<int>() >= 1 &&
                m.at("max_rank").get<int>() <= ops::kSteeringMaxRank,
            "MAX_RANK=32");
    out.norm_preserve  = m.at("norm_preserve").get<bool>();
    out.state.strength = m.at("default_steering_strength").get<double>();
    require(std::isfinite(out.state.strength) && out.state.strength >= 0 && out.state.strength <= 1,
            "default_steering_strength");
    for (const auto* key :
         {"model_sha", "model_sha_struct", "config_sha", "template_sha", "pack_sha", "created"})
        require(m.contains(key) && m.at(key).is_string(), std::string("metadata ") + key);
    out.state.pack_sha = m.at("pack_sha").get<std::string>();
    out.directions.resize(kDirections);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    for (auto it = header.begin(); it != header.end(); ++it) {
        if (it.key() == "__metadata__") continue;
        const std::string prefix = "layers.", suffix = ".directions";
        require(it.key().starts_with(prefix) && it.key().ends_with(suffix),
                "tensor_name " + it.key());
        const auto layer_text =
            it.key().substr(prefix.size(), it.key().size() - prefix.size() - suffix.size());
        std::size_t consumed = 0;
        const int layer      = std::stoi(layer_text, &consumed);
        require(consumed == layer_text.size() && layer >= 0 && layer < 48 &&
                    layer_text == std::to_string(layer),
                "layer_index");
        const auto& t    = it.value();
        const auto shape = t.at("shape").get<std::vector<int>>();
        require(t.at("dtype") == "F32" && shape.size() == 3 && shape[0] >= 1 &&
                    shape[0] <= ops::kSteeringMaxRank && shape[1] == 4 && shape[2] == 2560,
                "shape_dtype/MAX_RANK " + it.key());
        const auto& layer_meta = m.at("layers").at(layer_text);
        const int mask         = layer_meta.at("lanes").get<int>();
        require(mask >= 1 && mask <= 15 && layer_meta.at("k") == shape[0] &&
                    layer_meta.contains("provenance") && shape[0] <= m.at("max_rank").get<int>(),
                "layer_metadata");
        const auto offsets = t.at("data_offsets").get<std::vector<std::uint64_t>>();
        const auto count   = static_cast<std::size_t>(shape[0]) * 4 * 2560;
        require(offsets.size() == 2 && offsets[1] >= offsets[0] &&
                    offsets[1] - offsets[0] == count * 4 &&
                    offsets[1] <= static_cast<std::uint64_t>(bytes) - 8 - length,
                "data_offsets");
        ranges.emplace_back(offsets[0], offsets[1]);
        auto* destination = out.directions.data() +
                            static_cast<std::size_t>(layer) * ops::kSteeringMaxRank * 4 * 2560;
        file.seekg(8 + length + offsets[0]);
        file.read(reinterpret_cast<char*>(destination), count * 4);
        require(bool(file), "pack_read");
        for (int r = 0; r < shape[0]; ++r)
            for (int lane = 0; lane < 4; ++lane) {
                double norm = 0;
                for (int d = 0; d < 2560; ++d) {
                    const float value = destination[(r * 4 + lane) * 2560 + d];
                    require(std::isfinite(value), "direction_finite");
                    norm += double(value) * value;
                }
                require(std::abs(norm - 1) <= 1e-4, "direction_unit_norm");
            }
        out.ranks[layer] = shape[0];
        out.masks[layer] = mask;
        out.state.rank   = std::max(out.state.rank, shape[0]);
        out.state.layers.push_back(layer);
    }
    require(!out.state.layers.empty() && m.at("layers").size() == out.state.layers.size(),
            "present_layers");
    std::sort(out.state.layers.begin(), out.state.layers.end());
    std::sort(ranges.begin(), ranges.end());
    std::uint64_t covered = 0;
    for (auto [begin, end] : ranges) {
        require(begin == covered, "tensor_coverage");
        covered = end;
    }
    require(covered == static_cast<std::uint64_t>(bytes) - 8 - length, "tensor_coverage");
    return out;
}

ActivationControl::ActivationControl(cudaStream_t stream)
    : stream_(stream), controls_(sizeof(ops::ActivationDevice)), directions_(kDirections * 4),
      ranks_(48 * 4), masks_(48 * 4), device_(static_cast<ops::ActivationDevice*>(controls_.p)) {
    CUDA_CHECK(cudaMallocHost(&staging_, 2 * sizeof(ops::ActivationDevice)));
    for (auto& event : staged_) CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    host_.directions = static_cast<const float*>(directions_.p);
    host_.ranks      = static_cast<const int*>(ranks_.p);
    host_.masks      = static_cast<const int*>(masks_.p);
    activate(nullptr);
}

ActivationControl::~ActivationControl() {
    for (auto& event : staged_) {
        if (event != nullptr) cudaEventDestroy(event);
    }
    if (staging_ != nullptr) cudaFreeHost(staging_);
}

void ActivationControl::upload() {
    // Uploads occur at schedule boundaries, never inside graph capture. Kernels already queued
    // read the previous rows; kernels queued afterwards read these.
    if (uploaded_valid_ && std::memcmp(&uploaded_, &host_, sizeof(host_)) == 0) { return; }
    const int slot = next_slot_;
    next_slot_     = 1 - next_slot_;
    CUDA_CHECK(cudaEventSynchronize(staged_[slot]));
    staging_[slot] = host_;
    CUDA_CHECK(cudaMemcpyAsync(device_, &staging_[slot], sizeof(host_), cudaMemcpyHostToDevice,
                               stream_));
    CUDA_CHECK(cudaEventRecord(staged_[slot], stream_));
    uploaded_       = host_;
    uploaded_valid_ = true;
}

void ActivationControl::configure(const EngineOptions& options, std::string_view identity) {
    if (!identity.empty()) identities = Json::parse(identity);
    options_ = options;
    if (!options.capture_path.empty()) {
        host_.completion_capacity = options.speculative.backend == SpeculativeBackend::Mtp ? 4 : 1;
        const auto count = static_cast<std::size_t>(options.max_concurrency) *
                           (1 + host_.completion_capacity) * 48;
        samples_         = std::make_unique<DeviceBuffer>(count * ops::kActivationElements * 4);
        checks_          = std::make_unique<DeviceBuffer>(count * sizeof(ops::ActivationSample));
        samples_->fill();
        checks_->fill();
        host_.samples = static_cast<float*>(samples_->p);
        host_.checks  = static_cast<ops::ActivationSample*>(checks_->p);
    }
    upload();
}

void ActivationControl::activate(const SteeringPack* pack) {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    if (pack) {
        directions_.copy_from_host(pack->directions.data(), directions_.bytes);
        ranks_.copy_from_host(pack->ranks.data(), ranks_.bytes);
        masks_.copy_from_host(pack->masks.data(), masks_.bytes);
        host_.norm_preserve = pack->norm_preserve;
    } else {
        directions_.fill();
        ranks_.fill();
        masks_.fill();
        host_.norm_preserve = 0;
    }
    upload();
}

void ActivationControl::begin(int lane, std::uint64_t request, std::span<const TokenId> prompt,
                              bool capture, const SteeringState& steering) {
    auto& r = requests_.at(lane);
    r = Request{request, capture ? std::vector<TokenId>(prompt.begin(), prompt.end()) : std::vector<TokenId>{}, capture, steering};
    if (capture) {
        require(bool(checks_) && !prompt.empty(), "capture_path/prompt_nonempty");
        CUDA_CHECK(cudaMemsetAsync(static_cast<char*>(checks_->p) +
                                       lane * (1 + host_.completion_capacity) * 48 * sizeof(ops::ActivationSample),
                                   0, (1 + host_.completion_capacity) * 48 * sizeof(ops::ActivationSample), stream_));
    }
    const std::uint32_t selected = lane;
    select(std::span(&selected, 1));
}

void ActivationControl::select(std::span<const std::uint32_t> lanes) {
    for (auto& row : host_.rows) row = {};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const auto lane = lanes[row];
        const auto& r   = requests_.at(lane);
        host_.rows[row] = {static_cast<int>(lane), static_cast<int>(r.prompt.size()) - 1, r.capture,
                           static_cast<float>(r.steering.strength)};
        if (r.capture) {
            // Each completion snapshot is a whole round, so a missing layer cannot retain a
            // successful fire from the previous decode launch.
            CUDA_CHECK(cudaMemsetAsync(static_cast<char*>(checks_->p) +
                                           (lane * (1 + host_.completion_capacity) * 48 + 48) * sizeof(ops::ActivationSample),
                                       0, host_.completion_capacity * 48 * sizeof(ops::ActivationSample), stream_));
        }
    }
    upload();
}

std::string ActivationControl::flush(int lane, std::span<const TokenId> ledger,
                                     std::uint32_t execution_frontier) {
    const auto& r = requests_.at(lane);
    if (!r.capture) return {};
    // Uploads no longer synchronize the stream; the capture rows must be complete before reading.
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    const int sample_count = (1 + host_.completion_capacity) * 48;
    std::vector<ops::ActivationSample> checks(sample_count);
    checks_->copy_to_host(checks.data(), checks.size() * sizeof(checks[0]),
                           lane * checks.size() * sizeof(checks[0]));
    require(!r.prompt.empty() && ledger.size() >= r.prompt.size() && execution_frontier > 0 &&
                execution_frontier <= ledger.size(),
            "rendered_token_count/execution_frontier");
    Json records = Json::array();
    for (const auto& site : options_.capture_sites) {
        const int index    = site == "prompt_last" ? 0 : 1;
        const int expected = index == 0 ? static_cast<int>(r.prompt.size()) - 1
                                        : static_cast<int>(execution_frontier) - 1;
        int source_site = index;
        if (index == 1) {
            // The decoder may reject or truncate speculative columns. Resolve capture only
            // against the committed frontier, never against the last proposed token.
            bool found = false;
            for (int candidate = 1; candidate <= host_.completion_capacity && !found; ++candidate)
                for (int layer = 0; layer < 48; ++layer)
                    if (checks[candidate * 48 + layer].fires &&
                        checks[candidate * 48 + layer].position == expected) {
                        source_site = candidate;
                        found = true;
                        break;
                    }
            require(found, "fire_count/captured_index completion_last");
        }
        for (int layer = 0; layer < 48; ++layer) {
            const auto& c    = checks[source_site * 48 + layer];
            const auto where = site + " layer=" + std::to_string(layer);
            require(c.fires == 1, "fire_count " + where);
            require(c.position == expected, "captured_index " + where);
            require(c.token == (index == 0 ? r.prompt[expected] : ledger[expected]),
                    "token_id " + where);
            require(c.pad == 0, "pad_mask " + where);
            records.push_back({{"site", site},
                               {"layer", layer},
                               {"sequence_length", execution_frontier},
                               {"rendered_token_count", r.prompt.size()},
                               {"rendered_token_ids_ref", "/rendered_token_ids"},
                               {"captured_index", c.position},
                               {"token_id", c.token},
                               {"steering_strength", r.steering.strength},
                               {"steering_pack_sha", r.steering.pack_sha},
                               {"steering_generation", r.steering.generation},
                               {"offset", 0},
                               {"bytes", ops::kActivationElements * 4},
                               {"source_index", source_site * 48 + layer}});
        }
    }
    const auto id = std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) +
                    "-" + std::to_string(r.id);
    Json header{{"schema", "ninfer.activation/1"},
                {"request_id", r.id},
                {"record_id", id},
                {"artifact_identity", identities},
                {"steering", {{"pack_sha", r.steering.pack_sha},
                               {"strength", r.steering.strength},
                               {"generation", r.steering.generation}}},
                {"capture_stage", "post_steering_before_normalisation"},
                {"record_count", records.size()},
                {"rendered_token_ids", r.prompt},
                {"engine_flags",
                 {{"prefix_reuse", false},
                  {"prefix_cache_enabled", options_.context_cache.enabled},
                  {"cuda_graph", options_.use_cuda_graph},
                  {"kv_dtype", static_cast<int>(options_.kv_cache)},
                  {"speculative", options_.speculative.backend == SpeculativeBackend::Mtp}}},
                {"completion_semantics", "last_executed_token_before_final_sample"},
                {"record_layout", Json::array({{{"name", "pre_normalisation_stream"},
                                                {"shape", {4, 2560}},
                                                {"dtype", "F32"},
                                                {"source_dtype", "BF16"},
                                                {"byte_offset", 0}},
                                               {{"name", "post_normalisation"},
                                                {"shape", {4, 2560}},
                                                {"dtype", "F32"},
                                                {"source_dtype", "BF16"},
                                                {"byte_offset", 40960}},
                                               {{"name", "lane_mix_weights"},
                                                {"shape", {4, 2560}},
                                                {"dtype", "F32"},
                                                {"source_dtype", "BF16 logits, F32 sigmoid"},
                                                {"byte_offset", 81920}},
                                               {{"name", "post_mix"},
                                                {"shape", {2560}},
                                                {"dtype", "F32"},
                                                {"source_dtype", "BF16"},
                                                {"byte_offset", 122880}},
                                               {{"name", "adjustment_input"},
                                                {"shape", {4, 2560}},
                                                {"dtype", "F32"},
                                                {"source_dtype", "BF16"},
                                                {"byte_offset", 0},
                                                {"alias", "pre_normalisation_stream"}}})},
                {"records", records}};
    std::string encoded;
    for (;;) {
        encoded              = header.dump();
        std::uint64_t offset = 8 + encoded.size();
        bool stable          = true;
        for (auto& record : header["records"]) {
            stable &= record["offset"] == offset;
            record["offset"] = offset;
            offset += ops::kActivationElements * 4;
        }
        if (stable) break;
    }
    std::filesystem::create_directories(options_.capture_path);
    const auto final_path = options_.capture_path / (id + ".capture");
    const auto temp_path  = options_.capture_path / (id + ".tmp");
    try {
        std::ofstream output(temp_path, std::ios::binary);
        const std::uint64_t length = encoded.size();
        output.write(reinterpret_cast<const char*>(&length), 8);
        output.write(encoded.data(), encoded.size());
        std::vector<float> payload(ops::kActivationElements);
        for (const auto& record : records) {
            const auto index = record.at("source_index").get<int>();
            samples_->copy_to_host(payload.data(), payload.size() * 4,
                                   static_cast<std::size_t>(lane * sample_count + index) * payload.size() *
                                       4);
            output.write(reinterpret_cast<const char*>(payload.data()), payload.size() * 4);
        }
        output.close();
        require(bool(output), "capture_write");
        std::filesystem::rename(temp_path, final_path);
    } catch (...) {
        std::filesystem::remove(temp_path);
        throw;
    }
    return id;
}
} // namespace ninfer::models::qwen3_8_flash_next
