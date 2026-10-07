#include "models/qwen3_8_flash_next/impl/runtime/prefix_identity.h"
#include <bit>
#include "models/qwen3_8_flash_next/impl/activation_control.h"
#include "core/device.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <cmath>

namespace flash = ninfer::models::qwen3_8_flash_next;
using namespace ninfer;
using Json = nlohmann::json;

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void write_pack(const std::filesystem::path& path, int rank, int metadata_rank = 0) {
    Json metadata{{"schema", "ninfer.steering/1"},
                  {"model_sha", "model"},
                  {"model_sha_struct", "struct"},
                  {"config_sha", "config"},
                  {"template_sha", "template"},
                  {"pack_sha", "pack"},
                  {"created", "test"},
                  {"max_rank", metadata_rank ? metadata_rank : rank},
                  {"norm_preserve", false},
                  {"default_steering_strength", 0.35},
                  {"layers", {{"3", {{"k", rank}, {"lanes", 5}, {"provenance", "test"}}}}}};
    Json header{{"__metadata__", {{"ninfer.steering", metadata.dump()}}},
                {"layers.3.directions",
                 {{"dtype", "F32"},
                  {"shape", {rank, 4, 2560}},
                  {"data_offsets", {0, rank * 4 * 2560 * 4}}}}};
    auto text                = header.dump();
    const std::uint64_t size = text.size();
    std::vector<float> data(rank * 4 * 2560);
    for (int r = 0; r < rank; ++r)
        for (int lane = 0; lane < 4; ++lane) data[(r * 4 + lane) * 2560 + r % 2560] = 1;
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(&size), 8);
    file.write(text.data(), text.size());
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * 4);
}
} // namespace

int main(int argc, char** argv) {
    const auto dir = std::filesystem::temp_directory_path() /
                     ("ninfer-activation-test-" +
                      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        std::filesystem::create_directories(dir);
        write_pack(dir / "valid.safetensors", 2);
        const auto pack = flash::read_steering_pack(dir / "valid.safetensors");
        require(pack.state.layers == std::vector<int>{3} && pack.ranks[3] == 2 &&
                    pack.masks[3] == 5,
                "pack layers/rank/masks failed");
        for (auto [rank, max_rank] : {std::pair{33, 33}, std::pair{33, 32}, std::pair{2, 33}}) {
            write_pack(dir / "oversize.safetensors", rank, max_rank);
            bool failed = false;
            try {
                (void)flash::read_steering_pack(dir / "oversize.safetensors");
            } catch (const std::exception& error) {
                failed = std::string(error.what()).find("MAX_RANK") != std::string::npos;
            }
            require(failed, "oversize pack was not declined by MAX_RANK");
        }
        flash::PreparedPromptData cache_prompt;
        cache_prompt.token_ids                       = {41, 99};
        cache_prompt.token_types                     = {0, 0};
        cache_prompt.positions                       = {0, 1, 0, 1, 0, 1};
        cache_prompt.identity.steering_generation    = 12;
        cache_prompt.identity.steering_strength_bits = std::bit_cast<std::uint32_t>(0.35F);
        flash::detail::ResidentPrefixIdentity cached;
        flash::detail::PrefixShortlistDigests cached_digest, incoming_digest;
        cached.assign(cache_prompt);
        cached_digest.assign(cache_prompt);
        require(cached.matches(cache_prompt, 2), "same-generation cache miss");
        cache_prompt.identity.steering_generation = 13;
        incoming_digest.assign(cache_prompt);
        require(!cached.matches(cache_prompt, 2) && cached_digest.at(2) != incoming_digest.at(2),
                "generation N cached entry matched generation N+1");
        cache_prompt.identity.steering_generation    = 12;
        cache_prompt.identity.steering_strength_bits = 0;
        require(!cached.matches(cache_prompt, 2), "different-strength cached entry matched");
        DeviceContext device;
        flash::ActivationControl control(device.stream);
        EngineOptions options;
        options.capture_path    = dir / "captures";
        options.max_concurrency = 1;
        control.configure(options, "{\"model_sha\":\"served\"}");
        control.activate(&pack);
        control.activate(nullptr);
        ops::ActivationDevice header;
        CUDA_CHECK(cudaMemcpy(&header, control.device(), sizeof(header), cudaMemcpyDeviceToHost));
        std::vector<float> zero_directions(48ULL * ops::kSteeringMaxRank * 4 * 2560);
        CUDA_CHECK(cudaMemcpy(zero_directions.data(), header.directions, zero_directions.size() * 4,
                              cudaMemcpyDeviceToHost));
        for (float value : zero_directions) require(value == 0, "disable did not zero arena");
        const std::array<TokenId, 2> prompt{41, 99};
        const std::array<TokenId, 4> ledger{41, 99, 37, 777};
        control.activate(&pack);
        auto steered = pack.state;
        steered.strength = 0.35F;
        steered.generation = 12;
        control.begin(0, 1, prompt, true, steered);
        std::array<ops::ActivationSample, 96> checks;
        for (int i = 0; i < 96; ++i) checks[i] = {1, i < 48 ? 1 : 2, i < 48 ? 99 : 37, 0};
        const auto upload = [&] {
            CUDA_CHECK(
                cudaMemcpy(header.checks, checks.data(), sizeof(checks), cudaMemcpyHostToDevice));
        };
        const auto failure = [&](const char* name) {
            upload();
            bool failed = false;
            try {
                (void)control.flush(0, ledger, 3);
            } catch (const std::exception& error) {
                failed = std::string(error.what()).find(name) != std::string::npos;
            }
            require(failed, "injected assertion did not name its check");
            require(!std::filesystem::exists(options.capture_path),
                    "failed assertion wrote a file");
        };
        checks[3].fires = 0;
        failure("fire_count");
        checks[3].fires = 1;
        checks[2].token = 41;
        failure("token_id");
        checks[2].token  = 99;
        checks[55].token = 99;
        failure("token_id");
        checks[55].token = 37;
        checks[4].pad    = 1;
        failure("pad_mask");
        checks[4].pad      = 0;
        checks[1].position = -1;
        failure("captured_index");
        checks[1].position = 1;
        Tensor invalid(nullptr, DType::FP32, {10240, 1});
        bool shape_failed = false;
        try {
            ops::activation_capture(invalid, invalid, invalid, invalid, invalid, invalid, invalid,
                                    control.device(), 0, 1, device.stream);
        } catch (const std::exception& error) {
            shape_failed = std::string(error.what()).find("shape_dtype") != std::string::npos;
        }
        require(shape_failed, "shape assertion missing");
        upload();
        const auto id = control.flush(0, ledger, 3);
        std::ifstream file(options.capture_path / (id + ".capture"), std::ios::binary);
        std::uint64_t size;
        file.read(reinterpret_cast<char*>(&size), 8);
        std::string text(size, '\0');
        file.read(text.data(), text.size());
        const auto metadata = Json::parse(text);
        require(metadata["record_count"] == 96 && metadata["rendered_token_ids"] == prompt,
                "capture metadata failed");
        require(metadata["steering"]["pack_sha"] == steered.pack_sha &&
                    metadata["steering"]["generation"] == 12 &&
                    metadata["steering"]["strength"].get<float>() == 0.35F,
                "steered capture metadata failed");
        std::uint64_t offset = 8 + size;
        for (const auto& record : metadata["records"]) {
            require(record["steering_strength"].get<float>() == 0.35F &&
                        record["steering_pack_sha"] == steered.pack_sha &&
                        record["steering_generation"] == 12,
                    "steered record metadata failed");
            require(record["offset"] == offset, "absolute record offset failed");
            offset += record["bytes"].get<std::uint64_t>();
        }
        require(std::filesystem::file_size(options.capture_path / (id + ".capture")) == offset,
                "capture file size failed");
        std::filesystem::remove_all(options.capture_path);
        // All four verification candidates exist, but only the committed frontier may be
        // recorded. Exercise rejection/truncation at each possible accepted prefix length.
        {
            flash::ActivationControl mtp(device.stream);
            auto mtp_options = options;
            mtp_options.speculative.backend      = SpeculativeBackend::Mtp;
            mtp_options.speculative.draft_tokens = 3;
            mtp.configure(mtp_options, "{}");
            ops::ActivationDevice mtp_header;
            CUDA_CHECK(cudaMemcpy(&mtp_header, mtp.device(), sizeof(mtp_header), cudaMemcpyDeviceToHost));
            require(mtp_header.completion_capacity == 4, "MTP capture capacity missing");
            // Adaptive drafts verify up to the widest MTP row.
            flash::ActivationControl adaptive(device.stream);
            auto adaptive_options = options;
            adaptive_options.speculative.backend               = SpeculativeBackend::Mtp;
            adaptive_options.speculative.adaptive_draft_tokens = true;
            adaptive.configure(adaptive_options, "{}");
            ops::ActivationDevice adaptive_header;
            CUDA_CHECK(cudaMemcpy(&adaptive_header, adaptive.device(), sizeof(adaptive_header),
                                  cudaMemcpyDeviceToHost));
            require(adaptive_header.completion_capacity == 8, "adaptive MTP capture capacity missing");
            const std::vector<TokenId> accepted{41, 99, 37, 47, 57, 67, 77};
            for (int committed = 1; committed <= 4; ++committed) {
                mtp.begin(0, 100 + committed, prompt, true, {});
                std::vector<ops::ActivationSample> candidates(240);
                for (int layer = 0; layer < 48; ++layer) {
                    candidates[layer] = {1, 1, 99, 0};
                    for (int column = 0; column < 4; ++column)
                        candidates[(column + 1) * 48 + layer] =
                            {1, 2 + column, accepted[2 + column], 0};
                }
                CUDA_CHECK(cudaMemcpy(mtp_header.checks, candidates.data(),
                                       candidates.size() * sizeof(candidates[0]), cudaMemcpyHostToDevice));
                const auto record_id = mtp.flush(0, std::span(accepted).first(committed + 3), committed + 2);
                std::ifstream recorded(options.capture_path / (record_id + ".capture"), std::ios::binary);
                std::uint64_t header_bytes;
                recorded.read(reinterpret_cast<char*>(&header_bytes), 8);
                std::string record_header(header_bytes, '\0');
                recorded.read(record_header.data(), record_header.size());
                const auto parsed = Json::parse(record_header);
                for (const auto& row : parsed["records"])
                    if (row["site"] == "completion_last") {
                        require(row["captured_index"] == committed + 1 &&
                                    row["token_id"] == accepted[committed + 1] &&
                                    row["source_index"] == committed * 48 + row["layer"].get<int>(),
                                "MTP capture selected a rejected candidate");
                    }
                std::filesystem::remove(options.capture_path / (record_id + ".capture"));
            }
        }
        if (argc > 1 && std::string(argv[1]) == "--stress") {
            for (int i = 0; i < 520; ++i) {
                control.begin(0, i + 2, prompt, true, {});
                upload();
                const auto record = control.flush(0, ledger, 3);
                std::filesystem::remove(options.capture_path / (record + ".capture"));
            }
            std::cout << "520 requests x 2 sites, per-request flush: PASS\n";
        }
        std::filesystem::remove_all(dir);
        std::cout << "pack/MAX_RANK; zeroed disable; capture site IDs, offsets and injected "
                     "assertions: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove_all(dir);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
