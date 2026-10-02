#pragma once
#include "ninfer/ops/activation_steering.h"
#include "ninfer/types.h"
#include "core/arena.h"
#include <nlohmann/json.hpp>
#include <span>
#include <array>
#include <memory>

namespace ninfer::models::qwen3_8_flash_next {
struct SteeringPack {
    nlohmann::json metadata;
    std::vector<float> directions;
    std::array<int, 48> ranks{}, masks{};
    SteeringState state;
    bool norm_preserve = false;
};

inline constexpr std::size_t activation_device_bytes(bool capture, std::uint32_t concurrency,
                                                    SpeculativeBackend backend = SpeculativeBackend::None) {
    return sizeof(ops::ActivationDevice) + 48ULL * ops::kSteeringMaxRank * 4 * 2560 * 4 + 48 * 8 +
           (capture ? concurrency * (backend == SpeculativeBackend::Mtp ? 240ULL : 96ULL) *
                          (ops::kActivationElements * 4 + sizeof(ops::ActivationSample))
                    : 0);
}

SteeringPack read_steering_pack(const std::filesystem::path& path);

class ActivationControl {
public:
    explicit ActivationControl(cudaStream_t stream);
    ~ActivationControl();
    ActivationControl(const ActivationControl&)            = delete;
    ActivationControl& operator=(const ActivationControl&) = delete;
    void configure(const EngineOptions& options, std::string_view identity);
    nlohmann::json identities;
    void activate(const SteeringPack* pack);
    void begin(int lane, std::uint64_t request, std::span<const TokenId> prompt, bool capture,
               const SteeringState& steering);
    void select(std::span<const std::uint32_t> lanes);
    std::string flush(int lane, std::span<const TokenId> ledger, std::uint32_t execution_frontier);

    ops::ActivationDevice* device() const { return device_; }

    std::size_t capacity_bytes() const {
        return controls_.bytes + directions_.bytes + ranks_.bytes + masks_.bytes +
               (samples_ ? samples_->bytes : 0) + (checks_ ? checks_->bytes : 0);
    }
private:
    cudaStream_t stream_;
    DeviceBuffer controls_, directions_, ranks_, masks_;
    std::unique_ptr<DeviceBuffer> samples_, checks_;
    ops::ActivationDevice* device_;
    ops::ActivationDevice host_{};
    // Control uploads are stream-ordered and never block: a changed control block is staged
    // through one of two pinned slots (reused only after its copy event), unchanged ones are
    // skipped.
    ops::ActivationDevice uploaded_{};
    bool uploaded_valid_ = false;
    ops::ActivationDevice* staging_ = nullptr; // pinned [2]
    cudaEvent_t staged_[2]{};
    int next_slot_ = 0;
    EngineOptions options_;

    struct Request {
        std::uint64_t id = 0;
        std::vector<TokenId> prompt;
        bool capture   = false;
        SteeringState steering;
    };

    std::array<Request, 8> requests_;
    void upload();
};
} // namespace ninfer::models::qwen3_8_flash_next
