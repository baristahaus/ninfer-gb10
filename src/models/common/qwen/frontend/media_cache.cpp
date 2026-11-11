#include "models/common/qwen/frontend/media_cache.h"

#include "models/common/qwen/frontend/media_cache_impl.h"

#include "core/cpu/system.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace ninfer::models::qwen::frontend {

std::uint32_t CommonMediaCacheFamily::resolve_threads(std::uint32_t requested) {
    if (requested > 64) {
        throw std::invalid_argument("media preprocessing threads must be in [1,64]");
    }
    if (requested != 0) { return requested; }
    const CpuTopology topology;
    return static_cast<std::uint32_t>(std::min<std::size_t>(topology.info().processors.size(), 16));
}

MediaPreparationPermit::MediaPreparationPermit(std::shared_ptr<MediaPreparationGate> gate) noexcept
    : gate_(std::move(gate)) {}

void MediaPreparationPermit::release() noexcept {
    if (!gate_) { return; }
    {
        std::lock_guard lock(gate_->mutex);
        if (gate_->active == 0) { std::terminate(); }
        --gate_->active;
    }
    gate_->changed.notify_one();
    gate_.reset();
}

MediaPreparationPermit::~MediaPreparationPermit() { release(); }

MediaPreparationPermit::MediaPreparationPermit(MediaPreparationPermit&&) noexcept = default;

MediaPreparationPermit& MediaPreparationPermit::operator=(MediaPreparationPermit&& other) noexcept {
    if (this == &other) { return *this; }
    release();
    gate_ = std::move(other.gate_);
    return *this;
}

void check_preparation_control(const PreparationControl& control, std::string_view stage) {
    if (control.cancellation.requested()) {
        throw RequestError(RequestErrorKind::Cancelled,
                           "inference request was cancelled during " + std::string(stage));
    }
    if (control.deadline != std::chrono::steady_clock::time_point{} &&
        std::chrono::steady_clock::now() >= control.deadline) {
        throw RequestError(RequestErrorKind::QueueTimeout,
                           "inference request expired during " + std::string(stage));
    }
}

template class MediaCache<CommonMediaCacheFamily>;

} // namespace ninfer::models::qwen::frontend
