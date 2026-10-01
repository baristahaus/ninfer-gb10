#include "models/qwen3_8_flash_next/impl/ple_stage.h"

#include "models/qwen3_8_flash_next/impl/ple_table.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <span>
#include <stdexcept>

#include <sys/prctl.h>

namespace ninfer::models::qwen3_8_flash_next {
namespace {

static_assert(ops::kFlashNextPleStageHeads == kPleHeads);
static_assert(sizeof(PleIds) == kPleHeads * sizeof(std::uint32_t));

std::uint32_t load_acquire(std::uint32_t& value) noexcept {
    return std::atomic_ref<std::uint32_t>(value).load(std::memory_order_acquire);
}

} // namespace

PleGatherStage::PleGatherStage(const artifact::MappedRange& table)
    : table_(table), mailbox_(sizeof(ops::FlashNextPleStageMailbox)),
      staging_(static_cast<std::size_t>(ops::kFlashNextPleStageMaxTokens) * kPleEmbeddingDim) {
    *mailbox() = {};
    thread_    = std::jthread([this](std::stop_token stop) { run(stop); });
}

PleGatherStage::~PleGatherStage() {
    thread_.request_stop();
    wake_.notify_all();
}

ops::FlashNextPleStageMailbox* PleGatherStage::mailbox() const noexcept {
    return static_cast<ops::FlashNextPleStageMailbox*>(mailbox_.data());
}

void PleGatherStage::begin_round() {
    {
        const std::lock_guard lock(mutex_);
        active_ = true;
    }
    wake_.notify_one();
}

void PleGatherStage::end_round() noexcept {
    const std::lock_guard lock(mutex_);
    active_ = false;
}

void PleGatherStage::check() {
    std::exception_ptr failure;
    {
        const std::lock_guard lock(mutex_);
        failure  = failure_;
        failure_ = nullptr;
    }
    std::atomic_ref<std::uint32_t> late(mailbox()->late);
    if (late.exchange(0U, std::memory_order_acq_rel) != 0U) {
        throw std::runtime_error("Flash-Next PLE gather missed the round's deadline");
    }
    if (failure) { std::rethrow_exception(failure); }
}

void PleGatherStage::run(std::stop_token stop) {
    // Sleeps between polls are microseconds; the default 50 us timer slack would dominate them.
    prctl(PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL);
    ops::FlashNextPleStageMailbox& box = *mailbox();
    std::uint32_t served               = load_acquire(box.request_sequence);
    std::array<PleIds, ops::kFlashNextPleStageMaxTokens> ids{};
    while (!stop.stop_requested()) {
        {
            std::unique_lock lock(mutex_);
            if (!wake_.wait(lock, stop, [this] { return active_; })) { return; }
        }
        const std::uint32_t sequence = load_acquire(box.request_sequence);
        if (sequence == served) {
            std::this_thread::sleep_for(std::chrono::microseconds(5));
            continue;
        }
        try {
            const std::int32_t tokens = box.tokens;
            if (tokens < 1 || tokens > ops::kFlashNextPleStageMaxTokens) {
                throw std::runtime_error("Flash-Next PLE request has an invalid token count");
            }
            const auto count = static_cast<std::size_t>(tokens);
            std::memcpy(ids.data(), box.ids, count * sizeof(PleIds));
            gather_ple_fp8(table_, std::span<const PleIds>(ids.data(), count),
                           std::span<std::byte>(static_cast<std::byte*>(staging_.data()),
                                                count * kPleEmbeddingDim));
        } catch (...) {
            const std::lock_guard lock(mutex_);
            failure_ = std::current_exception();
        }
        served = sequence;
        // Acknowledge even a failed gather, so the device never waits out its deadline for it.
        std::atomic_ref<std::uint32_t>(box.answer_sequence)
            .store(sequence, std::memory_order_release);
    }
}

} // namespace ninfer::models::qwen3_8_flash_next
