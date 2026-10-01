#pragma once

#include "artifact/reader.h"
#include "core/arena.h"
#include "ninfer/ops/flash_next_ple_stage.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <stop_token>
#include <thread>

namespace ninfer::models::qwen3_8_flash_next {

// The host half of the in-graph PLE gather. While a round is active, a Program-owned thread
// polls the pinned mailbox for the row IDs a graph publishes (ops::flash_next_ple_publish_ids),
// gathers those FP8 rows from the file-backed table into pinned staging, and acknowledges them.
// The graph waits for the acknowledgement (ops::flash_next_ple_wait_staged) before it copies the
// staging to the device for its first PLE consumer, so the gather overlaps the layers before it.
class PleGatherStage {
public:
    explicit PleGatherStage(const artifact::MappedRange& table);
    ~PleGatherStage();

    PleGatherStage(const PleGatherStage&)            = delete;
    PleGatherStage& operator=(const PleGatherStage&) = delete;

    [[nodiscard]] ops::FlashNextPleStageMailbox* mailbox() const noexcept;

    // Token-major FP8 rows of the last answered request, kPleEmbeddingDim bytes per token.
    [[nodiscard]] const void* staging() const noexcept { return staging_.data(); }

    // Serves requests from now until end_round(); call before submitting work that publishes.
    void begin_round();
    // Stops serving. Safe on every path, including after a failed submission.
    void end_round() noexcept;
    // After the round's work completed: throws if the device gave up waiting (a stalled or
    // dead stage) or a gather failed. Clears both conditions.
    void check();

private:
    void run(std::stop_token stop);

    const artifact::MappedRange& table_;
    PinnedHostBuffer mailbox_;
    PinnedHostBuffer staging_;
    std::mutex mutex_;
    std::condition_variable_any wake_;
    bool active_ = false;
    std::exception_ptr failure_;
    // Declared last: stopped and joined before the buffers it uses are released.
    std::jthread thread_;
};

} // namespace ninfer::models::qwen3_8_flash_next
