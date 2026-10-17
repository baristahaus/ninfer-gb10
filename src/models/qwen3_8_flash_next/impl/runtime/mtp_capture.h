#pragma once

// Disk capture of one batched MTP decode round's per-row proposal and verification
// outcome. The engine writes fixed-size records from data that is already host-resident
// at the round boundary (pinned ingress/egress mirrors), so capture costs no device work
// and analysis runs offline. Format: little-endian header, then row records.

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

#include "core/dtype.h"
#include "ninfer/types.h"

namespace ninfer::models::qwen3_8_flash_next::detail {

inline constexpr char kMtpCaptureMagic[8] = {'N', 'M', 'T', 'P', 'C', 'A', 'P', 1};
inline constexpr std::uint32_t kMtpCaptureVersion = 1;
inline constexpr std::uint32_t kMtpCaptureMaximumDrafts = 5;

#pragma pack(push, 1)
struct MtpCaptureHeader {
    char magic[8];
    std::uint32_t version      = kMtpCaptureVersion;
    std::uint32_t header_size  = sizeof(MtpCaptureHeader);
    std::uint32_t draft_window = 0; // k, drafts per round
    std::uint32_t record_width = 0; // k + 1 committed columns
    std::uint32_t token_domain = 0; // target vocabulary size
    std::uint32_t reserved     = 0;
};

// One row of one round. drafts[] holds the k proposed ids; entries at or past the
// round's extent repeat the anchor token and carry no acceptance meaning. committed[]
// holds the verified published tokens (accepted drafts plus one target token); entries
// at or past committed_count are zero. extent == 0 marks a fallback step: no draft was
// proposed, committed[0] is the pure target step.
struct MtpCaptureRecord {
    std::uint64_t round_index     = 0;
    std::uint32_t lane            = 0;
    std::uint32_t base_frontier   = 0; // KV position of the row's first committed token
    std::uint32_t extent          = 0; // drafts proposed this round (0 = fallback)
    std::uint32_t committed_count = 0;
    std::uint32_t pad             = 0;
};
#pragma pack(pop)

// Per-row record payload after the fixed prefix: k draft ids, then k+1 committed ids.
[[nodiscard]] inline constexpr std::uint32_t mtp_capture_record_bytes(std::uint32_t draft_window) {
    return static_cast<std::uint32_t>(sizeof(MtpCaptureRecord)) +
           (draft_window + draft_window + 1) * sizeof(TokenId);
}

class MtpCaptureSink {
public:
    // Opens (truncates) the capture file and starts the writer thread. Throws on
    // filesystem errors.
    [[nodiscard]] static std::unique_ptr<MtpCaptureSink> open(std::filesystem::path path,
                                                              std::uint32_t draft_window,
                                                              std::uint32_t token_domain);
    ~MtpCaptureSink();

    MtpCaptureSink(const MtpCaptureSink&)            = delete;
    MtpCaptureSink& operator=(const MtpCaptureSink&) = delete;

    // Append one row's record. drafts points at draft_window ids, committed at up to
    // draft_window+1 ids. Safe to call from the decode thread; never throws.
    void record(std::uint64_t round_index, std::uint32_t lane, std::uint32_t base_frontier,
                std::uint32_t extent, std::span<const TokenId> drafts,
                std::span<const TokenId> committed) noexcept;

    // True when the writer thread hit a write error and dropped later records.
    [[nodiscard]] bool failed() const noexcept { return failed_.load(std::memory_order_relaxed); }

private:
    MtpCaptureSink() = default;

    void writer_loop();

    std::uint32_t draft_window_ = 0;
    std::vector<std::byte> staging_;
    std::deque<std::vector<std::byte>> queue_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable space_;
    std::atomic<bool> failed_{false};
    std::atomic<bool> closing_{false};
    std::thread writer_;
    int fd_ = -1;
};

} // namespace ninfer::models::qwen3_8_flash_next::detail
