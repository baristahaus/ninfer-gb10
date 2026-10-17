#include "models/qwen3_8_flash_next/impl/runtime/mtp_capture.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include <fcntl.h>
#include <unistd.h>

namespace ninfer::models::qwen3_8_flash_next::detail {
namespace {

// Records are appended into a staging buffer on the decode thread and the buffer moves
// to the writer queue at this size, so the decode thread takes the lock once per chunk
// instead of once per round.
constexpr std::size_t kChunkBytes = 256 * 1024;
// Backpressure bound: decoded chunks not yet on disk. At record rates of a few kilobytes
// per second this never fills; the bound keeps a pathological burst bounded instead of
// growing without limit.
constexpr std::size_t kMaximumQueuedChunks = 512;

void write_all(int fd, const std::byte* data, std::size_t size) {
    while (size > 0) {
        const ssize_t written = ::write(fd, data, size);
        if (written < 0) {
            if (errno == EINTR) { continue; }
            throw std::runtime_error(std::string("MTP capture write failed: ") +
                                     std::strerror(errno));
        }
        data += static_cast<std::size_t>(written);
        size -= static_cast<std::size_t>(written);
    }
}

} // namespace

std::unique_ptr<MtpCaptureSink> MtpCaptureSink::open(std::filesystem::path path,
                                                     std::uint32_t draft_window,
                                                     std::uint32_t token_domain) {
    if (draft_window == 0 || draft_window > kMtpCaptureMaximumDrafts) {
        throw std::invalid_argument("MTP capture requires a draft window in [1,5]");
    }
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error("MTP capture could not open " + path.string() + ": " +
                                 std::strerror(errno));
    }
    auto sink              = std::unique_ptr<MtpCaptureSink>(new MtpCaptureSink());
    sink->fd_              = fd;
    sink->draft_window_    = draft_window;
    sink->staging_.reserve(kChunkBytes + mtp_capture_record_bytes(draft_window));
    MtpCaptureHeader header{};
    std::memcpy(header.magic, kMtpCaptureMagic, sizeof(header.magic));
    header.draft_window = draft_window;
    header.record_width = draft_window + 1;
    header.token_domain = token_domain;
    try {
        write_all(fd, reinterpret_cast<const std::byte*>(&header), sizeof(header));
    } catch (...) {
        ::close(fd);
        throw;
    }
    sink->writer_ = std::thread([sink = sink.get()] { sink->writer_loop(); });
    return sink;
}

MtpCaptureSink::~MtpCaptureSink() {
    if (writer_.joinable()) {
        {
            std::lock_guard lock(mutex_);
            if (!staging_.empty()) {
                queue_.push_back(std::move(staging_));
                staging_.clear();
            }
            closing_.store(true, std::memory_order_release);
        }
        ready_.notify_all();
        writer_.join();
    }
    if (fd_ >= 0) { ::close(fd_); }
}

void MtpCaptureSink::writer_loop() {
    try {
        std::vector<std::byte> chunk;
        while (true) {
            {
                std::unique_lock lock(mutex_);
                ready_.wait(lock, [&] {
                    return closing_.load(std::memory_order_acquire) || !queue_.empty();
                });
                if (queue_.empty()) { return; }
                chunk = std::move(queue_.front());
                queue_.pop_front();
                space_.notify_all();
            }
            write_all(fd_, chunk.data(), chunk.size());
            chunk.clear();
            chunk.shrink_to_fit();
        }
    } catch (...) {
        failed_.store(true, std::memory_order_release);
    }
}

void MtpCaptureSink::record(std::uint64_t round_index, std::uint32_t lane,
                            std::uint32_t base_frontier, std::uint32_t extent,
                            std::span<const TokenId> drafts,
                            std::span<const TokenId> committed) noexcept {
    if (failed_.load(std::memory_order_relaxed)) { return; }
    const std::uint32_t k = draft_window_;
    const std::size_t draft_count = std::min(drafts.size(), static_cast<std::size_t>(k));
    const std::size_t committed_count =
        std::min(committed.size(), static_cast<std::size_t>(k) + 1);
    std::array<std::byte, sizeof(MtpCaptureRecord) + (2 * kMtpCaptureMaximumDrafts + 1) *
                                                              sizeof(TokenId)>
        bytes{};
    auto* record = reinterpret_cast<MtpCaptureRecord*>(bytes.data());
    record->round_index     = round_index;
    record->lane            = lane;
    record->base_frontier   = base_frontier;
    record->extent          = extent;
    record->committed_count = static_cast<std::uint32_t>(committed_count);
    record->pad             = 0;
    const auto* drafts_bytes = reinterpret_cast<const std::byte*>(drafts.data());
    std::memcpy(bytes.data() + sizeof(MtpCaptureRecord), drafts_bytes,
                draft_count * sizeof(TokenId));
    const auto* committed_bytes = reinterpret_cast<const std::byte*>(committed.data());
    std::memcpy(bytes.data() + sizeof(MtpCaptureRecord) + k * sizeof(TokenId), committed_bytes,
                committed_count * sizeof(TokenId));
    const std::size_t record_bytes = mtp_capture_record_bytes(k);
    {
        std::unique_lock lock(mutex_);
        if (staging_.size() + record_bytes > kChunkBytes) {
            queue_.push_back(std::move(staging_));
            staging_.clear();
            staging_.reserve(kChunkBytes + record_bytes);
            ready_.notify_one();
            if (queue_.size() > kMaximumQueuedChunks) {
                // Block the decode thread until the writer catches up; capture stays
                // lossless and the decode rate is far below what the writer sustains.
                space_.wait(lock, [&] { return queue_.size() <= kMaximumQueuedChunks; });
            }
        }
        staging_.insert(staging_.end(), bytes.begin(), bytes.begin() + record_bytes);
    }
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
