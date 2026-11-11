#pragma once

#include "models/common/qwen/frontend/processor.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string_view>

namespace ninfer {
class HostWorkerPool;
class TaskPool;
} // namespace ninfer

namespace ninfer::models::qwen::frontend {

template <class Family>
class MediaCache;

enum class MediaCacheDisposition : std::uint8_t {
    Hit,
    Producer,
    SingleflightWaiter,
};

namespace media_cache_detail {

template <class ModalityT, ModalityT DefaultModality>
struct Key {
    std::array<std::uint8_t, 32> digest{};
    ModalityT modality = DefaultModality;

    [[nodiscard]] bool operator==(const Key&) const noexcept = default;
};

template <class VisionItemT, class PayloadT>
struct Prepared {
    VisionItemT item;
    std::shared_ptr<const PayloadT> payload;
};

template <class PreparedMediaT>
struct Flight {
    std::mutex mutex;
    std::condition_variable changed;
    PreparedMediaT result;
    std::exception_ptr error;
    std::chrono::steady_clock::time_point started;
    std::chrono::steady_clock::time_point finished;
    std::size_t bytes = 0;
    bool done         = false;
};

template <class PreparedMediaT>
struct Pending {
    MediaCacheDisposition disposition = MediaCacheDisposition::Hit;
    PreparedMediaT ready;
    std::shared_ptr<Flight<PreparedMediaT>> flight;
};

} // namespace media_cache_detail

struct MediaPreparationGate {
    explicit MediaPreparationGate(std::size_t maximum_) : maximum(maximum_) {}

    std::mutex mutex;
    std::condition_variable changed;
    const std::size_t maximum;
    std::size_t active = 0;
};

class MediaPreparationPermit {
public:
    MediaPreparationPermit() noexcept = default;
    ~MediaPreparationPermit();
    MediaPreparationPermit(MediaPreparationPermit&&) noexcept;
    MediaPreparationPermit& operator=(MediaPreparationPermit&&) noexcept;

    MediaPreparationPermit(const MediaPreparationPermit&)            = delete;
    MediaPreparationPermit& operator=(const MediaPreparationPermit&) = delete;

    void reset() noexcept { release(); }

private:
    explicit MediaPreparationPermit(std::shared_ptr<MediaPreparationGate> gate) noexcept;
    void release() noexcept;
    std::shared_ptr<MediaPreparationGate> gate_;

    template <class Family>
    friend class MediaCache;
};

struct MediaCacheRequestStats {
    std::size_t hits               = 0;
    std::size_t misses             = 0;
    std::size_t singleflight_waits = 0;
    std::size_t built_patch_bytes  = 0;
    std::size_t reused_patch_bytes = 0;
    double build_seconds           = 0.0;
};

struct MediaCacheStats {
    std::size_t capacity_bytes       = 0;
    std::size_t live_capacity_bytes  = 0;
    std::size_t retained_bytes       = 0;
    std::size_t live_bytes           = 0;
    std::size_t entries              = 0;
    std::size_t inflight             = 0;
    std::size_t queued_tasks         = 0;
    std::size_t active_tasks         = 0;
    std::uint32_t preprocess_threads = 0;
    std::uint64_t hits               = 0;
    std::uint64_t misses             = 0;
    std::uint64_t singleflight_waits = 0;
    std::uint64_t evictions          = 0;
    std::uint64_t oversize_bypasses  = 0;
};

// The cache admission/lookup/eviction machinery, keying, accounting and worker-pool mechanics
// live here once. A family supplies its prepared-media payload type, modality, vision item and
// worker/thread policy through the trait struct passed as Family.
template <class Family>
class MediaCache {
public:
    using Payload       = typename Family::Payload;
    using Key           = typename Family::Key;
    using PreparedMedia = typename Family::Media;
    using Pending       = typename Family::Pending;
    using Builder       = std::function<PreparedMedia()>;

    MediaCache(std::size_t capacity_bytes, std::size_t live_capacity_bytes,
               std::uint32_t preprocess_threads = 0, std::size_t maximum_request_bytes = 0);
    ~MediaCache();

    MediaCache(const MediaCache&)            = delete;
    MediaCache& operator=(const MediaCache&) = delete;

    [[nodiscard]] std::shared_ptr<Payload> allocate_payload(std::size_t elements,
                                                            const PreparationControl& control);

    [[nodiscard]] MediaPreparationPermit acquire_request(const PreparationControl& control) const;

    [[nodiscard]] Pending begin_prepare(const Key& key, const PreparationControl& control,
                                        Builder builder);
    [[nodiscard]] PreparedMedia get_or_prepare(const Key& key, const PreparationControl& control,
                                               Builder builder,
                                               MediaCacheRequestStats& request_stats);
    [[nodiscard]] PreparedMedia await(const Pending& pending, const PreparationControl& control,
                                      MediaCacheRequestStats& request_stats) const;

    [[nodiscard]] MediaCacheStats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

void check_preparation_control(const PreparationControl& control,
                               std::string_view stage = "preparation");

using MediaCacheKey = media_cache_detail::Key<Modality, Modality::Image>;
using PreparedMedia = media_cache_detail::Prepared<VisionItem, PreparedMediaPayload>;
using PendingMedia  = media_cache_detail::Pending<PreparedMedia>;

// The common Qwen frontend selects the common worker/thread policy and its own payload type.
struct CommonMediaCacheFamily {
    using Payload = ninfer::models::qwen::PreparedMediaPayload;
    using Key     = ninfer::models::qwen::frontend::MediaCacheKey;
    using Media   = ninfer::models::qwen::frontend::PreparedMedia;
    using Pending = ninfer::models::qwen::frontend::PendingMedia;
    using Workers = ninfer::TaskPool;

    static std::uint32_t resolve_threads(std::uint32_t requested);
};

using MediaPreprocessCache = MediaCache<CommonMediaCacheFamily>;

} // namespace ninfer::models::qwen::frontend
