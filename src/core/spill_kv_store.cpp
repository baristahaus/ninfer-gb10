#include "core/spill_kv_store.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer {
namespace {

constexpr std::size_t kSpillAlignment = 256;

std::size_t align_up(std::size_t value) { return (value + kSpillAlignment - 1) & ~(kSpillAlignment - 1); }

std::string errno_text(const char* what) {
    return std::string(what) + ": " + std::strerror(errno);
}

} // namespace

SpillKVStore::SpillKVStore(std::filesystem::path path, std::size_t capacity_bytes)
    : path_(std::move(path)), capacity_bytes_(capacity_bytes) {
    if (capacity_bytes_ == 0) {
        throw std::invalid_argument("spill kv store: capacity must be positive");
    }
    if (path_.empty()) {
        throw std::invalid_argument("spill kv store: path must not be empty");
    }
    descriptor_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (descriptor_ < 0) {
        throw std::runtime_error(errno_text("spill kv store: open failed"));
    }
}

SpillKVStore::~SpillKVStore() {
    if (descriptor_ >= 0) { ::close(descriptor_); }
}

SpillKVStore::Extent SpillKVStore::write(const std::byte* bytes, std::size_t count) {
    const Extent extent = reserve(count);
    write_at(extent, bytes, count);
    return extent;
}

SpillKVStore::Extent SpillKVStore::reserve(std::size_t count) {
    if (count == 0) {
        throw std::invalid_argument("spill kv store: cannot write an empty extent");
    }
    const std::size_t aligned = align_up(count);

    // First fit over released extents; append when no released extent fits. A released extent
    // larger than the request is split, keeping the remainder as a new free extent. Appending
    // grows the file, so the capacity bounds the file size, not just live bytes.
    std::uint64_t offset  = append_offset_;
    bool append           = true;
    std::size_t remainder = 0;
    auto fit = std::find_if(free_extents_.begin(), free_extents_.end(),
                            [aligned](const FreeExtent& extent) { return extent.bytes >= aligned; });
    if (fit != free_extents_.end()) {
        offset    = fit->offset;
        remainder = static_cast<std::size_t>(fit->bytes - aligned);
        free_extents_.erase(fit);
        if (remainder != 0) {
            free_extents_.push_back(FreeExtent{offset + aligned, remainder});
        }
        append = false;
    } else if (aligned > capacity_bytes_ - append_offset_) {
        throw std::invalid_argument("spill kv store: capacity cannot hold the extent");
    }
    if (append) {
        append_offset_ += aligned;
    }

    live_.emplace(offset, next_generation_);
    reserved_bytes_ += aligned;
    return Extent{offset, count, next_generation_++};
}

void SpillKVStore::write_at(const Extent& extent, const std::byte* bytes, std::size_t count) {
    const auto found = live_.find(extent.offset);
    if (found == live_.end() || found->second != extent.generation) {
        throw std::invalid_argument("spill kv store: foreign extent");
    }
    if (bytes == nullptr) {
        throw std::invalid_argument("spill kv store: null write source");
    }
    if (count != extent.bytes) {
        throw std::invalid_argument("spill kv store: write size does not match the extent");
    }
    checked_write(bytes, count, extent.offset);
}

void SpillKVStore::cancel(const Extent& extent) noexcept { release(extent); }

void SpillKVStore::read(const Extent& extent, std::byte* destination, std::size_t count) const {
    const auto found = live_.find(extent.offset);
    if (found == live_.end() || found->second != extent.generation) {
        throw std::invalid_argument("spill kv store: foreign extent");
    }
    if (destination == nullptr) {
        throw std::invalid_argument("spill kv store: null read destination");
    }
    if (count != extent.bytes) {
        throw std::invalid_argument("spill kv store: read size does not match the extent");
    }
    checked_read(destination, count, extent.offset);
}

void SpillKVStore::release(const Extent& extent) noexcept {
    const auto found = live_.find(extent.offset);
    if (found == live_.end() || found->second != extent.generation) { return; }
    live_.erase(found);

    const std::size_t aligned = align_up(static_cast<std::size_t>(extent.bytes));
    reserved_bytes_ -= aligned;

    // Coalesce with directly adjacent free extents so a released region becomes reusable in
    // full rather than in fragments.
    std::uint64_t begin = extent.offset;
    std::uint64_t bytes = aligned;
    for (auto it = free_extents_.begin(); it != free_extents_.end();) {
        if (it->offset + it->bytes == begin) {
            begin = it->offset;
            bytes += it->bytes;
            it = free_extents_.erase(it);
        } else if (begin + bytes == it->offset) {
            bytes += it->bytes;
            it = free_extents_.erase(it);
        } else {
            ++it;
        }
    }
    free_extents_.push_back(FreeExtent{begin, bytes});
}

void SpillKVStore::checked_write(const void* bytes, std::size_t count, std::uint64_t offset) const {
    const auto* cursor = static_cast<const std::byte*>(bytes);
    std::size_t done   = 0;
    while (done < count) {
        const ssize_t written =
            ::pwrite(descriptor_, cursor + done, count - done,
                     static_cast<off_t>(offset + done));
        if (written <= 0) {
            if (written < 0 && (errno == EINTR)) { continue; }
            throw std::runtime_error(errno_text("spill kv store: write failed"));
        }
        done += static_cast<std::size_t>(written);
    }
}

void SpillKVStore::checked_read(void* bytes, std::size_t count, std::uint64_t offset) const {
    auto* cursor = static_cast<std::byte*>(bytes);
    std::size_t done = 0;
    while (done < count) {
        const ssize_t got = ::pread(descriptor_, cursor + done, count - done,
                                    static_cast<off_t>(offset + done));
        if (got < 0) {
            if (errno == EINTR) { continue; }
            throw std::runtime_error(errno_text("spill kv store: read failed"));
        }
        if (got == 0) {
            throw std::runtime_error("spill kv store: short read at offset " +
                                     std::to_string(offset + done));
        }
        done += static_cast<std::size_t>(got);
    }
}

} // namespace ninfer
