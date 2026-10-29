#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <unordered_map>
#include <vector>

namespace ninfer {

// File-backed spill tier for retained KV page coverage. The spill page layout equals the host
// KV page layout, so a spill-out is a Host-to-Spill copy of an allocation's bytes and a
// page-in is a read back into a host arena destination followed by the existing device-copy
// work. Extents are byte-addressed and 256-aligned, matching the host arena's alignment.
//
// The spill file is a cache, not a record: it is truncated at construction, so an engine
// restart drops it wholesale. Checkpoints that existed only in spill are gone; the next
// admission recomputes them. No fsync and no recovery: no published state ever depends on
// spill residency.
class SpillKVStore {
public:
    // One retained byte range in the file. `generation` distinguishes a stale extent value
    // from a reuse of the same offset after release.
    struct Extent {
        std::uint64_t offset     = 0;
        std::uint64_t bytes      = 0;
        std::uint32_t generation = 0;

        [[nodiscard]] friend bool operator==(const Extent&, const Extent&) = default;
    };

    SpillKVStore(std::filesystem::path path, std::size_t capacity_bytes);
    ~SpillKVStore();

    SpillKVStore(const SpillKVStore&)            = delete;
    SpillKVStore& operator=(const SpillKVStore&) = delete;
    SpillKVStore(SpillKVStore&&)                 = delete;
    SpillKVStore& operator=(SpillKVStore&&)      = delete;

    // The bound on the file size. Live (reserved) bytes never exceed it because the file only
    // grows through appends that the bound checks.
    [[nodiscard]] std::size_t capacity_bytes() const noexcept { return capacity_bytes_; }

    // Bytes currently claimed by live extents. Spill-resident bytes are billed once, like
    // host-resident bytes; this is the spill axis of the global occupancy model.
    [[nodiscard]] std::size_t reserved_bytes() const noexcept { return reserved_bytes_; }

    // Writes `bytes` into a new extent (first fit over released extents, append otherwise)
    // and returns it. Throws std::invalid_argument when the bound cannot hold the extent and
    // std::runtime_error when the write is short.
    [[nodiscard]] Extent write(const std::byte* bytes, std::size_t count);

    // Split form for described transfers: reserve commits the capacity and returns the extent,
    // write_at fills it (exactly extent.bytes), cancel gives an unused reservation back. The
    // executor of a spill-out uses this form because the describing side records the extent
    // before the copy runs.
    [[nodiscard]] Extent reserve(std::size_t count);
    void write_at(const Extent& extent, const std::byte* bytes, std::size_t count);
    void cancel(const Extent& extent) noexcept;

    // Reads an extent back exactly. Throws std::invalid_argument for a foreign extent and
    // std::runtime_error when the read is short.
    void read(const Extent& extent, std::byte* destination, std::size_t count) const;

    // Returns the extent's bytes to the free pool. A foreign extent is ignored.
    void release(const Extent& extent) noexcept;

private:
    struct FreeExtent {
        std::uint64_t offset = 0;
        std::uint64_t bytes  = 0;
    };

    void checked_write(const void* bytes, std::size_t count, std::uint64_t offset) const;
    void checked_read(void* bytes, std::size_t count, std::uint64_t offset) const;

    std::filesystem::path path_;
    int descriptor_             = -1;
    std::size_t capacity_bytes_ = 0;
    std::size_t reserved_bytes_ = 0;
    std::uint64_t append_offset_   = 0;
    std::uint32_t next_generation_ = 1;
    // Offsets of live extents to their generations: a released-then-reused offset cannot be
    // read or released through a stale Extent value.
    std::unordered_map<std::uint64_t, std::uint32_t> live_;
    std::vector<FreeExtent> free_extents_;
};

} // namespace ninfer
