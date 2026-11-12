#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <filesystem>
#include <span>

namespace ninfer::io {

enum class ReadMode { Buffered, Direct };

struct FileView {
    int descriptor               = -1;
    std::uint64_t bytes          = 0;
    ReadMode mode                = ReadMode::Buffered;
    std::size_t memory_alignment = 1;
    std::size_t offset_alignment = 1;
};

// One immutable regular-file handle. Views borrow it; all I/O completes before destruction.
class ReadOnlyFile {
public:
    ReadOnlyFile(const std::filesystem::path& path, ReadMode mode = ReadMode::Buffered);
    ~ReadOnlyFile();
    ReadOnlyFile(ReadOnlyFile&&) noexcept;
    ReadOnlyFile& operator=(ReadOnlyFile&&) noexcept;
    ReadOnlyFile(const ReadOnlyFile&)            = delete;
    ReadOnlyFile& operator=(const ReadOnlyFile&) = delete;

    [[nodiscard]] FileView view() const noexcept { return view_; }

    [[nodiscard]] std::uint64_t bytes() const noexcept { return view_.bytes; }

    // Buffered exact read. A missing required byte is an error.
    void read_exact(std::uint64_t offset, std::span<std::byte> destination) const;
    // One aligned direct read, including a possible short final block.
    [[nodiscard]] std::size_t read_direct(std::uint64_t offset,
                                          std::span<std::byte> destination) const;
    // Shared read-only mapping of [offset, offset+bytes). No payload is copied or pinned in
    // host RAM; the mapping dies with the last holder of the returned owner.
    [[nodiscard]] std::shared_ptr<const std::byte> map(std::uint64_t offset,
                                                       std::uint64_t bytes) const;

private:
    std::filesystem::path path_;
    FileView view_;
};

// The network filesystem a direct read cannot use, named for an error message, or null when the
// loader accepts the filesystem. Pure: the caller supplies the f_type statfs reported, so the
// mapping is exercised without mounting that filesystem.
//
// Direct I/O on a network filesystem is not merely slow: a read can enter an uninterruptible wait
// inside the filesystem's own direct path (nfs_direct_wait), where it ignores SIGKILL and is
// indistinguishable from an engine deadlock. The loader therefore refuses those filesystems at open
// rather than starting a read that cannot be cancelled.
[[nodiscard]] const char* unsupported_direct_io_filesystem(std::uint64_t f_type) noexcept;

// The f_type of the filesystem holding `path`. Throws std::system_error when statfs fails, which
// includes a path that does not exist.
[[nodiscard]] std::uint64_t filesystem_type(const std::filesystem::path& path);

struct DirectReadWindow {
    std::uint64_t offset;
    std::size_t bytes;
    std::size_t prefix;   // Requested bytes start here within the aligned buffer.
    std::size_t required; // Minimum returned byte count covering the requested range.
};

// Maps a nonempty in-file range to one enclosing aligned read. Its end may exceed EOF.
[[nodiscard]] DirectReadWindow direct_read_window(FileView file, std::uint64_t offset,
                                                  std::size_t bytes);
void validate_direct_read(FileView file, std::uint64_t offset, std::span<std::byte> destination);

} // namespace ninfer::io
