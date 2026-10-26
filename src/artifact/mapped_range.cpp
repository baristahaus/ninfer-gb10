// ninfer::artifact - master-generation mapped-range support over the current Reader and
// io::ReadOnlyFile. The Flash-Next PLE disk tier reads its hash table through these mappings.

#include "artifact/reader.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#include <sys/mman.h>
#include <unistd.h>

namespace ninfer::artifact {

MappedRange Reader::map_range(std::uint64_t offset, std::uint64_t bytes) const {
    MappedRange result;
    for (const auto& segment : segments(offset, bytes)) {
        result.segments.push_back(
            {open_file(segment.file_index, io::ReadMode::Buffered).map(segment.file_offset,
                                                                      segment.bytes),
             segment.destination_offset, segment.bytes});
    }
    return result;
}

void MappedRange::copy(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > size() || destination.size() > size() - offset) {
        throw std::invalid_argument("mapped read exceeds range");
    }
    if (destination.empty()) { return; }
    auto it = std::upper_bound(
        segments.begin(), segments.end(), offset,
        [](std::uint64_t value, const MappedSegment& segment) { return value < segment.begin; });
    --it;
    while (!destination.empty()) {
        const auto local = offset - it->begin;
        const auto count = std::min<std::uint64_t>(destination.size(), it->bytes - local);
        std::copy_n(it->storage.get() + local, count, destination.data());
        offset += count;
        destination = destination.subspan(count);
        ++it;
    }
}

namespace {

// madvise needs page-aligned starts; a segment's storage begins inside its first mapped page.
void advise_pages(const std::byte* begin, std::uint64_t bytes, int advice) {
    static const auto page = static_cast<std::uintptr_t>(::sysconf(_SC_PAGESIZE));
    const auto first       = reinterpret_cast<std::uintptr_t>(begin) / page * page;
    const auto last        = reinterpret_cast<std::uintptr_t>(begin) + bytes;
    // Advice only: a refused hint (for example EAGAIN under memory pressure) leaves the later
    // read to fault the page in as before.
    (void)::madvise(reinterpret_cast<void*>(first), last - first, advice);
}

} // namespace

void MappedRange::advise_random() const {
    for (const auto& segment : segments) {
        advise_pages(segment.storage.get(), segment.bytes, MADV_RANDOM);
    }
}

void MappedRange::prefetch(std::uint64_t offset, std::uint64_t bytes) const {
    if (offset > size() || bytes > size() - offset) {
        throw std::invalid_argument("mapped prefetch exceeds range");
    }
    if (bytes == 0) { return; }
    auto it = std::upper_bound(
        segments.begin(), segments.end(), offset,
        [](std::uint64_t value, const MappedSegment& segment) { return value < segment.begin; });
    --it;
    while (bytes != 0) {
        const auto local = offset - it->begin;
        const auto count = std::min<std::uint64_t>(bytes, it->bytes - local);
        advise_pages(it->storage.get() + local, count, MADV_WILLNEED);
        offset += count;
        bytes -= count;
        ++it;
    }
}

} // namespace ninfer::artifact
