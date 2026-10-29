#include "core/host_kv_arena.h"
#include "core/spill_kv_store.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

template <typename Exception, typename Fn>
int expect_throws(Fn&& fn, const char* label) {
    try {
        fn();
    } catch (const Exception&) { return 0; }
    std::cerr << label << " did not throw expected exception\n";
    return 1;
}

ninfer::KVPageGeometry test_geometry() {
    ninfer::KVPlaneGeometry keys{};
    keys.dtype          = ninfer::DType::BF16;
    keys.leading_extent = 8;
    keys.head_extent    = 128;
    keys.tokens_per_row = 1;
    ninfer::KVPlaneGeometry values = keys;
    ninfer::KVPageGeometry page{};
    page.page_tokens = 256;
    page.planes      = {keys, values};
    return page;
}

std::vector<std::byte> patterned_page_bytes(const ninfer::HostKVPageLayout& layout,
                                            std::uint32_t pages, std::uint32_t seed) {
    std::vector<std::byte> bytes(layout.page_stride * pages);
    std::mt19937_64 random(seed);
    for (auto& value : bytes) {
        value = static_cast<std::byte>(random() & 0xffU);
    }
    return bytes;
}

int run() {
    const ninfer::HostKVPageLayout layout = ninfer::plan_host_kv_page_layout(test_geometry());
    if (layout.page_stride == 0) { return fail("planned host layout has an empty page stride"); }

    const std::filesystem::path path =
        (std::filesystem::temp_directory_path() / "ninfer_spill_kv_store_test.bin");
    std::error_code ignored;
    std::filesystem::remove(path, ignored);

    int failures = 0;
    try {
        // The capacity bound holds exactly two pages per store in the bound test below.
        const std::size_t capacity = layout.page_stride * 4;
        ninfer::SpillKVStore store(path, capacity);
        if (store.capacity_bytes() != capacity || store.reserved_bytes() != 0) {
            failures += fail("fresh store accounting is wrong");
        }

        // Two interleaved writes read back bit-identical.
        const std::vector<std::byte> first  = patterned_page_bytes(layout, 1, 11U);
        const std::vector<std::byte> second = patterned_page_bytes(layout, 1, 22U);
        const auto first_extent             = store.write(first.data(), first.size());
        const auto second_extent            = store.write(second.data(), second.size());
        if (store.reserved_bytes() != 2 * layout.page_stride) {
            failures += fail("reserved bytes do not track two live pages");
        }

        std::vector<std::byte> back(first.size());
        store.read(first_extent, back.data(), back.size());
        if (back != first) { failures += fail("first page round-trip is not bit-identical"); }
        store.read(second_extent, back.data(), back.size());
        if (back != second) { failures += fail("second page round-trip is not bit-identical"); }

        // A multi-page extent spanning released space reads back whole.
        store.release(first_extent);
        if (store.reserved_bytes() != layout.page_stride) {
            failures += fail("release does not free the page");
        }
        ninfer::SpillKVStore::Extent stale = first_extent;
        try {
            store.read(stale, back.data(), back.size());
            failures += fail("reading a released extent did not throw");
        } catch (const std::invalid_argument&) {}
        store.release(stale);
        if (store.reserved_bytes() != layout.page_stride) {
            failures += fail("double release changed the accounting");
        }

        const std::vector<std::byte> pair = patterned_page_bytes(layout, 2, 33U);
        const auto pair_extent            = store.write(pair.data(), pair.size());
        back.assign(pair.size(), std::byte{0});
        store.read(pair_extent, back.data(), back.size());
        if (back != pair) { failures += fail("reused-space round-trip is not bit-identical"); }

        store.release(second_extent);
        store.release(pair_extent);
        if (store.reserved_bytes() != 0) {
            failures += fail("full release does not empty the store");
        }

        // The coalesced released region is reused in full and reads back whole.
        const std::vector<std::byte> reuse = patterned_page_bytes(layout, 2, 55U);
        const auto reuse_extent            = store.write(reuse.data(), reuse.size());
        if (reuse_extent.offset != 0) {
            failures += fail("coalesced free space was not reused");
        }
        back.assign(reuse.size(), std::byte{0});
        store.read(reuse_extent, back.data(), back.size());
        if (back != reuse) { failures += fail("reuse round-trip is not bit-identical"); }

        // The bound is on the file size, not just live bytes: with no fitting free extent, an
        // extent whose append would grow the file past the capacity is rejected even though
        // live bytes would stay within it.
        const std::vector<std::byte> overflow = patterned_page_bytes(layout, 3, 44U);
        failures += expect_throws<std::invalid_argument>(
            [&] { static_cast<void>(store.write(overflow.data(), overflow.size())); },
            "over-capacity write");
        store.release(reuse_extent);

        // Beyond the whole capacity no arrangement helps.
        const std::vector<std::byte> impossible = patterned_page_bytes(layout, 5, 66U);
        failures += expect_throws<std::invalid_argument>(
            [&] { static_cast<void>(store.write(impossible.data(), impossible.size())); },
            "over-capacity write");
    } catch (const std::exception& error) {
        std::cerr << "spill store test threw: " << error.what() << '\n';
        ++failures;
    }

    std::filesystem::remove(path, ignored);
    return failures;
}

} // namespace

int main() { return run(); }
