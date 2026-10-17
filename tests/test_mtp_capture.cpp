#include "models/qwen3_8_flash_next/impl/runtime/mtp_capture.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::models::qwen3_8_flash_next::detail;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        failures += 1;
    }
}

std::vector<std::byte> read_file(const std::filesystem::path& path) {
    FILE* file = std::fopen(path.c_str(), "rb");
    assert(file != nullptr);
    std::fseek(file, 0, SEEK_END);
    const long size = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    std::vector<std::byte> data(static_cast<std::size_t>(size));
    if (size > 0) { std::fread(data.data(), 1, data.size(), file); }
    std::fclose(file);
    return data;
}

} // namespace

int main() {
    const std::filesystem::path path =
        (std::filesystem::temp_directory_path() / "ninfer_mtp_capture_test.bin");

    // Rejections: bad window.
    bool bad_window_rejected = false;
    try {
        (void)MtpCaptureSink::open(path, 0, 248320);
    } catch (const std::invalid_argument&) { bad_window_rejected = true; }
    check(bad_window_rejected, "a zero draft window was accepted");

    // Unwritable path.
    bool unwritable_rejected = false;
    try {
        (void)MtpCaptureSink::open("/proc/ninfer-no-such-dir/capture.bin", 2, 248320);
    } catch (const std::runtime_error&) { unwritable_rejected = true; }
    check(unwritable_rejected, "an unwritable capture path was accepted");

    constexpr std::uint32_t k = 2;
    {
        std::unique_ptr<MtpCaptureSink> sink =
            MtpCaptureSink::open(path, k, 248320);
        for (std::uint64_t round = 1; round <= 4000; ++round) {
            for (std::uint32_t lane = 0; lane < 3; ++lane) {
                const std::array<TokenId, 5> drafts{
                    static_cast<TokenId>(round * 10 + lane), 7, -1, 0, 1};
                const std::array<TokenId, 3> committed{
                    static_cast<TokenId>(round * 10 + lane), 42, 43};
                sink->record(round, lane, round * 100 + lane, 2, drafts,
                             committed);
            }
        }
        check(!sink->failed(), "the writer thread reported a failure");
    }
    // Destructor joined: the file must be complete.

    const std::vector<std::byte> data = read_file(path);
    MtpCaptureHeader header{};
    std::memcpy(&header, data.data(), sizeof(header));
    check(std::memcmp(header.magic, kMtpCaptureMagic, sizeof(header.magic)) == 0,
          "header magic mismatch");
    check(header.version == kMtpCaptureVersion, "header version mismatch");
    check(header.draft_window == k, "header draft window mismatch");
    check(header.record_width == k + 1, "header record width mismatch");
    check(header.token_domain == 248320, "header token domain mismatch");

    const std::size_t record_bytes = mtp_capture_record_bytes(k);
    const std::size_t expected = sizeof(header) + 4000 * 3 * record_bytes;
    check(data.size() == expected, "file size does not match header plus records");

    bool records_ok = true;
    std::size_t at = sizeof(header);
    for (std::uint64_t round = 1; round <= 4000 && records_ok; ++round) {
        for (std::uint32_t lane = 0; lane < 3; ++lane) {
            MtpCaptureRecord record{};
            std::memcpy(&record, data.data() + at, sizeof(record));
            at += sizeof(record);
            const auto* drafts = reinterpret_cast<const TokenId*>(data.data() + at);
            at += record_bytes - sizeof(record);
            records_ok = record.round_index == round && record.lane == lane &&
                         record.base_frontier == round * 100 + lane && record.extent == 2 &&
                         record.committed_count == 3 && drafts[0] == static_cast<int>(round * 10 + lane);
        }
    }
    check(records_ok, "record contents mismatch");

    std::filesystem::remove(path);
    if (failures != 0) { return 1; }
    std::puts("ok");
    return 0;
}
