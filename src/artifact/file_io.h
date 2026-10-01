#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <memory>

namespace ninfer::artifact {

enum class MappingAccess { Normal, Random };

// Direct reads require aligned offsets and buffers. A short final direct block is allowed;
// read_exact always requires the complete requested byte range.
class InputFile {
public:
    explicit InputFile(std::filesystem::path path);
    ~InputFile();
    InputFile(const InputFile&)            = delete;
    InputFile& operator=(const InputFile&) = delete;

    [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }

    [[nodiscard]] std::shared_ptr<const std::byte> map(
        std::uint64_t offset, std::size_t bytes, MappingAccess access) const;

    void read_exact(std::uint64_t offset, std::span<std::byte> destination) const;
    [[nodiscard]] std::size_t read_direct(std::uint64_t offset,
                                          std::span<std::byte> destination) const;

private:
    std::filesystem::path path_;
    int fd_                = -1;
    mutable int direct_fd_ = -1;
    std::uint64_t bytes_   = 0;
};

} // namespace ninfer::artifact
