// The artifact loader refuses a network filesystem before it starts a direct read, because a direct
// read there can wait uninterruptibly inside the kernel. The mapping is exercised directly, so this
// test needs no NFS mount; the acceptance side uses a real tmpfs and the repository's own directory.

#include "core/io/file.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <sys/vfs.h>

namespace {

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

// The kernel's superblock magic numbers, restated here rather than read from the header so the test
// fails if the mapping drifts rather than silently following it.
constexpr std::uint64_t kNfs  = 0x6969;
constexpr std::uint64_t kCifs = 0xFF534D42;
constexpr std::uint64_t kSmb  = 0xFE534D42;
constexpr std::uint64_t kFuse = 0x65735546;
constexpr std::uint64_t kNine = 0x01021997;
constexpr std::uint64_t kExt  = 0xEF53;
constexpr std::uint64_t kTmp  = 0x01021994;

int exercise_mapping() {
    int failures = 0;
    const auto named = [](std::uint64_t type) {
        const char* name = ninfer::io::unsupported_direct_io_filesystem(type);
        return name == nullptr ? std::string() : std::string(name);
    };
    failures += check(named(kNfs) == "NFS", "NFS magic is not refused as NFS");
    failures += check(named(kCifs) == "CIFS", "CIFS magic is not refused as CIFS");
    failures += check(named(kSmb) == "SMB", "SMB2 magic is not refused as SMB");
    failures += check(named(kFuse) == "FUSE", "FUSE magic is not refused as FUSE");
    failures += check(named(kNine) == "9P", "9P magic is not refused as 9P");
    // A local filesystem must be accepted: refusing an unknown type would break the lane this
    // loader exists for.
    failures += check(named(kExt).empty(), "ext4 was refused");
    failures += check(named(kTmp).empty(), "tmpfs was refused");
    failures += check(named(0).empty(), "an unknown filesystem was refused");
    return failures;
}

// A real tmpfs and the repository's own directory must both be accepted, and their reported type
// must be the one the kernel says rather than something the helper invented.
int exercise_real_filesystems() {
    int failures = 0;
    struct statfs status {};
    if (::statfs("/dev/shm", &status) == 0) {
        const auto type = static_cast<std::uint64_t>(status.f_type);
        failures += check(ninfer::io::filesystem_type("/dev/shm") == type,
                          "filesystem_type disagrees with statfs for /dev/shm");
        failures += check(ninfer::io::unsupported_direct_io_filesystem(type) == nullptr,
                          "a real tmpfs was refused");
    } else {
        std::cout << "note: /dev/shm is not a tmpfs here; skipped the tmpfs acceptance case\n";
    }
    const auto local = ninfer::io::filesystem_type(".");
    failures += check(ninfer::io::unsupported_direct_io_filesystem(local) == nullptr,
                      "the repository directory's filesystem was refused");
    return failures;
}

} // namespace

int main() {
    int failures = exercise_mapping();
    failures += exercise_real_filesystems();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
