// GB10 file-backed page probe.
//
// Answers the plan step 6.3 gate (docs/maintainer/plan-2026-09-gb10.md): can a device kernel
// gather PLE rows — random 2560 B reads from a file-backed mmap — and what does the GPU pay
// when the file page is resident in the page cache versus cold (page-cache miss, NVMe page-in
// serviced by the fault path)?
//
// Phases run as separate invocations (`--phase`) so a hung cold-fault phase can be killed by a
// timeout without taking the rest. The row offset sequence is deterministic (splitmix64, fixed
// seed), so resident/cold runs read identical rows.
//
// Phases:
//   prep           create + fill + fsync the scratch file
//   warm           host-read the whole mapping (populate the page cache)
//   res-seq        GPU full sequential read (pages resident)
//   res-rows       GPU random 2560 B rows (PLE pattern; resident)
//   evict          host-read the evictor file to push the scratch mapping out of the page cache
//   cold-serial    1 thread, sequential cold rows (serialized per-fault latency)
//   cold-rows      GPU random cold rows (parallel fault throughput)
//   res-rows-again same rows as res-rows, now resident (sanity: must match res-rows)
//
// Build (no libcuda link needed):
//   nvcc -O3 -std=c++17 -arch=sm_121a tools/gb10/file_page_probe.cu -o file_page_probe

#include <cuda_runtime.h>

#include <chrono>
#include <functional>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

#define CUDA_CHECK(expr)                                                                          \
    do {                                                                                          \
        const cudaError_t e_ = (expr);                                                            \
        if (e_ != cudaSuccess) {                                                                  \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), __FILE__, \
                         __LINE__, cudaGetErrorString(e_));                                       \
            std::exit(1);                                                                         \
        }                                                                                         \
    } while (false)

constexpr std::uint32_t kRowBytes   = 2560; // PLE per-token size: 16 heads x 160 B
constexpr std::uint32_t kRowWords   = kRowBytes / sizeof(std::uint32_t); // 640 u32
constexpr std::uint32_t kPageBytes  = 4096;
constexpr std::uint64_t kSeed       = 0x9E3779B97F4A7C15ull;

struct Options {
    std::string phase;
    std::string file;
    std::string evict;
    std::size_t bytes       = std::size_t{512} << 20;
    std::size_t evict_bytes = std::size_t{130} << 30;
    int rows               = 100000;
    int serial_rows        = 512;
};

std::size_t parse_bytes(std::string s) {
    auto mult = std::size_t{1};
    if (!s.empty()) {
        const char u = s.back();
        if (u == 'G' || u == 'g') { mult = std::size_t{1} << 30; s.pop_back(); }
        else if (u == 'M' || u == 'm') { mult = std::size_t{1} << 20; s.pop_back(); }
        else if (u == 'K' || u == 'k') { mult = std::size_t{1} << 10; s.pop_back(); }
    }
    return std::stoull(s) * mult;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next     = [&] {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value after %s\n", a.c_str()); std::exit(2); }
            return std::string(argv[++i]);
        };
        if (a == "--phase") { o.phase = next(); }
        else if (a == "--file") { o.file = next(); }
        else if (a == "--evict") { o.evict = next(); }
        else if (a == "--bytes") { o.bytes = parse_bytes(next()); }
        else if (a == "--evict-bytes") { o.evict_bytes = parse_bytes(next()); }
        else if (a == "--rows") { o.rows = std::stoi(next()); }
        else if (a == "--serial-rows") { o.serial_rows = std::stoi(next()); }
        else {
            std::fprintf(stderr, "usage: file_page_probe --phase prep|warm|res-seq|res-rows|evict|"
                                 "cold-serial|cold-rows|res-rows-again --file PATH "
                                 "[--bytes 512M] [--evict PATH] [--evict-bytes 130G] "
                                 "[--rows 100000] [--serial-rows 512]\n");
            std::exit(a == "--help" ? 0 : 2);
        }
    }
    if (o.file.empty()) { std::fprintf(stderr, "--file is required\n"); std::exit(2); }
    o.bytes = o.bytes / kPageBytes * kPageBytes;
    return o;
}

// Deterministic row-page index for row i: splitmix64 of (seed ^ i*golden) mod page count.
std::uint64_t row_page(std::uint64_t i, std::uint64_t pages) {
    std::uint64_t x = kSeed ^ (i * 0x9E3779B97F4A7C15ull);
    x += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z  = z ^ (z >> 31);
    return z % pages;
}

// ---- kernels ------------------------------------------------------------------------------------

// One block per row; the block's 640 u32 (2560 B) are read coalesced from a 4 KiB-aligned offset.
__global__ void row_read_kernel(const std::uint32_t* __restrict__ base, const std::uint32_t* __restrict__ offsets,
                                int n, std::uint32_t* sink) {
    const int row = blockIdx.x;
    if (row >= n) { return; }
    const std::uint32_t* p = base + static_cast<std::size_t>(offsets[row]) * (kPageBytes / sizeof(std::uint32_t));
    std::uint32_t acc = threadIdx.x;
    for (std::uint32_t w = threadIdx.x; w < kRowWords; w += blockDim.x) {
        acc ^= __ldcs(p + w);
    }
    if (acc == 0x9E3779B9u) { sink[row] = acc; }
}

// Single thread, sequential rows: serialized per-fault latency.
__global__ void serial_row_kernel(const std::uint32_t* __restrict__ base, const std::uint32_t* __restrict__ offsets,
                                  int n, std::uint32_t* sink) {
    if (threadIdx.x != 0 || blockIdx.x != 0) { return; }
    std::uint32_t acc = 0;
    for (int row = 0; row < n; ++row) {
        const std::uint32_t* p = base + static_cast<std::size_t>(offsets[row]) * (kPageBytes / sizeof(std::uint32_t));
        for (std::uint32_t w = 0; w < kRowWords; ++w) { acc ^= p[w]; }
    }
    if (acc == 0x9E3779B9u) { sink[0] = acc; }
}

__global__ void seq_read_kernel(const std::uint32_t* __restrict__ data, std::size_t words, std::uint32_t* sink) {
    std::uint32_t acc = threadIdx.x;
    for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < words;
         i += std::size_t(gridDim.x) * blockDim.x) {
        acc ^= __ldcs(data + i);
    }
    if (acc == 0x9E3779B9u) { sink[blockIdx.x] = acc; }
}

double timed_kernel(std::function<void()> launch, const char* label, bool warm) {
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    if (warm) {
        // Resident phases only: cold phases must fault inside the timed launch.
        launch();
        CUDA_CHECK(cudaGetLastError());
    }
    const auto h0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(a));
    launch();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    const double host_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - h0).count();
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    std::printf("| %s | %9.3f ms | %9.3f ms |\n", label, ms, host_s * 1e3);
    return ms / 1e3;
}

// ---- host phases --------------------------------------------------------------------------------

void phase_prep(const Options& o) {
    const int fd = open(o.file.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd < 0) { std::perror("open scratch"); std::exit(1); }
    if (ftruncate(fd, static_cast<off_t>(o.bytes)) != 0) { std::perror("ftruncate"); std::exit(1); }
    std::vector<char> buf(1 << 20);
    std::uint64_t x = 0x4559;
    for (std::size_t off = 0; off < o.bytes; off += buf.size()) {
        for (char& c : buf) { x ^= x << 13; x ^= x >> 7; c = char(x & 0xFF); }
        if (write(fd, buf.data(), buf.size()) != static_cast<ssize_t>(buf.size())) {
            std::perror("write scratch"); std::exit(1);
        }
    }
    if (fsync(fd) != 0) { std::perror("fsync scratch"); std::exit(1); }
    close(fd);
    std::printf("scratch file %s: %zu bytes (%zu pages), pattern written + fsynced\n", o.file.c_str(),
                o.bytes, o.bytes / kPageBytes);
}

void* map_file(const std::string& path, std::size_t bytes) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { std::perror("open file"); std::exit(1); }
    void* p = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { std::perror("mmap"); std::exit(1); }
    return p;
}

void phase_warm(const Options& o) {
    char* p = static_cast<char*>(map_file(o.file, o.bytes));
    const auto t0 = std::chrono::steady_clock::now();
    volatile std::uint64_t acc = 0;
    for (std::size_t off = 0; off < o.bytes; off += kPageBytes) {
        acc += reinterpret_cast<const std::uint64_t*>(p + off)[0];
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("host warm read: %zu pages, %6.3f ms, %8.2f GB/s (acc=%llu)\n", o.bytes / kPageBytes,
                s * 1e3, double(o.bytes) / s / 1e9, static_cast<unsigned long long>(acc));
    munmap(p, o.bytes);
}
} // namespace


int main(int argc, char** argv) {
    const Options o = parse(argc, argv);
    int sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    unsigned* sink = nullptr;
    CUDA_CHECK(cudaMalloc(&sink, static_cast<std::size_t>(o.rows) * sizeof(unsigned)));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    if (o.phase == "prep") { phase_prep(o); return 0; }
    if (o.phase == "warm") { phase_warm(o); return 0; }

    char* base = static_cast<char*>(map_file(o.file, o.bytes));
    const std::uint64_t pages = o.bytes / kPageBytes;

    if (o.phase == "res-seq" || o.phase == "res-rows" || o.phase == "evict" ||
        o.phase == "cold-serial" || o.phase == "cold-rows" || o.phase == "res-rows-again") {
        if (o.phase == "evict") {
            if (o.evict.empty()) { std::fprintf(stderr, "--evict PATH is required for the evict phase\n"); return 2; }
            const int fd = open(o.evict.c_str(), O_RDONLY);
            if (fd < 0) { std::perror("open evictor"); return 1; }
            std::vector<char> buf(1 << 18);
            std::uint64_t total = 0, acc = 0;
            const auto t0 = std::chrono::steady_clock::now();
            while (total < o.evict_bytes) {
                const ssize_t r = read(fd, buf.data(), buf.size());
                if (r <= 0) { if (lseek(fd, 0, SEEK_SET) < 0) { break; } continue; } // loop the file
                for (const char c : buf) { if (c == 0x5A) { ++acc; } }
                total += static_cast<std::uint64_t>(r);
            }
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("evictor %s: %zu MiB read in %6.1f s (%8.2f GB/s, acc=%llu) — scratch mapping should now be "
                        "out of the page cache\n",
                        o.evict.c_str(), total >> 20, s, double(total) / s / 1e9,
                        static_cast<unsigned long long>(acc));
            close(fd);
            return 0;
        }
    }

    const int n = (o.phase == "cold-serial") ? o.serial_rows : o.rows;
    std::vector<std::uint32_t> offsets(n);
    for (int i = 0; i < n; ++i) { offsets[i] = static_cast<std::uint32_t>(row_page(std::uint64_t(i), pages)); }
    std::uint32_t* d_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_offsets, offsets.size() * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_offsets, offsets.data(), offsets.size() * sizeof(std::uint32_t),
                               cudaMemcpyHostToDevice, stream));

    const std::size_t bytes_moved = std::size_t(n) * kRowBytes;
    std::printf("phase %s: %d rows x %u B = %zu KiB, %d SMs\n", o.phase.c_str(), n, kRowBytes,
                bytes_moved >> 10, sms);
    std::printf("| measurement | GPU ms | wall ms |\n|---|---|---|\n");
    if (o.phase == "res-seq") {
        const std::size_t words = o.bytes / sizeof(std::uint32_t);
        const double s = timed_kernel([&] {
            seq_read_kernel<<<sms * 16, 256, 0, stream>>>(reinterpret_cast<const std::uint32_t*>(base), words, sink);
        }, "sequential full-file read", true);
        std::printf("effective bandwidth: %8.2f GB/s\n", double(o.bytes) / s / 1e9);
    } else if (o.phase == "cold-serial") {
        const double s = timed_kernel([&] {
            serial_row_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<const std::uint32_t*>(base), d_offsets, n, sink);
        }, "serial cold rows (1 thread)", false);
        std::printf("per-row: %9.2f us  (serialized page-fault path, NVMe page-in if cold)\n",
                    s * 1e6 / n);
    } else {
        const double s = timed_kernel([&] {
            row_read_kernel<<<n, 64, 0, stream>>>(reinterpret_cast<const std::uint32_t*>(base), d_offsets, n, sink);
        }, "random 2560 B rows", o.phase != "cold-rows");
        std::printf("effective bandwidth: %8.2f MB/s | per-row (amortized): %8.2f us\n",
                    double(bytes_moved) / s / 1e6, s * 1e6 / n);
    }
    std::uint32_t h = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h, sink, sizeof(h), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::printf("sink sanity: 0x%X\n", h);
    return 0;
}

