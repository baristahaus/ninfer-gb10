// GB10 allocation-versus-page-cache probe.
//
// The engine sizes weights and automatic KV from cudaMemGetInfo's free bytes. On GB10 that value
// excludes clean file pages (the page cache holding the PLE table and recently read artifact
// volumes), so a warm machine refuses a load that the unified pool can hold. Whether the fix is
// "size from /proc/meminfo MemAvailable" depends on what this probe measures: when cudaMalloc
// asks for more than cudaMemGetInfo reports free, does the kernel reclaim clean page cache and
// succeed, and how do cudaMemGetInfo, MemFree, MemAvailable and Cached move as it does?
//
// It allocates and touches CHUNK_GIB blocks until it has taken TARGET bytes (default: MemAvailable
// at start minus the reserve) or cudaMalloc fails, printing one Markdown row per block, then frees
// everything. Run it after warming the page cache (read the artifact volumes) and with no other
// GPU job.
//
// Build: nvcc -O2 -std=c++17 -arch=sm_121a tools/gb10/alloc_probe.cu -o alloc_probe
// Usage: alloc_probe [--chunk-gib N (default 4)] [--reserve-gib N (default 8)]

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct MemInfo {
    double free_gib      = 0;
    double available_gib = 0;
    double cached_gib    = 0;
};

MemInfo read_meminfo() {
    MemInfo m;
    std::ifstream in("/proc/meminfo");
    std::string key;
    long long kib = 0;
    std::string unit;
    while (in >> key >> kib >> unit) {
        const double gib = static_cast<double>(kib) / (1024.0 * 1024.0);
        if (key == "MemFree:") { m.free_gib = gib; }
        if (key == "MemAvailable:") { m.available_gib = gib; }
        if (key == "Cached:") { m.cached_gib = gib; }
    }
    return m;
}

double cuda_free_gib() {
    std::size_t free_bytes = 0, total = 0;
    if (cudaMemGetInfo(&free_bytes, &total) != cudaSuccess) { return -1; }
    return static_cast<double>(free_bytes) / (1ULL << 30);
}

void row(const char* label, double taken_gib, const char* result) {
    const MemInfo m = read_meminfo();
    std::printf("| %s | %.0f | %.1f | %.1f | %.1f | %.1f | %s |\n", label, taken_gib,
                cuda_free_gib(), m.free_gib, m.available_gib, m.cached_gib, result);
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    double chunk_gib = 4, reserve_gib = 8;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (std::strcmp(argv[i], "--chunk-gib") == 0) { chunk_gib = std::atof(argv[i + 1]); }
        else if (std::strcmp(argv[i], "--reserve-gib") == 0) { reserve_gib = std::atof(argv[i + 1]); }
        else {
            std::fprintf(stderr, "usage: alloc_probe [--chunk-gib N] [--reserve-gib N]\n");
            return 2;
        }
    }
    if (cudaFree(nullptr) != cudaSuccess) {
        std::fprintf(stderr, "CUDA initialization failed\n");
        return 1;
    }
    const MemInfo start          = read_meminfo();
    const double start_cuda_free = cuda_free_gib();
    const double target_gib = start.available_gib - reserve_gib;
    std::printf("Target %.1f GiB (MemAvailable %.1f - reserve %.1f), chunk %.1f GiB.\n\n",
                target_gib, start.available_gib, reserve_gib, chunk_gib);
    std::printf("| step | taken GiB | cudaMemGetInfo free GiB | MemFree GiB | MemAvailable GiB "
                "| Cached GiB | result |\n|---|---:|---:|---:|---:|---:|---|\n");
    row("start", 0, "-");
    const std::size_t chunk = static_cast<std::size_t>(chunk_gib * (1ULL << 30));
    std::vector<void*> blocks;
    double taken = 0;
    int step = 0;
    while (taken + chunk_gib <= target_gib) {
        void* p = nullptr;
        const cudaError_t e = cudaMalloc(&p, chunk);
        if (e != cudaSuccess) {
            row(std::to_string(++step).c_str(), taken, cudaGetErrorName(e));
            cudaGetLastError();
            break;
        }
        // Touch every page so the allocation is backed, not just reserved.
        if (cudaMemset(p, 1, chunk) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
            row(std::to_string(++step).c_str(), taken, "touch failed");
            cudaFree(p);
            cudaGetLastError();
            break;
        }
        blocks.push_back(p);
        taken += chunk_gib;
        row(std::to_string(++step).c_str(), taken, "ok");
    }
    for (void* p : blocks) { cudaFree(p); }
    cudaDeviceSynchronize();
    row("freed", 0, "-");
    std::printf("\nTaken %.1f GiB against a starting cudaMemGetInfo free of %.1f GiB (%+.1f GiB).\n",
                taken, start_cuda_free, taken - start_cuda_free);
    return 0;
}
