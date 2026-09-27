// GB10 unified-memory probe.
//
// Standalone measurements for the questions in docs/maintainer/plan-2026-09-gb10.md that driver
// source cannot settle: what CUDA reports about the shared LPDDR5X pool, whether generic
// ("compute data") compression is granted and changes effective read bandwidth, how fast the GPU
// reads each kind of host-visible memory, and how CPU and GPU traffic contend. Prints Markdown.
// --sample LABEL=PATH (repeatable; tools/gb10/weight_samples.py writes them from an artifact)
// repeats the compression comparison on real weight bytes.
//
// Build (no libcuda link needed; driver-API calls go through the runtime entry-point query):
//   nvcc -O3 -std=c++17 -arch=sm_121a tools/gb10/memory_probe.cu -o memory_probe
// Run on an otherwise idle machine; nothing else should hold the GPU.

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

#define CUDA_CHECK(expr)                                                                           \
    do {                                                                                           \
        const cudaError_t e_ = (expr);                                                             \
        if (e_ != cudaSuccess) {                                                                   \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), __FILE__,  \
                         __LINE__, cudaGetErrorString(e_));                                        \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (false)

struct Options {
    std::size_t bytes     = std::size_t{1} << 30; // per buffer
    double seconds        = 0.5;                  // target duration of one measurement
    unsigned cpu_threads  = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::pair<std::string, std::string>> samples; // label, file
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next     = [&] {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after %s\n", a.c_str());
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if (a == "--gib") {
            o.bytes = static_cast<std::size_t>(std::stod(next()) * double(std::size_t{1} << 30));
        } else if (a == "--seconds") {
            o.seconds = std::stod(next());
        } else if (a == "--cpu-threads") {
            o.cpu_threads = static_cast<unsigned>(std::stoul(next()));
        } else if (a == "--sample") {
            const std::string v = next();
            const auto eq       = v.find('=');
            if (eq == std::string::npos || eq == 0 || eq + 1 == v.size()) {
                std::fprintf(stderr, "--sample expects LABEL=PATH, got %s\n", v.c_str());
                std::exit(2);
            }
            o.samples.emplace_back(v.substr(0, eq), v.substr(eq + 1));
        } else {
            std::fprintf(stderr,
                         "usage: memory_probe [--gib N (default 1)] [--seconds S (default 0.5)] "
                         "[--cpu-threads N (default all)] [--sample LABEL=PATH]...\n");
            std::exit(a == "--help" ? 0 : 2);
        }
    }
    o.bytes = o.bytes / 4096 * 4096;
    return o;
}

// ---- driver API through the runtime entry-point query (no -lcuda) -------------------------------

struct Driver {
    CUresult (*deviceGetAttribute)(int*, CUdevice_attribute, CUdevice)                      = nullptr;
    CUresult (*memGetAllocationGranularity)(size_t*, const CUmemAllocationProp*,
                                            CUmemAllocationGranularity_flags)                  = nullptr;
    CUresult (*memCreate)(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*,
                          unsigned long long)                                                 = nullptr;
    CUresult (*memGetAllocationPropertiesFromHandle)(CUmemAllocationProp*,
                                                     CUmemGenericAllocationHandle)            = nullptr;
    CUresult (*memAddressReserve)(CUdeviceptr*, size_t, size_t, CUdeviceptr,
                                  unsigned long long)                                          = nullptr;
    CUresult (*memMap)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle,
                       unsigned long long)                                                     = nullptr;
    CUresult (*memSetAccess)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t)             = nullptr;
    CUresult (*memUnmap)(CUdeviceptr, size_t)                                                 = nullptr;
    CUresult (*memRelease)(CUmemGenericAllocationHandle)                                      = nullptr;
    CUresult (*memAddressFree)(CUdeviceptr, size_t)                                           = nullptr;
};

template <typename Fn>
bool resolve(const char* symbol, Fn& fn) {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult status{};
    if (cudaGetDriverEntryPointByVersion(symbol, &ptr, 12000, cudaEnableDefault, &status) !=
            cudaSuccess ||
        status != cudaDriverEntryPointSuccess || ptr == nullptr) {
        return false;
    }
    fn = reinterpret_cast<Fn>(ptr);
    return true;
}

bool load_driver(Driver& d) {
    return resolve("cuDeviceGetAttribute", d.deviceGetAttribute) &&
           resolve("cuMemGetAllocationGranularity", d.memGetAllocationGranularity) &&
           resolve("cuMemCreate", d.memCreate) &&
           resolve("cuMemGetAllocationPropertiesFromHandle",
                   d.memGetAllocationPropertiesFromHandle) &&
           resolve("cuMemAddressReserve", d.memAddressReserve) && resolve("cuMemMap", d.memMap) &&
           resolve("cuMemSetAccess", d.memSetAccess) && resolve("cuMemUnmap", d.memUnmap) &&
           resolve("cuMemRelease", d.memRelease) && resolve("cuMemAddressFree", d.memAddressFree);
}

// ---- kernels ------------------------------------------------------------------------------------

__global__ void fill_kernel(uint4* data, std::size_t n, std::uint32_t seed, bool random) {
    for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < n;
         i += std::size_t(gridDim.x) * blockDim.x) {
        if (!random) {
            data[i] = make_uint4(0, 0, 0, 0);
            continue;
        }
        std::uint32_t x = seed ^ static_cast<std::uint32_t>(i * 2654435761u) ^
                          static_cast<std::uint32_t>(i >> 32);
        uint4 v;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5; v.x = x;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5; v.y = x;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5; v.z = x;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5; v.w = x;
        data[i] = v;
    }
}

// Fills `n` 16-byte words of `dst` by repeating the `period` words of `src`. A kernel store, so the
// data reaches a compressible allocation the same way section 2's fills do.
__global__ void tile_kernel(uint4* dst, std::size_t n, const uint4* src, std::size_t period) {
    for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < n;
         i += std::size_t(gridDim.x) * blockDim.x) {
        dst[i] = src[i % period];
    }
}

// Streams the buffer once; the conditional store keeps the loads alive.
__global__ void read_kernel(const uint4* __restrict__ data, std::size_t n, unsigned* sink) {
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < n;
         i += std::size_t(gridDim.x) * blockDim.x) {
        const uint4 v = __ldcs(data + i);
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9E3779B9u) { sink[blockIdx.x] = acc.x; }
}

struct Gpu {
    int sms       = 0;
    unsigned* sink = nullptr;
    cudaStream_t stream{};
};

// GB/s of one full read of `bytes` at `ptr`, averaged over enough launches to fill `seconds`.
double gpu_read_gbps(const Gpu& g, const void* ptr, std::size_t bytes, double seconds) {
    const std::size_t n = bytes / sizeof(uint4);
    const dim3 grid(g.sms * 8), block(256);
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    read_kernel<<<grid, block, 0, g.stream>>>(static_cast<const uint4*>(ptr), n, g.sink); // warm
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(a, g.stream));
    read_kernel<<<grid, block, 0, g.stream>>>(static_cast<const uint4*>(ptr), n, g.sink);
    CUDA_CHECK(cudaEventRecord(b, g.stream));
    CUDA_CHECK(cudaEventSynchronize(b));
    float one_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&one_ms, a, b));
    const int reps = std::max(1, static_cast<int>(seconds * 1000.0 / std::max(one_ms, 0.01f)));
    CUDA_CHECK(cudaEventRecord(a, g.stream));
    for (int r = 0; r < reps; ++r) {
        read_kernel<<<grid, block, 0, g.stream>>>(static_cast<const uint4*>(ptr), n, g.sink);
    }
    CUDA_CHECK(cudaEventRecord(b, g.stream));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return double(bytes) * reps / (ms * 1e-3) / 1e9;
}

void gpu_fill(const Gpu& g, void* ptr, std::size_t bytes, bool random) {
    fill_kernel<<<g.sms * 8, 256, 0, g.stream>>>(static_cast<uint4*>(ptr), bytes / sizeof(uint4),
                                                 0x1234567u, random);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(g.stream));
}

// ---- CPU side -----------------------------------------------------------------------------------

void cpu_fill_random(void* ptr, std::size_t bytes, unsigned threads) {
    std::vector<std::thread> pool;
    const std::size_t words = bytes / 8, per = (words + threads - 1) / threads;
    for (unsigned t = 0; t < threads; ++t) {
        pool.emplace_back([=] {
            auto* p               = static_cast<std::uint64_t*>(ptr);
            std::uint64_t x       = 0x9E3779B97F4A7C15ull * (t + 1);
            const std::size_t end = std::min(words, (t + 1) * per);
            for (std::size_t i = t * per; i < end; ++i) {
                x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                p[i] = x;
            }
        });
    }
    for (auto& th : pool) { th.join(); }
}

// Reads `bytes` repeatedly with `threads` threads until `stop` or `seconds` elapse; returns GB/s.
double cpu_read_gbps(const void* ptr, std::size_t bytes, unsigned threads, double seconds,
                     const std::atomic<bool>* stop = nullptr) {
    std::atomic<std::uint64_t> total{0};
    std::atomic<std::uint64_t> sink{0};
    const std::size_t words = bytes / 8, per = (words + threads - 1) / threads;
    const auto t0           = std::chrono::steady_clock::now();
    const auto deadline     = t0 + std::chrono::duration<double>(seconds);
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            const auto* p         = static_cast<const std::uint64_t*>(ptr);
            const std::size_t beg = t * per, end = std::min(words, (t + 1) * per);
            std::uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, done = 0;
            while ((stop ? !stop->load(std::memory_order_relaxed) : true) &&
                   std::chrono::steady_clock::now() < deadline) {
                std::size_t i = beg;
                for (; i + 4 <= end; i += 4) {
                    a0 ^= p[i]; a1 ^= p[i + 1]; a2 ^= p[i + 2]; a3 ^= p[i + 3];
                }
                for (; i < end; ++i) { a0 ^= p[i]; }
                done += (end - beg) * 8;
            }
            total += done;
            sink ^= a0 ^ a1 ^ a2 ^ a3;
        });
    }
    for (auto& th : pool) { th.join(); }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (sink.load() == 0x123) { std::printf(" "); }
    return double(total.load()) / s / 1e9;
}

// ---- report helpers -----------------------------------------------------------------------------

int attr(cudaDeviceAttr a, int dev) {
    int v = -1;
    if (cudaDeviceGetAttribute(&v, a, dev) != cudaSuccess) {
        (void)cudaGetLastError();
        return -1;
    }
    return v;
}

struct VmmBuffer {
    CUdeviceptr ptr = 0;
    std::size_t size = 0;
    CUmemGenericAllocationHandle handle = 0;
    bool compression_granted = false;
    std::string error;
};

VmmBuffer vmm_alloc(const Driver& d, std::size_t bytes, bool request_compression) {
    VmmBuffer b;
    CUmemAllocationProp prop{};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = 0;
    if (request_compression) { prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC; }
    std::size_t gran = 0;
    if (d.memGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) !=
            CUDA_SUCCESS ||
        gran == 0) {
        b.error = "cuMemGetAllocationGranularity failed";
        return b;
    }
    b.size = (bytes + gran - 1) / gran * gran;
    if (d.memCreate(&b.handle, b.size, &prop, 0) != CUDA_SUCCESS) {
        b.error = "cuMemCreate failed";
        return b;
    }
    CUmemAllocationProp granted{};
    if (d.memGetAllocationPropertiesFromHandle(&granted, b.handle) == CUDA_SUCCESS) {
        b.compression_granted = granted.allocFlags.compressionType == CU_MEM_ALLOCATION_COMP_GENERIC;
    }
    CUmemAccessDesc access{};
    access.location = prop.location;
    access.flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (d.memAddressReserve(&b.ptr, b.size, gran, 0, 0) != CUDA_SUCCESS ||
        d.memMap(b.ptr, b.size, 0, b.handle, 0) != CUDA_SUCCESS ||
        d.memSetAccess(b.ptr, b.size, &access, 1) != CUDA_SUCCESS) {
        b.error = "reserve/map/set-access failed";
    }
    return b;
}

std::vector<unsigned char> read_file(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot open sample %s\n", path.c_str());
        std::exit(2);
    }
    std::vector<unsigned char> data;
    unsigned char chunk[1 << 16];
    for (std::size_t got; (got = std::fread(chunk, 1, sizeof(chunk), f)) > 0;) {
        data.insert(data.end(), chunk, chunk + got);
    }
    std::fclose(f);
    data.resize(data.size() / sizeof(uint4) * sizeof(uint4));
    if (data.empty()) {
        std::fprintf(stderr, "sample %s holds fewer than 16 bytes\n", path.c_str());
        std::exit(2);
    }
    return data;
}

void vmm_free(const Driver& d, VmmBuffer& b) {
    if (b.ptr) {
        d.memUnmap(b.ptr, b.size);
        d.memAddressFree(b.ptr, b.size);
    }
    if (b.handle) { d.memRelease(b.handle); }
    b = {};
}

} // namespace

int main(int argc, char** argv) {
    const Options opt = parse(argc, argv);
    const int dev     = 0;
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaFree(nullptr)); // create the primary context for the driver calls below

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int driver_version = 0, runtime_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));

    Driver drv;
    const bool have_driver = load_driver(drv);
    int generic_compression = -1;
    if (have_driver) {
        drv.deviceGetAttribute(&generic_compression,
                               CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, dev);
    }

    Gpu g;
    g.sms = attr(cudaDevAttrMultiProcessorCount, dev);
    CUDA_CHECK(cudaStreamCreateWithFlags(&g.stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(&g.sink, sizeof(unsigned) * g.sms * 8));

    const double gib = double(opt.bytes) / double(std::size_t{1} << 30);
    std::printf("## GB10 memory probe\n\n");
    std::printf("- Device: %s, compute capability %d.%d, %d SMs, L2 %.1f MiB, bus width %d bits\n",
                prop.name, attr(cudaDevAttrComputeCapabilityMajor, dev),
                attr(cudaDevAttrComputeCapabilityMinor, dev), g.sms,
                attr(cudaDevAttrL2CacheSize, dev) / 1048576.0,
                attr(cudaDevAttrGlobalMemoryBusWidth, dev));
    std::printf("- CUDA driver %d, runtime %d; buffer %.2f GiB; %u CPU threads; %.2f s per "
                "measurement\n\n",
                driver_version, runtime_version, gib, opt.cpu_threads, opt.seconds);

    std::printf("### 1. What CUDA reports\n\n| Attribute | Value |\n|---|---:|\n");
    const std::pair<const char*, cudaDeviceAttr> attrs[] = {
        {"integrated", cudaDevAttrIntegrated},
        {"pageableMemoryAccess", cudaDevAttrPageableMemoryAccess},
        {"pageableMemoryAccessUsesHostPageTables", cudaDevAttrPageableMemoryAccessUsesHostPageTables},
        {"concurrentManagedAccess", cudaDevAttrConcurrentManagedAccess},
        {"directManagedMemAccessFromHost", cudaDevAttrDirectManagedMemAccessFromHost},
        {"hostNativeAtomicSupported", cudaDevAttrHostNativeAtomicSupported},
        {"canUseHostPointerForRegisteredMem", cudaDevAttrCanUseHostPointerForRegisteredMem},
        {"unifiedAddressing", cudaDevAttrUnifiedAddressing},
        {"memoryPoolsSupported", cudaDevAttrMemoryPoolsSupported},
    };
    for (const auto& [name, a] : attrs) { std::printf("| %s | %d |\n", name, attr(a, dev)); }
    std::printf("| GENERIC_COMPRESSION_SUPPORTED (driver API) | %s |\n\n",
                have_driver ? std::to_string(generic_compression).c_str()
                            : "driver entry points unavailable");

    // 2. Compression.
    std::printf("### 2. Generic compression (cuMemCreate)\n\n");
    if (!have_driver) {
        std::printf("Skipped: driver entry points unavailable.\n\n");
    } else {
        std::printf("| Allocation | Compression granted | Read GB/s, zero-filled | Read GB/s, "
                    "random |\n|---|---|---:|---:|\n");
        for (const bool want : {false, true}) {
            VmmBuffer b = vmm_alloc(drv, opt.bytes, want);
            const char* label = want ? "compression requested" : "plain";
            if (!b.error.empty()) {
                std::printf("| %s | — | %s | — |\n", label, b.error.c_str());
                vmm_free(drv, b);
                continue;
            }
            void* p = reinterpret_cast<void*>(b.ptr);
            gpu_fill(g, p, b.size, false);
            const double zero = gpu_read_gbps(g, p, b.size, opt.seconds);
            gpu_fill(g, p, b.size, true);
            const double rnd = gpu_read_gbps(g, p, b.size, opt.seconds);
            std::printf("| %s | %s | %.1f | %.1f |\n", label, b.compression_granted ? "yes" : "no",
                        zero, rnd);
            vmm_free(drv, b);
        }
        std::printf("\nCompression is doing something only if the granted, zero-filled row reads "
                    "clearly faster than the plain rows.\n\n");
    }

    // 2b. Compression on real weight bytes.
    if (have_driver && !opt.samples.empty()) {
        std::printf("### 2b. Generic compression on weight samples\n\n"
                    "Each sample is the tensor's leading bytes, repeated to fill the %.2f GiB "
                    "buffer.\n\n| Sample | Sample MiB | Plain GB/s | Compressed GB/s | Gain |\n"
                    "|---|---:|---:|---:|---:|\n",
                    gib);
        for (const auto& [label, path] : opt.samples) {
            const std::vector<unsigned char> host = read_file(path);
            void* staged                          = nullptr;
            CUDA_CHECK(cudaMalloc(&staged, host.size()));
            CUDA_CHECK(cudaMemcpy(staged, host.data(), host.size(), cudaMemcpyHostToDevice));
            double gbps[2] = {0, 0};
            std::string error;
            for (const bool want : {false, true}) {
                VmmBuffer b = vmm_alloc(drv, opt.bytes, want);
                if (!b.error.empty() || (want && !b.compression_granted)) {
                    error = b.error.empty() ? "compression not granted" : b.error;
                    vmm_free(drv, b);
                    break;
                }
                tile_kernel<<<g.sms * 8, 256, 0, g.stream>>>(
                    reinterpret_cast<uint4*>(b.ptr), b.size / sizeof(uint4),
                    static_cast<const uint4*>(staged), host.size() / sizeof(uint4));
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaStreamSynchronize(g.stream));
                gbps[want] = gpu_read_gbps(g, reinterpret_cast<void*>(b.ptr), b.size, opt.seconds);
                vmm_free(drv, b);
            }
            CUDA_CHECK(cudaFree(staged));
            if (!error.empty()) {
                std::printf("| %s | %.1f | %s | — | — |\n", label.c_str(),
                            host.size() / 1048576.0, error.c_str());
            } else {
                std::printf("| %s | %.1f | %.1f | %.1f | %.2fx |\n", label.c_str(),
                            host.size() / 1048576.0, gbps[0], gbps[1], gbps[1] / gbps[0]);
            }
        }
        std::printf("\n");
    }

    // 3. GPU reads of each memory kind, and copies.
    std::printf("### 3. GPU read bandwidth by memory kind\n\n| Source | GB/s |\n|---|---:|\n");
    void* dev_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&dev_buf, opt.bytes));
    gpu_fill(g, dev_buf, opt.bytes, true);
    std::printf("| cudaMalloc | %.1f |\n", gpu_read_gbps(g, dev_buf, opt.bytes, opt.seconds));

    void* pinned = nullptr;
    CUDA_CHECK(cudaHostAlloc(&pinned, opt.bytes, cudaHostAllocMapped));
    cpu_fill_random(pinned, opt.bytes, opt.cpu_threads);
    std::printf("| cudaHostAlloc (pinned, mapped), read in place | %.1f |\n",
                gpu_read_gbps(g, pinned, opt.bytes, opt.seconds));

    void* pageable = std::malloc(opt.bytes);
    cpu_fill_random(pageable, opt.bytes, opt.cpu_threads);
    if (attr(cudaDevAttrPageableMemoryAccess, dev) == 1) {
        std::printf("| malloc (pageable), read in place | %.1f |\n",
                    gpu_read_gbps(g, pageable, opt.bytes, opt.seconds));
    } else {
        std::printf("| malloc (pageable), read in place | not supported |\n");
    }

    const auto copy_gbps = [&](const void* src) {
        cudaEvent_t a, b;
        CUDA_CHECK(cudaEventCreate(&a));
        CUDA_CHECK(cudaEventCreate(&b));
        CUDA_CHECK(cudaMemcpyAsync(dev_buf, src, opt.bytes, cudaMemcpyHostToDevice, g.stream));
        CUDA_CHECK(cudaEventRecord(a, g.stream));
        CUDA_CHECK(cudaMemcpyAsync(dev_buf, src, opt.bytes, cudaMemcpyHostToDevice, g.stream));
        CUDA_CHECK(cudaEventRecord(b, g.stream));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        CUDA_CHECK(cudaEventDestroy(a));
        CUDA_CHECK(cudaEventDestroy(b));
        return double(opt.bytes) / (ms * 1e-3) / 1e9;
    };
    std::printf("| cudaMemcpy H2D from pinned (payload; DRAM traffic is ~2x) | %.1f |\n",
                copy_gbps(pinned));
    std::printf("| cudaMemcpy H2D from pageable (payload) | %.1f |\n\n", copy_gbps(pageable));

    // 4. CPU bandwidth and contention.
    std::printf("### 4. CPU bandwidth and CPU/GPU contention\n\n| Measurement | GB/s |\n|---|---:|\n");
    const double cpu_alone = cpu_read_gbps(pageable, opt.bytes, opt.cpu_threads, opt.seconds * 2);
    const double gpu_alone = gpu_read_gbps(g, dev_buf, opt.bytes, opt.seconds * 2);
    std::printf("| CPU read alone (%u threads) | %.1f |\n", opt.cpu_threads, cpu_alone);
    std::printf("| GPU read alone | %.1f |\n", gpu_alone);

    std::atomic<bool> stop{false};
    double cpu_shared = 0;
    std::thread cpu_side([&] {
        cpu_shared = cpu_read_gbps(pageable, opt.bytes, opt.cpu_threads, 3600, &stop);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const double gpu_shared = gpu_read_gbps(g, dev_buf, opt.bytes, opt.seconds * 2);
    stop = true;
    cpu_side.join();
    std::printf("| GPU read while CPU reads | %.1f |\n", gpu_shared);
    std::printf("| CPU read while GPU reads (whole window, includes 100 ms CPU-only lead-in) | "
                "%.1f |\n",
                cpu_shared);
    std::printf("| Sum while contending | %.1f |\n\n", gpu_shared + cpu_shared);

    std::free(pageable);
    CUDA_CHECK(cudaFreeHost(pinned));
    CUDA_CHECK(cudaFree(dev_buf));
    CUDA_CHECK(cudaFree(g.sink));
    CUDA_CHECK(cudaStreamDestroy(g.stream));
    return 0;
}
