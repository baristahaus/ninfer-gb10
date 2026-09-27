// GB10 host/GPU round-boundary probe.
//
// Separates the two host costs that fill the MTP decode round boundary on GB10
// (docs/maintainer/plan-2026-09-gb10.md, step 6):
//
// 1. Completion lag: how long after a kernel finishes does cudaStreamSynchronize return, under
//    the CUDA device schedule given by --schedule (the engine sets blocking; src/core/device.cu)?
//    Kernels busy-wait for a set duration so the host thread sleeps as long as it does in a
//    decode round. The kernel's end time comes from %globaltimer, mapped to the host clock by
//    a polled calibration that never calls a CUDA synchronize.
// 2. Graph launch: for chains of N empty kernels, the cudaGraphLaunch call duration and the time
//    from the call's start to the first kernel's start.
//
// The schedule is fixed per process (cudaSetDeviceFlags before the context exists), so
// probe_sync.sh runs one process per schedule. Prints Markdown.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_121a tools/gb10/sync_probe.cu -o sync_probe

#include <cuda_runtime.h>

#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
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

std::int64_t host_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

__device__ std::uint64_t global_ns() {
    std::uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Writes the GPU time and then a sequence number the host polls for.
__global__ void stamp_kernel(volatile std::uint64_t* out, std::uint64_t seq) {
    out[0] = global_ns();
    __threadfence_system();
    out[1] = seq;
}

// Busy-waits for `ns`, then records its end time.
__global__ void spin_kernel(std::uint64_t ns, std::uint64_t* end) {
    const std::uint64_t start = global_ns();
    while (global_ns() - start < ns) {}
    *end = global_ns();
}

__global__ void empty_kernel() {}

__global__ void first_node_kernel(std::uint64_t* start) { *start = global_ns(); }

struct Stats {
    double p50, p90, max;
};

Stats stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const auto at = [&](double q) { return v[std::min(v.size() - 1, std::size_t(q * v.size()))]; };
    return {at(0.5), at(0.9), v.back()};
}

// Host-minus-GPU clock offset from the calibration sample with the smallest round trip.
std::int64_t calibrate(cudaStream_t stream, std::uint64_t* device_view,
                       volatile std::uint64_t* host_view) {
    std::int64_t best_rtt = INT64_MAX, offset = 0;
    for (std::uint64_t seq = 1; seq <= 200; ++seq) {
        const std::int64_t t0 = host_ns();
        stamp_kernel<<<1, 1, 0, stream>>>(device_view, seq);
        while (host_view[1] != seq) {}
        const std::int64_t t1 = host_ns();
        if (t1 - t0 < best_rtt) {
            best_rtt = t1 - t0;
            offset   = (t0 + t1) / 2 - static_cast<std::int64_t>(host_view[0]);
        }
    }
    return offset;
}

std::string read_line(const std::string& path) {
    std::ifstream f(path);
    std::string s;
    std::getline(f, s);
    return s.empty() ? "?" : s;
}

} // namespace

int main(int argc, char** argv) {
    std::string schedule = "blocking";
    int iterations       = 30;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--schedule" && i + 1 < argc) {
            schedule = argv[++i];
        } else if (a == "--iterations" && i + 1 < argc) {
            iterations = std::max(5, std::atoi(argv[++i]));
        } else {
            std::fprintf(stderr, "usage: sync_probe [--schedule auto|spin|yield|blocking] "
                                 "[--iterations N (default 30)]\n");
            return a == "--help" ? 0 : 2;
        }
    }
    unsigned flag = cudaDeviceScheduleBlockingSync;
    if (schedule == "auto") {
        flag = cudaDeviceScheduleAuto;
    } else if (schedule == "spin") {
        flag = cudaDeviceScheduleSpin;
    } else if (schedule == "yield") {
        flag = cudaDeviceScheduleYield;
    } else if (schedule != "blocking") {
        std::fprintf(stderr, "unknown schedule %s\n", schedule.c_str());
        return 2;
    }
    CUDA_CHECK(cudaSetDeviceFlags(flag | cudaDeviceMapHost));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::uint64_t* mapped_host = nullptr;
    CUDA_CHECK(cudaHostAlloc(&mapped_host, 2 * sizeof(std::uint64_t), cudaHostAllocMapped));
    std::uint64_t* mapped_dev = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&mapped_dev, mapped_host, 0));
    mapped_host[1] = 0;
    std::uint64_t* stamps = nullptr; // [0] spin end, [1] first graph node start
    CUDA_CHECK(cudaMallocHost(&stamps, 2 * sizeof(std::uint64_t)));
    std::uint64_t* dev_stamps = nullptr;
    CUDA_CHECK(cudaMalloc(&dev_stamps, 2 * sizeof(std::uint64_t)));

    const int cpu = sched_getcpu();
    const std::string cpu_dir = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
    std::printf("### Schedule `%s`\n\n", schedule.c_str());
    std::printf("Host thread on CPU %d (max %s kHz, idle governor %s); %d iterations per row.\n\n",
                cpu, read_line(cpu_dir + "/cpufreq/cpuinfo_max_freq").c_str(),
                read_line("/sys/devices/system/cpu/cpuidle/current_governor_ro").c_str(),
                iterations);

    // 1. Completion lag.
    std::printf("| Kernel duration | Sync return lag p50 | p90 | max |\n|---|---:|---:|---:|\n");
    for (const std::uint64_t us : {50ull, 1000ull, 10000ull, 60000ull}) {
        std::vector<double> lag;
        for (int it = 0; it < iterations; ++it) {
            const std::int64_t offset = calibrate(stream, mapped_dev, mapped_host);
            spin_kernel<<<1, 1, 0, stream>>>(us * 1000, dev_stamps);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const std::int64_t returned = host_ns();
            CUDA_CHECK(cudaMemcpy(stamps, dev_stamps, sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
            lag.push_back(double(returned - (static_cast<std::int64_t>(stamps[0]) + offset)) / 1e3);
        }
        const Stats s = stats(lag);
        std::printf("| %llu µs | %.1f µs | %.1f µs | %.1f µs |\n",
                    static_cast<unsigned long long>(us), s.p50, s.p90, s.max);
    }

    // 2. Graph launch cost against node count.
    std::printf("\n| Graph nodes | cudaGraphLaunch call p50 | Call start → first kernel p50 | "
                "max |\n|---:|---:|---:|---:|\n");
    for (const int nodes : {100, 1000, 4000}) {
        cudaGraph_t graph;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        first_node_kernel<<<1, 1, 0, stream>>>(dev_stamps + 1);
        for (int n = 1; n < nodes; ++n) { empty_kernel<<<1, 32, 0, stream>>>(); }
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        cudaGraphExec_t exec;
        CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
        CUDA_CHECK(cudaGraphUpload(exec, stream));
        CUDA_CHECK(cudaGraphLaunch(exec, stream)); // warm
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<double> call, first;
        for (int it = 0; it < iterations; ++it) {
            const std::int64_t offset = calibrate(stream, mapped_dev, mapped_host);
            const std::int64_t t0 = host_ns();
            CUDA_CHECK(cudaGraphLaunch(exec, stream));
            const std::int64_t t1 = host_ns();
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaMemcpy(stamps + 1, dev_stamps + 1, sizeof(std::uint64_t),
                                  cudaMemcpyDeviceToHost));
            call.push_back(double(t1 - t0) / 1e3);
            first.push_back(double(static_cast<std::int64_t>(stamps[1]) + offset - t0) / 1e3);
        }
        const Stats c = stats(call), f = stats(first);
        std::printf("| %d | %.1f µs | %.1f µs | %.1f µs |\n", nodes, c.p50, f.p50, f.max);
        CUDA_CHECK(cudaGraphExecDestroy(exec));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    std::printf("\n");

    CUDA_CHECK(cudaFree(dev_stamps));
    CUDA_CHECK(cudaFreeHost(stamps));
    CUDA_CHECK(cudaFreeHost(mapped_host));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
