// A PER-RANK, PER-ROUND LAUNCH COUNTER, linked with -Wl,--wrap so it sees every launch on every
// path without touching a call site.
//
// WHY IT EXISTS: the two-rank lane fails with the peer's exchange epoch exactly one below this
// rank's target, and the collective audit cannot see it because the audit counts COLLECTIVE LABELS
// while the question is about KERNEL LAUNCHES. This counts launches per device; in this process the
// device ordinal IS the rank, so the counters are per rank by construction.
//
// THE INTERVENTION IS ONE ATOMIC-FREE INCREMENT PER LAUNCH and a printf only at the round
// boundary, so it must not perturb the timing. If the lane behaves differently with it linked, that
// is itself information: it says the failure is marginal in timing.
//
// ARMED CHECK: the boundary hook prints every round, including round 0, so a run that prints
// nothing or prints zeros is a run where the instrument was NOT linked -- check that before reading
// equal counts as a result.
//
// THE LIMITATION, AND IT MUST BE READ BEFORE THE NUMBERS: `launch=0` means the kernel launches in
// libninfer_ops.so ARE NOT COUNTED; only async copies and memsets issued from the executable are.
// -Wl,--wrap rewrites calls made from the executable being linked, and this process launches its
// kernels from inside the shared library, which is not wrapped (and deliberately will not be: a
// shipped library's link line is not a diagnostic's business). An instrument whose zero is a blind
// spot rather than an agreement must say so where it is read.
//
// THE SECOND BLIND SPOT, same boundary, and it must be read the same way: the per-site tags live in
// the model code inside the shared library, and a WEAK undefined symbol referenced from a shared
// library does not bind to a definition in the executable at load -- it stays null. So the site
// column prints nothing until the executable exports its symbols (`-rdynamic` / `--export-dynamic`,
// which the test target now passes). An empty site column means the export did not bind, NOT that
// no tagged site ran. Two documented blind spots in one instrument is an instrument that tells you
// what it cannot see, which is the thing five rounds of silence could not do.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace {

struct Counts {
    std::uint64_t launch = 0;
    std::uint64_t memcpy = 0;
    std::uint64_t memset = 0;
    std::uint64_t graph  = 0;
    std::uint64_t wait   = 0;
    std::uint64_t total() const { return launch + memcpy + memset + graph + wait; }
};

Counts g_counts[2];
Counts g_snapshot[2];
int g_round = 0;
bool g_printed_startup = false;

int device_index() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device > 1) return 0;
    return device;
}

void bump(std::uint64_t Counts::*member) {
    g_counts[device_index()].*member += 1;
    if (!g_printed_startup) {
        g_printed_startup = true;
        std::fprintf(stderr, "[launch-count] armed: the wrap shims are linked and counting\n");
    }
}

} // namespace

extern "C" {

cudaError_t __real_cudaLaunchKernel(const void*, dim3, dim3, void**, std::size_t, cudaStream_t);
cudaError_t __real_cudaMemcpyAsync(void*, const void*, std::size_t, cudaMemcpyKind, cudaStream_t);
cudaError_t __real_cudaMemsetAsync(void*, int, std::size_t, cudaStream_t);
cudaError_t __real_cudaGraphLaunch(cudaGraphExec_t, cudaStream_t);
cudaError_t __real_cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned int);

cudaError_t __wrap_cudaLaunchKernel(const void* func, dim3 grid, dim3 block, void** args,
                                    std::size_t shared, cudaStream_t stream) {
    bump(&Counts::launch);
    return __real_cudaLaunchKernel(func, grid, block, args, shared, stream);
}

cudaError_t __wrap_cudaMemcpyAsync(void* dst, const void* src, std::size_t bytes,
                                   cudaMemcpyKind kind, cudaStream_t stream) {
    bump(&Counts::memcpy);
    return __real_cudaMemcpyAsync(dst, src, bytes, kind, stream);
}

cudaError_t __wrap_cudaMemsetAsync(void* dst, int value, std::size_t bytes,
                                   cudaStream_t stream) {
    bump(&Counts::memset);
    return __real_cudaMemsetAsync(dst, value, bytes, stream);
}

cudaError_t __wrap_cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream) {
    bump(&Counts::graph);
    return __real_cudaGraphLaunch(exec, stream);
}

cudaError_t __wrap_cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    bump(&Counts::wait);
    return __real_cudaStreamWaitEvent(stream, event, flags);
}

// PER-SITE TAGGING. The API counters above cannot distinguish six sites running once from one
// site running six times, and that distinction is the question. Sites tag themselves by calling
// ninfer_launch_site("name") immediately before their cudaMemcpyAsync; the weak declaration is
// resolved to null in production.
static const char* g_site_names[32];
static std::uint64_t g_site_counts[32][2];
static int g_site_count = 0;

static int site_index(const char* site) {
    for (int i = 0; i < g_site_count; ++i)
        if (g_site_names[i] == site) return i;
    if (g_site_count == 32) return 0;
    g_site_names[g_site_count] = site;
    return g_site_count++;
}

extern "C" void ninfer_launch_site(const char* site) {
    g_site_counts[site_index(site)][device_index()] += 1;
}

// Called by LocalExecutor at each round release. Prints the DELTA per rank for this round, because
// a cumulative count says nothing after round 1.
void ninfer_launch_round_boundary(void) {
    for (int rank = 0; rank < 2; ++rank) {
        const Counts delta{g_counts[rank].launch - g_snapshot[rank].launch,
                           g_counts[rank].memcpy - g_snapshot[rank].memcpy,
                           g_counts[rank].memset - g_snapshot[rank].memset,
                           g_counts[rank].graph - g_snapshot[rank].graph,
                           g_counts[rank].wait - g_snapshot[rank].wait};
        std::fprintf(stderr,
                     "[launch-count] round=%d rank=%d launch=%llu memcpy=%llu memset=%llu "
                     "graph=%llu wait=%llu total=%llu\n",
                     g_round, rank, static_cast<unsigned long long>(delta.launch),
                     static_cast<unsigned long long>(delta.memcpy),
                     static_cast<unsigned long long>(delta.memset),
                     static_cast<unsigned long long>(delta.graph),
                     static_cast<unsigned long long>(delta.wait),
                     static_cast<unsigned long long>(delta.total()));
        g_snapshot[rank] = g_counts[rank];
    }
    if (g_counts[0].total() + g_counts[1].total() == 0) {
        std::fprintf(stderr,
                     "[launch-count] ZERO AFTER A ROUND: the instrument is not counting; do not read "
                     "equal counts as a result\n");
    }
    for (int i = 0; i < g_site_count; ++i) {
        std::fprintf(stderr, "[launch-count] round=%d site=%s rank0=%llu rank1=%llu\n", g_round,
                     g_site_names[i], static_cast<unsigned long long>(g_site_counts[i][0]),
                     static_cast<unsigned long long>(g_site_counts[i][1]));
    }
    for (int i = 0; i < g_site_count; ++i) g_site_counts[i][0] = g_site_counts[i][1] = 0;
    ++g_round;
}

} // extern "C"
