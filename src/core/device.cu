#include "core/device.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer {
namespace {

struct SyncSchedule {
    std::string_view name;
    unsigned int flags;
};

constexpr SyncSchedule kSyncSchedules[] = {
    {"spin", cudaDeviceScheduleSpin},
    {"blocking", cudaDeviceScheduleBlockingSync},
    {"yield", cudaDeviceScheduleYield},
    {"auto", cudaDeviceScheduleAuto},
};

unsigned int sync_schedule_from_environment() {
    const char* value = std::getenv("NINFER_CUDA_SYNC");
    if (value == nullptr) { return cudaDeviceScheduleSpin; }
    for (const auto& schedule : kSyncSchedules) {
        if (schedule.name == value) { return schedule.flags; }
    }
    throw std::invalid_argument("NINFER_CUDA_SYNC must be spin, blocking, yield, or auto");
}

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

void destroy_stream(cudaStream_t& stream) noexcept {
    if (stream != nullptr) {
        log_cuda_error("cudaStreamDestroy", cudaStreamDestroy(stream));
        stream = nullptr;
    }
}

void destroy_event(cudaEvent_t& event) noexcept {
    if (event != nullptr) {
        log_cuda_error("cudaEventDestroy", cudaEventDestroy(event));
        event = nullptr;
    }
}

// DIAGNOSTIC (NINFER_CTX_READY_TRACE=1): report every poll of a labelled completion event with
// the raw cudaEventQuery code, before ready() maps it to a bool. The context frame's completion
// event is the one labelled call site (qwen3_5/program/lifecycle.cpp); every other completion
// event stays silent. Unset, neither ready() nor record() takes a diagnostic path.
bool ctx_ready_trace_enabled() {
    static const bool enabled = std::getenv("NINFER_CTX_READY_TRACE") != nullptr;
    return enabled;
}

constexpr unsigned long long kCtxReadyTracePolls = 20;

struct CtxReadyTraceCounters {
    std::atomic<unsigned long long> polls{0};
    std::atomic<int> last{static_cast<int>(cudaSuccess)};
    std::atomic<bool> summary{false};
};

// The local two-GPU lane's rank is its device index, and that lane's device list is {0, 1}; one
// counter set per rank. A rank outside that pair cannot occur on the lane this diagnostic targets
// and shares slot 0, which is preferable to an out-of-bounds read in a diagnostic.
CtxReadyTraceCounters& ctx_ready_trace_counters(int rank) {
    static std::array<CtxReadyTraceCounters, 2> counters{};
    const std::size_t slot = rank >= 0 && rank < 2 ? static_cast<std::size_t>(rank) : 0;
    return counters[slot];
}

int current_cuda_device_or_negative() {
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess) { current = -1; }
    return current;
}

void print_ctx_ready_poll(int rank, unsigned long long poll, cudaEvent_t event, cudaStream_t stream,
                          int created_device, int recorded_device, int current_device,
                          cudaError_t cleared, cudaError_t status) {
    std::fprintf(stderr,
                 "[ctx-ready] rank=%d poll=%llu event=%p stream=%p created-device=%d "
                 "recorded-device=%d current-device=%d cleared=%d(%s) query=%d(%s)\n",
                 rank, poll, static_cast<const void*>(event), static_cast<const void*>(stream),
                 created_device, recorded_device, current_device, static_cast<int>(cleared),
                 cudaGetErrorName(cleared), static_cast<int>(status), cudaGetErrorName(status));
}

void print_ctx_ready_summary(int rank, unsigned long long polls, cudaError_t status) {
    std::fprintf(stderr, "[ctx-ready] rank=%d summary polls=%llu last=%d(%s)\n", rank, polls,
                 static_cast<int>(status), cudaGetErrorName(status));
}

} // namespace

DeviceAllocationError::DeviceAllocationError(const char* operation, std::size_t requested_bytes)
    : std::runtime_error(std::string(operation) + ": insufficient Device memory"),
      requested_bytes_(requested_bytes) {}

void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) { return; }
    std::fprintf(stderr, "%s:%d: CUDA_CHECK(%s) failed: %s: %s\n", file, line, expr,
                 cudaGetErrorName(err), cudaGetErrorString(err));
    std::abort();
}

DeviceContext::DeviceContext(int device_id) : device_(Device::cuda(device_id)) {
    const unsigned int sync_flags = sync_schedule_from_environment();
    int count                     = 0;
    cudaError_t err               = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceCount failed", err));
    }
    if (count <= 0) { throw std::runtime_error("no CUDA devices available"); }
    if (device_id >= count) { throw std::invalid_argument("invalid CUDA device id"); }

    bind_to_current_thread();

    err = cudaSetDeviceFlags(sync_flags);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDeviceFlags failed", err));
    }

    err = cudaGetDeviceProperties(&props_, device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceProperties failed", err));
    }

    cudaStream_t compute = nullptr;
    cudaStream_t load    = nullptr;
    err                  = cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(stream) failed", err));
    }

    err = cudaStreamCreateWithFlags(&load, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        destroy_stream(compute);
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(transfer_stream) failed", err));
    }

    stream_          = compute;
    transfer_stream_ = load;
}

DeviceContext::~DeviceContext() {
    if (stream_ != nullptr || transfer_stream_ != nullptr) { bind_to_current_thread_noexcept(); }
    destroy_stream(transfer_stream_);
    destroy_stream(stream_);
}

DeviceContext::DeviceContext(DeviceContext&& other) noexcept
    : device_(other.device_), stream_(other.stream_), transfer_stream_(other.transfer_stream_),
      props_(other.props_) {
    other.stream_          = nullptr;
    other.transfer_stream_ = nullptr;
}

DeviceContext& DeviceContext::operator=(DeviceContext&& other) noexcept {
    if (this == &other) { return *this; }

    if (stream_ != nullptr || transfer_stream_ != nullptr) { bind_to_current_thread_noexcept(); }
    destroy_stream(transfer_stream_);
    destroy_stream(stream_);

    device_          = other.device_;
    props_           = other.props_;
    stream_          = other.stream_;
    transfer_stream_ = other.transfer_stream_;

    other.stream_          = nullptr;
    other.transfer_stream_ = nullptr;
    return *this;
}

void DeviceContext::bind_to_current_thread() const {
    const cudaError_t err = cudaSetDevice(device_.index());
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice failed", err));
    }
}

void DeviceContext::bind_to_current_thread_noexcept() const noexcept {
    log_cuda_error("cudaSetDevice", cudaSetDevice(device_.index()));
}

int DeviceContext::compute_capability() const noexcept { return props_.major * 10 + props_.minor; }

int DeviceContext::multiprocessor_count() const noexcept { return props_.multiProcessorCount; }

CudaExecutionView DeviceContext::execution_view() const noexcept {
    return {device(), stream_, multiprocessor_count()};
}

CudaExecutionView DeviceContext::transfer_view() const noexcept {
    return {device(), transfer_stream_, multiprocessor_count()};
}

CudaExecutionView DeviceContext::execution_view(cudaStream_t borrowed_stream) const {
    CudaDeviceScope scope(device());
    int ordinal      = -1;
    const auto error = cudaStreamGetDevice(borrowed_stream, &ordinal);
    if (error != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaStreamGetDevice failed", error));
    }
    if (ordinal != device_.index()) {
        throw std::invalid_argument("stream belongs to another device");
    }
    return {device(), borrowed_stream, multiprocessor_count()};
}

CudaDeviceScope::CudaDeviceScope(Device device) {
    if (device.type() != DeviceType::CUDA) {
        throw std::invalid_argument("CUDA operation requires a CUDA device");
    }
    auto error = cudaGetDevice(&previous_);
    if (error == cudaSuccess && previous_ != device.index()) {
        error    = cudaSetDevice(device.index());
        changed_ = error == cudaSuccess;
    }
    if (error != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("CUDA device selection failed", error));
    }
}

CudaDeviceScope::~CudaDeviceScope() {
    if (changed_) { log_cuda_error("restore CUDA device", cudaSetDevice(previous_)); }
}

CudaStream::CudaStream(const DeviceContext& device) : device_(device.device()) {
    CudaDeviceScope scope(device_);
    const auto error = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (error != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaStreamCreateWithFlags failed", error));
    }
}

CudaStream::~CudaStream() {
    try {
        CudaDeviceScope scope(device_);
        destroy_stream(stream_);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "CUDA stream cleanup failed: %s\n", error.what());
    }
}

std::size_t DeviceContext::total_vram() const noexcept { return props_.totalGlobalMem; }

const char* DeviceContext::sync_mode() const {
    bind_to_current_thread();
    unsigned int flags    = 0;
    const cudaError_t err = cudaGetDeviceFlags(&flags);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceFlags failed", err));
    }
    for (const auto& schedule : kSyncSchedules) {
        if ((flags & cudaDeviceScheduleMask) == schedule.flags) { return schedule.name.data(); }
    }
    throw std::runtime_error("unknown CUDA synchronization schedule");
}

void DeviceContext::synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx) : CudaEventTimer(ctx, ctx.stream()) {}

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream) : stream_(stream) {
    if (stream == nullptr) { throw std::invalid_argument("CUDA timer stream is null"); }
    ctx.bind_to_current_thread();

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    cudaError_t err   = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreate(start) failed", err));
    }

    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        destroy_event(start);
        throw std::runtime_error(cuda_error_message("cudaEventCreate(stop) failed", err));
    }

    start_ = start;
    stop_  = stop;
}

CudaEventTimer::~CudaEventTimer() {
    destroy_event(stop_);
    destroy_event(start_);
}

CudaEventTimer::CudaEventTimer(CudaEventTimer&& other) noexcept
    : stream_(other.stream_), start_(other.start_), stop_(other.stop_) {
    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
}

CudaEventTimer& CudaEventTimer::operator=(CudaEventTimer&& other) noexcept {
    if (this == &other) { return *this; }

    destroy_event(stop_);
    destroy_event(start_);

    stream_ = other.stream_;
    start_  = other.start_;
    stop_   = other.stop_;

    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
    return *this;
}

void CudaEventTimer::start() { CUDA_CHECK(cudaEventRecord(start_, stream_)); }

void CudaEventTimer::record_stop() { CUDA_CHECK(cudaEventRecord(stop_, stream_)); }

float CudaEventTimer::elapsed_ms() const {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
}

float CudaEventTimer::stop_ms() {
    record_stop();
    CUDA_CHECK(cudaEventSynchronize(stop_));
    return elapsed_ms();
}

CudaCompletionEvent::CudaCompletionEvent(const DeviceContext& ctx, const char* trace_label)
    : trace_label_(trace_label), trace_rank_(ctx.device().index()) {
    ctx.bind_to_current_thread();
    if (trace_label_ != nullptr && ctx_ready_trace_enabled()) {
        created_device_ = current_cuda_device_or_negative();
        // A diagnostic query must not leave its own failure for a later check.
        (void)cudaGetLastError();
    }
    const cudaError_t err = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreateWithFlags failed", err));
    }
}

CudaCompletionEvent::~CudaCompletionEvent() { destroy_event(event_); }

CudaCompletionEvent::CudaCompletionEvent(CudaCompletionEvent&& other) noexcept
    : event_(std::exchange(other.event_, nullptr)), trace_label_(other.trace_label_),
      trace_rank_(other.trace_rank_), created_device_(other.created_device_),
      recorded_stream_(other.recorded_stream_), recorded_device_(other.recorded_device_) {
    other.trace_label_     = nullptr;
    other.trace_rank_      = -1;
    other.created_device_  = -1;
    other.recorded_stream_ = nullptr;
    other.recorded_device_ = -1;
}

CudaCompletionEvent& CudaCompletionEvent::operator=(CudaCompletionEvent&& other) noexcept {
    if (this == &other) { return *this; }
    destroy_event(event_);
    event_           = std::exchange(other.event_, nullptr);
    trace_label_     = std::exchange(other.trace_label_, nullptr);
    trace_rank_      = std::exchange(other.trace_rank_, -1);
    created_device_  = std::exchange(other.created_device_, -1);
    recorded_stream_ = std::exchange(other.recorded_stream_, nullptr);
    recorded_device_ = std::exchange(other.recorded_device_, -1);
    return *this;
}

void CudaCompletionEvent::record(cudaStream_t stream) {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not recordable");
    }
    const bool trace    = ctx_ready_trace_enabled() && trace_label_ != nullptr;
    int current         = -1;
    cudaError_t cleared = cudaSuccess;
    if (trace) {
        current = current_cuda_device_or_negative();
        cleared = cudaGetLastError();
    }
    const auto status = cudaEventRecord(event_, stream);
    if (trace) {
        std::fprintf(stderr,
                     "[ctx-ready] rank=%d record event=%p stream=%p created-device=%d "
                     "current-device=%d cleared=%d(%s) status=%d(%s)\n",
                     trace_rank_, static_cast<const void*>(event_),
                     static_cast<const void*>(stream), created_device_, current,
                     static_cast<int>(cleared), cudaGetErrorName(cleared), static_cast<int>(status),
                     cudaGetErrorName(status));
    }
    if (status != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventRecord failed", status));
    }
    recorded_stream_ = stream;
    if (trace) { recorded_device_ = current; }
}

void CudaCompletionEvent::record_external(cudaStream_t stream) {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not recordable");
    }
    CUDA_CHECK(cudaEventRecordWithFlags(event_, stream, cudaEventRecordExternal));
    recorded_stream_ = stream;
    if (ctx_ready_trace_enabled() && trace_label_ != nullptr) {
        recorded_device_ = current_cuda_device_or_negative();
        (void)cudaGetLastError();
    }
}

void CudaCompletionEvent::wait(cudaStream_t stream) const {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not waitable");
    }
    const auto status = cudaStreamWaitEvent(stream, event_, 0);
    if (status != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaStreamWaitEvent failed", status));
    }
}



void CudaCompletionEvent::wait_external(cudaStream_t stream) const {
    if (event_ == nullptr || stream == nullptr)
        throw std::logic_error("CUDA completion event is not waitable");
    const auto status = cudaStreamWaitEvent(stream, event_, cudaEventWaitExternal);
    if (status != cudaSuccess)
        throw std::runtime_error(cuda_error_message("cudaStreamWaitEvent external failed", status));
}

void validate_execution_device_ids(std::span<const int> device_ids) {
    if (device_ids.empty()) {
        throw std::invalid_argument("an execution context needs at least one device");
    }
    for (std::size_t index = 0; index < device_ids.size(); ++index) {
        if (device_ids[index] < 0) {
            throw std::invalid_argument("a device id must not be negative");
        }
        for (std::size_t earlier = 0; earlier < index; ++earlier) {
            if (device_ids[earlier] == device_ids[index]) {
                throw std::invalid_argument(
                    "an execution context must not name the same device twice");
            }
        }
    }
}

ExecutionContext::ExecutionContext(std::vector<int> device_ids) {
    validate_execution_device_ids(device_ids);
    devices_.reserve(device_ids.size());
    for (const int device_id : device_ids) {
        devices_.push_back(std::make_unique<DeviceContext>(device_id));
    }
    const int capability = devices_.front()->compute_capability();
    for (const auto& device : devices_) {
        if (device->compute_capability() != capability) {
            throw std::invalid_argument(
                "every device in an execution context must share one compute capability");
        }
    }
}

DeviceContext& ExecutionContext::device(int rank) {
    return *devices_.at(static_cast<std::size_t>(rank));
}

const DeviceContext& ExecutionContext::device(int rank) const {
    return *devices_.at(static_cast<std::size_t>(rank));
}

bool CudaCompletionEvent::ready() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    const bool trace    = ctx_ready_trace_enabled() && trace_label_ != nullptr;
    int current         = -1;
    cudaError_t cleared = cudaSuccess;
    if (trace) {
        current = current_cuda_device_or_negative();
        // Clear a sticky error first: cudaEventQuery may also return an earlier launch's error,
        // which would otherwise be indistinguishable from this query's own result.
        cleared = cudaGetLastError();
    }
    const cudaError_t status = cudaEventQuery(event_);
    if (trace) {
        auto& counters  = ctx_ready_trace_counters(trace_rank_);
        const auto poll = counters.polls.fetch_add(1, std::memory_order_relaxed) + 1;
        counters.last.store(static_cast<int>(status), std::memory_order_relaxed);
        // The first N polls are the readable part; after that only a non-NotReady result is
        // interesting, and the summary is the one line that survives the throttle.
        if (poll <= kCtxReadyTracePolls || status != cudaErrorNotReady) {
            print_ctx_ready_poll(trace_rank_, poll, event_, recorded_stream_, created_device_,
                                 recorded_device_, current, cleared, status);
        }
        if ((status != cudaErrorNotReady || poll > kCtxReadyTracePolls) &&
            !counters.summary.exchange(true, std::memory_order_relaxed)) {
            print_ctx_ready_summary(trace_rank_, poll, status);
        }
    }
    if (status == cudaSuccess) { return true; }
    if (status == cudaErrorNotReady) { return false; }
    throw std::runtime_error(cuda_error_message("cudaEventQuery failed", status));
}

void CudaCompletionEvent::synchronize() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    const auto status = cudaEventSynchronize(event_);
    if (status != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventSynchronize failed", status));
    }
}

} // namespace ninfer
