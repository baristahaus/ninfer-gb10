#include "models/common/context/transfer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <limits>
#include <optional>
#include <stdexcept>
#include <thread>
#include <type_traits>

namespace ninfer::models::context {
namespace {
std::size_t mapping_bytes(const ContextStorage& storage, std::size_t requested) {
    const auto elements = requested ? requested : storage.mapping_capacity_elements();
    if (elements > SIZE_MAX / sizeof(std::int32_t))
        throw std::overflow_error("context mapping staging size overflow");
    return elements * sizeof(std::int32_t);
}

// DIAGNOSTIC (NINFER_CTX_SEGMENTS=1): name every transfer segment a context migration enqueues,
// before it is executed. A segment is the enqueue unit the migration's own bookkeeping already
// defines (same resource class and direction, contiguous actions), so this prints the work the
// transfer stream is about to carry without reading device memory or waiting for completion.
// Unset, the enqueue path below is unchanged.
bool segments_trace_enabled() {
    static const bool enabled = std::getenv("NINFER_CTX_SEGMENTS") != nullptr;
    return enabled;
}

// DIAGNOSTIC (NINFER_STATE_CLEAR_TRACE=1): report the fence a StateClear action inherits on the
// transfer stream. The clear itself is traced where the memset is enqueued (core/state_image.cpp),
// which cannot see this event. A bounded poll separates a record the device has not processed yet
// from one that never signals; the clear's own trace then reads the same stream before and after
// its enqueue.
bool state_clear_trace_enabled() {
    static const bool enabled = std::getenv("NINFER_STATE_CLEAR_TRACE") != nullptr;
    return enabled;
}

struct EventObservation {
    const char* state = "none";
    long wait_us      = 0;
    std::string error;
};

EventObservation observe_event(const CudaCompletionEvent& event, bool fenced) {
    EventObservation result;
    if (!fenced) { return result; }
    const auto started   = std::chrono::steady_clock::now();
    constexpr auto limit = std::chrono::milliseconds(20);
    for (;;) {
        try {
            if (event.ready()) {
                result.state = "ready";
                break;
            }
        } catch (const std::exception& failure) {
            result.state = "error";
            result.error = failure.what();
            break;
        }
        if (std::chrono::steady_clock::now() - started >= limit) {
            result.state = "pending";
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    result.wait_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::steady_clock::now() - started)
                         .count();
    // cudaEventQuery leaves cudaErrorNotReady as the last error; do not hand it to a later check.
    (void)cudaGetLastError();
    return result;
}

const char* stream_query_name(cudaError_t status) {
    if (status == cudaSuccess) return "idle";
    if (status == cudaErrorNotReady) return "busy";
    return cudaGetErrorName(status);
}

const char* resource_class_name(runtime::ContextResourceClass value) {
    switch (value) {
    case runtime::ContextResourceClass::State: return "state";
    case runtime::ContextResourceClass::MainKV: return "main-kv";
    case runtime::ContextResourceClass::BackendKV: return "backend-kv";
    }
    return "unknown";
}

const char* transfer_direction_name(runtime::ContextTransferDirection value) {
    switch (value) {
    case runtime::ContextTransferDirection::DeviceToHost: return "d2h";
    case runtime::ContextTransferDirection::HostToDevice: return "h2d";
    case runtime::ContextTransferDirection::DeviceToDevice: return "d2d";
    case runtime::ContextTransferDirection::HostToSpill: return "host-to-spill";
    case runtime::ContextTransferDirection::SpillToHost: return "spill-to-host";
    }
    return "unknown";
}

const char* state_direction_name(StateCopyDirection value) {
    return value == StateCopyDirection::DeviceToHost ? "d2h" : "h2d";
}

void append_page_slot(std::string& out, const char* label, const PageSlot& slot) {
    out += label;
    out += "=pool";
    out += std::to_string(slot.pool_id);
    out += ":page";
    out += std::to_string(slot.index);
    out += ":gen";
    out += std::to_string(slot.generation);
}

void append_host_extent(std::string& out, const HostExtent& host) {
    out += " host-pool";
    out += std::to_string(host.pool_id);
    out += ":off";
    out += std::to_string(host.offset);
    out += ":bytes";
    out += std::to_string(host.bytes);
}

// One action's identity: the slot, page or host extent it reads and writes. Kept compact because a
// segment may hold several actions and every identity stays on the segment's one line.
std::string describe_action(const ContextAction& action) {
    std::string out;
    std::visit(
        [&](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, StateClear>) {
                out = "state-clear slot=" + std::to_string(value.slot);
            } else if constexpr (std::is_same_v<T, StateCopy>) {
                out = "state-copy src-slot=" + std::to_string(value.source) +
                      " dst-slot=" + std::to_string(value.destination) +
                      " offset=" + std::to_string(value.offset) +
                      (value.bytes ? " bytes=" + std::to_string(value.bytes)
                                   : std::string(" bytes=whole-image"));
            } else if constexpr (std::is_same_v<T, StateTransferDescription>) {
                out = "state-transfer ";
                out += state_direction_name(value.direction);
                out += " src-slot=";
                out += std::to_string(value.slots.source);
                out += " dst-slot=";
                out += std::to_string(value.slots.destination);
                out += " bytes=";
                out += std::to_string(value.bytes);
                append_host_extent(out, value.host);
            } else if constexpr (std::is_same_v<T, PageCopy>) {
                out = "page-copy";
                append_page_slot(out, " src", value.source);
                append_page_slot(out, " dst", value.destination);
            } else if constexpr (std::is_same_v<T, PageTransfer>) {
                out = "page-transfer ";
                out += state_direction_name(value.direction);
                out += " pages=";
                out += std::to_string(value.pages.size());
                if (!value.pages.empty()) {
                    append_page_slot(out, " first", value.pages.front());
                    append_page_slot(out, " last", value.pages.back());
                }
                append_host_extent(out, value.host);
            } else if constexpr (std::is_same_v<T, MappingUpdate>) {
                out = "mapping row-pool" + std::to_string(value.row.pool_id) + ":row" +
                      std::to_string(value.row.index) + ":gen" +
                      std::to_string(value.row.generation) + " logical-begin=" +
                      std::to_string(value.logical_begin) + " pages=" +
                      std::to_string(value.pages.size());
                if (!value.pages.empty()) {
                    append_page_slot(out, " first", value.pages.front());
                    append_page_slot(out, " last", value.pages.back());
                }
            } else if constexpr (std::is_same_v<T, StateSpillOut>) {
                out = "spill-out spill-off=" + std::to_string(value.spill.offset) +
                      " spill-bytes=" + std::to_string(value.spill.bytes) + " spill-gen=" +
                      std::to_string(value.spill.generation);
                append_host_extent(out, value.host);
            } else if constexpr (std::is_same_v<T, StateSpillIn>) {
                out = "spill-in spill-off=" + std::to_string(value.spill.offset) +
                      " spill-bytes=" + std::to_string(value.spill.bytes) + " spill-gen=" +
                      std::to_string(value.spill.generation);
                append_host_extent(out, value.host);
            }
        },
        action);
    return out;
}

const char* action_kind_name(const ContextAction& action) {
    return std::visit(
        [](const auto& value) -> const char* {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, StateClear>) return "state-clear";
            else if constexpr (std::is_same_v<T, StateCopy>) return "state-copy";
            else if constexpr (std::is_same_v<T, StateTransferDescription>) return "state-transfer";
            else if constexpr (std::is_same_v<T, PageCopy>) return "page-copy";
            else if constexpr (std::is_same_v<T, PageTransfer>) return "page-transfer";
            else if constexpr (std::is_same_v<T, MappingUpdate>) return "mapping-update";
            else if constexpr (std::is_same_v<T, StateSpillOut>) return "spill-out";
            else if constexpr (std::is_same_v<T, StateSpillIn>) return "spill-in";
            else return "unknown";
        },
        action);
}

// What an action puts on the transfer stream. Actions with copy work are covered by a segment
// line; this exists for the actions the segment bookkeeping deliberately skips because
// ContextStorage::describe reports no copy operations for them (a clear is a memset and a mapping
// publication is an H2D copy, so "no copy operations" is not "no stream work").
std::string action_stream_work(const ContextAction& action, const ContextStorage& storage) {
    return std::visit(
        [&](const auto& value) -> std::string {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, StateClear>) {
                return "memset:" + std::to_string(storage.state().image_bytes()) + "B";
            } else if constexpr (std::is_same_v<T, MappingUpdate>) {
                if (value.pages.empty()) return "none";
                return "h2d:" + std::to_string(value.pages.size() * sizeof(std::int32_t)) + "B";
            } else if constexpr (std::is_same_v<T, StateSpillOut> ||
                                 std::is_same_v<T, StateSpillIn>) {
                return "file-io";
            } else {
                return "copy";
            }
        },
        action);
}

// The device pool a segment addresses; a State segment has no device pool id, only the state image
// pool. The per-action identity carries the Host pool id for host transfers.
std::string segment_pool(const ContextActions& actions, std::size_t begin, std::size_t end) {
    for (auto i = begin; i < end; ++i) {
        std::string result;
        std::visit(
            [&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, PageCopy>) {
                    result = std::to_string(value.source.pool_id);
                } else if constexpr (std::is_same_v<T, PageTransfer>) {
                    if (!value.pages.empty())
                        result = std::to_string(value.pages.front().pool_id);
                } else if constexpr (std::is_same_v<T, MappingUpdate>) {
                    result = std::to_string(value.row.pool_id);
                }
            },
            actions.operations[i]);
        if (!result.empty()) return result;
    }
    return "state";
}
} // namespace

ContextMigration::ContextMigration(DeviceContext& device, ContextStorage& storage,
                                   std::size_t mapping_capacity_elements,
                                   std::size_t copy_segment_capacity)
    : ContextMigration(device, storage, device.transfer_view(), mapping_capacity_elements,
                       copy_segment_capacity) {}

ContextMigration::ContextMigration(DeviceContext& device, ContextStorage& storage,
                                   CudaExecutionView execution,
                                   std::size_t mapping_capacity_elements,
                                   std::size_t copy_segment_capacity)
    : device_(&device), storage_(&storage), execution_(execution),
      mapping_(mapping_bytes(storage, mapping_capacity_elements)), source_ready_(device),
      completed_(device) {
    if (execution.device() != device.device())
        throw std::invalid_argument("context action stream belongs to another device");
    const auto capacity =
        copy_segment_capacity ? copy_segment_capacity : storage.copy_segment_capacity();
    timers_.reserve(capacity);
    segments_.reserve(capacity);
    observations_.reserve(capacity);
    for (std::size_t i = 0; i < capacity; ++i) timers_.emplace_back(device, execution.stream());
}

ContextMigration::~ContextMigration() {
    if (!ready()) std::terminate();
}

void ContextMigration::validate(const ContextActions& actions) const {
    storage_->validate(actions, mapping_.span());
    std::size_t segments = 0;
    std::optional<runtime::ContextTransferObservation> previous;
    for (const auto& action : actions.operations) {
        const auto observed = storage_->describe(action);
        if (!observed.work.copy_operations) {
            previous.reset();
            continue;
        }
        if (!previous || previous->resource != observed.resource ||
            previous->direction != observed.direction)
            ++segments;
        previous = observed;
    }
    if (segments > timers_.size())
        throw std::out_of_range("context action segments exceed the prepared slot");
}

void ContextMigration::prepare(const ContextActions& actions) {
    if (occupied_) throw std::logic_error("context action slot is occupied");
    validate(actions);
    segments_.clear();
    observations_.clear();
    for (std::size_t i = 0; i < actions.operations.size(); ++i) {
        const auto observed = storage_->describe(actions.operations[i]);
        if (!observed.work.copy_operations) continue;
        if (segments_.empty() || segments_.back().end != i ||
            segments_.back().observed.resource != observed.resource ||
            segments_.back().observed.direction != observed.direction) {
            if (segments_.size() == timers_.size())
                throw std::out_of_range("context action segments exceed the prepared slot");
            segments_.push_back({observed, i, i + 1});
        } else {
            auto& segment = segments_.back();
            segment.end   = i + 1;
            segment.observed.work.payload_bytes += observed.work.payload_bytes;
            segment.observed.work.copy_operations += observed.work.copy_operations;
            segment.observed.page_count += observed.page_count;
            segment.observed.units += observed.units;
        }
    }
}

void ContextMigration::enqueue(const ContextActions& actions, CudaExecutionView execution) {
    bind_rank();
    occupied_ = true;
    if (actions.empty()) {
        submitted_stream_ = nullptr;
        fenced_           = true;
        return;
    }
    fenced_           = false;
    submitted_stream_ = execution.stream();
    if (execution.stream() != device_->stream()) {
        source_ready_.record(device_->stream());
        source_ready_.wait(execution.stream());
    }
    auto staging        = mapping_.span();
    std::size_t segment = 0;
    // DIAGNOSTIC (NINFER_CTX_SEGMENTS=1). The local two-device lane's rank is its device index:
    // LocalExecutor drives its workers in rank order and that lane's device list is {0, 1}.
    // ContextMigration is not handed a rank, so the device index is the only rank fact available
    // here. A segment line prints once, at its first action, so the lines follow enqueue order and
    // the last line before the stall names the last segment the transfer stream received. Actions
    // the segment bookkeeping skips (no copy operations in ContextStorage::describe) get their own
    // line: a StateClear is a memset and a MappingUpdate is an H2D publication, and both are stream
    // work even though neither is a copy segment.
    const bool trace = segments_trace_enabled();
    const int rank    = device_->device().index();
    if (trace && segments_.empty())
        std::fprintf(stderr, "[ctx-seg] rank=%d no-copy-segments actions=%zu\n", rank,
                     actions.operations.size());
    for (std::size_t i = 0; i < actions.operations.size(); ++i) {
        const bool segmented = segment < segments_.size() && i >= segments_[segment].begin &&
                               i < segments_[segment].end;
        if (trace) {
            if (segmented && segments_[segment].begin == i) {
                const auto& part     = segments_[segment];
                const auto& observed = part.observed;
                std::string identity;
                for (auto j = part.begin; j < part.end; ++j) {
                    if (!identity.empty()) identity += "; ";
                    identity += describe_action(actions.operations[j]);
                }
                std::fprintf(
                    stderr,
                    "[ctx-seg] rank=%d segment=%zu/%zu resource=%s direction=%s units=%" PRIu64
                    " pages=%u payload=%" PRIu64 " ops=%u pool=%s identity=%s\n",
                    rank, segment, segments_.size(), resource_class_name(observed.resource),
                    transfer_direction_name(observed.direction), observed.units,
                    observed.page_count, observed.work.payload_bytes,
                    observed.work.copy_operations,
                    segment_pool(actions, part.begin, part.end).c_str(), identity.c_str());
            } else if (!segmented) {
                std::fprintf(
                    stderr, "[ctx-seg] rank=%d action=%zu/%zu kind=%s stream=%s identity=%s\n",
                    rank, i, actions.operations.size(), action_kind_name(actions.operations[i]),
                    action_stream_work(actions.operations[i], *storage_).c_str(),
                    describe_action(actions.operations[i]).c_str());
            }
        }
        if (state_clear_trace_enabled()) {
            if (const auto* clear = std::get_if<StateClear>(&actions.operations[i])) {
                const bool fenced        = execution.stream() != device_->stream();
                const auto event         = observe_event(source_ready_, fenced);
                const auto compute_state = cudaStreamQuery(device_->stream());
                (void)cudaGetLastError();
                std::fprintf(stderr,
                             "[state-clear] rank=%d slot=%d stage=fence fence=%s event=%s "
                             "wait-us=%ld compute-stream=%s\n",
                             rank, clear->slot, fenced ? "enqueued" : "none", event.state,
                             event.wait_us, stream_query_name(compute_state));
                if (!event.error.empty()) {
                    std::fprintf(stderr,
                                 "[state-clear] rank=%d slot=%d stage=fence event-error=%s\n", rank,
                                 clear->slot, event.error.c_str());
                }
            }
        }
        if (segment < segments_.size() && segments_[segment].begin == i) timers_[segment].start();
        const auto consumed =
            storage_->execute(actions.operations[i], actions.spill_store, staging, execution);
        if (consumed) {
            staging.data = static_cast<std::byte*>(staging.data) + consumed;
            staging.bytes -= consumed;
        }
        if (segment < segments_.size() && segments_[segment].end == i + 1) {
            timers_[segment].record_stop();
            ++segment;
        }
    }
    completed_.record(execution.stream());
    fenced_ = true;
}

void ContextMigration::submit(const ContextActions& actions) {
    prepare(actions);
    enqueue(actions, execution_);
}

bool ContextMigration::ready() const {
    bind_rank();
    return !occupied_ || (fenced_ && (!submitted_stream_ || completed_.ready()));
}

std::span<const runtime::ContextTransferObservation> ContextMigration::observations() {
    bind_rank();
    if (!occupied_ || !fenced_ || !ready())
        throw std::logic_error("context observations require completed execution");
    observations_.clear();
    for (std::size_t i = 0; i < segments_.size(); ++i) {
        auto observed = segments_[i].observed;
        observed.elapsed_ns =
            std::max<std::uint64_t>(1, std::llround(timers_[i].elapsed_ms() * 1.0e6));
        const auto existing =
            std::find_if(observations_.begin(), observations_.end(), [&](const auto& item) {
                return item.resource == observed.resource && item.direction == observed.direction;
            });
        if (existing == observations_.end())
            observations_.push_back(observed);
        else {
            existing->work.payload_bytes += observed.work.payload_bytes;
            existing->work.copy_operations += observed.work.copy_operations;
            existing->page_count += observed.page_count;
            existing->units += observed.units;
            existing->elapsed_ns += observed.elapsed_ns;
        }
    }
    return observations_;
}

void ContextMigration::finish() {
    if (!occupied_ || !fenced_ || !ready())
        throw std::logic_error("cannot reuse an unfinished context action slot");
    occupied_ = false;
    fenced_   = false;
    segments_.clear();
    observations_.clear();
}

void ContextMigration::drain() {
    bind_rank();
    if (!occupied_) return;
    std::exception_ptr error;
    try {
        if (fenced_ && submitted_stream_)
            completed_.synchronize();
        else if (!fenced_) {
            const auto status = cudaStreamSynchronize(submitted_stream_);
            if (status != cudaSuccess)
                throw std::runtime_error(std::string("context action drain failed: ") +
                                         cudaGetErrorString(status));
        }
    } catch (...) { error = std::current_exception(); }
    occupied_ = false;
    fenced_   = false;
    segments_.clear();
    observations_.clear();
    if (error) std::rethrow_exception(error);
}

} // namespace ninfer::models::context
