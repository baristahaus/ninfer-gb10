#include "core/state_image.h"

#include "core/transfer.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace ninfer {
namespace {

constexpr std::size_t kAlignment = 256;

// DIAGNOSTIC (NINFER_STATE_CLEAR_TRACE=1): report what a state-image clear hands to the stream.
// cudaPointerGetAttributes reads allocation metadata, not device memory, so it needs no stream
// ordering; cudaStreamQuery is a point-in-time observation and changes no scheduling. Unset, the
// clear path below is unchanged.
bool state_clear_trace_enabled() {
    static const bool enabled = std::getenv("NINFER_STATE_CLEAR_TRACE") != nullptr;
    return enabled;
}

const char* memory_type_name(cudaMemoryType type) {
    if (type == cudaMemoryTypeDevice) return "device";
    if (type == cudaMemoryTypeHost) return "host";
    if (type == cudaMemoryTypeManaged) return "managed";
    if (type == cudaMemoryTypeUnregistered) return "unregistered";
    return "other";
}

const char* stream_state_name(cudaError_t status) {
    if (status == cudaSuccess) return "idle";
    if (status == cudaErrorNotReady) return "busy";
    return cudaGetErrorName(status);
}

struct PointerFacts {
    cudaError_t status = cudaSuccess;
    cudaPointerAttributes attributes{};
};

PointerFacts query_pointer(const void* pointer) {
    PointerFacts facts;
    if (pointer == nullptr) {
        facts.status = cudaErrorInvalidValue;
        return facts;
    }
    facts.status = cudaPointerGetAttributes(&facts.attributes, pointer);
    // A diagnostic query must not leave its own failure for a later CUDA_CHECK(cudaGetLastError()).
    (void)cudaGetLastError();
    return facts;
}

std::size_t product(std::size_t a, std::size_t b) {
    if (b != 0 && a > static_cast<std::size_t>(INT64_MAX) / b) {
        throw std::overflow_error("StateImage extent exceeds int64");
    }
    return a * b;
}

void clear_memory(MemorySpan memory, CudaExecutionView execution) {
    if (memory.device != execution.device()) {
        throw std::invalid_argument("StateImage clearing requires its CUDA execution device");
    }
    CudaDeviceScope scope(execution.device());
    CUDA_CHECK(cudaMemsetAsync(memory.data, 0, memory.bytes, execution.stream()));
}

} // namespace

TransferWork StateImageLayout::transfer_work() const noexcept {
    return {.payload_bytes = image_stride, .copy_operations = 1};
}

StateField StateImageBuilder::add(DType dtype, std::initializer_list<std::int32_t> shape,
                                  std::uint32_t layers, std::string_view label) {
    if (shape.size() == 0 || shape.size() > 3 || layers == 0 || fields_.size() >= UINT32_MAX ||
        std::any_of(shape.begin(), shape.end(), [](auto extent) { return extent <= 0; })) {
        throw std::invalid_argument(
            "StateImage field requires 1..3 inner axes and positive layers");
    }
    LayoutBuilder field;
    TensorRegion tensor  = field.add_tensor(dtype, shape, kAlignment, label);
    const auto stride    = field.finish(kAlignment, label);
    const auto region    = image_.add(product(stride, layers), kAlignment, label);
    tensor.region.offset = region.offset;
    const StateField id{static_cast<std::uint32_t>(fields_.size())};
    fields_.push_back({tensor, layers, stride, static_cast<int>(shape.size())});
    return id;
}

LayoutRegion StateImageBuilder::group(StateField first, StateField last) const {
    if (first.index > last.index || last.index >= fields_.size()) {
        throw std::out_of_range("StateImage field group is out of range");
    }
    const auto begin = fields_[first.index].tensor.region.offset;
    const auto& end  = fields_[last.index];
    return {begin, end.tensor.region.offset + product(end.layer_stride, end.layers) - begin,
            kAlignment};
}

StateImageLayout StateImageBuilder::finish(LayoutBuilder& backing, std::int32_t slots) const {
    if (slots <= 0 || fields_.empty()) {
        throw std::invalid_argument("StateImage requires fields and positive slot capacity");
    }
    const auto stride = image_.finish(kAlignment, "StateImage");
    return {backing.add(product(stride, slots), kAlignment, "StateImage storage"), stride, slots,
            fields_};
}

StateImageStorage::StateImageStorage(MemorySpan backing, const StateImageLayout& layout)
    : layout_(layout), storage_(layout.storage.bind(backing)) {
    if (layout.slot_count <= 0 || layout.image_stride == 0 || layout.fields.empty() ||
        storage_.bytes != product(layout.image_stride, layout.slot_count)) {
        throw std::invalid_argument("StateImage storage geometry is inconsistent");
    }
    std::size_t end = 0;
    for (const auto& field : layout.fields) {
        if (field.slot_axis < 1 || field.slot_axis > 3 || field.layers == 0 ||
            field.tensor.region.offset < end || field.layer_stride < field.tensor.region.bytes) {
            throw std::invalid_argument("StateImage fields overlap or have invalid geometry");
        }
        const auto extent = product(field.layer_stride, field.layers);
        if (field.tensor.region.offset > layout.image_stride ||
            extent > layout.image_stride - field.tensor.region.offset) {
            throw std::invalid_argument("StateImage field exceeds its image");
        }
        // TensorRegion checks dtype, shape, alignment and the addressed inner payload.
        const Tensor tensor = field.tensor.bind(storage_.subspan(0, layout.image_stride));
        for (int axis = field.slot_axis; axis < 4; ++axis) {
            if (tensor.ne[axis] != 1) {
                throw std::invalid_argument("StateImage field uses the reserved slot axis");
            }
        }
        if (field.layer_stride % tensor.nb[0] != 0 || layout.image_stride % tensor.nb[0] != 0) {
            throw std::invalid_argument("StateImage strides split a field element");
        }
        end = field.tensor.region.offset + extent;
    }
}

Tensor StateImageStorage::field(StateField id, std::uint32_t layer) const {
    const auto& field = layout_.fields.at(id.index);
    if (layer >= field.layers) { throw std::out_of_range("StateImage layer is out of range"); }
    Tensor tensor              = field.tensor.bind(storage_.subspan(layer * field.layer_stride));
    tensor.ne[field.slot_axis] = slot_count();
    for (int axis = field.slot_axis; axis < 4; ++axis) {
        tensor.nb[axis] = static_cast<std::int64_t>(axis == field.slot_axis ? layout_.image_stride
                                                                            : storage_.bytes);
    }
    return tensor;
}

std::int64_t StateImageStorage::layer_stride(StateField field) const {
    return static_cast<std::int64_t>(layout_.fields.at(field.index).layer_stride);
}

MemorySpan StateImageStorage::image(std::int32_t slot) const {
    if (slot < 0 || slot >= slot_count()) {
        throw std::out_of_range("StateImage slot out of range");
    }
    return storage_.subspan(static_cast<std::size_t>(slot) * image_bytes(), image_bytes());
}

void StateImageStorage::clear(std::int32_t slot, CudaExecutionView execution) const {
    const auto memory = image(slot);
    if (!state_clear_trace_enabled()) {
        clear_memory(memory, execution);
        return;
    }
    const int rank    = execution.device().index();
    const auto facts  = query_pointer(memory.data);
    const auto before = cudaStreamQuery(execution.stream());
    (void)cudaGetLastError();
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess) current_device = -1;
    (void)cudaGetLastError();
    const auto offset = static_cast<std::size_t>(static_cast<const std::byte*>(memory.data) -
                                                 static_cast<const std::byte*>(storage_.data));
    char attribute[64];
    char device_pointer[32];
    if (facts.status == cudaSuccess) {
        std::snprintf(attribute, sizeof attribute, "valid");
        std::snprintf(device_pointer, sizeof device_pointer, "%p", facts.attributes.devicePointer);
    } else {
        std::snprintf(attribute, sizeof attribute, "invalid(%s)", cudaGetErrorName(facts.status));
        std::snprintf(device_pointer, sizeof device_pointer, "unknown");
    }
    const bool owner_match = facts.status == cudaSuccess && facts.attributes.device == rank;
    std::fprintf(stderr,
                 "[state-clear] rank=%d slot=%d stage=enqueue ptr=%p attr=%s owner=%d type=%s "
                 "device-pointer=%s span-device=%d owner-match=%s storage=%p storage-bytes=%zu "
                 "slot-offset=%zu bytes=%zu stream=%p stream-before=%s current-device=%d\n",
                 rank, slot, memory.data, attribute,
                 facts.status == cudaSuccess ? facts.attributes.device : -1,
                 facts.status == cudaSuccess ? memory_type_name(facts.attributes.type) : "none",
                 device_pointer, memory.device.index(), owner_match ? "yes" : "no", storage_.data,
                 storage_.bytes, offset, memory.bytes, static_cast<const void*>(execution.stream()),
                 stream_state_name(before), current_device);
    clear_memory(memory, execution);
    const auto after = cudaStreamQuery(execution.stream());
    (void)cudaGetLastError();
    std::fprintf(stderr, "[state-clear] rank=%d slot=%d stage=enqueued stream-after=%s\n", rank,
                 slot, stream_state_name(after));
}

void StateImageStorage::clear_all(CudaExecutionView execution) const {
    clear_memory(storage_, execution);
}

void StateImageStorage::copy(std::int32_t source, std::int32_t destination,
                             CudaExecutionView execution) const {
    copy_bytes_async(image(destination), image(source), execution);
}

void StateImageStorage::copy_group(LayoutRegion group, std::int32_t source,
                                   std::int32_t destination, CudaExecutionView execution) const {
    copy_bytes_async(group.bind(image(destination)), group.bind(image(source)), execution);
}

void StateImageStorage::copy_to_host(std::int32_t source, MemorySpan destination,
                                     CudaExecutionView execution) const {
    if (destination.device != Device::cpu()) {
        throw std::invalid_argument("StateImage Host destination must be CPU memory");
    }
    copy_bytes_async(destination, image(source), execution);
}

void StateImageStorage::copy_from_host(ConstMemorySpan source, std::int32_t destination,
                                       CudaExecutionView execution) const {
    if (source.device != Device::cpu()) {
        throw std::invalid_argument("StateImage Host source must be CPU memory");
    }
    copy_bytes_async(image(destination), source, execution);
}

} // namespace ninfer
