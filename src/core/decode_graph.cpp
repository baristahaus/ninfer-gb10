#include "core/decode_graph.h"

#include "core/device.h"
#include "core/nvtx.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer {
namespace {

void check_graph(cudaError_t error, const char* operation) {
    if (error == cudaErrorMemoryAllocation) {
        (void)cudaGetLastError();
        throw DeviceAllocationError(operation);
    }
    cuda_check(error, operation, __FILE__, __LINE__);
}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

void destroy_graph_exec(cudaGraphExec_t& exec) noexcept {
    if (exec != nullptr) {
        log_cuda_error("cudaGraphExecDestroy", cudaGraphExecDestroy(exec));
        exec = nullptr;
    }
}

void destroy_graph(cudaGraph_t& graph) noexcept {
    if (graph != nullptr) {
        log_cuda_error("cudaGraphDestroy", cudaGraphDestroy(graph));
        graph = nullptr;
    }
}

void discard_capture(cudaStream_t stream) noexcept {
    cudaGraph_t discard = nullptr;
    log_cuda_error("cudaStreamEndCapture(discard)", cudaStreamEndCapture(stream, &discard));
    destroy_graph(discard);
}

} // namespace

DecodeGraphDefinition::~DecodeGraphDefinition() { reset(); }

DecodeGraphDefinition::DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept
    : graph_(other.graph_) {
    other.graph_ = nullptr;
}

DecodeGraphDefinition& DecodeGraphDefinition::operator=(DecodeGraphDefinition&& other) noexcept {
    if (this == &other) { return *this; }

    reset();
    graph_ = other.graph_;

    other.graph_ = nullptr;
    return *this;
}

void DecodeGraphDefinition::capture(cudaStream_t stream, const std::function<void()>& body) {
    nvtx::ScopedRange capture_range(nvtx::Name::CudaGraphCapture, nvtx::Category::Graph);
    reset();

    DecodeGraphCapture capture(stream);
    body();
    *this = capture.finish();
    dump_nodes("captured");
}

DecodeGraphChild DecodeGraphDefinition::append_to(cudaStream_t stream) const {
    cudaStreamCaptureStatus status;
    cudaGraph_t parent                  = nullptr;
    const cudaGraphNode_t* dependencies = nullptr;
    std::size_t count                   = 0;
    check_graph(
        cudaStreamGetCaptureInfo(stream, &status, nullptr, &parent, &dependencies, nullptr, &count),
        "cudaStreamGetCaptureInfo");
    if (!ready() || status != cudaStreamCaptureStatusActive)
        throw std::logic_error("appending a child requires a definition and active capture");
    DecodeGraphChild child;
    check_graph(cudaGraphAddChildGraphNode(&child.node, parent, dependencies, count, graph_),
                "cudaGraphAddChildGraphNode");
    check_graph(cudaStreamUpdateCaptureDependencies(stream, &child.node, nullptr, 1,
                                                    cudaStreamSetCaptureDependencies),
                "cudaStreamUpdateCaptureDependencies");
    return child;
}

DecodeGraphCapture::DecodeGraphCapture(cudaStream_t stream) : stream_(stream) {
    check_graph(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal),
                "cudaStreamBeginCapture");
}

DecodeGraphCapture::~DecodeGraphCapture() {
    if (stream_) discard_capture(stream_);
}

DecodeGraphDefinition DecodeGraphCapture::finish() {
    if (!stream_) throw std::logic_error("CUDA Graph capture is already finished");
    DecodeGraphDefinition definition;
    const auto error = cudaStreamEndCapture(stream_, &definition.graph_);
    stream_          = nullptr;
    check_graph(error, "cudaStreamEndCapture");
    return definition;
}

bool DecodeGraphDefinition::ready() const noexcept { return graph_ != nullptr; }

void DecodeGraphDefinition::reset() noexcept { destroy_graph(graph_); }

void DecodeGraphDefinition::dump_nodes(const char* label) const {
    const char* flag = std::getenv("NINFER_GRAPH_DUMP");
    if (flag == nullptr || std::string_view{flag} != "1" || graph_ == nullptr) { return; }
    std::size_t count = 0;
    if (cudaGraphGetNodes(graph_, nullptr, &count) != cudaSuccess) { return; }
    std::vector<cudaGraphNode_t> nodes(count);
    if (cudaGraphGetNodes(graph_, nodes.data(), &count) != cudaSuccess) { return; }
    std::fprintf(stderr, "[graphdump] %s nodes=%zu\n", label, count);
    std::unordered_map<std::string, int> kernels;
    for (const auto node : nodes) {
        cudaGraphNodeType type;
        cudaGraphNodeGetType(node, &type);
        if (type != cudaGraphNodeTypeKernel) { continue; }
        cudaKernelNodeParams params{};
        if (cudaGraphKernelNodeGetParams(node, &params) != cudaSuccess) { continue; }
        // params.func is the host stub pointer; cudaFuncGetName resolves it.
        const char* name = nullptr;
        if (cudaFuncGetName(&name, params.func) == cudaSuccess && name != nullptr) {
            kernels[std::string{name}] += 1;
        } else {
            kernels["<unresolved>"] += 1;
        }
    }
    for (const auto& [name, n] : kernels) {
        std::fprintf(stderr, "[graphdump]   %s x%d\n", name.c_str(), n);
    }
}

DecodeGraphExecutable::~DecodeGraphExecutable() { reset(); }

DecodeGraphExecutable::DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept
    : exec_(other.exec_),
      definition_(other.definition_),
      launch_stream_(other.launch_stream_),
      upload_stream_(other.upload_stream_),
      needs_upload_(other.needs_upload_),
      children_(std::move(other.children_)) {
    other.exec_         = nullptr;
    other.definition_   = nullptr;
    other.launch_stream_ = nullptr;
    other.upload_stream_ = nullptr;
    other.needs_upload_  = false;
    other.children_.clear();
}

DecodeGraphExecutable& DecodeGraphExecutable::operator=(DecodeGraphExecutable&& other) noexcept {
    if (this == &other) { return *this; }

    reset();
    exec_                = other.exec_;
    definition_          = other.definition_;
    launch_stream_       = other.launch_stream_;
    upload_stream_       = other.upload_stream_;
    needs_upload_        = other.needs_upload_;
    children_            = std::move(other.children_);
    other.exec_          = nullptr;
    other.definition_    = nullptr;
    other.launch_stream_ = nullptr;
    other.upload_stream_ = nullptr;
    other.needs_upload_  = false;
    other.children_.clear();
    return *this;
}

void DecodeGraphExecutable::instantiate(const DecodeGraphDefinition& definition) {
    nvtx::ScopedRange instantiate_range(nvtx::Name::CudaGraphInstantiate, nvtx::Category::Graph);
    if (!definition.ready()) {
        throw std::logic_error("cannot instantiate an empty CUDA Graph definition");
    }
    // Child bindings belong to the definition graph their node came from; keep the ones this
    // instantiation can honour and drop the rest.
    std::vector<ChildBinding> retained;
    for (const ChildBinding& binding : children_) {
        if (binding.parent == definition.graph_) { retained.push_back(binding); }
    }
    const cudaStream_t uploaded_on = upload_stream_;
    reset();

    cudaGraphExec_t exec  = nullptr;
    const cudaError_t err = cudaGraphInstantiate(&exec, definition.graph_, 0);
    if (err != cudaSuccess) {
        destroy_graph_exec(exec);
        check_graph(err, "cudaGraphInstantiate");
    }
    exec_          = exec;
    definition_    = definition.graph_;
    upload_stream_ = uploaded_on;
    needs_upload_  = uploaded_on != nullptr;
    children_      = std::move(retained);
    rebind_children();
}

void DecodeGraphExecutable::update(const DecodeGraphDefinition& definition) {
    nvtx::ScopedRange update_range(nvtx::Name::CudaGraphUpdate, nvtx::Category::Graph);
    if (!ready() || !definition.ready()) {
        throw std::logic_error("CUDA Graph update requires a definition and executable");
    }

    cudaGraphExecUpdateResultInfo result{};
    const cudaError_t err = cudaGraphExecUpdate(exec_, definition.graph_, &result);
    // An allocation failure is the caller's to handle; it is not a shape change.
    if (err == cudaErrorMemoryAllocation) check_graph(err, "cudaGraphExecUpdate");
    if (err == cudaSuccess && result.result == cudaGraphExecUpdateSuccess) {
        // The executable took the new parameters in place: no rebuild and no synchronisation, which
        // is the path every ordinary frontier change takes.
        definition_ = definition.graph_;
        retain_bindings(definition.graph_);
        return;
    }
    // Every other outcome is recoverable by construction: CUDA could not apply these node
    // parameters to the existing executable and documents that the caller must instantiate a new
    // one. Rebuilding is routine, not an engine failure; only a failed instantiate throws.
    rebuild(definition);
}

void DecodeGraphExecutable::rebuild(const DecodeGraphDefinition& definition) {
    // Instantiate first, so a failed instantiate leaves the current executable usable.
    cudaGraphExec_t exec  = nullptr;
    const cudaError_t err = cudaGraphInstantiate(&exec, definition.graph_, 0);
    if (err != cudaSuccess) {
        destroy_graph_exec(exec);
        check_graph(err, "cudaGraphInstantiate");
    }
    // The previous executable may still have a launch in flight on the stream it was last launched
    // on. Retire it before destroying the handle. This runs only on the rebuild path, so the
    // clean-update path above never synchronises.
    if (launch_stream_ != nullptr) {
        check_graph(cudaStreamSynchronize(launch_stream_), "cudaStreamSynchronize");
    }
    destroy_graph_exec(exec_);
    exec_       = exec;
    definition_ = definition.graph_;
    retain_bindings(definition.graph_);
    rebind_children();
    // The new executable has not been uploaded. The next launch uploads it on its own stream, so
    // the caller's upload-then-launch pre-warm is preserved without a second synchronisation here.
    needs_upload_ = upload_stream_ != nullptr;
}

void DecodeGraphExecutable::retain_bindings(cudaGraph_t parent) {
    std::erase_if(children_, [parent](const ChildBinding& binding) {
        return binding.parent != parent;
    });
}

void DecodeGraphExecutable::rebind_children() {
    for (const ChildBinding& binding : children_) {
        check_graph(cudaGraphExecChildGraphNodeSetParams(exec_, binding.node, binding.child),
                    "cudaGraphExecChildGraphNodeSetParams");
    }
}

void DecodeGraphExecutable::update_child(DecodeGraphChild child,
                                         const DecodeGraphDefinition& definition) {
    nvtx::ScopedRange update_range(nvtx::Name::CudaGraphUpdate, nvtx::Category::Graph);
    if (!ready() || !child.node || !definition.ready())
        throw std::logic_error("CUDA Graph child update requires a node and definition");
    check_graph(cudaGraphExecChildGraphNodeSetParams(exec_, child.node, definition.graph_),
                "cudaGraphExecChildGraphNodeSetParams");
    // Remember the binding so a later rebuild can re-apply it. The node handle belongs to the graph
    // the executable was instantiated from, which is what makes the binding replayable at all.
    for (ChildBinding& binding : children_) {
        if (binding.node == child.node) {
            binding.parent = definition_;
            binding.child  = definition.graph_;
            return;
        }
    }
    children_.push_back(ChildBinding{definition_, child.node, definition.graph_});
}

void DecodeGraphExecutable::upload(cudaStream_t stream) {
    nvtx::ScopedRange upload_range(nvtx::Name::CudaGraphUpload, nvtx::Category::Graph);
    if (!ready()) { throw std::logic_error("cannot upload an empty CUDA Graph executable"); }
    check_graph(cudaGraphUpload(exec_, stream), "cudaGraphUpload");
    upload_stream_ = stream;
    needs_upload_  = false;
}

void DecodeGraphExecutable::launch(cudaStream_t stream) {
    // This range executes for every replay; ranges in the captured body execute only at capture.
    nvtx::ScopedRange launch_range(nvtx::Name::CudaGraphLaunch, nvtx::Category::Graph);
    if (!ready()) { throw std::logic_error("cannot launch an empty CUDA Graph executable"); }
    if (needs_upload_) {
        // The executable was rebuilt after the caller uploaded the previous one. Upload the new one
        // on its launch stream: ordered before the launch, so no synchronisation is added.
        check_graph(cudaGraphUpload(exec_, stream), "cudaGraphUpload");
        needs_upload_ = false;
    }
    launch_stream_ = stream;
    check_graph(cudaGraphLaunch(exec_, stream), "cudaGraphLaunch");
}

bool DecodeGraphExecutable::ready() const noexcept { return exec_ != nullptr; }

void DecodeGraphExecutable::reset() noexcept {
    destroy_graph_exec(exec_);
    definition_    = nullptr;
    launch_stream_ = nullptr;
    upload_stream_ = nullptr;
    needs_upload_  = false;
    children_.clear();
}

} // namespace ninfer
