#pragma once

#include <cuda_runtime.h>

#include <functional>
#include <vector>

namespace ninfer {

// A child node belongs to its enclosing definition. Keep that definition alive while
// updating the executable instantiated from it.
struct DecodeGraphChild {
    cudaGraphNode_t node = nullptr;
};

class DecodeGraphDefinition {
public:
    DecodeGraphDefinition() = default;
    ~DecodeGraphDefinition();

    DecodeGraphDefinition(const DecodeGraphDefinition&)            = delete;
    DecodeGraphDefinition& operator=(const DecodeGraphDefinition&) = delete;
    DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept;
    DecodeGraphDefinition& operator=(DecodeGraphDefinition&& other) noexcept;

    void capture(cudaStream_t stream, const std::function<void()>& body);
    [[nodiscard]] DecodeGraphChild append_to(cudaStream_t capturing_stream) const;
    [[nodiscard]] bool ready() const noexcept;
    void reset() noexcept;
    // NINFER_GRAPH_DUMP=1: print every node's type and kernel function name to stderr
    // after a capture. Used to diff the tiered and untiered captured node sets.
    void dump_nodes(const char* label) const;

private:
    friend class DecodeGraphCapture;
    friend class DecodeGraphExecutable;
    cudaGraph_t graph_ = nullptr;
};

// An unfinished capture is discarded on scope exit. All participating streams must
// rejoin the origin before finish(); the definition owns the resulting graph.
class DecodeGraphCapture {
public:
    explicit DecodeGraphCapture(cudaStream_t stream);
    ~DecodeGraphCapture();
    DecodeGraphCapture(const DecodeGraphCapture&)            = delete;
    DecodeGraphCapture& operator=(const DecodeGraphCapture&) = delete;

    [[nodiscard]] DecodeGraphDefinition finish();

private:
    cudaStream_t stream_;
};

class DecodeGraphExecutable {
public:
    DecodeGraphExecutable() = default;
    ~DecodeGraphExecutable();

    DecodeGraphExecutable(const DecodeGraphExecutable&)            = delete;
    DecodeGraphExecutable& operator=(const DecodeGraphExecutable&) = delete;
    DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept;
    DecodeGraphExecutable& operator=(DecodeGraphExecutable&& other) noexcept;

    void instantiate(const DecodeGraphDefinition& definition);
    void update(const DecodeGraphDefinition& definition);
    void update_child(DecodeGraphChild child, const DecodeGraphDefinition& definition);
    void upload(cudaStream_t stream);
    void launch(cudaStream_t stream);
    [[nodiscard]] bool ready() const noexcept;
    void reset() noexcept;

private:
    // A child binding is valid only for the definition graph its node handle came from, so a rebuild
    // keeps just the bindings taken from the graph it instantiates.
    struct ChildBinding {
        cudaGraph_t parent   = nullptr;
        cudaGraphNode_t node = nullptr;
        cudaGraph_t child    = nullptr;
    };

    // Instantiate a fresh executable from `definition` and swap it in. The previous executable is
    // destroyed only after its last launch has retired.
    void rebuild(const DecodeGraphDefinition& definition);
    void retain_bindings(cudaGraph_t parent);
    void rebind_children();

    cudaGraphExec_t exec_ = nullptr;
    // The definition graph `exec_` was instantiated from; a child node handle is meaningful only
    // against the graph that owns it.
    cudaGraph_t definition_ = nullptr;
    // The stream of the last launch. Read only on the rebuild path, to retire in-flight work before
    // the old executable is destroyed; the clean-update path never touches it.
    cudaStream_t launch_stream_ = nullptr;
    // The stream of the last upload, and whether a rebuild left the executable un-uploaded. The next
    // launch uploads it on its own stream, keeping the caller's pre-warm.
    cudaStream_t upload_stream_ = nullptr;
    bool needs_upload_          = false;
    std::vector<ChildBinding> children_;
};

} // namespace ninfer
