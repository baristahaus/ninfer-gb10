#include "core/arena.h"
#include "core/buffer.h"
#include "core/decode_graph.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t err) {
    // cudaErrorStubLibrary is what a GPU-less host that links the CUDA driver stub reports; a graph
    // test cannot run there and must skip rather than fail.
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver ||
           err == cudaErrorStubLibrary;
}

int expect_value(void* device, std::uint32_t expected, const char* label) {
    std::uint32_t actual  = 0;
    const cudaError_t err = cudaMemcpy(&actual, device, sizeof(actual), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << label << " copy failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    if (actual == expected) { return 0; }
    std::cerr << label << " expected 0x" << std::hex << expected << ", got 0x" << actual << std::dec
              << '\n';
    return 1;
}

int check_external_event(ninfer::DeviceContext& device) {
    ninfer::DeviceBuffer storage(device.device(), sizeof(std::uint32_t));
    ninfer::PinnedHostBuffer host(sizeof(std::uint32_t));
    ninfer::CudaCompletionEvent ready(device);
    ninfer::DecodeGraphDefinition definitions[2];
    for (int i = 0; i < 2; ++i) {
        definitions[i].capture(device.stream(), [&] {
            CUDA_CHECK(
                cudaMemsetAsync(storage.data(), i + 1, sizeof(std::uint32_t), device.stream()));
            CUDA_CHECK(cudaMemcpyAsync(host.data(), storage.data(), sizeof(std::uint32_t),
                                       cudaMemcpyDeviceToHost, device.stream()));
            ready.record_external(device.stream());
            // The handoff exposes the prefix, even though later graph nodes overwrite the source.
            CUDA_CHECK(
                cudaMemsetAsync(storage.data(), 0xff, sizeof(std::uint32_t), device.stream()));
        });
    }
    ninfer::DecodeGraphExecutable executable;
    executable.instantiate(definitions[0]);
    int failures = 0;
    for (int iteration = 0; iteration < 32; ++iteration) {
        const auto value = static_cast<std::uint32_t>(iteration % 2 + 1) * 0x01010101U;
        executable.update(definitions[iteration % 2]);
        *static_cast<std::uint32_t*>(host.data()) = 0;
        executable.launch(device.stream());
        ready.synchronize();
        if (*static_cast<std::uint32_t*>(host.data()) != value) {
            std::cerr << "external event exposed stale data after replay/update\n";
            ++failures;
        }
        device.synchronize();
        failures += expect_value(storage.data(), 0xffffffffU, "graph suffix");
    }
    return failures;
}

int check_child_graph(ninfer::DeviceContext& device) {
    ninfer::DeviceBuffer source(device.device(), sizeof(std::uint32_t));
    ninfer::DeviceBuffer destination(device.device(), sizeof(std::uint32_t));
    ninfer::DecodeGraphDefinition children[2], parent;
    for (int i = 0; i < 2; ++i)
        children[i].capture(device.stream(), [&] {
            CUDA_CHECK(cudaMemsetAsync(source.data(), i + 1, source.size(), device.stream()));
        });
    ninfer::DecodeGraphChild child;
    parent.capture(device.stream(), [&] {
        CUDA_CHECK(cudaMemsetAsync(source.data(), 0xff, source.size(), device.stream()));
        child = children[0].append_to(device.stream());
        CUDA_CHECK(cudaMemcpyAsync(destination.data(), source.data(), source.size(),
                                   cudaMemcpyDeviceToDevice, device.stream()));
    });
    ninfer::DecodeGraphExecutable executable;
    executable.instantiate(parent);
    int failures = 0;
    for (int i = 0; i < 8; ++i) {
        executable.update_child(child, children[i % 2]);
        executable.launch(device.stream());
        device.synchronize();
        failures += expect_value(destination.data(), (i % 2 + 1) * 0x01010101U,
                                 "child update and surrounding capture dependencies");
    }
    return failures;
}


// A topology change is refused in place by CUDA (cudaGraphExecUpdateErrorTopologyChanged), and a
// changed kernel-node parameter set is refused with cudaGraphExecUpdateErrorParametersChanged. Both
// are documented as recoverable: the caller instantiates a new executable. This case proves the
// executable rebuilds instead of throwing, and that the rebuilt executable launches the new graph.
int check_reinstantiate(ninfer::DeviceContext& device) {
    ninfer::DeviceBuffer storage(device.device(), sizeof(std::uint32_t));
    ninfer::DecodeGraphDefinition single, pair;
    single.capture(device.stream(), [&] {
        CUDA_CHECK(cudaMemsetAsync(storage.data(), 0x11, sizeof(std::uint32_t), device.stream()));
    });
    pair.capture(device.stream(), [&] {
        CUDA_CHECK(cudaMemsetAsync(storage.data(), 0x22, sizeof(std::uint32_t), device.stream()));
        CUDA_CHECK(cudaMemsetAsync(storage.data(), 0x33, sizeof(std::uint32_t), device.stream()));
    });

    ninfer::DecodeGraphExecutable executable;
    executable.instantiate(single);
    executable.upload(device.stream());
    device.synchronize();
    int failures = 0;
    executable.launch(device.stream());
    device.synchronize();
    failures += expect_value(storage.data(), 0x11111111U, "graph before re-instantiation");

    // The fix: an update CUDA refuses in place rebuilds the executable. Before the fix this threw and
    // the caller's engine died; the launch below is the proof that a usable executable came back.
    executable.update(pair);
    executable.launch(device.stream());
    device.synchronize();
    failures += expect_value(storage.data(), 0x33333333U, "graph after re-instantiation");

    // The rebuilt executable stays usable: a later clean update takes the ordinary in-place path.
    executable.update(single);
    executable.launch(device.stream());
    device.synchronize();
    failures += expect_value(storage.data(), 0x11111111U,
                             "graph after a clean update on the rebuilt executable");
    return failures;
}

// Report which update results this driver produces for the mutations a re-instantiation must survive,
// so a passing run says which recoverable code was actually exercised.
int probe_update_results(ninfer::DeviceContext& device) {
    ninfer::DeviceBuffer storage(device.device(), sizeof(std::uint32_t));
    ninfer::PinnedHostBuffer host(sizeof(std::uint32_t));
    cudaGraph_t graphs[3] = {nullptr, nullptr, nullptr};
    CUDA_CHECK(cudaStreamBeginCapture(device.stream(), cudaStreamCaptureModeThreadLocal));
    CUDA_CHECK(cudaMemsetAsync(storage.data(), 0x11, sizeof(std::uint32_t), device.stream()));
    CUDA_CHECK(cudaStreamEndCapture(device.stream(), &graphs[0]));
    CUDA_CHECK(cudaStreamBeginCapture(device.stream(), cudaStreamCaptureModeThreadLocal));
    CUDA_CHECK(cudaMemsetAsync(storage.data(), 0x22, sizeof(std::uint32_t), device.stream()));
    CUDA_CHECK(cudaStreamEndCapture(device.stream(), &graphs[1]));
    CUDA_CHECK(cudaStreamBeginCapture(device.stream(), cudaStreamCaptureModeThreadLocal));
    CUDA_CHECK(cudaMemsetAsync(storage.data(), 0x22, sizeof(std::uint32_t), device.stream()));
    CUDA_CHECK(cudaMemcpyAsync(host.data(), storage.data(), sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost, device.stream()));
    CUDA_CHECK(cudaStreamEndCapture(device.stream(), &graphs[2]));

    int recoverable = 0;
    for (int target = 1; target < 3; ++target) {
        cudaGraphExec_t exec = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&exec, graphs[0], 0));
        cudaGraphExecUpdateResultInfo info{};
        const cudaError_t err = cudaGraphExecUpdate(exec, graphs[target], &info);
        const int code        = static_cast<int>(info.result);
        std::cout << "update to mutation " << target << ": " << cudaGetErrorName(err)
                  << ", result " << code << '\n';
        // A refusal is reported as cudaErrorGraphExecUpdateFailure *with* a non-success result —
        // that is precisely the recoverable case the caller must handle by re-instantiating.
        // Requiring `err == cudaSuccess` here, as the first version did, meant a refusal could never
        // be counted: the driver refused mutation 2 with result 2 (TopologyChanged) and the probe
        // still reported that nothing had been refused, which is the opposite of its own output.
        if (err == cudaErrorGraphExecUpdateFailure) {
            ++recoverable;
        } else {
            CUDA_CHECK(err);
            if (info.result != cudaGraphExecUpdateSuccess) { ++recoverable; }
        }
        CUDA_CHECK(cudaGraphExecDestroy(exec));
    }
    for (auto& graph : graphs) { CUDA_CHECK(cudaGraphDestroy(graph)); }
    if (recoverable == 0) {
        std::cerr << "this driver refused no mutation in place; the recoverable path was not exercised\n";
        return -1;
    }
    return 0;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cerr << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    try {
        ninfer::DeviceContext device(0);
        ninfer::DeviceBuffer storage(device.device(), sizeof(std::uint32_t));
        CUDA_CHECK(cudaMemsetAsync(storage.data(), 0x33, sizeof(std::uint32_t), device.stream()));
        device.synchronize();

        ninfer::DecodeGraphDefinition first;
        first.capture(device.stream(), [&] {
            CUDA_CHECK(
                cudaMemsetAsync(storage.data(), 0x11, sizeof(std::uint32_t), device.stream()));
        });
        ninfer::DecodeGraphDefinition second;
        second.capture(device.stream(), [&] {
            CUDA_CHECK(
                cudaMemsetAsync(storage.data(), 0x22, sizeof(std::uint32_t), device.stream()));
        });

        int failures = 0;
        ninfer::DecodeGraphExecutable executable;
        executable.instantiate(first);
        executable.upload(device.stream());
        device.synchronize();
        failures += expect_value(storage.data(), 0x33333333U, "initial graph upload");

        executable.launch(device.stream());
        device.synchronize();
        failures += expect_value(storage.data(), 0x11111111U, "first graph launch");

        executable.update(second);
        executable.upload(device.stream());
        device.synchronize();
        failures += expect_value(storage.data(), 0x11111111U, "updated graph upload");

        executable.launch(device.stream());
        device.synchronize();
        failures += expect_value(storage.data(), 0x22222222U, "updated graph launch");
        try {
            ninfer::DecodeGraphDefinition abandoned;
            abandoned.capture(device.stream(), [&] {
                CUDA_CHECK(
                    cudaMemsetAsync(storage.data(), 0x44, sizeof(std::uint32_t), device.stream()));
                throw ninfer::DeviceAllocationError("startup test");
            });
            ++failures;
        } catch (const ninfer::DeviceAllocationError&) {}
        ninfer::DecodeGraphDefinition recovered;
        recovered.capture(device.stream(), [&] {
            CUDA_CHECK(
                cudaMemsetAsync(storage.data(), 0x55, sizeof(std::uint32_t), device.stream()));
        });
        executable.instantiate(recovered);
        executable.launch(device.stream());
        device.synchronize();
        failures += expect_value(storage.data(), 0x55555555U, "capture after failed preparation");
        failures += check_external_event(device);
        failures += check_child_graph(device);
        const int probe = probe_update_results(device);
        if (probe < 0) {
            std::cerr << "SKIP: no mutation was refused in place\n";
            return 77;
        }
        failures += check_reinstantiate(device);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "decode graph test failed: " << error.what() << '\n';
        return 1;
    }
}
