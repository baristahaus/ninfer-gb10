#include "ninfer/ops/rmsnorm.h"
#include "core/device.h"
#include "ops/normalization/rmsnorm/dispatch.h"

namespace ninfer::ops {
namespace norm = detail::rmsnorm;

namespace {
void execute(const Tensor& x, const Tensor& weight, const Tensor* gate, float eps,
             norm::Epilogue kind, Tensor& out, Device device, int multiprocessor_count,
             cudaStream_t stream, const char* op) {
    const auto operands = norm::validate(x, weight, gate, eps, out, kind, device, op);
    if (operands.rows)
        norm::dispatch(x, weight, gate, eps, kind, out, operands, multiprocessor_count, stream);
}

int current_multiprocessor_count() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int multiprocessors = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
    return multiprocessors;
}
} // namespace

void rmsnorm(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, Tensor& out,
             CudaExecutionView execution) {
    execute(x, weight, nullptr, eps, unit_offset ? norm::Epilogue::Offset : norm::Epilogue::Plain,
            out, execution.device(), execution.multiprocessor_count(), execution.stream(),
            "rmsnorm");
}

void silu_gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& gate, float eps,
                        Tensor& out, CudaExecutionView execution) {
    execute(x, weight, &gate, eps, norm::Epilogue::SiluGated, out, execution.device(),
            execution.multiprocessor_count(), execution.stream(), "silu_gated_rmsnorm");
}

void sigmoid_gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& gate, float eps,
                           Tensor& out, CudaExecutionView execution) {
    execute(x, weight, &gate, eps, norm::Epilogue::SigmoidGated, out, execution.device(),
            execution.multiprocessor_count(), execution.stream(), "sigmoid_gated_rmsnorm");
}

// Stream-based forms for callers that own only a stream (the Flash-Next GDN block and its
// model runtime). Validation pins the operands to the x tensor's device; the SM count comes
// from the device the caller has current when entering the op.
void rmsnorm(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, Tensor& out,
             cudaStream_t stream) {
    execute(x, weight, nullptr, eps,
            unit_offset ? norm::Epilogue::Offset : norm::Epilogue::Plain, out, x.device,
            current_multiprocessor_count(), stream, "rmsnorm");
}

void sigmoid_gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& gate, float eps,
                           Tensor& out, cudaStream_t stream) {
    execute(x, weight, &gate, eps, norm::Epilogue::SigmoidGated, out, x.device,
            current_multiprocessor_count(), stream, "sigmoid_gated_rmsnorm");
}

void gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& z, float eps, Tensor& out,
                   cudaStream_t stream) {
    execute(x, weight, &z, eps, norm::Epilogue::SiluGated, out, x.device,
            current_multiprocessor_count(), stream, "gated_rmsnorm");
}
} // namespace ninfer::ops
