#include "ninfer/ops/linear.h"

#include <cublas_v2.h>

#include <sstream>
#include <stdexcept>

namespace ninfer::ops {
namespace {

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status == CUBLAS_STATUS_SUCCESS) { return; }
    std::ostringstream message;
    message << operation << " failed with cuBLAS status " << static_cast<int>(status);
    throw std::runtime_error(message.str());
}

} // namespace

struct Bf16GemmContext::Impl {
    cublasHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;
};

Bf16GemmContext::Bf16GemmContext(cudaStream_t stream) : impl_(std::make_unique<Impl>()) {
    if (stream == nullptr) { throw std::invalid_argument("BF16 GEMM context requires a stream"); }
    check_cublas(cublasCreate(&impl_->handle), "cublasCreate");
    try {
        check_cublas(cublasSetStream(impl_->handle, stream), "cublasSetStream");
        check_cublas(cublasSetMathMode(impl_->handle, CUBLAS_TENSOR_OP_MATH),
                     "cublasSetMathMode");
        check_cublas(cublasSetWorkspace(impl_->handle, nullptr, 0), "cublasSetWorkspace");
        impl_->stream = stream;
    } catch (...) {
        (void)cublasDestroy(impl_->handle);
        throw;
    }
}

Bf16GemmContext::~Bf16GemmContext() {
    if (impl_ && impl_->handle != nullptr) { (void)cublasDestroy(impl_->handle); }
}

void Bf16GemmContext::launch(const Tensor& x, const Weight& w, Tensor& out,
                             cudaStream_t stream) {
    if (stream != impl_->stream) {
        throw std::invalid_argument("BF16 GEMM context used with a different stream");
    }
    const float alpha = 1.0F;
    const float beta = 0.0F;
    check_cublas(cublasGemmEx(impl_->handle, CUBLAS_OP_T, CUBLAS_OP_N, w.n, x.ne[1], w.k,
                              &alpha, w.qdata, CUDA_R_16BF, w.k, x.data, CUDA_R_16BF, w.k,
                              &beta, out.data, CUDA_R_16BF, w.n, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                 "cublasGemmEx");
}

} // namespace ninfer::ops
