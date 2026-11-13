// The single instantiation point for the add/q5 schedules that more than one shape file selects.
// See launch.cuh for why these are declared extern there: two shape files selecting the same
// schedule used to emit the same __global__ kernel in both objects, which the module registered
// twice. The arithmetic is unchanged; only where the instantiation lives moved.

#include "ops/linear/add/q5/launch.cuh"

namespace ninfer::ops::detail {

template void launch_q5_add_sliced<q5_instances::SlicedR16T8W4S2>(
    const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream);
template void launch_q5_add_sliced<q5_instances::SlicedR16T16W4S2>(
    const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream);
template void launch_q5_add_sliced<q5_instances::SlicedR16T24W4S2>(
    const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream);
template void launch_q5_add_sliced<q5_instances::SlicedR32T24W4S2Pairwise>(
    const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream);
template void launch_q5_add_sliced<q5_instances::SlicedR32T32W2S2>(
    const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream);
template void launch_q5_add_sliced<q5_instances::SlicedR32T32W4S2>(
    const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream);
template void launch_q5_add_sliced<q5_instances::SlicedR32T64W2S1>(
    const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream);
template void launch_q5_add_collective<q5_instances::MmaR64T128>(
    const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream);
template void launch_q5_add_pointwise<q5_instances::MmaR32T128>(
    const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream);

} // namespace ninfer::ops::detail
