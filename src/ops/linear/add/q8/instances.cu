// The single instantiation point for the add/q8 MMA schedules that both n2048 shape files select.
// See launch.cuh for why these are declared extern there: two shape files selecting the same
// schedule used to emit the same __global__ kernel in both objects, which the module registered
// twice. The arithmetic is unchanged; only where the instantiation lives moved.

#include "ops/linear/add/q8/launch.cuh"

namespace ninfer::ops::detail {

template void launch_q8_add_mma<Q8A16MmaSchedule<32, 128, 64, 32, 16, 2, 2>>(
    const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream);
template void launch_q8_add_mma<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(
    const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream);

} // namespace ninfer::ops::detail
