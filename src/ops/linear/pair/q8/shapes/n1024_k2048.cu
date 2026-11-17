#include "ops/common/cuda/memory.cuh"
#include "ops/linear/pair/dispatch.h"
#include "ops/linear/kernels/q8/q8_geometry.h"
#include "ops/linear/kernels/q8/q8_instances.cuh"
#include "ops/linear/kernels/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/kernels/q8/q8_grouped_sliced_k_launch.cuh"
#include "ops/linear/kernels/q8/q8_mma_launch.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/reduce.cuh"

#include <utility>

namespace ninfer::ops::detail {
namespace {
constexpr int kRows   = 1024;
constexpr int kHidden = 2048;
using PairOutput      = LinearBf16SegmentedOutput<kRows, kRows>;

template <int TileCols, int KSplits, int NGroups, int MinBlocks>
void launch_medium(const Tensor& x, const Weight& first_weight, const Weight&, Tensor& first_out,
                   Tensor& second_out, cudaStream_t stream) {
    const auto* first_codes  = static_cast<const std::uint8_t*>(first_weight.qdata);
    const auto* first_scales = static_cast<const std::uint8_t*>(first_weight.scales);
    const PairOutput output{static_cast<__nv_bfloat16*>(first_out.data),
                            static_cast<__nv_bfloat16*>(second_out.data)};
    launch_q8_a16_grouped_sliced_k_mma<Q8A16GroupedSlicedKMmaSchedule<
        TileCols, KSplits, NGroups, 1, MinBlocks, kHidden, Cache::cg, Cache::cg, false>>(
        Q8LinearOperands{static_cast<const __nv_bfloat16*>(x.data), first_codes, first_scales,
                         ((2 * kRows) / 16) * 16, kHidden, x.ne[1], kHidden},
        output, LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_concat(const Tensor& x, const Weight& first, const Weight&, Tensor& y0, Tensor& y1,
                   cudaStream_t stream) {
    const PairOutput output{static_cast<__nv_bfloat16*>(y0.data),
                            static_cast<__nv_bfloat16*>(y1.data)};
    auto operands = q8_linear_operands(x, first);
    operands.rows = 2 * kRows;
    launch_q8_a16_mma<Schedule>(operands, output, LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_concat_tail(const Tensor& x, const Weight& first, const Weight& second, Tensor& y0,
                        Tensor& y1, cudaStream_t stream) {
    const int prefix = x.ne[1] / Schedule::kBlockTokens * Schedule::kBlockTokens;
    const int tail   = x.ne[1] - prefix;
    if (prefix <= 0 || tail < 1 || tail > 62)
        throw std::logic_error("linear_pair: invalid tuned prefix/tail decomposition");
    const auto prefix_x = x.slice(1, 0, prefix);
    auto prefix_y0      = y0.slice(1, 0, prefix);
    auto prefix_y1      = y1.slice(1, 0, prefix);
    launch_concat<Schedule>(prefix_x, first, second, prefix_y0, prefix_y1, stream);
    const auto tail_x = x.slice(1, prefix, tail);
    auto tail_y0      = y0.slice(1, prefix, tail);
    auto tail_y1      = y1.slice(1, prefix, tail);
    select_q8_pair_n1024_k2048(tail)(tail_x, first, second, tail_y0, tail_y1, stream);
}
} // namespace

LinearPairLaunch select_q8_pair_n1024_k2048(int tokens) {
    if (tokens <= 48) return launch_medium<48, 4, 2, 3>;
    if (tokens <= 64) return launch_medium<64, 4, 2, 2>;
    if (tokens <= 80) return launch_medium<80, 4, 2, 1>;
    if (tokens <= 88) return launch_medium<88, 4, 1, 1>;
    if (tokens <= 96) return launch_medium<96, 4, 1, 1>;
    if (tokens <= 104) return launch_medium<104, 4, 1, 1>;
    if (tokens <= 112) return launch_medium<112, 4, 1, 1>;
    if (tokens <= 128) return launch_medium<128, 2, 4, 2>;
    if (tokens <= 160) return launch_medium<160, 2, 5, 2>;
    if (tokens <= 192) return launch_medium<192, 2, 6, 2>;
    if (tokens <= 384) return launch_concat<Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>>;
    if (tokens <= 480) return launch_concat<Q8A16MmaSchedule<32, 96, 64, 32, 16, 2, 2>>;
    if (tokens <= 640) return launch_concat<Q8A16MmaSchedule<32, 128, 64, 32, 16, 2, 2>>;
    if (tokens <= 641) return launch_concat_tail<Q8A16MmaSchedule<32, 128, 64, 32, 16, 2, 2>>;
    if (tokens <= 672) return launch_concat<Q8A16MmaSchedule<48, 96, 64, 48, 16, 2, 2>>;
    if (tokens <= 680) return launch_concat_tail<Q8A16MmaSchedule<32, 96, 64, 32, 16, 2, 2>>;
    if (tokens <= 784) return launch_concat<Q8A16MmaSchedule<48, 112, 64, 48, 16, 2, 2>>;
    if (tokens <= 896) return launch_concat<Q8A16MmaSchedule<48, 128, 64, 48, 16, 2, 2>>;
    if (tokens <= 960) return launch_concat<Q8A16MmaSchedule<96, 64, 64, 48, 16, 2, 2>>;
    if (tokens <= 976) return launch_concat_tail<Q8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>;
    if (tokens <= 1280) return launch_concat<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
    if (tokens <= 1316) return launch_concat_tail<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
    if (tokens <= 1344) return launch_concat<Q8A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>;
    if (tokens <= 1345) return launch_concat_tail<Q8A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>;
    if (tokens <= 1440) return launch_concat<Q8A16MmaSchedule<96, 96, 64, 48, 16, 2, 2>>;
    if (tokens <= 1466) return launch_concat_tail<Q8A16MmaSchedule<96, 96, 64, 48, 16, 2, 2>>;
    if (tokens <= 1680) return launch_concat<Q8A16MmaSchedule<128, 80, 64, 64, 16, 2, 2>>;
    if (tokens <= 1708) return launch_concat_tail<Q8A16MmaSchedule<128, 80, 64, 64, 16, 2, 2>>;
    if (tokens <= 1920) return launch_concat<Q8A16MmaSchedule<48, 128, 64, 48, 16, 2, 2>>;
    if (tokens <= 1922) return launch_concat_tail<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
    if (tokens <= 2016) return launch_concat<Q8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>;
    if (tokens <= 2018) return launch_concat_tail<Q8A16MmaSchedule<64, 96, 64, 64, 16, 2, 2>>;
    if (tokens <= 2208) return launch_concat<Q8A16MmaSchedule<96, 96, 64, 48, 16, 2, 2>>;
    if (tokens <= 2270) return launch_concat_tail<Q8A16MmaSchedule<96, 96, 64, 48, 16, 2, 2>>;
    return launch_concat<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}
} // namespace ninfer::ops::detail
