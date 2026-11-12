#include "ops/linear/add/dispatch.h"
#include "ops/linear/add/q8/launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Access = Q8ScaleAccess;
using Stage  = Q8ActivationStage;

template <int Capacity, class Schedule>
void launch_sliced(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q8_add_sliced<typename Schedule::template with_problem<8704, Capacity>>(x, w, out, stream);
}
} // namespace

LinearAddLaunch select_q8_linear_add_n5120_k8704(std::int32_t tokens) {
    // mlp/down's fused residual add at tensor-parallel width 2: the 5120x17408 Columns shard halves K, so the rank presents 5120x8704 where width 1 presents 5120x17408. The MTP draft block reaches it, which is why a width-2 MTP round refused here.
    //
    // The ladder is N5120K17408's, and that is a derivation rather than a copy: every band is chosen
    // from (rows / kBlockRows) * ceil(tokens / BlockTokens) against the tuned wave, and that
    // expression mentions neither K nor the row count -- kBlockRows is the schedule family's fixed
    // 16-row tile. K only sets how many 512-wide K tiles each CTA walks (17 here, against the
    // parent's 34), so the bands transfer unchanged. The MTP draft block's hot band is T<=4, which
    // the first band covers.
    if (tokens > 64)
        return launch_q8_add_grouped<
            Q8A16GroupedSlicedKMmaSchedule<128, 2, 4, 1, 1, 8704, Cache::cg, Cache::cg, true>>;
    using C4  = Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Direct, Cache::ca, Cache::cg,
                                        Stage::RuntimeActive>;
    using C8  = Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg,
                                        Stage::RuntimeActive>;
    using C16 = Q8A16SlicedKMmaSchedule<16, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg,
                                        Stage::ActiveOnly>;
    using C24 = Q8A16SlicedKMmaSchedule<24, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg,
                                        Stage::RuntimeActive>;
    using C32 = Q8A16SlicedKMmaSchedule<32, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg,
                                        Stage::RuntimeActive>;
    using C40 = Q8A16SlicedKMmaSchedule<40, 4, 1, 3, Access::Shared, Cache::ca, Cache::cg,
                                        Stage::RuntimeActive>;
    using C48 = Q8A16SlicedKMmaSchedule<48, 4, 1, 3, Access::Shared, Cache::ca, Cache::cg,
                                        Stage::ActiveOnly>;
    using C56 = Q8A16SlicedKMmaSchedule<56, 4, 1, 3, Access::Shared, Cache::ca, Cache::cg,
                                        Stage::ActiveOnly>;
    using C64 = Q8A16SlicedKMmaSchedule<64, 4, 1, 3, Access::Shared, Cache::ca, Cache::cg,
                                        Stage::ActiveOnly>;
    if (tokens <= 4) return launch_sliced<4, C4>;
    if (tokens <= 8) return launch_sliced<8, C8>;
    if (tokens <= 16) return launch_sliced<16, C16>;
    if (tokens <= 24) return launch_sliced<24, C24>;
    if (tokens <= 32) return launch_sliced<32, C32>;
    if (tokens <= 40) return launch_sliced<40, C40>;
    if (tokens <= 48) return launch_sliced<48, C48>;
    if (tokens <= 56) return launch_sliced<56, C56>;
    return launch_sliced<64, C64>;
}
} // namespace ninfer::ops::detail
