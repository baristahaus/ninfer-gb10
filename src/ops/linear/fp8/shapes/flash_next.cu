// Flash-Next attention/GDN projections, router, shared-expert gate/up, HyperConnection Down and
// output head in the optional row-scaled FP8 profile. All routes keep BF16 activations
// (weight-only FP8): one token uses the GEMV, compact batches (MTP verification, small batches)
// the K-split Tensor Core kernel, and prefill the A16 GEMM.
// Schedules were measured on 48 cold weight copies per shape (RTX PRO 6000).
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/flash_next_launch.h"
#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/fp8/fp8_a16_gemm_mma.cuh"
#include "ops/linear/fp8/fp8_a16_ksplit_mma.cuh"
#include "ops/linear/fp8/fp8_gemv.cuh"
#include "ops/linear/fp8/fp8_output.cuh"

namespace ninfer::ops::detail {
namespace {

using Gemv     = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using Prefill  = Fp8A16GemmSchedule<64, 128, 64, 64, 16, 2, 2>;
constexpr int kKSplitMaxTokens = 16;

template <class Geometry>
void launch_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    fp8_gemv_kernel<Geometry, Gemv><<<Geometry::kOutputRows / Gemv::kRowsPerCta, Gemv::kThreads,
                                      0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales),
        Fp8ContiguousOutput{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows});
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, int TileTokens>
void launch_ksplit(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Fp8A16KSplitSchedule<8, TileTokens, 1>;
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);
    fp8_a16_ksplit_mma_kernel<Geometry, TileTokens, Schedule, Fp8ContiguousOutput, true>
        <<<Geometry::kOutputRows / Schedule::kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales),
            Fp8ContiguousOutput{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows},
            x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, bool FullTokens>
void launch_prefill_slice(const Tensor& x, const Weight& weight, Tensor& out,
                          cudaStream_t stream) {
    static_assert((Geometry::kOutputRows % Prefill::kBlockRows) == 0);
    static_assert((Geometry::kInputRows % Prefill::kBlockK) == 0);
    const dim3 grid(static_cast<unsigned>(Geometry::kOutputRows / Prefill::kBlockRows),
                    static_cast<unsigned>(div_up(x.ne[1], Prefill::kBlockTokens)), 1U);
    if constexpr (Prefill::kSharedBytes > 48 * 1024) {
        static const cudaError_t attribute = cudaFuncSetAttribute(
            fp8_a16_gemm_mma_kernel<Geometry, Prefill, FullTokens>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Prefill::kSharedBytes);
        CUDA_CHECK(attribute);
    }
    fp8_a16_gemm_mma_kernel<Geometry, Prefill, FullTokens>
        <<<grid, Prefill::kThreads, Prefill::kSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales),
            Fp8ContiguousOutput{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows},
            x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return launch_gemv<Geometry>(x, weight, out, stream);
    if (tokens <= 8) return launch_ksplit<Geometry, 8>(x, weight, out, stream);
    if (tokens <= kKSplitMaxTokens) return launch_ksplit<Geometry, 16>(x, weight, out, stream);
    for_each_token_slice(tokens, Prefill::kBlockTokens, [&](std::int32_t offset, std::int32_t count) {
        const Tensor input = x.slice(1, offset, count);
        Tensor output      = out.slice(1, offset, count);
        if ((count % Prefill::kBlockTokens) == 0) {
            launch_prefill_slice<Geometry, true>(input, weight, output, stream);
        } else {
            launch_prefill_slice<Geometry, false>(input, weight, output, stream);
        }
    });
}

bool uses_a8(std::int32_t, std::int32_t) { return false; }

template <int N, int K>
constexpr Fp8LinearShape flash_next_shape() {
    return {N, K, launch_a16<Fp8Geometry<N, K>>, nullptr, uses_a8};
}

// CTA-local rows of one shared-expert projection, rounded at the BF16 projection boundary.
struct SharedRows {
    float* rows;
    int base;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t, float value) const {
        rows[parent_row - base] = __bfloat162float(__float2bfloat16_rn(value));
    }
};

constexpr int kMoeHidden       = 2560;
constexpr int kMoeExperts      = 512;
constexpr int kMoeIntermediate = 640;
// Each 256-thread half of a 512-thread CTA runs one eight-row GEMV block. Rows keep the
// production lane/chain/phase accumulation, so results equal the separate GEMVs bit for bit.
using EntryGemv                = Fp8GemvSchedule<8, 1, 8, 4, Fp8CodeCache::Default, 2, 1>;
constexpr int kEntryThreads    = 2 * EntryGemv::kThreads;
constexpr int kRouterBlocks    = kMoeExperts / EntryGemv::kRowsPerCta / 2;
constexpr int kSharedBlocks    = kMoeIntermediate / EntryGemv::kRowsPerCta;

__global__ __launch_bounds__(kEntryThreads, 1) void fp8_moe_entry_decode_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ router_codes,
    const __nv_bfloat16* __restrict__ router_scales, __nv_bfloat16* __restrict__ scores,
    const std::uint8_t* __restrict__ gate_codes, const __nv_bfloat16* __restrict__ gate_scales,
    const std::uint8_t* __restrict__ up_codes, const __nv_bfloat16* __restrict__ up_scales,
    __nv_bfloat16* __restrict__ activation) {
    const int block  = static_cast<int>(blockIdx.x);
    const int half   = static_cast<int>(threadIdx.x) / EntryGemv::kThreads;
    const int thread = static_cast<int>(threadIdx.x) % EntryGemv::kThreads;
    if (block < kRouterBlocks) {
        fp8_gemv_block<Fp8Geometry<kMoeExperts, kMoeHidden>, EntryGemv>(
            2 * block + half, thread, x, router_codes, router_scales,
            Fp8ContiguousOutput{scores, kMoeExperts});
        return;
    }
    using SharedGeometry = Fp8Geometry<kMoeIntermediate, kMoeHidden>;
    __shared__ float projected[2][EntryGemv::kRowsPerCta];  // gate, up
    const int shared_block = block - kRouterBlocks;
    const int base         = shared_block * EntryGemv::kRowsPerCta;
    fp8_gemv_block<SharedGeometry, EntryGemv>(shared_block, thread, x,
                                              half == 0 ? gate_codes : up_codes,
                                              half == 0 ? gate_scales : up_scales,
                                              SharedRows{projected[half], base});
    __syncthreads();
    if (threadIdx.x < EntryGemv::kRowsPerCta) {
        const int row          = static_cast<int>(threadIdx.x);
        activation[base + row] = __float2bfloat16(silu(projected[0][row]) * projected[1][row]);
    }
}

// Compact batches (2-8 tokens): each 512-thread CTA hosts two production K-split blocks with
// their own dynamic shared storage. Router CTAs run two row blocks; shared CTAs run the gate and
// up blocks of the same rows, then apply SwiGLU at the BF16 projection boundary.
using EntrySplit          = Fp8A16KSplitSchedule<8, 8, 1>;
using EntrySplitShared    = Fp8A16KSplitShared<Fp8Geometry<kMoeExperts, kMoeHidden>, EntrySplit>;
using EntrySharedSplit    = Fp8A16KSplitShared<Fp8Geometry<kMoeIntermediate, kMoeHidden>, EntrySplit>;
static_assert(sizeof(EntrySplitShared) == sizeof(EntrySharedSplit));
constexpr int kSplitThreads       = 2 * EntrySplit::kThreads;
constexpr int kSplitRouterBlocks  = kMoeExperts / EntrySplit::kRowsPerCta / 2;
constexpr int kSplitSharedBlocks  = kMoeIntermediate / EntrySplit::kRowsPerCta;
constexpr std::size_t kSplitBytes = 2 * sizeof(EntrySplitShared);

// CTA-local [rows, tokens] values of one shared-expert projection at the BF16 boundary.
struct SharedTile {
    float (*values)[EntrySplit::kTileTokens];
    int base;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        values[parent_row - base][token] = __bfloat162float(__float2bfloat16_rn(value));
    }
};

__global__ __launch_bounds__(kSplitThreads, 1) void fp8_moe_entry_small_t_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ router_codes,
    const __nv_bfloat16* __restrict__ router_scales, __nv_bfloat16* __restrict__ scores,
    const std::uint8_t* __restrict__ gate_codes, const __nv_bfloat16* __restrict__ gate_scales,
    const std::uint8_t* __restrict__ up_codes, const __nv_bfloat16* __restrict__ up_scales,
    __nv_bfloat16* __restrict__ activation, int tokens) {
    extern __shared__ float4 entry_split_storage[];
    const int block  = static_cast<int>(blockIdx.x);
    const int half   = static_cast<int>(threadIdx.x) / EntrySplit::kThreads;
    const int thread = static_cast<int>(threadIdx.x) % EntrySplit::kThreads;
    if (block < kSplitRouterBlocks) {
        auto* shared = reinterpret_cast<EntrySplitShared*>(entry_split_storage) + half;
        fp8_a16_ksplit_block<Fp8Geometry<kMoeExperts, kMoeHidden>, EntrySplit::kTileTokens,
                             EntrySplit, Fp8ContiguousOutput, true>(
            2 * block + half, thread, *shared, x, router_codes, router_scales,
            Fp8ContiguousOutput{scores, kMoeExperts}, tokens);
        return;
    }
    __shared__ float projected[2][EntrySplit::kRowsPerCta][EntrySplit::kTileTokens];  // gate, up
    auto* shared           = reinterpret_cast<EntrySharedSplit*>(entry_split_storage) + half;
    const int shared_block = block - kSplitRouterBlocks;
    const int base         = shared_block * EntrySplit::kRowsPerCta;
    fp8_a16_ksplit_block<Fp8Geometry<kMoeIntermediate, kMoeHidden>, EntrySplit::kTileTokens,
                         EntrySplit, SharedTile, true>(
        shared_block, thread, *shared, x, half == 0 ? gate_codes : up_codes,
        half == 0 ? gate_scales : up_scales, SharedTile{projected[half], base}, tokens);
    __syncthreads();
    const int row   = static_cast<int>(threadIdx.x) % EntrySplit::kRowsPerCta;
    const int token = static_cast<int>(threadIdx.x) / EntrySplit::kRowsPerCta;
    if (token < tokens) {
        activation[static_cast<std::int64_t>(token) * kMoeIntermediate + base + row] =
            __float2bfloat16(silu(projected[0][row][token]) * projected[1][row][token]);
    }
}

} // namespace

void flash_next::launch_fp8_moe_entry_decode(const Tensor& x, const Weight& router,
                                             const Weight& shared_gate, const Weight& shared_up,
                                             Tensor& scores, Tensor& activation,
                                             cudaStream_t stream) {
    const auto fp8 = [](const Weight& weight, int n) {
        return weight.qtype == QType::FP8_E4M3FN_ROW_BF16 && weight.n == n &&
               weight.k == kMoeHidden && weight.scales != nullptr;
    };
    const int tokens = x.ne[1];
    if (tokens < 1 || tokens > kFp8MoeEntryMaxTokens || x.ne[0] != kMoeHidden ||
        !fp8(router, kMoeExperts) || !fp8(shared_gate, kMoeIntermediate) ||
        !fp8(shared_up, kMoeIntermediate) || scores.numel() != kMoeExperts * tokens ||
        activation.numel() != kMoeIntermediate * tokens) {
        throw std::invalid_argument("FP8 MoE entry: invalid exact problem");
    }
    if (tokens > 1) {
        static const cudaError_t configured = cudaFuncSetAttribute(
            fp8_moe_entry_small_t_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(kSplitBytes));
        CUDA_CHECK(configured);
        fp8_moe_entry_small_t_kernel<<<kSplitRouterBlocks + kSplitSharedBlocks, kSplitThreads,
                                       kSplitBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(router.qdata),
            static_cast<const __nv_bfloat16*>(router.scales),
            static_cast<__nv_bfloat16*>(scores.data),
            static_cast<const std::uint8_t*>(shared_gate.qdata),
            static_cast<const __nv_bfloat16*>(shared_gate.scales),
            static_cast<const std::uint8_t*>(shared_up.qdata),
            static_cast<const __nv_bfloat16*>(shared_up.scales),
            static_cast<__nv_bfloat16*>(activation.data), tokens);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    fp8_moe_entry_decode_kernel<<<kRouterBlocks + kSharedBlocks, kEntryThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(router.qdata),
        static_cast<const __nv_bfloat16*>(router.scales), static_cast<__nv_bfloat16*>(scores.data),
        static_cast<const std::uint8_t*>(shared_gate.qdata),
        static_cast<const __nv_bfloat16*>(shared_gate.scales),
        static_cast<const std::uint8_t*>(shared_up.qdata),
        static_cast<const __nv_bfloat16*>(shared_up.scales),
        static_cast<__nv_bfloat16*>(activation.data));
    CUDA_CHECK(cudaGetLastError());
}

const Fp8LinearShape kFp8N12288K2560 = flash_next_shape<12288, 2560>();
const Fp8LinearShape kFp8N10240K2560 = flash_next_shape<10240, 2560>();
const Fp8LinearShape kFp8N6144K2560  = flash_next_shape<6144, 2560>();
const Fp8LinearShape kFp8N2560K6144  = flash_next_shape<2560, 6144>();
const Fp8LinearShape kFp8N512K2560   = flash_next_shape<512, 2560>();
const Fp8LinearShape kFp8N640K2560   = flash_next_shape<640, 2560>();
const Fp8LinearShape kFp8N248320K2560 = flash_next_shape<248320, 2560>();
const Fp8LinearShape kFp8N320K10240  = flash_next_shape<320, 10240>();
} // namespace ninfer::ops::detail
