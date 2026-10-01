// Implements: include/ninfer/ops/flash_next_ple_stage.h
#include "ninfer/ops/flash_next_ple_stage.h"

#include "core/device.h"

#include <cuda/atomic>

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kEosToken = 248044;

__constant__ unsigned long long kMultipliers[3] = {
    23703573157769ULL,
    20109073645365ULL,
    8052911324071ULL,
};
__constant__ std::uint32_t kSizes[kFlashNextPleStageHeads] = {
    20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
    20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
};
__constant__ std::uint32_t kOffsets[kFlashNextPleStageHeads] = {
    0,         20000003,  40000026,  60000059,  80000106,  100000165, 120000228, 140000297,
    160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275,
};

__device__ unsigned long long product(std::int32_t token, unsigned long long multiplier) {
    return static_cast<unsigned long long>(static_cast<long long>(token)) * multiplier;
}

__global__ void publish_ids_kernel(const std::int32_t* tokens, const std::int32_t* history,
                                   FlashNextPleStageMailbox* mailbox, std::int32_t width,
                                   std::int32_t batch) {
    const int count = width * batch;
    for (int item = static_cast<int>(threadIdx.x); item < count * kFlashNextPleStageHeads;
         item += static_cast<int>(blockDim.x)) {
        const int token_index          = item / kFlashNextPleStageHeads;
        const int head                 = item % kFlashNextPleStageHeads;
        const int row                  = token_index / width;
        const int column               = token_index % width;
        const std::int32_t* row_tokens = tokens + row * width;
        // Position p of the row's extended sequence: p >= 0 is a column, -1/-2 the history.
        const auto at = [&](int p) { return p >= 0 ? row_tokens[p] : history[row * 2 + (-p - 1)]; };
        const std::int32_t current  = row_tokens[column];
        const std::int32_t previous = at(column - 1);
        const std::int32_t before   = at(column - 2);
        const bool has_previous     = previous >= 0;
        const std::int32_t shifted1 = has_previous ? previous : kEosToken;
        const std::int32_t shifted2 =
            has_previous && previous != kEosToken && before >= 0 ? before : kEosToken;
        const unsigned long long bigram =
            product(current, kMultipliers[0]) ^ product(shifted1, kMultipliers[1]);
        const unsigned long long hashed =
            head < 8 ? bigram : bigram ^ product(shifted2, kMultipliers[2]);
        const auto modulus  = static_cast<long long>(kSizes[head]);
        long long remainder = static_cast<long long>(hashed) % modulus;
        if (remainder < 0) { remainder += modulus; }
        mailbox->ids[item] = kOffsets[head] + static_cast<std::uint32_t>(remainder);
    }
    // Every writer orders its IDs at system scope before the barrier, so the host never sees
    // the new sequence ahead of any ID.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        mailbox->tokens = count;
        cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> request(
            mailbox->request_sequence);
        const std::uint32_t sequence = request.load(cuda::memory_order_relaxed) + 1U;
        request.store(sequence, cuda::memory_order_release);
    }
}

__global__ void wait_staged_kernel(FlashNextPleStageMailbox* mailbox,
                                   unsigned long long timeout_ns) {
    cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> request(mailbox->request_sequence);
    cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> answer(mailbox->answer_sequence);
    const std::uint32_t expected = request.load(cuda::memory_order_relaxed);
    unsigned long long start     = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    while (answer.load(cuda::memory_order_acquire) != expected) {
        unsigned long long now = 0;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
        if (now - start > timeout_ns) {
            cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> late(mailbox->late);
            late.store(1U, cuda::memory_order_release);
            return;
        }
        __nanosleep(256);
    }
}

} // namespace

void flash_next_ple_publish_ids(const Tensor& tokens, const Tensor& history,
                                FlashNextPleStageMailbox* mailbox, cudaStream_t stream) {
    const std::int32_t width = tokens.ne[0];
    const std::int32_t batch = tokens.ne[1];
    if (mailbox == nullptr || tokens.dtype != DType::I32 || history.dtype != DType::I32 ||
        width < 1 || batch < 1 || tokens.ne[2] != 1 || tokens.ne[3] != 1 ||
        width * batch > kFlashNextPleStageMaxTokens || !tokens.is_contiguous() ||
        history.ne[0] != 2 || history.ne[1] != batch || history.ne[2] != 1 || history.ne[3] != 1 ||
        !history.is_contiguous() || tokens.data == nullptr || history.data == nullptr) {
        throw std::invalid_argument("flash_next_ple_publish_ids: invalid tokens or history");
    }
    publish_ids_kernel<<<1, 256, 0, stream>>>(static_cast<const std::int32_t*>(tokens.data),
                                              static_cast<const std::int32_t*>(history.data),
                                              mailbox, width, batch);
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_ple_wait_staged(FlashNextPleStageMailbox* mailbox, std::uint64_t timeout_ns,
                                cudaStream_t stream) {
    if (mailbox == nullptr || timeout_ns == 0) {
        throw std::invalid_argument("flash_next_ple_wait_staged: invalid mailbox or timeout");
    }
    wait_staged_kernel<<<1, 1, 0, stream>>>(mailbox, timeout_ns);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
