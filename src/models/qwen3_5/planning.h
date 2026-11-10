#pragma once

#include "core/layout.h"

#include "models/common/context/directory.h"
#include "models/qwen3_5/execution/profile.h"
#include "models/common/qwen/vision/vision.h"
#include "ninfer/types.h"
#include "runtime/contract/resources.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::models::moe {
struct Plan;
}

namespace ninfer::models::qwen3_5::execution {
// The group-scoped resources of a two-GPU lane; defined in execution/sharded.h.
struct ShardedGroup;
}

namespace ninfer::models::qwen3_5 {

using ContextResources    = runtime::ContextResources<2>;
using ResourceReservation = runtime::ResourceReservation<ContextResources>;
inline constexpr runtime::KvPoolId kTextKvPool{0};
inline constexpr runtime::KvPoolId kBackendKvPool{1};
inline constexpr std::uint32_t kPrefillChunkAlignment     = 128;
inline constexpr std::uint32_t kMtpDecodeMaximumDrafts    = 5;
inline constexpr std::uint32_t kMtpDecodeMaximumWidth     = kMtpDecodeMaximumDrafts + 1;
inline constexpr std::uint32_t kDFlashDecodeMaximumDrafts = 15;
inline constexpr std::uint32_t kDFlashDecodeMaximumWidth  = kDFlashDecodeMaximumDrafts + 1;

namespace execution {
class Parameters;
}

namespace detail {
struct SequencePlanImpl;
struct SequencePlannerImpl;
} // namespace detail

class SequencePlanner;
class PreparedSession;
class RankProgram;

enum class TextPhase { Prefill, Verify };

struct SessionPlan {
    std::shared_ptr<const execution::ModelProfile> profile;
    context::ContextDirectorySpec context;
    std::vector<runtime::KvPoolDescriptor> kv_pools;
    std::uint32_t capacity                 = 0;
    std::uint32_t max_concurrency          = 1;
    std::uint32_t prefill_chunk            = 0;
    std::uint32_t draft_window             = 0;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    ContextCacheOptions context_cache;
    KvCacheStorage kv_storage  = KvCacheStorage::BFloat16;
    ProposalHead proposal_head = ProposalHead::Full;
    TransferWork state_transfer;
    std::optional<LayoutRegion> dflash_local_group;
    std::optional<TransferWork> dflash_local_transfer;
    std::optional<qwen::VisionWorkspacePlan> vision_workspace;
    bool causal_scoring = false;
};

class SequencePlan {
public:
    SequencePlan(SequencePlan&&) noexcept;
    SequencePlan& operator=(SequencePlan&&) noexcept;
    ~SequencePlan();

    SequencePlan(const SequencePlan&)            = delete;
    SequencePlan& operator=(const SequencePlan&) = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t kv_capacity() const noexcept;
    [[nodiscard]] std::uint32_t max_concurrency() const noexcept;
    [[nodiscard]] std::size_t device_reservation_bytes() const noexcept;
    [[nodiscard]] std::size_t workspace_capacity_bytes() const noexcept;
    [[nodiscard]] std::size_t host_capacity_bytes() const noexcept;

private:
    explicit SequencePlan(std::unique_ptr<detail::SequencePlanImpl>) noexcept;
    std::unique_ptr<detail::SequencePlanImpl> impl_;

    friend class SequencePlanner;
    friend class RankProgram;
    friend SessionPlan plan_session(const SequencePlan&);
    friend std::unique_ptr<RankProgram> create_rank_program(const execution::Parameters&,
                                                            SequencePlan&&, DeviceContext&,
                                                            const StartupObserver&,
                                                            const execution::ShardedGroup*);
    friend void attach_rank_mirrors(RankProgram&, RankProgram&, DeviceContext&,
                                    PreparedSession*);
};

class SequencePlanner {
public:
    SequencePlanner(SequencePlanner&&) noexcept;
    SequencePlanner& operator=(SequencePlanner&&) noexcept;
    ~SequencePlanner();

    SequencePlanner(const SequencePlanner&)            = delete;
    SequencePlanner& operator=(const SequencePlanner&) = delete;

    [[nodiscard]] const runtime::SequenceCapacityCurve& capacity_curve() const noexcept;
    [[nodiscard]] SequencePlan finalize(std::uint32_t main_page_groups) &&;

private:
    explicit SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl>) noexcept;
    std::unique_ptr<detail::SequencePlannerImpl> impl_;

    friend SequencePlanner make_sequence_planner(std::shared_ptr<const execution::ModelProfile>,
                                                 DeviceContext&, const EngineOptions&,
                                                 std::shared_ptr<const moe::Plan>);
};

[[nodiscard]] SessionPlan plan_session(const SequencePlan&);
[[nodiscard]] SequencePlanner make_sequence_planner(std::shared_ptr<const execution::ModelProfile>,
                                                    DeviceContext&, const EngineOptions&,
                                                    std::shared_ptr<const moe::Plan> = {});
[[nodiscard]] SequencePlanner make_sequence_planner(const execution::Parameters&, DeviceContext&,
                                                    const EngineOptions&);
// `sharded` is the group-scoped resource bundle of a two-GPU lane — the one mailbox and the one
// partial pair the whole group shares. Null on the single-device lane. A Program at width > 1
// refuses to run without it rather than creating its own: a mailbox per Program would never
// rendezvous and a partial pair per Program would diverge silently.
[[nodiscard]] std::unique_ptr<RankProgram> create_rank_program(
    const execution::Parameters&, SequencePlan&&, DeviceContext&, const StartupObserver&,
    const execution::ShardedGroup* sharded = nullptr);

// tp 2: attach `peer`'s KV page pool and execution tables to `origin`'s, so every physical mutation
// rank 0 issues is replayed on rank 1's stream. Called once, after both Programs exist and before
// either has run a round. Throws std::invalid_argument when the two ranks' KV capacity differs,
// because the mirror addresses rank 0's pages by index and unequal capacity is a configuration
// error rather than something to paper over.
void attach_rank_mirrors(RankProgram& origin, RankProgram& peer, DeviceContext& peer_device,
                         PreparedSession* session = nullptr);

} // namespace ninfer::models::qwen3_5
