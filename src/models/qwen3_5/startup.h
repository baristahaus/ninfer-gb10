#pragma once

#include "models/common/startup.h"
#include "models/qwen3_5/load.h"

#include <span>
#include <vector>
#include "models/qwen3_5/runtime_types.h"
#include "models/qwen3_5/program/graph_preparation.h"

namespace ninfer::models::qwen3_5 {

struct ExecutionPreparation {
    std::shared_ptr<const moe::Plan> experts;
};

// Native loading and execution planning. Runtime owns the budget, calibration loop and retries.
class Startup {
public:
    using Types         = RuntimeTypes;
    using ExecutionPlan = models::ExecutionPlan<SequencePlan, ExecutionPreparation>;
    using Calibration   = GraphCalibration;

    Startup(const artifact::Reader& definition_source, const artifact::Reader& rank_source,
            const EngineOptions&, Device);

    [[nodiscard]] Architecture architecture() const noexcept { return architecture_; }

    // `published_tokens` pins the capacity to a group decision instead of resolving one here. It is
    // the seam that lets both ranks of a group plan at one capacity: the caller decides once, then
    // every rank is planned at the same count.
    [[nodiscard]] std::optional<ExecutionPlan>
    plan(DeviceContext&, const EngineOptions&, WeightPlacement, std::size_t available,
         std::optional<std::uint32_t> published_tokens = std::nullopt) const;
    // The curve plan() would resolve against, without finalizing a sequence. A group decision needs
    // every rank's curve before either Program exists.
    [[nodiscard]] runtime::SequenceCapacityCurve capacity_curve(DeviceContext&,
                                                                const EngineOptions&,
                                                                WeightPlacement) const;
    // `rank` selects which of `rank_definitions()` the materialization reads, so a group's rank
    // materializes its own bindings on its own device. Rank 0 is the local rank and the default, so
    // every existing caller is unchanged.
    [[nodiscard]] std::unique_ptr<ModelShard> materialize(DeviceContext&, WeightPlacement,
                                                          const StartupObserver&, int rank = 0);
    [[nodiscard]] qwen::Frontend frontend(const EngineOptions&) const;
    [[nodiscard]] GraphPreparationPlan graph_plan(const EngineOptions&,
                                                  const ExecutionPreparation&) const;
    [[nodiscard]] Calibration calibrate(const execution::Parameters&, DeviceContext&,
                                        const EngineOptions&, const ExecutionPreparation&) const;

    [[nodiscard]] std::span<const WeightCandidate> placements() const { return placements_; }

    [[nodiscard]] std::string prefill_signature(const ExecutionPreparation&,
                                                std::uint32_t chunk) const;

    [[nodiscard]] const std::shared_ptr<const ModelDefinition>& definition() const {
        return definition_;
    }

    // One definition per rank of the load's execution group, rank 0 first. Rank 0 is the local
    // rank and is what `definition()` returns; a rank above 0 holds that rank's bindings, read with
    // the model's shard transform installed, and is what the peer rank's materialization needs.
    // Width 1 holds exactly one entry, so the single-device lane is unchanged.
    [[nodiscard]] std::span<const std::shared_ptr<const ModelDefinition>> rank_definitions() const {
        return rank_definitions_;
    }

    std::shared_ptr<const execution::ModelProfile> profile;

    // The rows of one column-parallel partial — the residual stream's width — or nullopt for a
    // family with no two-GPU lane. The startup reads this to size the shared partial pair; the token
    // capacity comes from the resolved plan, which every family has.
    [[nodiscard]] std::optional<std::int32_t> sharded_partial_rows() const {
        return profile != nullptr ? std::optional<std::int32_t>(profile->config.text.hidden_size)
                                  : std::nullopt;
    }

private:
    std::vector<WeightCandidate> placements_;
    std::shared_ptr<const moe::Plan> offload_;
    const artifact::Reader& reader_;
    std::shared_ptr<const ModelDefinition> definition_;
    std::vector<std::shared_ptr<const ModelDefinition>> rank_definitions_;
    Architecture architecture_;
};

} // namespace ninfer::models::qwen3_5
