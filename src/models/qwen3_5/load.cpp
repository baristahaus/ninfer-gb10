#include "models/qwen3_5/load.h"

#include "artifact/encoded_slice.h"

#include "models/common/bindings.h"

#include <cstdio>
#include <cstdlib>

namespace ninfer::models::qwen3_5 {

ResolvedLoadPlan::ResolvedLoadPlan(const artifact::Reader& reader,
                                   std::shared_ptr<const ModelDefinition> definition,
                                   artifact::MaterializationPlan materialization,
                                   std::vector<Device> devices, std::vector<Device> bank_devices)
    : reader_(&reader), definition_(std::move(definition)),
      materialization_(std::move(materialization)), devices_(std::move(devices)),
      bank_devices_(std::move(bank_devices)) {}


namespace {

// The slices a sharded reference describes. A whole-object reference keeps the `whole_tensor` form
// (it is exactly one object, unsliced), so the single-device lane never enters this. A sharded one
// gives parts that are contiguous element ranges of the parent: a row block is one slice, a column
// prefix is one slice per row. Destination rows accumulate in part order over the rank's own shape.
std::vector<artifact::MatrixSlice> matrix_slices(const artifact::Reader& reader,
                                                 const artifact::ParameterReference& reference) {
    std::vector<artifact::MatrixSlice> slices;
    // A 1-D parameter has one column and `shape[0]` rows; a matrix has `shape[1]` columns and
    // `shape[0]` rows. Reading `shape.back()` as the column count would make a 1-D parameter look
    // like a single row of `shape[0]` columns, so the coverage check below would compare one row
    // against the whole extent.
    const auto k = reference.shape.size() > 1 ? reference.shape.back() : 1;
    const auto n = weight_element_count(reference.shape) / k;
    std::uint64_t destination_row    = 0;
    std::uint64_t destination_column = 0;
    bool saw_column_slice            = false;
    for (std::size_t next = 0; next < reference.binding.parts.size(); ++next) {
        const auto& part   = reference.binding.parts[next];
        const auto extent = part.end - part.begin;
        if (extent == 0) { continue; }
        const auto parent = reader.geometry(part.object).shape;
        if (parent.size() == 1) {
            // A 1-D parameter is a single row of `shape[0]` columns as far as the slicer is
            // concerned (`slice_tensor` derives its row count as `elements / columns`), so a
            // sharded range is a COLUMN slice of that one row, and the destination accumulates in
            // destination columns rather than rows. Reading it as `extent` rows would claim more
            // rows than the parent has.
            slices.push_back({part.object, 0, part.begin, 1, extent, 0, destination_column});
            destination_column += extent;
            // The rank's own extent is the coverage this form needs; `n` already equals it for a
            // 1-D parameter, and its destination column is an accumulated offset rather than a
            // column of a matrix, so it is not the check the matrix path makes.
            destination_row = n;
            continue;
        }
        if (parent.size() != 2) {
            throw artifact::ArtifactError(reference.name + ": a sharded weight must be a matrix");
        }
        const std::uint64_t columns = parent.back();
        const std::uint64_t row     = part.begin / columns;
        const std::uint64_t column  = part.begin % columns;
        if (extent % columns == 0) {
            const std::uint64_t rows = extent / columns;
            slices.push_back({part.object, row, column, rows, columns, destination_row, 0});
            destination_row += rows;
        } else {
            // A column shard: the parts are one row-length range per row, grouped by segment
            // (segment 0's rows, then segment 1's). Emit ONE slice per segment covering every row,
            // not one slice per row: a quantized format's scale block spans rows, so a per-row
            // slice splits it and the slicer refuses it outright. The destination column is the
            // segment's own offset within the rank's columns.
            const std::uint64_t segment_columns = extent;
            std::uint64_t rows                  = 1;
            // Start at the part AFTER this one: the outer loop has already counted this part, so
            // starting at `next` would count it twice and emit one row too many.
            // `next` advances only to a part actually consumed: setting it before the break test
            // skipped the first part of the next segment, which is how the groups came out three
            // rows instead of four.
            std::size_t last_consumed = next;
            for (std::size_t j = next + 1; j < reference.binding.parts.size(); ++j) {
                const auto& following = reference.binding.parts[j];
                if (following.end <= following.begin) { break; }
                if (following.begin % columns != column) { break; }
                if (following.end - following.begin != extent) { break; }
                ++rows;
                last_consumed = j;
            }
            next = last_consumed;
            slices.push_back({part.object, 0, column, rows, segment_columns, 0,
                              destination_column});
            destination_column += segment_columns;
            saw_column_slice = true;
            destination_row = n;
        }
    }
    // A row-block shard never touches destination columns, so the column check applies only when
    // the shard actually sliced columns; applying it unconditionally would refuse every row shard.
    const auto columns_of_rank = reference.shape.size() > 1 ? reference.shape[1] : 1;
    if (destination_row != n || (saw_column_slice && destination_column != columns_of_rank)) {
        throw artifact::ArtifactError(reference.name + ": shard slices do not cover its own rows");
    }
    return slices;
}

} // namespace

ResolvedLoadPlan resolve_load(const artifact::Reader& reader,
                              std::shared_ptr<const ModelDefinition> definition, Device cuda,
                              std::optional<Device> main_experts) {
    if (!definition || definition->artifact_id() != reader.artifact_id())
        throw artifact::ArtifactError("model definition belongs to another artifact");
    const auto expert_device = main_experts.value_or(cuda);
    if (cuda.type() != DeviceType::CUDA ||
        (expert_device != cuda && expert_device != Device::cpu()))
        throw artifact::ArtifactError("weight placement requires the selected GPU or CPU");
    std::vector<Device> devices(definition->weight_data().size(), cuda);
    std::vector<Device> banks(definition->expert_banks().size(), cuda);
    for (const auto& layer : definition->weights().text.layers) {
        if (const auto* moe = std::get_if<MoeWeights>(&layer.ffn)) {
            banks[moe->bank.index] = expert_device;
            for (const auto& expert : moe->experts)
                for (const auto id : {expert.gate, expert.up, expert.down})
                    devices[id.index] = expert_device;
        }
    }
    // Place slices, not objects. A `PlacementRequest` names an object and the materializer places
    // it whole (`placement.cpp:61`), so a rank would plan and allocate the entire artifact at any
    // width; the reference's sliced binding has to reach the plan. A whole-object reference keeps
    // `whole_tensor`, which is why width 1 takes exactly the path it took before.
    std::vector<artifact::TensorRequest> requests;
    for (std::size_t i = 0; i < devices.size(); ++i) {
        const auto& reference = definition->parameter({i});
        const auto access     = devices[i] == Device::cpu() ? artifact::HostAccess::CudaTransfer
                                                            : artifact::HostAccess::CpuOnly;
        if (reference.binding.whole_object) {
            for (const auto& part : reference.binding.parts) {
                requests.push_back({artifact::whole_tensor(reader, part.object), devices[i],
                                    access});
            }
            continue;
        }
        const auto slices = matrix_slices(reader, reference);
        if (std::getenv("NINFER_DEBUG_SLICES") != nullptr) {
            std::fprintf(stderr, "[slices] %s rank_shape=%llu,%llu n=%zu\n", reference.name.c_str(),
                         (unsigned long long)(reference.shape.size() > 0 ? reference.shape[0] : 0),
                         (unsigned long long)(reference.shape.size() > 1 ? reference.shape[1] : 0),
                         slices.size());
            for (const auto& sl : slices) {
                std::fprintf(stderr, "  obj=%d row=%llu col=%llu rows=%llu cols=%llu drow=%llu dcol=%llu\n",
                             sl.object.index, (unsigned long long)sl.row, (unsigned long long)sl.column,
                             (unsigned long long)sl.rows, (unsigned long long)sl.columns,
                             (unsigned long long)sl.destination_row, (unsigned long long)sl.destination_column);
            }
        }
        try {
            requests.push_back({artifact::slice_tensor(reader, reference.shape, slices), devices[i],
                                access});
        } catch (const std::exception& error) {
            // The slicer has no parameter name, so a rejection there is anonymous. Name it here.
            throw artifact::ArtifactError(reference.name + ": " + error.what());
        }
    }
    auto storage = artifact::plan_materialization(reader, requests);
    return {reader, std::move(definition), std::move(storage), std::move(devices),
            std::move(banks)};
}

std::unique_ptr<ModelShard> materialize_model(ResolvedLoadPlan&& plan, DeviceContext& device,
                                              const StartupObserver* observer) {
    if (!plan.definition_) throw artifact::ArtifactError("load plan was already consumed");
    auto definition = std::move(plan.definition_);
    auto backing    = artifact::materialize(*plan.reader_, std::move(plan.materialization_),
                                            device.transfer_view(), observer);
    std::vector<PendingWeight> pending;
    pending.reserve(definition->weight_data().size());
    for (std::size_t i = 0; i < definition->weight_data().size(); ++i) {
        const auto& description = definition->weight_data()[i];
        pending.push_back(
            {definition->parameter({i}), description.uses, description.source_objects});
    }
    auto bound = resolve_weights(std::move(pending), backing, plan.devices_);
    // A column-sharded weight is many small ranges in its parent, which no consumer can take; the
    // gather turns each into one contiguous parent. The condition is structural (non-contiguous
    // parts in one parent), so width 1 never enters it and needs no guard.
    RepackedWeights repacked;
    repacked.gather(bound);
    auto banks = moe::bind_banks(definition->expert_banks(), backing, plan.bank_devices_);
    return std::unique_ptr<ModelShard>(new ModelShard(std::move(definition), std::move(backing),
                                                      std::move(bound), std::move(banks),
                                                      std::move(repacked)));
}

} // namespace ninfer::models::qwen3_5
