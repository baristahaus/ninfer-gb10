#include "models/qwen3_5/load.h"

#include "artifact/encoded_slice.h"

#include "models/common/bindings.h"

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
    const auto k = reference.shape.back();
    const auto n = weight_element_count(reference.shape) / k;
    std::uint64_t destination_row = 0;
    for (const auto& part : reference.binding.parts) {
        const auto extent = part.end - part.begin;
        if (extent == 0) { continue; }
        const auto parent = reader.geometry(part.object).shape;
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
            slices.push_back({part.object, row, column, 1, extent, destination_row, 0});
            ++destination_row;
        }
    }
    if (destination_row != n) {
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
        requests.push_back({artifact::slice_tensor(reader, reference.shape,
                                                   matrix_slices(reader, reference)),
                            devices[i], access});
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
