#include "models/qwen3_5/model.h"

namespace ninfer::models::qwen3_5 {

ModelShard::ModelShard(std::shared_ptr<const ModelDefinition> definition,
                       artifact::MaterializedArtifact backing, std::vector<BoundWeight> bound,
                       std::vector<ExpertBankSource> banks, RepackedWeights repacked)
    : definition_(std::move(definition)), backing_(std::move(backing)), bound_(std::move(bound)),
      banks_(std::move(banks)), repacked_(std::move(repacked)) {}

ModelShard::~ModelShard() = default;

} // namespace ninfer::models::qwen3_5
