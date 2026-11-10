#pragma once

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "models/common/weights.h"

#include <map>
#include <span>

namespace ninfer::models {

// One parameter's place in a materialization plan: the request that holds its bytes and the element
// range within it. A plan that places *slices* needs this, because an object handle cannot
// translate between the artifact object's coordinates and a slice's.
struct WeightStorage {
    artifact::StorageId request;
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

struct PendingWeight {
    artifact::ParameterReference reference;
    std::vector<WeightUse> uses;
    std::vector<std::string> source_objects;
};

class Bindings {
public:
    explicit Bindings(artifact::Binder& binder) : binder(binder) {}

    [[nodiscard]] WeightId parameter(std::string name, artifact::Shape shape,
                                     std::vector<std::string> inputs   = {},
                                     std::optional<QType> exact_format = {});
    [[nodiscard]] WeightId direct(std::string name, artifact::Shape shape,
                                  QType format = QType::BF16);

    [[nodiscard]] const PendingWeight& at(WeightId id) const { return weights.at(id.index); }

    [[nodiscard]] WeightUseId use(WeightId id, std::string_view input) const;

    artifact::Binder& binder;
    std::vector<PendingWeight> weights;

private:
    std::map<std::string, WeightId, std::less<>> parameters_;
};

[[nodiscard]] std::vector<BoundWeight>
resolve_weights(std::vector<PendingWeight>&& pending,
                const artifact::MaterializedArtifact& materialized,
                std::span<const Device> devices);

// The sliced-plan form: bind each parameter to the request that holds its bytes rather than to its
// artifact object. A sliced storage's coordinate space is the rank's own shape, so `bind_view`'s
// object-handle resolution cannot reach it -- see `WeightStorage`.
[[nodiscard]] std::vector<BoundWeight>
resolve_weights(std::vector<PendingWeight>&& pending,
                const artifact::MaterializedArtifact& materialized,
                std::span<const Device> devices, std::span<const WeightStorage> storage);

} // namespace ninfer::models
