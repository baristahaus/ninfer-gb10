#pragma once

#include "runtime/engine/kv_capacity.h"

namespace ninfer::models {

enum class WeightPlacement { Resident, HostExperts };

struct WeightCandidate {
    WeightPlacement placement;
    std::size_t device_bytes;
};

template <class SequencePlan, class Preparation>
struct ExecutionPlan {
    SequencePlan sequence;
    runtime::KvCapacityResolution capacity;
    WeightPlacement placement;
    std::uint32_t chunk;
    Preparation preparation;
    // The curve this capacity was resolved against. A group decision needs every rank's own curve
    // and budget before either Program exists, so the plan publishes the curve it used instead of
    // making the caller re-derive it.
    runtime::SequenceCapacityCurve curve;
};

} // namespace ninfer::models
