#include "ninfer/ops/sparse_moe.h"
#include "ninfer/ops/weight_input.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

const std::vector<std::uint64_t>& matrix(const WeightInput& input) {
    require(input.weight.shape.size() == 2, "projection weight must be a matrix");
    const auto elements = weight_element_count(input.weight.shape);
    require(!input.weight.parts.empty(), "projection weight has no parent regions");
    std::uint64_t covered = 0;
    for (const auto& part : input.weight.parts) {
        require(part.parent && part.begin < part.end &&
                    part.end <= part.parent->geometry.elements &&
                    part.end - part.begin <= elements - covered,
                "projection region exceeds its parent or logical matrix");
        covered += part.end - part.begin;
    }
    require(covered == elements, "projection regions do not cover its logical matrix");
    return input.weight.shape;
}

WeightView concatenate_rows(std::span<const WeightInput> inputs) {
    require(!inputs.empty(), "projection bank is empty");
    const auto columns = matrix(inputs.front())[1];
    WeightView out{{0, columns}, {}};
    for (const auto& input : inputs) {
        const auto& shape = matrix(input);
        require(shape[1] == columns &&
                    shape[0] <= std::numeric_limits<std::uint64_t>::max() - out.shape[0],
                "projection bank has inconsistent columns or overflowing rows");
        out.shape[0] += shape[0];
        out.parts.insert(out.parts.end(), input.weight.parts.begin(), input.weight.parts.end());
    }
    return out;
}

bool contiguous(const WeightView& view) {
    if (view.parts.empty()) { return false; }
    const auto* parent = view.parts.front().parent;
    auto end           = view.parts.front().begin;
    for (const auto& part : view.parts) {
        if (!parent || part.parent != parent || part.begin != end || part.end <= part.begin) {
            return false;
        }
        end = part.end;
    }
    return true;
}

LinearPolicy common_policy(std::span<const WeightInput> inputs) {
    auto result = LinearPolicy::AllowA4;
    for (const auto& input : inputs) {
        switch (input.policy) {
        case LinearPolicy::A16Only:
            result = LinearPolicy::A16Only;
            break;
        case LinearPolicy::AllowA8:
            if (result == LinearPolicy::AllowA4) { result = LinearPolicy::AllowA8; }
            break;
        case LinearPolicy::AllowA4:
            break;
        default:
            throw std::invalid_argument("invalid projection activation policy");
        }
    }
    return result;
}

// SP-4 phase 2a: every expert view must reference exactly one parent region owner.
const WeightParent* view_parent(const WeightInput& input) {
    require(!input.weight.parts.empty(), "expert view has no parent region");
    const auto* parent = input.weight.parts.front().parent;
    for (const auto& part : input.weight.parts) {
        require(part.parent == parent, "expert view spans multiple parents");
    }
    return parent;
}

// SP-4: partitions expert views by residency parent — device parents form the hot tier,
// pinned-host parents the warm tier — with arbitrary expert ids on either side. Each
// partition keeps expert id order (the input order is expert id order, so filtering
// preserves it). An empty warm partition means the bank is untiered.
std::pair<std::vector<std::size_t>, std::vector<std::size_t>>
tier_partition(std::span<const WeightInput> inputs) {
    require(!inputs.empty(), "expert bank is empty");
    const auto* hot  = static_cast<const WeightParent*>(nullptr);
    const auto* warm = static_cast<const WeightParent*>(nullptr);
    std::pair<std::vector<std::size_t>, std::vector<std::size_t>> out;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto* parent = view_parent(inputs[i]);
        if (parent->device.type() != DeviceType::CUDA) {
            require(warm == nullptr || parent == warm,
                    "tiered routed bank requires one host warm parent");
            warm = parent;
            out.second.push_back(i);
        } else {
            require(hot == nullptr || hot->device == parent->device,
                    "tiered routed bank requires one device hot parent");
            hot = parent;
            out.first.push_back(i);
        }
    }
    return out;
}

SingleProjectionWeight single(const WeightView& view, std::span<const WeightInput> inputs) {
    const auto region = contiguous_weight_region(view);
    const auto policy = common_policy(inputs);
    float divisor     = 0;
    if (region.parent->geometry.format == QType::NVFP4) {
        for (const auto& input : inputs) {
            if (!input.activation_input_divisor) {
                require(!allows_a4(policy), "NVFP4 A4 native input requires an activation divisor");
                continue;
            }
            require(std::isfinite(*input.activation_input_divisor) &&
                        *input.activation_input_divisor > 0,
                    "NVFP4 native input requires a positive activation divisor");
            if (divisor == 0) {
                // A16 does not read this auxiliary; retain a stored positive value for the ABI.
                divisor = *input.activation_input_divisor;
            } else if (allows_a4(policy)) {
                // Current NVFP4 consumers use A16 for AllowA8 as well. Only their A4 route
                // quantizes the shared activation and therefore requires a common divisor.
                require(std::bit_cast<std::uint32_t>(divisor) ==
                            std::bit_cast<std::uint32_t>(*input.activation_input_divisor),
                        "shared NVFP4 native input requires identical activation divisors");
            }
        }
        // The legacy native ABI validates this field even on A16, where it is never read.
        // Keep that ABI detail out of the artifact's optional Use auxiliaries.
        if (divisor == 0) { divisor = 1.0F; }
    }
    return {native_weight(view, divisor), policy};
}

SingleProjectionWeight single(std::span<const WeightInput> inputs) {
    return single(concatenate_rows(inputs), inputs);
}

LinearProfile describe_single(const WeightView& view, std::span<const WeightInput> inputs) {
    const auto region = contiguous_weight_region(view);
    require(view.shape.size() == 2 && view.shape[0] <= INT32_MAX && view.shape[1] <= INT32_MAX,
            "projection dimensions exceed int32");
    const auto& geometry = region.parent->geometry;
    const auto layout    = geometry.layout == QuantLayout::BatchedBlockScaleK16M128x4
                               ? QuantLayout::BlockScaleK16M128x4
                               : geometry.layout;
    return {geometry.format, static_cast<std::int32_t>(view.shape[0]),
            static_cast<std::int32_t>(view.shape[1]), common_policy(inputs), layout};
}

template <class Make>
auto projection_regions(std::span<const WeightInput> rows, Make make) {
    const auto view = concatenate_rows(rows);
    if (contiguous(view)) return std::vector{make(view, rows)};

    const auto columns = view.shape[1];
    WeightView bank{{0, columns}, {}};
    std::uint64_t elements = 0;
    std::vector<WeightInput> bank_rows;
    using Result = decltype(make(view, rows));
    std::vector<Result> banks;
    const auto finish_bank = [&] {
        require(elements % columns == 0, "projection parent boundary must align to a matrix row");
        bank.shape[0] = elements / columns;
        banks.push_back(make(bank, bank_rows));
    };
    for (const auto& input : rows) {
        bool included = false;
        for (const auto& part : input.weight.parts) {
            if (!bank.parts.empty() &&
                (part.parent != bank.parts.back().parent || part.begin != bank.parts.back().end)) {
                finish_bank();
                require(banks.size() < 2, "projection bank requires more than two native regions");
                bank.parts.clear();
                bank_rows.clear();
                elements = 0;
                included = false;
            }
            bank.parts.push_back(part);
            elements += part.end - part.begin;
            if (!included) {
                bank_rows.push_back(input);
                included = true;
            }
        }
    }
    finish_bank();
    return banks;
}

ProjectionWeights input_projection(std::span<const WeightInput, 4> inputs, bool attention) {
    const auto& q      = matrix(inputs[0]);
    const auto& k      = matrix(inputs[1]);
    const auto& third  = matrix(inputs[2]);
    const auto& fourth = matrix(inputs[3]);
    const bool dense =
        attention ? q == std::vector<std::uint64_t>{6144, 5120} &&
                        k == std::vector<std::uint64_t>{1024, 5120} && third == q && fourth == k
                  : q == std::vector<std::uint64_t>{2048, 5120} && k == q &&
                        third == std::vector<std::uint64_t>{6144, 5120} && fourth == third;
    const bool moe =
        attention ? q == std::vector<std::uint64_t>{4096, 2048} &&
                        k == std::vector<std::uint64_t>{512, 2048} && third == q && fourth == k
                  : q == std::vector<std::uint64_t>{2048, 2048} && k == q &&
                        third == std::vector<std::uint64_t>{4096, 2048} && fourth == third;
    require(dense || moe, "input projection: unsupported logical projection geometry");
    const auto joined = concatenate_rows(inputs);
    if (contiguous(joined)) {
        auto result       = single(inputs);
        const auto format = result.weight.qtype;
        const bool supported =
            (moe && format == QType::Q8_G32_FP16) ||
            (dense && (format == QType::NVFP4 || format == QType::FP8_E4M3FN_ROW_BF16 ||
                       format == QType::EXL3 || (attention && format == QType::BF16)));
        require(supported, "input projection: unsupported single-parent format");
        return result;
    }
    require(dense, "input projection: unsupported multi-parent geometry");
    const auto first  = single(inputs.first<2>());
    const auto second = single(inputs.last<2>());
    require(first.weight.qtype == QType::Q4_G64_FP16 && second.weight.qtype == QType::Q5_G64_FP16,
            "input projection: paired native form requires Q4 and Q5");
    return PairedProjectionWeights{first.weight, second.weight};
}

} // namespace

SingleProjectionWeight prepare_linear_weight(const WeightInput& input) {
    return single({&input, 1});
}

SingleProjectionWeight prepare_linear_weight(std::span<const WeightInput> rows) {
    return single(rows);
}

LinearProfile describe_linear_weight(std::span<const WeightInput> rows) {
    return describe_single(concatenate_rows(rows), rows);
}

ProjectionProfile describe_projection_weights(std::span<const WeightInput> rows) {
    auto banks = projection_regions(rows, describe_single);
    if (banks.size() == 1) return {banks[0], {}};
    banks[0].policy = banks[1].policy = LinearPolicy::A16Only;
    return {banks[0], banks[1]};
}

ProjectionWeights prepare_projection_weights(std::span<const WeightInput> rows) {
    const auto banks =
        projection_regions(rows, [](const WeightView& view, std::span<const WeightInput> inputs) {
            return single(view, inputs);
        });
    if (banks.size() == 1) return banks[0];
    return PairedProjectionWeights{banks[0].weight, banks[1].weight};
}

ProjectionWeights prepare_gdn_control_proj_weights(const WeightInput& a, const WeightInput& b) {
    const auto& shape = matrix(a);
    require(shape == matrix(b) && (shape == std::vector<std::uint64_t>{48, 5120} ||
                                   shape == std::vector<std::uint64_t>{32, 2048} ||
                                   shape == std::vector<std::uint64_t>{48, 2560} ||
                                   shape == std::vector<std::uint64_t>{24, 2560}),
            "GDN control: unsupported A/B geometry");
    const std::array inputs{a, b};
    if (contiguous(concatenate_rows(inputs))) {
        auto result = single(inputs);
        require(result.weight.qtype == QType::BF16, "GDN control requires BF16 weights");
        return result;
    }
    require(shape == std::vector<std::uint64_t>{48, 5120},
            "GDN control: this geometry requires a combined parent");
    const auto first  = prepare_linear_weight(a);
    const auto second = prepare_linear_weight(b);
    require(first.weight.qtype == QType::BF16 && second.weight.qtype == QType::BF16,
            "GDN control requires BF16 weights");
    return PairedProjectionWeights{first.weight, second.weight};
}

SingleProjectionWeight prepare_linear_swiglu_weight(const WeightInput& gate,
                                                    const WeightInput& up) {
    require(matrix(gate) == matrix(up), "SwiGLU gate and up geometry differs");
    const std::array inputs{gate, up};
    return single(inputs);
}

SparseMoeWeights
prepare_sparse_moe_weights(const WeightInput& router, const WeightInput& shared_score,
                           std::span<const WeightInput> expert_gate_up,
                           std::span<const WeightInput> expert_down, const WeightInput& shared_gate,
                           const WeightInput& shared_up, const WeightInput& shared_down) {
    require(expert_gate_up.size() == 512 && expert_down.size() == 256,
            "SparseMoe native input requires 256 experts");
    require(matrix(router) == std::vector<std::uint64_t>{256, 2048} &&
                matrix(shared_score) == std::vector<std::uint64_t>{1, 2048},
            "SparseMoe router geometry differs");
    for (const auto& input : expert_gate_up) {
        require(matrix(input) == std::vector<std::uint64_t>{512, 2048},
                "SparseMoe gate/up geometry differs");
    }
    for (const auto& input : expert_down) {
        require(matrix(input) == std::vector<std::uint64_t>{2048, 512},
                "SparseMoe down geometry differs");
    }
    require(matrix(shared_gate) == std::vector<std::uint64_t>{512, 2048} &&
                matrix(shared_up) == matrix(shared_gate) &&
                matrix(shared_down) == std::vector<std::uint64_t>{2048, 512},
            "SparseMoe shared expert geometry differs");
    const std::array router_inputs{router, shared_score};
    const auto router_bank = single(router_inputs);
    const auto shared      = prepare_linear_swiglu_weight(shared_gate, shared_up);
    const auto down        = prepare_linear_weight(shared_down);
    const auto gate_format = view_parent(expert_gate_up.front())->geometry.format;
    const auto down_format = view_parent(expert_down.front())->geometry.format;
    // SP-7 EXL3: gate and up are separate complete banks (contract 2.2), paired per expert
    // in binding order (gate, up, gate, up, ...). The route is untiered: SP-4's tiered
    // packing assumes one constant per-expert byte size, which per-expert bitrates break
    // (contract section 5 refuses the split at plan time).
    if (gate_format == QType::EXL3) {
        require(down_format == QType::EXL3,
                 "SparseMoe EXL3 gate/up banks require an EXL3 down bank");
        require(expert_gate_up.size() % 2 == 0, "SparseMoe EXL3 gate/up views must pair");
        std::vector<WeightInput> gates;
        std::vector<WeightInput> ups;
        gates.reserve(expert_gate_up.size() / 2);
        ups.reserve(expert_gate_up.size() / 2);
        for (std::size_t index = 0; index < expert_gate_up.size(); ++index) {
            (index % 2 == 0 ? gates : ups).push_back(expert_gate_up[index]);
        }
        const auto gate_bank = single(gates);
        const auto up_bank   = single(ups);
        const auto down_bank = single(expert_down);
        for (const auto* bank : {&gate_bank.weight, &up_bank.weight, &down_bank.weight}) {
            require(bank->qtype == QType::EXL3 && bank->layout == QuantLayout::Exl3ExpertBank,
                    "SparseMoe EXL3 banks must resolve as complete EXL3 expert banks");
            require(bank->exl3_codebook == 1 || bank->exl3_codebook == 2,
                    "SparseMoe refuses 3INST EXL3 banks until a qualified route exists");
            require(bank->exl3_experts == 256 && bank->payload != nullptr,
                    "SparseMoe EXL3 bank geometry differs from the routed bank contract");
        }
        require(gate_bank.weight.k == 2048 && gate_bank.weight.exl3_rows_per_expert == 512 &&
                    up_bank.weight.k == 2048 && up_bank.weight.exl3_rows_per_expert == 512 &&
                    down_bank.weight.k == 512 && down_bank.weight.exl3_rows_per_expert == 2048,
                "SparseMoe EXL3 bank shapes differ from the 35B bank contract");
        return {router_bank.weight, Weight{},       down_bank.weight, shared.weight,
                down.weight,        Weight{},       Weight{},         0,
                nullptr,            nullptr,        gate_bank.weight, up_bank.weight};
    }
    require(router_bank.weight.qtype == QType::BF16 && shared.weight.qtype == QType::Q8_G32_FP16 &&
                down.weight.qtype == QType::Q8_G32_FP16 &&
                ((gate_format == QType::Q4_G64_FP16 &&
                  (down_format == QType::Q5_G64_FP16 || down_format == QType::Q6_G64_FP16)) ||
                 (gate_format == QType::Q8_G32_FP16 && down_format == QType::Q8_G32_FP16)),
            "SparseMoe native bank formats are unsupported");
    // SP-4: a tiered layer partitions its routed expert views by residency parent — the
    // device hot mini-bank's parent versus the pinned-host warm mini-bank's parent — with
    // arbitrary expert ids on either side. Gate and up views pair per expert, so the gate/up
    // partitions must select matching expert pairs; down views are one per expert and both
    // banks must partition identically.
    const auto [gate_up_hot, gate_up_warm] = tier_partition(expert_gate_up);
    const auto [down_hot, down_warm]       = tier_partition(expert_down);
    if (gate_up_warm.empty() && down_warm.empty()) {
        const auto gate_up_bank = single(expert_gate_up);
        const auto down_bank    = single(expert_down);
        return {router_bank.weight, gate_up_bank.weight, down_bank.weight, shared.weight,
                down.weight};
    }
    require(!gate_up_warm.empty() && !down_warm.empty(),
            "tiered routed banks must split together");
    const auto hot_experts = down_hot.size();
    require(gate_up_hot.size() == 2 * hot_experts && gate_up_warm.size() % 2 == 0 &&
                gate_up_warm.size() / 2 == down_warm.size(),
            "tiered gate/up and down banks must partition the same experts");
    auto gather = [](std::span<const WeightInput> inputs,
                     const std::vector<std::size_t>& indices) {
        std::vector<WeightInput> picked;
        picked.reserve(indices.size());
        for (const auto i : indices) { picked.push_back(inputs[i]); }
        return picked;
    };
    const auto hot_gate_up  = single(gather(expert_gate_up, gate_up_hot));
    const auto warm_gate_up = single(gather(expert_gate_up, gate_up_warm));
    const auto hot_down     = single(gather(expert_down, down_hot));
    const auto warm_down    = single(gather(expert_down, down_warm));
    return {router_bank.weight,
            hot_gate_up.weight,
            hot_down.weight,
            shared.weight,
            down.weight,
            warm_gate_up.weight,
            warm_down.weight,
            static_cast<std::uint32_t>(hot_experts)};
}
} // namespace ninfer::ops
