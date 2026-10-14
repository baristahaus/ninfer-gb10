#include "models/qwen3_5/execution/parameters.h"

#include "core/device.h"
#include "core/weight_view.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5::execution {

// SP-4 adaptive cache: runtime adaptation state. One Layer per routed MoE layer records the
// tier geometry and the per-expert warm ranks; the shadow mirrors the residency table, and
// the counter mirrors carry the current pass and the previous snapshot for recency deltas.
struct ExpertCacheState {
    struct Layer {
        const WeightParent* gate_device   = nullptr;
        const WeightParent* gate_warm     = nullptr;
        const WeightParent* down_device   = nullptr;
        const WeightParent* down_warm     = nullptr;
        std::uint64_t gate_rows_per_expert = 0;
        std::uint64_t down_rows_per_expert = 0;
        std::uint32_t hot                 = 0;
        std::uint32_t warm_count          = 0;
        std::uint32_t slots               = 0;
        // Warm-tier experts carry their compacted rank in the warm mini-bank; hot-tier
        // experts carry kNoWarmRank.
        std::vector<std::uint32_t> warm_rank;
    };
    static constexpr std::uint32_t kNoWarmRank = 0xffffffffu;
    std::vector<Layer> layers;
    std::vector<std::uint32_t> shadow;
    std::vector<std::uint32_t> counters;
    std::vector<std::uint32_t> snapshot;
    std::uint32_t decode_gate = 0;
    std::uint64_t warm_hits   = 0;
    std::uint64_t warm_total  = 0;
    std::uint64_t promotions  = 0;
    std::uint64_t evictions   = 0;
};

namespace {

template <class Function>
auto with_context(const std::string& context, Function&& function) {
    try {
        return function();
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(context + ": " + error.what());
    }
}

class Prepare {
public:
    explicit Prepare(const Model& model) : model_(model) {}

    LinearParameters linear(WeightId id) const {
        return with_context(model_.weight(id).name,
                            [&] { return ops::prepare_linear_weight(model_.input(id)); });
    }

    LinearParameters linear(WeightUseId id) const {
        return with_context(model_.weight(id.parameter).name,
                            [&] { return ops::prepare_linear_weight(model_.input(id)); });
    }

    Tensor tensor(WeightId id) const {
        const auto& bound = model_.weight(id);
        return with_context(bound.name, [&] {
            const auto& view = bound.view;
            if (view.shape.size() > 4) {
                throw std::invalid_argument("direct parameter exceeds Tensor rank");
            }
            std::array<std::int32_t, 4> axes{1, 1, 1, 1};
            for (std::size_t i = 0; i < view.shape.size(); ++i) {
                const auto extent = view.shape[view.shape.size() - 1 - i];
                if (extent > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
                    throw std::invalid_argument("direct parameter exceeds Tensor extent");
                }
                axes[i] = static_cast<std::int32_t>(extent);
            }
            return weight_tensor(view, {axes[0], axes[1], axes[2], axes[3]});
        });
    }

    DenseParameters dense(const DenseWeights& w) const {
        return {with_context(model_.weight(w.gate).name,
                             [&] {
                                 return ops::prepare_linear_swiglu_weight(model_.input(w.gate),
                                                                          model_.input(w.up));
                             }),
                linear(w.down)};
    }

    FfnParameters ffn(const BlockWeights& w) const {
        if (const auto* d = std::get_if<DenseWeights>(&w.ffn)) { return dense(*d); }
        const auto& moe = std::get<MoeWeights>(w.ffn);
        if (std::get<MoeConfig>(model_.config().text.ffn).num_experts_per_tok != 8) {
            throw std::invalid_argument("SparseMoe implements top-8 routing");
        }
        std::vector<ops::WeightInput> gate_up, down;
        gate_up.reserve(2 * moe.experts.size());
        down.reserve(moe.experts.size());
        for (const auto& expert : moe.experts) {
            gate_up.push_back(model_.input(expert.gate));
            gate_up.push_back(model_.input(expert.up));
            down.push_back(model_.input(expert.down));
        }
        return with_context(model_.weight(moe.router).name, [&] {
            return ops::prepare_sparse_moe_weights(
                model_.input(moe.router), model_.input(moe.shared_score), gate_up, down,
                model_.input(moe.shared.gate), model_.input(moe.shared.up),
                model_.input(moe.shared.down));
        });
    }

    BlockParameters block(const BlockWeights& w) const {
        BlockParameters out;
        out.input_norm          = tensor(w.input_norm);
        out.post_attention_norm = tensor(w.post_attention_norm);
        out.ffn                 = ffn(w);
        if (const auto* a = std::get_if<AttentionWeights>(&w.mixer)) {
            out.mixer = AttentionParameters{
                ops::prepare_attn_input_proj_weights(model_.input(a->query), model_.input(a->key),
                                                     model_.input(a->gate), model_.input(a->value)),
                tensor(a->query_norm), tensor(a->key_norm), linear(a->output)};
            out.projection_prefetch =
                prefetch(std::get<AttentionParameters>(out.mixer).projection, a->query);
        } else {
            const auto& g = std::get<GdnWeights>(w.mixer);
            out.mixer     = GdnParameters{
                ops::prepare_gdn_input_proj_weights(model_.input(g.query), model_.input(g.key),
                                                        model_.input(g.value), model_.input(g.z)),
                ops::prepare_gdn_gating_proj_weights(model_.input(g.a_projection),
                                                         model_.input(g.b_projection)),
                tensor(g.a_log),
                tensor(g.dt_bias),
                tensor(g.convolution),
                tensor(g.norm),
                linear(g.output)};
            out.projection_prefetch =
                prefetch(std::get<GdnParameters>(out.mixer).projection, g.query);
        }
        return out;
    }

    ops::SparseMoeHints prefetch(const ops::ProjectionWeights& projection, WeightId query) const {
        const auto* single = std::get_if<LinearParameters>(&projection);
        const auto& weight =
            single ? single->weight : std::get<ops::PairedProjectionWeights>(projection).first;
        const auto& geometry = model_.weight(query).view.parts.front().parent->geometry;
        const auto row_bytes = geometry.layout == QuantLayout::Contiguous
                                   ? std::uint64_t(weight.k) * dtype_size(DType::BF16)
                                   : geometry.code_bytes_per_row;
        return {weight.qdata, static_cast<std::size_t>(row_bytes * weight.n)};
    }

    MtpParameters mtp(const MtpWeights& w) const {
        const auto& a = std::get<AttentionWeights>(w.layer.mixer);
        const std::array inputs{model_.input(a.query), model_.input(a.key), model_.input(a.gate),
                                model_.input(a.value)};
        MtpParameters out;
        out.input_projection    = linear(w.input_projection);
        out.embedding_norm      = tensor(w.embedding_norm);
        out.hidden_norm         = tensor(w.hidden_norm);
        out.input_norm          = tensor(w.layer.input_norm);
        out.post_attention_norm = tensor(w.layer.post_attention_norm);
        out.final_norm          = tensor(w.final_norm);
        out.projection.packed   = ops::prepare_linear_weight(inputs);
        if (model_.config().text.architecture == Architecture::Qwen3_5) {
            out.projection.rows = {linear(a.query), linear(a.key), linear(a.gate), linear(a.value)};
        }
        out.query_norm  = tensor(a.query_norm);
        out.key_norm    = tensor(a.key_norm);
        out.output      = linear(a.output);
        out.ffn         = ffn(w.layer);
        out.output_head = linear(w.output_head_use);
        return out;
    }

    NormParameters norm(const NormWeights& w) const { return {tensor(w.weight), tensor(w.bias)}; }

    std::optional<Tensor> joined_bias(const std::array<WeightId, 3>& ids) const {
        WeightView view;
        std::uint64_t count = 0;
        for (const auto id : ids) {
            const auto& input = model_.weight(id).view;
            count += weight_element_count(input.shape);
            for (const auto& part : input.parts) {
                if (!view.parts.empty() && (view.parts.back().parent != part.parent ||
                                            view.parts.back().end != part.begin)) {
                    return std::nullopt;
                }
                view.parts.push_back(part);
            }
        }
        if (count > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument("Vision bias exceeds Tensor extent");
        }
        view.shape = {count};
        return weight_tensor(view, {static_cast<std::int32_t>(count)});
    }

    VisionParameters vision(const VisionWeights& w) const {
        VisionParameters out;
        out.patch_embedding      = linear(w.patch_embedding);
        out.patch_embedding_bias = tensor(w.patch_embedding_bias);
        out.position_embedding   = tensor(w.position_embedding);
        out.layers.reserve(w.layers.size());
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            out.layers.push_back(with_context("vision/layers/" + std::to_string(i), [&] {
                const auto& layer = w.layers[i];
                const std::array qkv{model_.input(layer.query), model_.input(layer.key),
                                     model_.input(layer.value)};
                const std::array ids{layer.query_bias, layer.key_bias, layer.value_bias};
                const auto bias = joined_bias(ids);
                if (!bias) {
                    throw std::invalid_argument(
                        "Vision QKV bias: this fixed call requires a contiguous bias bank");
                }
                return VisionBlockParameters{norm(layer.norm1),
                                             norm(layer.norm2),
                                             ops::prepare_linear_weight(qkv),
                                             *bias,
                                             linear(layer.output),
                                             linear(layer.fc1),
                                             linear(layer.fc2),
                                             tensor(layer.output_bias),
                                             tensor(layer.fc1_bias),
                                             tensor(layer.fc2_bias)};
            }));
        }
        out.merger_norm     = norm(w.merger_norm);
        out.merger_fc1      = linear(w.merger_fc1);
        out.merger_fc2      = linear(w.merger_fc2);
        out.merger_fc1_bias = tensor(w.merger_fc1_bias);
        out.merger_fc2_bias = tensor(w.merger_fc2_bias);
        return out;
    }

    DynamicConvParameters convolution(const DynamicConvWeights& w) const {
        return {tensor(w.base_kernel), linear(w.kernel_projection)};
    }

    DraftParameters draft(const DraftWeights& w) const {
        DraftParameters out;
        out.feature_projection = linear(w.feature_projection);
        out.context_norm       = tensor(w.context_norm);
        out.final_norm         = tensor(w.final_norm);
        out.output_head        = linear(w.output_head_use);
        out.layers.reserve(w.layers.size());
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            out.layers.push_back(with_context(
                std::string(model_.options().speculative_component()) + "/layers/" +
                    std::to_string(i),
                [&] {
                    const auto& layer = w.layers[i];
                    const auto& a     = layer.attention;
                    DraftBlockParameters result;
                    result.input_norm          = tensor(layer.input_norm);
                    result.post_attention_norm = tensor(layer.post_attention_norm);
                    result.query_key_value     = ops::prepare_attn_input_proj_weights(
                        model_.input(a.query), model_.input(a.key), model_.input(a.value));
                    result.context_key   = linear(a.context_key);
                    result.context_value = linear(a.context_value);
                    result.query_norm    = tensor(a.query_norm);
                    result.key_norm      = tensor(a.key_norm);
                    result.output        = linear(a.output);
                    result.mlp           = dense(layer.mlp);
                    if (layer.attention_conv) {
                        result.attention_conv = convolution(*layer.attention_conv);
                    }
                    if (layer.mlp_conv) { result.mlp_conv = convolution(*layer.mlp_conv); }
                    return result;
                }));
        }
        if (w.selector) {
            out.selector = SelectorParameters{linear(w.selector->hidden_projection),
                                              tensor(w.selector->predecessor_codebook),
                                              tensor(w.selector->successor_codebook)};
        }
        return out;
    }

private:
    const Model& model_;
};

} // namespace

Parameters::Parameters(const Model& source) : model(source) {
    const Prepare prepare(model);
    const auto& w        = model.weights();
    text.token_embedding = native_weight(model.weight(w.text.token_embedding).view);
    text.output_head     = prepare.linear(w.text.output_head_use);
    text.final_norm      = prepare.tensor(w.text.final_norm);
    text.layers.reserve(w.text.layers.size());
    for (std::size_t i = 0; i < w.text.layers.size(); ++i) {
        text.layers.push_back(with_context("text/layers/" + std::to_string(i),
                                           [&] { return prepare.block(w.text.layers[i]); }));
    }
    if (w.mtp) {
        mtp = with_context("mtp", [&] { return prepare.mtp(*w.mtp); });
    }
    if (w.vision) {
        vision = with_context("vision", [&] { return prepare.vision(*w.vision); });
    }
    if (w.draft) {
        draft = with_context(std::string(model.options().speculative_component()),
                             [&] { return prepare.draft(*w.draft); });
    }
    if (w.proposal) {
        proposal =
            ProposalParameters{prepare.linear(w.proposal->head), std::nullopt, w.proposal->rows};
        if (w.proposal->token_ids) { proposal->token_ids = prepare.tensor(*w.proposal->token_ids); }
    }
    // SP-4 phase 2b: allocate per-expert activation counters for the routed MoE layers and
    // wire them into the assembled weights; the routing stage accumulates one count per
    // selected expert per token. Tiered layers additionally receive a per-expert residency
    // table derived from the assembled views: an expert's gate part begins at its compacted
    // rank in the gate_up bank, and the part's parent marks the tier. Null tables keep the
    // untiered direct path where every expert reads the device bank at its own rank.
    struct RoutedLayer {
        std::string label;
        ops::SparseMoeWeights* parameters;
        const MoeWeights* weights;
    };
    std::vector<RoutedLayer> routed;
    for (std::size_t i = 0; i < text.layers.size(); ++i) {
        auto* moe = std::get_if<ops::SparseMoeWeights>(&text.layers[i].ffn);
        const auto* src = std::get_if<MoeWeights>(&w.text.layers[i].ffn);
        if (moe != nullptr && src != nullptr) {
            routed.push_back({"text/layers/" + std::to_string(i), moe, src});
        }
    }
    if (w.mtp && mtp) {
        auto* moe = std::get_if<ops::SparseMoeWeights>(&mtp->ffn);
        const auto* src = std::get_if<MoeWeights>(&w.mtp->layer.ffn);
        if (moe != nullptr && src != nullptr) {
            routed.push_back({"mtp/layer", moe, src});
        }
    }
    if (!routed.empty()) {
        counter_experts_ = std::get<MoeConfig>(model.config().text.ffn).num_experts;
        expert_counters_bytes_ =
            routed.size() * static_cast<std::size_t>(counter_experts_) * sizeof(std::uint32_t);
        CUDA_CHECK(cudaMalloc(&expert_counters_, expert_counters_bytes_));
        CUDA_CHECK(cudaMemset(expert_counters_, 0, expert_counters_bytes_));
        for (std::size_t i = 0; i < routed.size(); ++i) {
            routed[i].parameters->expert_activations =
                expert_counters_ + i * static_cast<std::size_t>(counter_experts_);
            counter_layers_.push_back(routed[i].label);
        }
        std::vector<std::uint32_t> tables(
            routed.size() * static_cast<std::size_t>(counter_experts_), 0u);
        std::vector<std::size_t> tiered_layers;
        expert_cache_                     = std::make_unique<ExpertCacheState>();
        expert_cache_->layers.resize(routed.size());
        expert_cache_->counters.assign(routed.size() * counter_experts_, 0u);
        expert_cache_->snapshot.assign(routed.size() * counter_experts_, 0u);
        for (auto& layer : expert_cache_->layers) {
            layer.warm_rank.assign(counter_experts_, ExpertCacheState::kNoWarmRank);
        }
        for (std::size_t i = 0; i < routed.size(); ++i) {
            bool tiered = false;
            for (std::size_t e = 0; e < routed[i].weights->experts.size(); ++e) {
                const auto& gate = model.input(routed[i].weights->experts[e].gate).weight;
                const auto& part = gate.parts.front();
                const std::uint64_t gate_rows = 2 * gate.shape[0];
                const auto rank = static_cast<std::uint32_t>(
                    part.begin / (gate_rows * gate.shape[1]));
                const bool warm = part.parent->host;
                const auto& down = model.input(routed[i].weights->experts[e].down).weight;
                const auto& down_part = down.parts.front();
                const auto down_rank = static_cast<std::uint32_t>(
                    down_part.begin / (down.shape[0] * down.shape[1]));
                if (down_part.parent->host != warm || down_rank != rank) {
                    throw std::invalid_argument(
                        routed[i].label + ": tiered routed banks disagree on expert " +
                        std::to_string(e) + " residency");
                }
                if (warm) {
                    tables[i * counter_experts_ + e] = 0x80000000u | rank;
                    expert_cache_->layers[i].warm_rank[e] = rank;
                    tiered = true;
                } else {
                    tables[i * counter_experts_ + e] = rank;
                    if (rank != e) { tiered = true; }
                }
            }
            if (tiered) { tiered_layers.push_back(i); }
        }
        // SP-4 adaptive cache: promote warm experts into the cache tail. A tiered layer's
        // device mini-bank carries the hot experts plus tail slots; every warm expert with a
        // tail slot copies its per-plane blocks from the pinned-host warm mini-bank into the
        // device tail and flips its residency entry to the device rank, so the tiered model
        // serves device-resident experts wherever the tail covers its warm tier. Warm slots
        // and tail slots are both rank ordered, so each plane copies as one contiguous block,
        // and the flipped entries ride the same upload below. Layers without a tail, and
        // banks that are already fully hot, keep their entries unchanged.
        for (std::size_t i = 0; i < routed.size(); ++i) {
            const auto& experts = routed[i].weights->experts;
            const WeightParent* gate_dev  = nullptr;
            const WeightParent* gate_host = nullptr;
            const WeightParent* down_dev  = nullptr;
            const WeightParent* down_host = nullptr;
            std::uint32_t hot  = 0;
            std::uint32_t warm = 0;
            std::uint64_t gate_rows_per_expert = 0;
            std::uint64_t down_rows_per_expert = 0;
            for (std::size_t e = 0; e < experts.size(); ++e) {
                const auto& gate = model.input(experts[e].gate).weight;
                const auto& part = gate.parts.front();
                gate_rows_per_expert = 2 * gate.shape[0];
                if (part.parent->host) {
                    gate_host = part.parent;
                    ++warm;
                } else {
                    gate_dev = part.parent;
                    ++hot;
                }
                const auto& down = model.input(experts[e].down).weight;
                const auto& down_part = down.parts.front();
                down_rows_per_expert = down.shape[0];
                if (down_part.parent->host) {
                    down_host = down_part.parent;
                } else {
                    down_dev = down_part.parent;
                }
            }
            if (gate_dev == nullptr || gate_host == nullptr || down_dev == nullptr ||
                down_host == nullptr) {
                continue;
            }
            auto& cache_layer      = expert_cache_->layers[i];
            cache_layer.gate_device = gate_dev;
            cache_layer.gate_warm  = gate_host;
            cache_layer.down_device = down_dev;
            cache_layer.down_warm  = down_host;
            cache_layer.gate_rows_per_expert = gate_rows_per_expert;
            cache_layer.down_rows_per_expert = down_rows_per_expert;
            cache_layer.hot        = hot;
            cache_layer.warm_count = warm;
            // One plane-aligned copy per bank; the tail is mutable by design (the promotion
            // and eviction transactions own it), while the hot rows stay untouched.
            const auto promote = [&](const WeightParent* device, const WeightParent* host_bank,
                                     std::uint64_t rows_per_expert, std::uint32_t hot_count,
                                     std::uint32_t warm_count) {
                const auto tail_rows =
                    device->geometry.shape[0] -
                    static_cast<std::uint64_t>(hot_count) * rows_per_expert;
                if (tail_rows == 0) { return static_cast<std::uint32_t>(0); }
                const auto slots = static_cast<std::uint32_t>(tail_rows / rows_per_expert);
                const auto count = std::min(slots, warm_count);
                const struct Plane {
                    std::uint64_t device_offset;
                    std::uint64_t host_offset;
                    std::uint64_t per_row;
                } planes[] = {
                    {0, 0, device->geometry.code_bytes_per_row},
                    {device->geometry.high_offset, host_bank->geometry.high_offset,
                     device->geometry.high_bytes_per_row},
                    {device->geometry.scale_offset, host_bank->geometry.scale_offset,
                     device->geometry.scale_bytes_per_row},
                };
                for (const auto& plane : planes) {
                    if (plane.per_row == 0) { continue; }
                    const auto block =
                        static_cast<std::uint64_t>(count) * rows_per_expert * plane.per_row;
                    const auto hot_prefix =
                        static_cast<std::uint64_t>(hot_count) * rows_per_expert * plane.per_row;
                    CUDA_CHECK(cudaMemcpy(
                        const_cast<std::byte*>(device->data) + plane.device_offset + hot_prefix,
                        host_bank->data + plane.host_offset, block, cudaMemcpyHostToDevice));
                }
                return count;
            };
            const auto gate_count =
                promote(gate_dev, gate_host, gate_rows_per_expert, hot, warm);
            const auto down_count =
                promote(down_dev, down_host, down_rows_per_expert, hot, warm);
            if (gate_count != down_count) {
                throw std::invalid_argument(routed[i].label +
                                            ": tiered cache tails disagree between banks");
            }
            cache_layer.slots = gate_count;
            for (std::size_t e = 0; e < experts.size(); ++e) {
                auto& entry = tables[i * counter_experts_ + e];
                if ((entry & 0x80000000u) != 0u && (entry & 0x7fffffffu) < gate_count) {
                    entry = (entry & 0x7fffffffu) + hot;
                }
            }
        }
        expert_cache_->shadow = tables;
        if (!tiered_layers.empty()) {
            CUDA_CHECK(cudaMalloc(&expert_residency_, tables.size() * sizeof(std::uint32_t)));
            CUDA_CHECK(cudaMemcpy(expert_residency_, tables.data(),
                                  tables.size() * sizeof(std::uint32_t),
                                  cudaMemcpyHostToDevice));
            for (const auto i : tiered_layers) {
                routed[i].parameters->expert_residency =
                    expert_residency_ + i * static_cast<std::size_t>(counter_experts_);
            }
        }
    }
}

// SP-4 adaptive cache: one promotion/eviction pass. The counters ride the stream (ordered
// after the boundary's kernels), the host then swaps the weakest promoted warm experts for the
// hottest uncovered ones with hysteresis — a newcomer must clear a minimum recent count and
// double the incumbent it evicts, at most two swaps per layer per pass — and the flipped
// entries follow on the stream. Tiers the tail fully covers only accumulate the warm-tier
// hit rate.
void Parameters::adapt_expert_cache(cudaStream_t stream, bool prefill_boundary) {
    if (expert_cache_ == nullptr || expert_cache_->layers.empty()) { return; }
    auto& state = *expert_cache_;
    if (!prefill_boundary && ++state.decode_gate % 8 != 1) { return; }
    CUDA_CHECK(cudaMemcpyAsync(state.counters.data(), expert_counters_, expert_counters_bytes_,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (std::size_t i = 0; i < state.layers.size(); ++i) {
        auto& layer = state.layers[i];
        if (layer.gate_warm == nullptr || layer.gate_device == nullptr ||
            layer.down_warm == nullptr || layer.down_device == nullptr) {
            continue;
        }
        const std::uint32_t base = static_cast<std::uint32_t>(i) * counter_experts_;
        for (std::uint32_t e = 0; e < counter_experts_; ++e) {
            if (layer.warm_rank[e] == ExpertCacheState::kNoWarmRank) { continue; }
            const std::uint32_t delta = state.counters[base + e] - state.snapshot[base + e];
            state.warm_total += delta;
            if ((state.shadow[base + e] & 0x80000000u) == 0u) { state.warm_hits += delta; }
        }
        if (layer.slots >= layer.warm_count) { continue; }
        std::vector<std::pair<std::uint32_t, std::uint32_t>> promoted;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> candidates;
        promoted.reserve(layer.slots);
        candidates.reserve(layer.warm_count - layer.slots);
        for (std::uint32_t e = 0; e < counter_experts_; ++e) {
            if (layer.warm_rank[e] == ExpertCacheState::kNoWarmRank) { continue; }
            const std::uint32_t delta = state.counters[base + e] - state.snapshot[base + e];
            if ((state.shadow[base + e] & 0x80000000u) == 0u) {
                promoted.emplace_back(e, delta);
            } else {
                candidates.emplace_back(e, delta);
            }
        }
        std::sort(promoted.begin(), promoted.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        constexpr std::uint32_t kMinimumRecentCount = 4;
        constexpr double kEvictionMargin           = 2.0;
        constexpr std::size_t kMaximumSwaps        = 2;
        std::size_t swaps = 0;
        for (std::size_t k = 0;
             k < promoted.size() && k < candidates.size() && swaps < kMaximumSwaps; ++k) {
            const auto [newcomer, newcomer_count]   = candidates[k];
            const auto [incumbent, incumbent_count] = promoted[k];
            if (newcomer_count < kMinimumRecentCount) { break; }
            if (static_cast<double>(newcomer_count) <=
                static_cast<double>(incumbent_count) * kEvictionMargin) {
                break;
            }
            const std::uint32_t slot =
                (state.shadow[base + incumbent] & 0x7fffffffu) - layer.hot;
            const std::uint32_t warm_rank = layer.warm_rank[newcomer];
            const auto copy_expert         = [&](const WeightParent* device,
                                                 const WeightParent* warm_bank,
                                                 std::uint64_t rows_per_expert) {
                const struct Plane {
                    std::uint64_t device_offset;
                    std::uint64_t warm_offset;
                    std::uint64_t per_row;
                } planes[] = {
                    {0, 0, device->geometry.code_bytes_per_row},
                    {device->geometry.high_offset, warm_bank->geometry.high_offset,
                     device->geometry.high_bytes_per_row},
                    {device->geometry.scale_offset, warm_bank->geometry.scale_offset,
                     device->geometry.scale_bytes_per_row},
                };
                for (const auto& plane : planes) {
                    if (plane.per_row == 0) { continue; }
                    const auto block = rows_per_expert * plane.per_row;
                    CUDA_CHECK(cudaMemcpyAsync(
                        const_cast<std::byte*>(device->data) + plane.device_offset +
                            static_cast<std::uint64_t>(layer.hot + slot) * block,
                        warm_bank->data + plane.warm_offset +
                            static_cast<std::uint64_t>(warm_rank) * block,
                        block, cudaMemcpyHostToDevice, stream));
                }
            };
            copy_expert(layer.gate_device, layer.gate_warm, layer.gate_rows_per_expert);
            copy_expert(layer.down_device, layer.down_warm, layer.down_rows_per_expert);
            state.shadow[base + incumbent] = 0x80000000u | layer.warm_rank[incumbent];
            state.shadow[base + newcomer]  = layer.hot + slot;
            ++state.promotions;
            ++state.evictions;
            ++swaps;
        }
        if (swaps != 0) {
            CUDA_CHECK(cudaMemcpyAsync(expert_residency_ + base, state.shadow.data() + base,
                                       counter_experts_ * sizeof(std::uint32_t),
                                       cudaMemcpyHostToDevice, stream));
        }
    }
    state.snapshot = state.counters;
}

Parameters::~Parameters() {
    if (expert_counters_ != nullptr) { CUDA_CHECK(cudaFree(expert_counters_)); }
    if (expert_cache_ != nullptr && expert_cache_->warm_total != 0) {
        std::fprintf(stderr,
                     "ninfer: expert cache warm-tier hit rate %.2f%% (%llu/%llu selections, "
                     "%llu promotions, %llu evictions)\n",
                     100.0 * static_cast<double>(expert_cache_->warm_hits) /
                         static_cast<double>(expert_cache_->warm_total),
                     static_cast<unsigned long long>(expert_cache_->warm_hits),
                     static_cast<unsigned long long>(expert_cache_->warm_total),
                     static_cast<unsigned long long>(expert_cache_->promotions),
                     static_cast<unsigned long long>(expert_cache_->evictions));
    }
    if (expert_residency_ != nullptr) { CUDA_CHECK(cudaFree(expert_residency_)); }
}

MoeActivationProfile Parameters::moe_activation_profile() const {
    MoeActivationProfile out;
    if (expert_counters_ == nullptr) { return out; }
    std::vector<std::uint32_t> host(expert_counters_bytes_ / sizeof(std::uint32_t));
    CUDA_CHECK(cudaMemcpy(host.data(), expert_counters_, expert_counters_bytes_,
                          cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < counter_layers_.size(); ++i) {
        MoeActivationProfileLayer layer;
        layer.component = counter_layers_[i];
        layer.counts.assign(host.begin() + static_cast<std::ptrdiff_t>(i) * counter_experts_,
                            host.begin() + static_cast<std::ptrdiff_t>(i + 1) * counter_experts_);
        out.layers.push_back(std::move(layer));
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::execution
