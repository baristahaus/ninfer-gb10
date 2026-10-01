#include "models/qwen3_8_flash_next/impl/runtime/instance.h"
#include "models/qwen3_8_flash_next/impl/runtime/text_context.h"
#include "models/qwen3_8_flash_next/impl/runtime/diagnostics.h"
#include "models/qwen3_8_flash_next/impl/runtime/workspace_recipe.h"

#include "core/nvtx.h"
#include "core/performance.h"
#include "models/qwen3_8_flash_next/impl/runtime/visual_scatter.h"
#include "models/qwen3_8_flash_next/impl/runtime/vision_context.h"
#include <ninfer/models/qwen3_8_flash_next/vision_control.h>
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/flash_next_gdn.h"
#include "ninfer/ops/flash_next_moe.h"
#include "ninfer/ops/flash_next_ple.h"
#include "ninfer/ops/flash_next_qsa.h"
#include "ninfer/ops/hyperconnection.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/softmax_attention.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "models/qwen3_8_flash_next/impl/ple_table.h"

namespace ninfer::models::qwen3_8_flash_next::detail::NINFER_QWEN38_FLASH_NEXT_RUNTIME_NS::
    schedule {
namespace {

void copy_i32(const std::int32_t* source, Tensor& destination, cudaStream_t stream) {
    if (source == nullptr || destination.dtype != DType::I32 || !destination.is_contiguous() ||
        destination.data == nullptr) {
        throw std::invalid_argument("copy_i32: invalid host source or I32 destination");
    }
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source, destination.bytes(),
                               cudaMemcpyHostToDevice, stream));
}

void require_tensor_shape(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape,
                          const char* label) {
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    int i = 0;
    for (const std::int32_t dim : shape) {
        if (t.ne[i] != dim) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
        ++i;
    }
    for (; i < 4; ++i) {
        if (t.ne[i] != 1) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

void require_tensor_window(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t cols,
                           const char* label) {
    if (cols <= 0) { throw std::invalid_argument(std::string(label) + " cols must be positive"); }
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    if (t.ne[0] != rows || t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(label) + " shape mismatch");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

Tensor matrix_window(Tensor& t, std::int32_t cols) {
    if (cols <= 0) { throw std::invalid_argument("matrix_window cols must be positive"); }
    if (t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument("matrix_window shape mismatch");
    }
    return t.slice(1, 0, cols);
}

class ScopedPositions {
public:
    ScopedPositions(const Tensor*& slot, const Tensor& positions) : slot_(slot) {
        slot_ = &positions;
    }

    ScopedPositions(const ScopedPositions&)            = delete;
    ScopedPositions& operator=(const ScopedPositions&) = delete;

    ~ScopedPositions() { slot_ = nullptr; }

private:
    const Tensor*& slot_;
};

class ScopedEnvelope {
public:
    ScopedEnvelope(const ops::CausalAttentionExecutionEnvelope*& slot,
                   const ops::CausalAttentionExecutionEnvelope& envelope)
        : slot_(slot) {
        slot_ = &envelope;
    }

    ScopedEnvelope(const ScopedEnvelope&)            = delete;
    ScopedEnvelope& operator=(const ScopedEnvelope&) = delete;

    ~ScopedEnvelope() { slot_ = nullptr; }

private:
    const ops::CausalAttentionExecutionEnvelope*& slot_;
};

template <class T>
class ScopedValue {
public:
    ScopedValue(T& slot, T value) : slot_(slot), previous_(slot) { slot_ = value; }

    ScopedValue(const ScopedValue&)            = delete;
    ScopedValue& operator=(const ScopedValue&) = delete;

    ~ScopedValue() { slot_ = previous_; }

private:
    T& slot_;
    T previous_;
};

} // namespace

void DFlashFeatureSink::begin(const Tensor& value) {
    const bool prefill = features != nullptr && positions != nullptr && batch_features == nullptr;
    const bool batch   = batch_features != nullptr && batch_lanes != nullptr &&
                         batch_valid_columns != nullptr && batch_width > 0 && batch_size > 0;
    if ((!prefill && !batch) || layers.empty()) {
        throw std::logic_error("DFlash feature sink is incomplete");
    }
    captured_mask = 0;
    active_tokens = batch ? batch_width * batch_size : value.ne[1];
    if (value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash batch feature source has an invalid width");
    }
}

void DFlashFeatureSink::capture_layer(int layer, const Tensor& value, cudaStream_t stream) {
    const auto it = std::find(layers.begin(), layers.end(), layer);
    if (it == layers.end()) { return; }
    const std::size_t index = static_cast<std::size_t>(it - layers.begin());
    Tensor* destination     = batch_features != nullptr ? batch_features : features;
    if (layers.size() > 32 || active_tokens <= 0 || value.dtype != DType::BF16 ||
        destination == nullptr ||
        value.ne[0] * static_cast<std::int32_t>(layers.size()) != destination->ne[0] ||
        value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash feature capture shape is invalid");
    }
    if (batch_features != nullptr) {
        Tensor source = value.view({value.ne[0], batch_width, batch_size});
        Tensor target =
            batch_features->slice(0, static_cast<std::int32_t>(index) * value.ne[0], value.ne[0]);
        ops::scatter_bf16_batch(source, *batch_lanes, *batch_valid_columns, target, stream);
        captured_mask |= 1U << index;
        return;
    }
    if (active_tokens > features->ne[1]) {
        throw std::logic_error("DFlash prefill feature capture exceeds its buffer");
    }
    const std::size_t element_bytes = dtype_size(DType::BF16);
    const std::size_t width_bytes   = static_cast<std::size_t>(value.ne[0]) * element_bytes;
    const std::size_t source_pitch  = static_cast<std::size_t>(value.nb[1]);
    const std::size_t target_pitch  = static_cast<std::size_t>(features->nb[1]);
    auto* target                    = static_cast<std::byte*>(features->data) + index * width_bytes;
    CUDA_CHECK(cudaMemcpy2DAsync(target, target_pitch, value.data, source_pitch, width_bytes,
                                 static_cast<std::size_t>(active_tokens), cudaMemcpyDeviceToDevice,
                                 stream));
    captured_mask |= 1U << index;
}

void DFlashFeatureSink::capture_positions(const Tensor& source, cudaStream_t stream) {
    const std::uint32_t complete_mask = layers.size() == 32 ? ~0U : ((1U << layers.size()) - 1U);
    if (captured_mask != complete_mask) {
        throw std::logic_error("DFlash target call did not publish every feature layer");
    }
    if (batch_features != nullptr) {
        if (source.dtype != DType::I32 || source.ne[0] != batch_width ||
            source.ne[1] != batch_size) {
            throw std::logic_error("DFlash batch feature positions are invalid");
        }
        return;
    }
    if (active_tokens <= 0 || source.dtype != DType::I32 || source.ne[0] != active_tokens ||
        positions == nullptr || active_tokens > positions->ne[0]) {
        throw std::logic_error("DFlash feature positions are invalid");
    }
    CUDA_CHECK(cudaMemcpyAsync(positions->data, source.data,
                               static_cast<std::size_t>(active_tokens) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, stream));
}

void DFlashFeatureSink::consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint) {
    if (!consume_prefill || tokens != active_tokens) {
        throw std::logic_error("DFlash prefill feature consumer is unavailable");
    }
    Tensor feature_window  = features->slice(1, 0, tokens);
    Tensor position_window = positions->slice(0, 0, tokens);
    consume_prefill(feature_window, position_window, rewrite_checkpoint);
}

TextContext::TextContext(DeviceContext& ctx, const LoadedModelData& weights, WorkspaceArena& work,
                         qwen3_8_flash_next::PagedKVCacheView kv, LinearAttentionStatePool& state,
                         qwen3_8_flash_next::RoundState& io, Tensor& prefill_hidden,
                         std::uint32_t prefill_chunk, std::uint32_t text_kv_base,
                         qwen3_8_flash_next::PagedKVCacheView mtp_kv,
                         const qwen3_8_flash_next::PagedKVCache* batch_text_kv,
                         const qwen3_8_flash_next::PagedKVCache* batch_mtp_kv, Tensor* ple_state)
    : ctx_(ctx), weights_(weights), work_(work), kv_(kv), mtp_kv_(mtp_kv), state_(state), io_(io),
      prefill_hidden_(prefill_hidden), prefill_chunk_(prefill_chunk), text_kv_base_(text_kv_base),
      batch_text_kv_(batch_text_kv), batch_mtp_kv_(batch_mtp_kv), ple_state_(ple_state) {
    if (prefill_chunk_ == 0 ||
        prefill_chunk_ > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("TextContext effective prefill chunk must fit positive int32");
    }
    if (mtp_enabled() && !io_.mtp_decode && !io_.mtp) {
        throw std::invalid_argument("MTP TextContext requires MTP round state");
    }
    set_linear_state_slots(0, 0);
    bind();
}

TextContext::~TextContext() = default;

void TextContext::set_linear_state_slots(std::int32_t source_slot, std::int32_t destination_slot) {
    if (source_slot < 0 || source_slot >= state_.slot_count() || destination_slot < 0 ||
        destination_slot >= state_.slot_count()) {
        throw std::invalid_argument("TextContext Linear Attention slots are invalid");
    }
    linear_state_source_slot_      = source_slot;
    linear_state_destination_slot_ = destination_slot;
}

void TextContext::set_gdn_state_action(GdnStateAction action,
                                       const GdnReplayRecords* replay_records) {
    if ((action == GdnStateAction::RecordForReplay) != (replay_records != nullptr)) {
        throw std::invalid_argument("TextContext GDN state action has inconsistent records");
    }
    gdn_state_action_ = action;
    replay_records_   = replay_records;
}

void TextContext::bind() {
    using TargetBindings = LoadedModelData;
    using TargetMlp      = MlpWeights;
    const auto bind_mlp  = [](const TargetMlp& source) { return MlpW{&source}; };

    embed_      = &weights_.token_embedding;
    final_norm_ = &weights_.final_norm;
    lm_head_    = &weights_.output_head;
    if (weights_.optimized_proposal) {
        const auto& proposal = *weights_.optimized_proposal;
        set_proposal_head(&proposal.head, static_cast<const std::int32_t*>(proposal.token_ids.data),
                          proposal.head.n);
    }

    if (mtp_enabled()) {
        if (!weights_.mtp) {
            throw std::invalid_argument("MTP state was enabled without materialized MTP weights");
        }
        const auto& source = *weights_.mtp;
        mtp_               = MtpW{&source,
                                  &source.input_projection,
                                  &source.embedding_norm,
                                  &source.hidden_norm,
                                  &source.input_norm,
                                  &source.query_norm,
                                  &source.key_norm,
                                  &source.output,
                                  &source.post_attention_norm,
                                  &source.final_norm};
    }

    for (int layer = 0; layer < kCfg.n_layers; ++layer) {
        if (ModelConfig::is_full(layer)) {
            FullLayerW& out = full_[static_cast<std::size_t>(ModelConfig::full_idx(layer))];
            const auto& source =
                weights_.full_layers[static_cast<std::size_t>(ModelConfig::full_idx(layer))];
            out.input_norm     = &source.input_norm;
            out.projection     = &source.projection;
            out.o_proj         = &source.output;
            out.q_norm         = &source.query_norm;
            out.k_norm         = &source.key_norm;
            out.post_attn_norm = &source.post_attention_norm;
            out.mlp            = bind_mlp(source.post_mixer);
        } else {
            const std::size_t gidx = static_cast<std::size_t>(ModelConfig::gdn_idx(layer));
            GdnLayerW& out         = gdn_[gidx];
            const auto& source     = weights_.gdn_layers[gidx];
            out.input_norm         = &source.input_norm;
            out.projection         = &source.projection;
            out.conv1d             = &source.convolution;
            out.gdn_norm           = &source.norm;
            out.out_proj           = &source.output;
            out.post_attn_norm     = &source.post_attention_norm;
            out.mlp                = bind_mlp(source.post_mixer);
        }
    }
}

const MtpW& TextContext::mtp_weights() const {
    if (!mtp_enabled()) { throw std::runtime_error("MTP draft weights are not enabled"); }
    return mtp_;
}

void TextContext::mtp_forward_stem(const Tensor& ids, const Tensor& hidden,
                                   const Tensor* input_embeddings, Tensor& x, Tensor& ah) {
    cudaStream_t s     = ctx_.stream;
    const int T        = ids.ne[0] * ids.ne[1];
    Tensor flat_ids    = ids.view({T});
    Tensor flat_hidden = hidden.view({kCfg.hidden, T});

    auto roots = workspace_recipe::mtp_stem<TextConfig>(work_, T, input_embeddings == nullptr);
    Tensor emb;
    if (input_embeddings != nullptr) {
        if (input_embeddings->dtype != DType::BF16 || input_embeddings->ne[0] != kCfg.hidden ||
            input_embeddings->numel() != static_cast<std::int64_t>(kCfg.hidden) * T ||
            !input_embeddings->is_contiguous() || input_embeddings->data == nullptr) {
            throw std::invalid_argument("MTP input embeddings shape mismatch");
        }
        emb = input_embeddings->view({kCfg.hidden, T});
    } else {
        emb = roots.embedding;
        ops::embedding(flat_ids, *embed_, emb, s);
    }

    Tensor e = roots.normalized_embedding;
    Tensor h = roots.normalized_hidden;
    ops::rmsnorm(emb, *mtp_.pre_fc_norm_embedding, kCfg.rms_eps, true, e, s);
    ops::rmsnorm(flat_hidden, *mtp_.pre_fc_norm_hidden, kCfg.rms_eps, true, h, s);

    Tensor fc_in = roots.packed_input;
    ops::mtp_pack_fc_input(e, h, fc_in, s);

    x = roots.residual;
    ops::linear(fc_in, *mtp_.fc, x, s, bf16_gemm_);

    ah = roots.attention_hidden;
    ops::rmsnorm(x, *mtp_.input_norm, kCfg.rms_eps, true, ah, s);
}

void TextContext::mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                                   const Tensor& rope_positions,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   Tensor& mtp_hidden) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    const auto projection = workspace_recipe::mtp_attention_projection<TextConfig>(work_, T);
    Tensor q              = projection.query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor k              = projection.key.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor gate           = projection.gate.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor v              = projection.value.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor q_flat         = q.view({kCfg.q_size, T});
    Tensor gate_flat      = gate.view({kCfg.q_size, T});
    Tensor k_flat         = k.view({kCfg.kv_size, T});
    Tensor v_flat         = v.view({kCfg.kv_size, T});
    Variant::mtp_attention_projection(ah, mtp_.payload->attention, q_flat, gate_flat, k_flat,
                                      v_flat, work_, s);

    const auto results = workspace_recipe::mtp_attention_results<TextConfig>(work_, T);
    Tensor qn          = results.normalized_query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor kn          = results.normalized_key.view({kCfg.head_dim, kCfg.n_kv, T});
    ops::rmsnorm(q, *mtp_.q_norm, kCfg.rms_eps, true, qn, s);
    ops::rmsnorm(k, *mtp_.k_norm, kCfg.rms_eps, true, kn, s);
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    ops::rope(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);

    Tensor a = results.attention.view({kCfg.head_dim, kCfg.n_q, T});
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T ||
            active_backend_kv_table_rows_ == nullptr || active_valid_columns_ == nullptr) {
            throw std::logic_error("MTP sequence batch binding is incomplete");
        }
        Tensor q_batch        = qn.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor k_batch        = kn.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor v_batch        = v.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor a_batch        = a.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor position_batch = positions.view({width, active_sequence_batch_});
        ops::causal_softmax_attention(
            q_batch, k_batch, v_batch, position_batch, *active_valid_columns_,
            *active_backend_kv_table_rows_, {kCfg.head_dim, kCfg.n_q, kCfg.n_kv}, kAttnScale,
            batch_mtp_kv_->batch_layer_view(0), envelope, work_, a_batch, s);
    } else {
        ops::causal_softmax_attention(qn, kn, v, positions, Tensor{}, io_.backend_kv_table_row,
                                      {kCfg.head_dim, kCfg.n_q, kCfg.n_kv}, kAttnScale,
                                      batch_mtp_kv_->batch_layer_view(0), envelope, work_, a, s);
    }
    ops::sigmoid_mul(gate, a, s);

    const auto post = workspace_recipe::mtp_post_attention<TextConfig>(work_, T);
    Tensor o        = post.output;
    ops::linear(a.view({kCfg.q_size, T}), *mtp_.o_proj, o, s, bf16_gemm_);
    ops::residual_add(o, x, s);

    Tensor mh = post.post_mixer_hidden;
    ops::rmsnorm(x, *mtp_.post_attn_norm, kCfg.rms_eps, true, mh, s);

    {
        auto post_mixer_scope = work_.scope();
        Variant::mtp_post_mixer(mh, mtp_.payload->post_mixer, x, work_, s);
    }

    Tensor flat_mtp_hidden = mtp_hidden.view({kCfg.hidden, T});
    ops::rmsnorm(x, *mtp_.norm, kCfg.rms_eps, true, flat_mtp_hidden, s);
}

void TextContext::mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                                   const Tensor& rope_positions,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   Tensor& mtp_hidden, const Tensor* input_embeddings,
                                   Tensor* predictor_hidden, Tensor* selected_qsa_indices,
                                   const Tensor* reused_qsa_indices) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    nvtx::ScopedRange forward_range(nvtx::Name::MtpForward, nvtx::Category::Mtp,
                                    static_cast<std::uint64_t>(ids.numel()));
    auto scratch_scope = work_.scope();
#ifdef NINFER_QWEN38_FLASH_NEXT
    if (predictor_hidden == nullptr) {
        throw std::invalid_argument("Flash-Next MTP requires predictor hidden output");
    }
    mtp_forward_flash_next(ids, hidden, positions, rope_positions, envelope, mtp_hidden,
                           input_embeddings, *predictor_hidden, selected_qsa_indices,
                           reused_qsa_indices);
    return;
#endif
    Tensor x;
    Tensor ah;
    mtp_forward_stem(ids, hidden, input_embeddings, x, ah);
    mtp_forward_tail(x, ah, positions, rope_positions, envelope, mtp_hidden);
}

void TextContext::mtp_forward_flash_next(const Tensor& ids, const Tensor& hidden,
                                         const Tensor& positions, const Tensor& rope_positions,
                                         ops::CausalAttentionExecutionEnvelope envelope,
                                         Tensor& sample_hidden, const Tensor* input_embeddings,
                                         Tensor& predictor_hidden, Tensor* selected_qsa_indices,
                                         const Tensor* reused_qsa_indices) {
    NINFER_PERF_SCOPE("ninfer.region/1|predictor");
#ifndef NINFER_QWEN38_FLASH_NEXT
    (void)ids;
    (void)hidden;
    (void)positions;
    (void)rope_positions;
    (void)envelope;
    (void)sample_hidden;
    (void)input_embeddings;
    (void)predictor_hidden;
    throw std::logic_error("Flash-Next MTP selected by a non-Flash target");
#else
    cudaStream_t stream = ctx_.stream;
    const int tokens    = static_cast<int>(ids.numel());
    const int batch     = active_sequence_batch_ == 0 ? 1 : active_sequence_batch_;
    const int width     = tokens / batch;
    if (tokens <= 0 || width * batch != tokens) {
        throw std::invalid_argument("Flash-Next MTP token geometry is invalid");
    }
    require_tensor_shape(hidden, DType::BF16, {10240, tokens},
                         "Flash-Next MTP input predictor hidden");
    require_tensor_shape(sample_hidden, DType::BF16, {2560, tokens},
                         "Flash-Next MTP sample hidden");
    require_tensor_shape(predictor_hidden, DType::BF16, {10240, tokens},
                         "Flash-Next MTP output predictor hidden");
    Tensor embedding = work_.alloc(DType::BF16, {2560, tokens});
    if (input_embeddings != nullptr) {
        require_tensor_shape(*input_embeddings, DType::BF16, {2560, tokens},
                             "Flash-Next MTP input embeddings");
        CUDA_CHECK(cudaMemcpyAsync(embedding.data, input_embeddings->data, embedding.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
    } else {
        ops::embedding(ids.view({tokens}), *embed_, embedding, stream);
    }
    Tensor normalized_embedding = work_.alloc(DType::BF16, {2560, tokens});
    Tensor projected_embedding  = work_.alloc(DType::BF16, {2560, tokens});
    ops::rmsnorm(embedding, mtp_.payload->embedding_norm, kCfg.rms_eps, true, normalized_embedding,
                 stream);
    ops::linear(normalized_embedding, mtp_.payload->embedding_projection, projected_embedding,
                stream, bf16_gemm_);

    Tensor normalized_hidden = work_.alloc(DType::BF16, {10240, tokens});
    Tensor hyper             = work_.alloc(DType::BF16, {10240, tokens});
    ops::rmsnorm(hidden, mtp_.payload->hidden_norm, kCfg.rms_eps, true, normalized_hidden, stream);
    Tensor normalized_hidden_branches = normalized_hidden.view({2560, 4 * tokens});
    Tensor hyper_branches             = hyper.view({2560, 4 * tokens});
    ops::linear(normalized_hidden_branches, mtp_.payload->hidden_projection, hyper_branches, stream,
                bf16_gemm_);
    ops::hyperconnection_add_repeated(projected_embedding, hyper, stream);
    Tensor block_input  = work_.alloc(DType::BF16, {2560, tokens});
    Tensor block_output = work_.alloc(DType::BF16, {2560, tokens});
    Tensor injection    = work_.alloc(DType::BF16, {4, tokens});
    ops::hyperconnection_mix(hyper, mtp_.payload->attention_hc, block_input, &injection, work_,
                             stream);

    Tensor cache_positions = positions.view({width, batch});
    Tensor mrope_positions;
    if (rope_positions.ne[2] == 3 || rope_positions.ne[1] == 3) {
        mrope_positions = rope_positions.view({width, batch, 3});
    } else {
        mrope_positions = work_.alloc(DType::I32, {width, batch, 3});
        ops::flash_next_expand_text_positions(rope_positions.view({width, batch}), mrope_positions,
                                              stream);
    }
    Tensor valid;
    if (active_valid_columns_ != nullptr) {
        valid = *active_valid_columns_;
    } else {
        valid = work_.alloc(DType::I32, {batch});
        // Captured schedules must not retain a pointer into a temporary host vector.
        ops::fill_i32(valid, width, stream);
    }
    const Tensor& rows = active_backend_kv_table_rows_ != nullptr ? *active_backend_kv_table_rows_
                                                                  : io_.backend_kv_table_row;
    const auto cache   = active_sequence_batch_ == 0
                             ? single_row_paged_kv_batch_view(mtp_kv_.layer_view(0))
                             : batch_mtp_kv_->batch_layer_view(0);
    ops::flash_next_qsa(
        block_input, cache_positions, mrope_positions, valid, rows, mtp_.payload->attention, cache,
        envelope, block_output, work_, stream, bf16_gemm_,
        {.selected_indices = selected_qsa_indices, .reused_indices = reused_qsa_indices});
    ops::hyperconnection_combine_mix(hyper, block_output, injection, mtp_.payload->mlp_hc,
                                     block_input, &injection, work_, stream);
    ops::flash_next_moe(block_input, mtp_.payload->post_mixer, block_output, work_, stream,
                        bf16_gemm_, envelope.max_visible_keys > ops::kFlashNextQsaSelectedTokens);
    ops::hyperconnection_combine(hyper, block_output, injection, stream);
    CUDA_CHECK(cudaMemcpyAsync(predictor_hidden.data, hyper.data, hyper.bytes(),
                               cudaMemcpyDeviceToDevice, stream));

    const ops::HyperConnectionWeights final_weights{
        .norm = mtp_.payload->final_hc.norm,
        .down = mtp_.payload->final_hc.down,
        .up   = mtp_.payload->final_hc.up,
    };
    ops::hyperconnection_mix(hyper, final_weights, sample_hidden, nullptr, work_, stream);
    (void)envelope;
#endif
}

void TextContext::mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden,
                                    const Tensor* input_embeddings, const Tensor& positions,
                                    const Tensor& rope_positions,
                                    ops::CausalAttentionExecutionEnvelope envelope,
                                    bool final_chunk, Tensor* final_hidden, Tensor* logits,
                                    Tensor* draft_token) {
    if (!mtp_kv_.valid()) { throw std::runtime_error("MTP prefill is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP prefill chunk T must be in [1,prefill_chunk]");
    }
    nvtx::ScopedRange mtp_prefill_range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                                        static_cast<std::uint64_t>(T));
    require_tensor_shape(ids, DType::I32, {T}, "MTP prefill ids");
#ifndef NINFER_QWEN38_FLASH_NEXT
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "MTP prefill hidden");
#endif
    require_tensor_shape(positions, DType::I32, {T}, "MTP prefill positions");
    if (rope_positions.dtype != DType::I32 || rope_positions.ne[0] != T ||
        (rope_positions.ne[1] != 1 && rope_positions.ne[1] != 3) || rope_positions.ne[2] != 1 ||
        rope_positions.ne[3] != 1 || !rope_positions.is_contiguous() ||
        rope_positions.data == nullptr) {
        throw std::invalid_argument("MTP prefill rope positions must be [T] or [T,3]");
    }
#ifdef NINFER_QWEN38_FLASH_NEXT
    require_tensor_shape(hidden, DType::BF16, {4 * kCfg.hidden, T},
                         "Flash-Next MTP prefill predictor hidden");
    if (final_chunk) {
        if (final_hidden == nullptr || logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("Flash-Next MTP final prefill outputs are required");
        }
        require_tensor_shape(*final_hidden, DType::BF16, {4 * kCfg.hidden, 1},
                             "Flash-Next MTP final predictor hidden");
        require_tensor_shape(*logits, DType::BF16, {kCfg.vocab, 1},
                             "Flash-Next MTP final prefill logits");
        require_tensor_shape(*draft_token, DType::I32, {1},
                             "Flash-Next MTP final prefill draft token");
    }
    auto flash_scope = work_.scope();
    Tensor sample    = work_.alloc(DType::BF16, {kCfg.hidden, T});
    Tensor predictor = work_.alloc(DType::BF16, {4 * kCfg.hidden, T});
    mtp_forward_core(ids, hidden, positions, rope_positions, envelope, sample, input_embeddings,
                     &predictor);
    if (final_chunk) {
        const Tensor last_predictor = predictor.slice(1, T - 1, 1);
        CUDA_CHECK(cudaMemcpyAsync(final_hidden->data, last_predictor.data, last_predictor.bytes(),
                                   cudaMemcpyDeviceToDevice, ctx_.stream));
        Tensor last_sample = sample.slice(1, T - 1, 1);
        proposal_argmax(last_sample, *logits, *draft_token);
    }
    return;
#else
    if (final_chunk) {
        if (final_hidden == nullptr || logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP final prefill outputs are required");
        }
        require_tensor_shape(*final_hidden, DType::BF16, {kCfg.hidden, 1},
                             "MTP final prefill hidden");
        require_tensor_shape(*logits, DType::BF16, {kCfg.vocab, 1}, "MTP final prefill logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP final prefill draft token");
    }

    cudaStream_t s     = ctx_.stream;
    auto scratch_scope = work_.scope();
    Tensor x_last;
    Tensor ah_last;
    if (final_chunk) {
        x_last  = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ah_last = work_.alloc(DType::BF16, {kCfg.hidden, 1});
    }

    {
        auto bulk_scope = work_.scope();
        Tensor x;
        Tensor ah;
        mtp_forward_stem(ids, hidden, input_embeddings, x, ah);

        Tensor k_flat = work_.alloc(DType::BF16, {kCfg.kv_size, T});
        Tensor v_flat = work_.alloc(DType::BF16, {kCfg.kv_size, T});
        Variant::mtp_kv_projection(ah, mtp_.payload->attention, k_flat, v_flat, work_, s);
        Tensor k  = k_flat.view({kCfg.head_dim, kCfg.n_kv, T});
        Tensor v  = v_flat.view({kCfg.head_dim, kCfg.n_kv, T});
        Tensor kn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
        ops::rmsnorm(k, *mtp_.k_norm, kCfg.rms_eps, true, kn, s);
        ops::rope(rope_positions, kCfg.rotary_dim, kCfg.rope_theta, kn, s);
        ops::kv_cache_append(kn, v, positions, mtp_kv_.layer_view(0), s);

        if (final_chunk) {
            const std::size_t column_bytes =
                static_cast<std::size_t>(kCfg.hidden) * dtype_size(DType::BF16);
            const auto* x_src  = static_cast<const unsigned char*>(x.data) +
                                 static_cast<std::size_t>(T - 1) * column_bytes;
            const auto* ah_src = static_cast<const unsigned char*>(ah.data) +
                                 static_cast<std::size_t>(T - 1) * column_bytes;
            CUDA_CHECK(
                cudaMemcpyAsync(x_last.data, x_src, column_bytes, cudaMemcpyDeviceToDevice, s));
            CUDA_CHECK(
                cudaMemcpyAsync(ah_last.data, ah_src, column_bytes, cudaMemcpyDeviceToDevice, s));
        }
    }

    if (final_chunk) {
        Tensor q_flat    = work_.alloc(DType::BF16, {kCfg.q_size, 1});
        Tensor gate_flat = work_.alloc(DType::BF16, {kCfg.q_size, 1});
        Variant::mtp_q_gate_projection(ah_last, mtp_.payload->attention, q_flat, gate_flat, work_,
                                       s);
        Tensor q    = q_flat.view({kCfg.head_dim, kCfg.n_q, 1});
        Tensor gate = gate_flat.view({kCfg.head_dim, kCfg.n_q, 1});
        Tensor qn   = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, 1});
        ops::rmsnorm(q, *mtp_.q_norm, kCfg.rms_eps, true, qn, s);
        Tensor last_position = positions.slice(0, T - 1, 1);
        Tensor last_rope_position;
        if (rope_positions.ne[1] == 1) {
            last_rope_position = rope_positions.slice(0, T - 1, 1);
        } else {
            last_rope_position = work_.alloc(DType::I32, {1, 3});
            for (int axis = 0; axis < 3; ++axis) {
                const auto* src = static_cast<const std::int32_t*>(rope_positions.data) +
                                  static_cast<std::size_t>(axis) * T + (T - 1);
                auto* dst       = static_cast<std::int32_t*>(last_rope_position.data) + axis;
                CUDA_CHECK(
                    cudaMemcpyAsync(dst, src, sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
            }
        }
        ops::rope(last_rope_position, kCfg.rotary_dim, kCfg.rope_theta, qn, s);

        Tensor a = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, 1});
        ops::causal_softmax_attention_cached(qn, last_position,
                                             {kCfg.head_dim, kCfg.n_q, kCfg.n_kv}, kAttnScale,
                                             mtp_kv_.layer_view(0), envelope, work_, a, s);
        ops::sigmoid_mul(gate, a, s);

        Tensor o = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ops::linear(a.view({kCfg.q_size, 1}), *mtp_.o_proj, o, s, bf16_gemm_);
        ops::residual_add(o, x_last, s);

        Tensor mh = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ops::rmsnorm(x_last, *mtp_.post_attn_norm, kCfg.rms_eps, true, mh, s);
        {
            auto post_mixer_scope = work_.scope();
            Variant::mtp_post_mixer(mh, mtp_.payload->post_mixer, x_last, work_, s);
        }
        ops::rmsnorm(x_last, *mtp_.norm, kCfg.rms_eps, true, *final_hidden, s);
        proposal_argmax(*final_hidden, *logits, *draft_token);
    }
#endif
}

void TextContext::proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens) {
    const int T = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "proposal hidden");
    require_tensor_shape(proposal_tokens, DType::I32, {T}, "proposal tokens");
    require_tensor_window(logits, DType::BF16, kCfg.vocab, T, "proposal logits");
    nvtx::ScopedRange proposal_range(nvtx::Name::MtpProposal, nvtx::Category::Mtp,
                                     static_cast<std::uint64_t>(T));
    if (proposal_head_ != nullptr) {
        Tensor proposal_logits = work_.alloc(DType::BF16, {proposal_head_n_, T});
        ops::linear(hidden, *proposal_head_, proposal_logits, ctx_.stream, bf16_gemm_);
        if constexpr (Variant::flash_next) {
            constexpr int kRerankTile        = 512;
            constexpr int kCandidatesPerTile = 2;
            const int candidate_rows =
                ((Variant::draft_head_valid_rows + kRerankTile - 1) / kRerankTile) *
                kCandidatesPerTile;
            Tensor candidate_ids    = work_.alloc(DType::I32, {candidate_rows, T});
            Tensor candidate_scores = work_.alloc(DType::FP32, {candidate_rows, T});
            ops::shortlist_exact_argmax(hidden, proposal_logits, Variant::draft_head_valid_rows,
                                        *lm_head_, proposal_head_ids_, candidate_ids,
                                        candidate_scores, proposal_tokens, ctx_.stream);
        } else {
            ops::argmax(proposal_logits, proposal_tokens, proposal_head_n_, ctx_.stream);
            ops::proposal_remap_token_ids(proposal_tokens, proposal_head_ids_, proposal_head_n_,
                                          ctx_.stream);
        }
    } else {
        Tensor output_logits = matrix_window(logits, T);
        ops::linear(hidden, *lm_head_, output_logits, ctx_.stream, bf16_gemm_);
        ops::argmax(output_logits, proposal_tokens, kCfg.token_domain, ctx_.stream);
    }
}

void TextContext::mtp_forward_batch(const Tensor& ids, const Tensor& hidden,
                                    const Tensor& positions,
                                    ops::CausalAttentionExecutionEnvelope envelope,
                                    Tensor& mtp_hidden, int logits_column, Tensor* logits,
                                    Tensor* draft_token, const Tensor* explicit_rope_positions,
                                    const Tensor* input_embeddings, Tensor* predictor_hidden) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP batch T must be in [1,prefill_chunk]");
    }
    require_tensor_shape(ids, DType::I32, {T}, "MTP ids");
    require_tensor_shape(positions, DType::I32, {T}, "MTP positions");
#ifdef NINFER_QWEN38_FLASH_NEXT
    require_tensor_shape(hidden, DType::BF16, {4 * kCfg.hidden, T}, "MTP predictor hidden");
    if (predictor_hidden == nullptr) {
        throw std::invalid_argument("Flash-Next MTP predictor output is required");
    }
#else
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "MTP hidden");
#endif
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, T}, "MTP output hidden");
    if (logits_column >= T) { throw std::invalid_argument("MTP logits column out of range"); }
    if (logits_column >= 0) {
        if (logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP logits and draft_token outputs are required");
        }
        require_tensor_shape(*logits, DType::BF16, {kCfg.vocab, 1}, "MTP logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP draft token");
    }

    auto position_scope = work_.scope();
    Tensor generated_rope_positions;
    const Tensor* rope_positions = explicit_rope_positions;
    if (rope_positions == nullptr) {
        generated_rope_positions = work_.alloc(DType::I32, {T});
        ops::offset_i32_positions(positions, io_.rope_delta, generated_rope_positions, ctx_.stream);
        rope_positions = &generated_rope_positions;
    } else if (rope_positions->dtype != DType::I32 || rope_positions->ne[0] != T ||
               (rope_positions->ne[1] != 1 && rope_positions->ne[1] != 3) ||
               rope_positions->ne[2] != 1 || rope_positions->ne[3] != 1 ||
               !rope_positions->is_contiguous() || rope_positions->data == nullptr) {
        throw std::invalid_argument("MTP explicit rope positions must be [T] or [T,3]");
    }
    mtp_forward_core(ids, hidden, positions, *rope_positions, envelope, mtp_hidden,
                     input_embeddings, predictor_hidden);

    if (logits_column >= 0) {
        auto logits_scope = work_.scope();
        Tensor col        = mtp_hidden.slice(1, logits_column, 1);
        proposal_argmax(col, *logits, *draft_token);
    }
}

void TextContext::mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                                      const Tensor& position,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token,
                                      Tensor* predictor_hidden) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    require_tensor_shape(token, DType::I32, {1}, "MTP AR token");
    require_tensor_shape(position, DType::I32, {1}, "MTP AR position");
#ifdef NINFER_QWEN38_FLASH_NEXT
    require_tensor_shape(previous_hidden, DType::BF16, {4 * kCfg.hidden, 1},
                         "MTP AR previous predictor hidden");
    if (predictor_hidden == nullptr) {
        throw std::invalid_argument("Flash-Next MTP AR predictor output is required");
    }
#else
    require_tensor_shape(previous_hidden, DType::BF16, {kCfg.hidden, 1}, "MTP AR previous hidden");
#endif
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, 1}, "MTP AR output hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, 1}, "MTP AR logits");
    require_tensor_shape(draft_token, DType::I32, {1}, "MTP AR draft token");

    auto position_scope  = work_.scope();
    Tensor rope_position = work_.alloc(DType::I32, {1});
    ops::offset_i32_positions(position, io_.rope_delta, rope_position, ctx_.stream);
    mtp_forward_core(token, previous_hidden, position, rope_position, envelope, mtp_hidden, nullptr,
                     predictor_hidden);
    auto logits_scope = work_.scope();
    proposal_argmax(mtp_hidden, logits, draft_token);
}

void TextContext::ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                                        const Tensor& rope_positions, const Tensor& kv_table_rows,
                                        const Tensor& linear_state_source_slots,
                                        const Tensor& linear_state_destination_slots,
                                        ops::CausalAttentionExecutionEnvelope envelope,
                                        Tensor& hidden, Tensor& logits,
                                        const Tensor* ple_embeddings) {
    const std::int32_t batch = ids.ne[0];
    if (batch <= 0 || batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("ordinary decode batch size must be in [1,8]");
    }
    require_tensor_shape(ids, DType::I32, {batch}, "ordinary decode ids");
    require_tensor_shape(cache_positions, DType::I32, {batch}, "ordinary decode cache positions");
    require_tensor_shape(rope_positions, DType::I32, {batch}, "ordinary decode RoPE positions");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "ordinary decode KV rows");
    require_tensor_shape(linear_state_source_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention source slots");
    require_tensor_shape(linear_state_destination_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention destination slots");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, batch}, "ordinary decode hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, batch}, "ordinary decode logits");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_causal_attention_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> source_binding(active_linear_state_source_slots_,
                                                  &linear_state_source_slots);
        ScopedValue<const Tensor*> destination_binding(active_linear_state_destination_slots_,
                                                       &linear_state_destination_slots);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, 1);
        ScopedValue<const Tensor*> ple_binding(active_ple_embeddings_, ple_embeddings);

        Tensor x = work_.alloc(DType::BF16, {kCfg.hidden, batch});
        ops::embedding(ids, *embed_, x, stream);
        NullTap tap;
        run_layers(x, Phase::Verify, tap);
        if constexpr (Variant::flash_next) {
            CUDA_CHECK(
                cudaMemcpyAsync(hidden.data, x.data, x.bytes(), cudaMemcpyDeviceToDevice, stream));
        } else {
            ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, hidden, stream);
        }
        ops::linear(hidden, *lm_head_, logits, stream, bf16_gemm_);
    }
    work_.reset();
}

template <class Tap>
void TextContext::target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           const Tensor& linear_state_source_slots,
                                           ops::CausalAttentionExecutionEnvelope envelope,
                                           Tensor& hidden, Tensor& logits, Tensor& target_tokens,
                                           Tap& tap) {
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kDFlashDecodeMaximumWidth) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("target verify batch shape is outside the supported domain");
    }
    const std::int32_t columns = width * batch;
    require_tensor_shape(ids, DType::I32, {width, batch}, "target verify batch ids");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "target verify batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "target verify batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "target verify batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "target verify batch KV rows");
    require_tensor_shape(linear_state_source_slots, DType::I32, {batch},
                         "target verify batch Linear Attention slots");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, width, batch},
                         "target verify batch hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, width, batch},
                         "target verify batch logits");
    require_tensor_shape(target_tokens, DType::I32, {width, batch}, "target verify batch tokens");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_causal_attention_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> state_binding(active_linear_state_source_slots_,
                                                 &linear_state_source_slots);
        ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);

        Tensor x        = work_.alloc(DType::BF16, {kCfg.hidden, columns});
        Tensor flat_ids = ids.view({columns});
        ops::embedding(flat_ids, *embed_, x, stream);
        if constexpr (Tap::enabled) { tap.begin(x); }
        run_layers(x, Phase::Verify, tap);
        if constexpr (requires { tap.capture_positions(cache_positions, stream); }) {
            tap.capture_positions(cache_positions, stream);
        }
        Tensor flat_hidden = hidden.view({kCfg.hidden, columns});
        Tensor flat_logits = logits.view({kCfg.vocab, columns});
        Tensor flat_tokens = target_tokens.view({columns});
        if constexpr (Variant::flash_next) {
            CUDA_CHECK(cudaMemcpyAsync(flat_hidden.data, x.data, x.bytes(),
                                       cudaMemcpyDeviceToDevice, stream));
        } else {
            ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, flat_hidden, stream);
        }
        ops::linear(flat_hidden, *lm_head_, flat_logits, stream, bf16_gemm_);
        qwen3_8_flash_next::detail::capture_target_logits(
            "verify", ids, cache_positions, &valid_columns, kv_table_rows, flat_logits, stream);
        ops::argmax(flat_logits, flat_tokens, kCfg.token_domain, stream);
    }
    work_.reset();
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows,
                                      const Tensor& linear_state_source_slots,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& hidden, Tensor& logits, Tensor& target_tokens) {
    NullTap tap;
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_source_slots, envelope, hidden, logits, target_tokens,
                             tap);
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows,
                                      const Tensor& linear_state_source_slots,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& hidden, Tensor& logits, Tensor& target_tokens,
                                      DFlashFeatureSink& sink) {
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_source_slots, envelope, hidden, logits, target_tokens,
                             sink);
}

void TextContext::mtp_forward_decode_batch(
    const Tensor& ids, const Tensor& hidden, const Tensor& cache_positions,
    const Tensor& rope_positions, const Tensor& valid_columns, const Tensor& kv_table_rows,
    ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden, Tensor* predictor_hidden,
    Tensor* selected_qsa_indices, const Tensor* reused_qsa_indices) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kMaximumMtpDraftTokens + 1) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("MTP decode batch shape is outside the supported domain");
    }
    require_tensor_shape(ids, DType::I32, {width, batch}, "MTP decode batch ids");
#ifdef NINFER_QWEN38_FLASH_NEXT
    require_tensor_shape(hidden, DType::BF16, {4 * kCfg.hidden, width, batch},
                         "MTP decode batch target predictor hidden");
    if (predictor_hidden == nullptr) {
        throw std::invalid_argument("Flash-Next MTP decode predictor output is required");
    }
#else
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, width, batch},
                         "MTP decode batch target hidden");
#endif
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "MTP decode batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "MTP decode batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "MTP decode batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "MTP decode batch KV rows");
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, width, batch},
                         "MTP decode batch hidden");

    ScopedValue<const Tensor*> backend_binding(active_backend_kv_table_rows_, &kv_table_rows);
    ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
    ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
    ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
#ifdef NINFER_QWEN38_FLASH_NEXT
    Tensor flat_hidden = hidden.view({4 * kCfg.hidden, width * batch});
    Tensor flat_mtp_hidden = mtp_hidden.view({kCfg.hidden, width * batch});
    Tensor flat_predictor_hidden =
        predictor_hidden->view({4 * kCfg.hidden, width * batch});
    mtp_forward_core(ids, flat_hidden, cache_positions, rope_positions, envelope, flat_mtp_hidden,
                     nullptr, &flat_predictor_hidden, selected_qsa_indices, reused_qsa_indices);
#else
    mtp_forward_core(ids, hidden, cache_positions, rope_positions, envelope, mtp_hidden, nullptr,
                     predictor_hidden, selected_qsa_indices, reused_qsa_indices);
#endif
}

void TextContext::mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens) {
    const std::int32_t batch = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, batch}, "MTP proposal batch hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, batch}, "MTP proposal batch logits");
    require_tensor_shape(draft_tokens, DType::I32, {batch}, "MTP proposal batch tokens");
    proposal_argmax(hidden, logits, draft_tokens);
}

void TextContext::attn_mix(const FullLayerW& w, Tensor& x, int fidx, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    if (active_causal_attention_envelope_ == nullptr) {
        throw std::logic_error("Text GQA execution envelope is not set");
    }

    const auto projection = workspace_recipe::text_attention_projection<TextConfig>(work_, T);
    Tensor h              = projection.hidden;
    ops::rmsnorm(x, *w.input_norm, kCfg.rms_eps, true, h, s);

    Tensor q         = projection.query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor gate      = projection.gate.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor k         = projection.key.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor v         = projection.value.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor q_flat    = q.view({kCfg.q_size, T});
    Tensor gate_flat = gate.view({kCfg.q_size, T});
    Tensor k_flat    = k.view({kCfg.kv_size, T});
    Tensor v_flat    = v.view({kCfg.kv_size, T});
    Variant::attention_projection(h, *w.projection, q_flat, gate_flat, k_flat, v_flat, ph, work_,
                                  s);

    const auto results = workspace_recipe::text_attention_results<TextConfig>(work_, T);
    Tensor qn          = results.normalized_query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor kn          = results.normalized_key.view({kCfg.head_dim, kCfg.n_kv, T});
    ops::rmsnorm(q, *w.q_norm, kCfg.rms_eps, true, qn, s);
    ops::rmsnorm(k, *w.k_norm, kCfg.rms_eps, true, kn, s);
    const Tensor& cache_positions =
        active_cache_positions_ != nullptr ? *active_cache_positions_ : io_.pos;
    const Tensor& rope_positions =
        active_rope_positions_ != nullptr ? *active_rope_positions_ : io_.rope_pos;
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    ops::rope(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);

    Tensor a = results.attention.view({kCfg.head_dim, kCfg.n_q, T});
    const Tensor& kv_table_rows =
        active_kv_table_rows_ != nullptr ? *active_kv_table_rows_ : io_.text_kv_table_row;
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("Text sequence batch binding does not match aggregate columns");
        }
        Tensor q_batch        = qn.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor k_batch        = kn.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor v_batch        = v.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor a_batch        = a.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor position_batch = cache_positions.view({width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        ops::causal_softmax_attention(q_batch, k_batch, v_batch, position_batch, valid,
                                      kv_table_rows, {kCfg.head_dim, kCfg.n_q, kCfg.n_kv},
                                      kAttnScale, batch_text_kv_->batch_layer_view(fidx),
                                      *active_causal_attention_envelope_, work_, a_batch, s);
    } else {
        ops::causal_softmax_attention(qn, kn, v, cache_positions, Tensor{}, kv_table_rows,
                                      {kCfg.head_dim, kCfg.n_q, kCfg.n_kv}, kAttnScale,
                                      batch_text_kv_->batch_layer_view(fidx),
                                      *active_causal_attention_envelope_, work_, a, s);
    }
    ops::sigmoid_mul(gate, a, s);

    Variant::attention_output_projection(a.view({kCfg.q_size, T}), *w.o_proj, x, ph, work_, s);
}

void TextContext::gdn_mix(const GdnLayerW& w, Tensor& x, int gidx, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    const auto control = workspace_recipe::gdn_control<TextConfig>(work_, T);
    Tensor h           = control.hidden;
    Tensor g           = control.g;
    Tensor beta        = control.beta;
    Variant::gdn_norm_control_projection(x, *w.input_norm, kCfg.rms_eps, *w.projection, h, g, beta,
                                         work_, ctx_.execution_view());

    const auto projection = workspace_recipe::gdn_projection<TextConfig>(work_, T);
    Tensor z              = projection.output_gate.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor qc             = projection.query;
    Tensor kc             = projection.key;
    Tensor vc             = projection.value;
    if (ph == Phase::Verify) {
        if (active_sequence_batch_ == 0 || active_linear_state_source_slots_ == nullptr) {
            throw std::logic_error(
                "Verify GDN requires an explicit sequence batch and state slots");
        }
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("GDN sequence batch binding does not match aggregate columns");
        }
        if (gdn_state_action_ == GdnStateAction::UpdateInPlace && width != 1) {
            throw std::logic_error("In-place batched GDN update requires width one");
        }
        Tensor projection_input = h.view({kCfg.hidden, width, active_sequence_batch_});
        Tensor query_output     = qc.view({kCfg.key_dim, width, active_sequence_batch_});
        Tensor key_output       = kc.view({kCfg.key_dim, width, active_sequence_batch_});
        Tensor value_output     = vc.view({kCfg.value_dim, width, active_sequence_batch_});
        Tensor gate_output      = z.view({kCfg.value_dim, width, active_sequence_batch_});
        Tensor conv_states      = state_.layer_view(static_cast<std::uint32_t>(gidx)).conv;
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            if (replay_records_ == nullptr) {
                throw std::logic_error("Replay-record GDN has no record storage");
            }
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            Variant::gdn_input_projection_record(
                projection_input, *w.projection, *w.conv1d, conv_states, valid,
                *active_linear_state_source_slots_, records.conv, query_output, key_output,
                value_output, gate_output, ph, work_, s);
        } else {
            Variant::gdn_input_projection_snapshot(
                projection_input, *w.projection, *w.conv1d, conv_states, valid,
                *active_linear_state_source_slots_, *active_linear_state_destination_slots_,
                query_output, key_output, value_output, gate_output, ph, work_, s);
        }
    } else {
        Tensor qkv = workspace_recipe::gdn_prefill_conv<TextConfig>(work_, T);
        Variant::gdn_input_projection(h, *w.projection, qkv, z, ph, work_, s);
        Tensor conv_state_in =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_source_slot_);
        Tensor conv_state_out =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_destination_slot_);
        ops::causal_conv1d_silu_split(qkv, *w.conv1d, conv_state_in, conv_state_out, qc, kc, vc, s);
    }

    Tensor q_recurrent = qc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});
    Tensor k_recurrent = kc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});

    Tensor vv = vc.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor o  = workspace_recipe::gdn_recurrent_output<TextConfig>(work_, T).view(
        {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    if (ph == Phase::Verify) {
        Tensor recurrent_states  = state_.layer_view(static_cast<std::uint32_t>(gidx)).recurrent;
        const std::int32_t width = active_sequence_width_;
        Tensor q_batch =
            q_recurrent.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, width, active_sequence_batch_});
        Tensor k_batch =
            k_recurrent.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, width, active_sequence_batch_});
        Tensor v_batch = vv.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, width, active_sequence_batch_});
        Tensor g_batch = g.view({kCfg.gdn_v_heads, width, active_sequence_batch_});
        Tensor beta_batch = beta.view({kCfg.gdn_v_heads, width, active_sequence_batch_});
        Tensor out_batch =
            o.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            ops::gated_delta_net_replay_record(q_batch, k_batch, v_batch, g_batch, beta_batch,
                                               kGdnScale, recurrent_states, valid,
                                               *active_linear_state_source_slots_, records.key,
                                               records.value, records.gate, out_batch, s);
        } else {
            ops::gated_delta_net_batch_update(
                q_batch, k_batch, v_batch, g_batch, beta_batch, kGdnScale,
                /*normalize_qk=*/true, recurrent_states, *active_linear_state_source_slots_,
                *active_linear_state_destination_slots_, out_batch, s);
        }
    } else {
        Tensor recurrent_state_in =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_source_slot_);
        Tensor recurrent_state_out =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_destination_slot_);
        ops::gated_delta_net(q_recurrent, k_recurrent, vv, g, beta, kGdnScale,
                             /*normalize_qk=*/true, work_, recurrent_state_in, recurrent_state_out,
                             o, s);
    }

    Tensor on = workspace_recipe::gdn_normalized_output<TextConfig>(work_, T).view(
        {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    ops::gated_rmsnorm(o, *w.gdn_norm, z, kCfg.rms_eps, on, s);

    Variant::gdn_output_projection(on.view({kCfg.value_dim, T}), *w.out_proj, x, ph, work_, s);
}

void TextContext::mlp_tail(const Tensor* post_norm, const MlpW& m, Tensor& x, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    Tensor h       = workspace_recipe::post_mixer_hidden<TextConfig>(work_, T);
    ops::rmsnorm(x, *post_norm, kCfg.rms_eps, true, h, s);

    Variant::post_mixer(h, *m.payload, x, ph, work_, s);
}

template <class Tap>
void TextContext::run_layers(Tensor& x, Phase ph, Tap& tap) {
    if constexpr (Variant::flash_next) {
        run_flash_next_layers(x, ph);
        return;
    }
    const bool prefill = ph == Phase::Prefill;
    for (int layer = 0; layer < kCfg.n_layers; ++layer) {
        if (ModelConfig::is_full(layer)) {
            const int fidx         = ModelConfig::full_idx(layer);
            const FullLayerW& full = full_.at(static_cast<std::size_t>(fidx));
            nvtx::ScopedRange layer_range(
                prefill ? nvtx::Name::PrefillLayerFull : nvtx::Name::VerifyLayerFull,
                nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
            {
                nvtx::ScopedRange mixer_range(
                    prefill ? nvtx::Name::PrefillAttention : nvtx::Name::VerifyAttention,
                    nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
                auto mixer_scope = work_.scope();
                attn_mix(full, x, fidx, ph);
            }
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto mlp_scope = work_.scope();
                mlp_tail(full.post_attn_norm, full.mlp, x, ph);
                if constexpr (Tap::enabled) { tap.capture_layer(layer, x, ctx_.stream); }
            }
        } else {
            const int gidx       = ModelConfig::gdn_idx(layer);
            const GdnLayerW& gdn = gdn_.at(static_cast<std::size_t>(gidx));
            nvtx::ScopedRange layer_range(prefill ? nvtx::Name::PrefillLayerGdn
                                                  : nvtx::Name::VerifyLayerGdn,
                                          nvtx::Category::Gdn, static_cast<std::uint64_t>(layer));
            {
                nvtx::ScopedRange mixer_range(
                    prefill ? nvtx::Name::PrefillGdn : nvtx::Name::VerifyGdn, nvtx::Category::Gdn,
                    static_cast<std::uint64_t>(layer));
                auto mixer_scope = work_.scope();
                gdn_mix(gdn, x, gidx, ph);
            }
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto mlp_scope = work_.scope();
                mlp_tail(gdn.post_attn_norm, gdn.mlp, x, ph);
                if constexpr (Tap::enabled) { tap.capture_layer(layer, x, ctx_.stream); }
            }
        }
    }
}

void TextContext::run_flash_next_layers(Tensor& x, Phase ph) {
    NINFER_PERF_SCOPE(ph == Phase::Prefill ? "ninfer.region/1|target.prefill"
                                         : "ninfer.region/1|target.verify");
#ifndef NINFER_QWEN38_FLASH_NEXT
    (void)x;
    (void)ph;
    throw std::logic_error("Flash-Next schedule selected by a non-Flash target");
#else
    const int tokens = x.ne[1];
    const int batch  = active_sequence_batch_ == 0 ? 1 : active_sequence_batch_;
    const int width  = tokens / batch;
    if (tokens <= 0 || width * batch != tokens) {
        throw std::logic_error("Flash-Next schedule has invalid token geometry");
    }
    if (ph == Phase::Verify && (active_linear_state_source_slots_ == nullptr ||
                                (gdn_state_action_ != GdnStateAction::RecordForReplay &&
                                 active_linear_state_destination_slots_ == nullptr))) {
        throw std::logic_error("Flash-Next verify state selectors are unavailable");
    }
    cudaStream_t stream = ctx_.stream;
    auto schedule_scope = work_.scope();
    Tensor hyper        = work_.alloc(DType::BF16, {10240, tokens});
    Tensor block_input  = work_.alloc(DType::BF16, {TextConfig::hidden, tokens});
    Tensor block_output = work_.alloc(DType::BF16, {TextConfig::hidden, tokens});
    Tensor injection    = work_.alloc(DType::BF16, {4, tokens});
    ops::hyperconnection_repeat(x, hyper, stream);

    Tensor positions = active_cache_positions_->view({width, batch});
    Tensor rope_positions;
    if (active_rope_positions_->numel() == static_cast<std::int64_t>(tokens) * 3) {
        rope_positions = active_rope_positions_->view({width, batch, 3});
    } else {
        if (active_rope_positions_->numel() != tokens) {
            throw std::logic_error("Flash-Next schedule has invalid RoPE geometry");
        }
        rope_positions = work_.alloc(DType::I32, {width, batch, 3});
        ops::flash_next_expand_text_positions(active_rope_positions_->view({width, batch}),
                                              rope_positions, stream);
    }
    Tensor valid;
    if (active_valid_columns_ != nullptr) {
        valid = *active_valid_columns_;
    } else {
        valid = work_.alloc(DType::I32, {batch});
        // Captured schedules must not retain a pointer into a temporary host vector.
        ops::fill_i32(valid, width, stream);
    }
    const Tensor& table_rows =
        active_kv_table_rows_ != nullptr ? *active_kv_table_rows_ : io_.text_kv_table_row;

    bool pending             = false;
    std::size_t full_index   = 0;
    std::size_t gdn_index    = 0;
    const auto execute_layer = [&](const auto& source) {
        if (source.has_ple) {
            if (pending) {
                ops::hyperconnection_combine(hyper, block_output, injection, stream);
                pending = false;
            }
            if (active_ple_embeddings_ == nullptr) {
                throw std::logic_error("Flash-Next PLE input was not prepared");
            }
            auto ple_scope    = work_.scope();
            Tensor ple_output = work_.alloc(DType::BF16, {10240, tokens});
            if (ple_state_ == nullptr) {
                throw std::logic_error("Flash-Next PLE state is unavailable");
            }
            Tensor gathered = active_ple_embeddings_->view({2560, tokens});
            if (ph == Phase::Verify) {
                if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
                    if (ple_replay_records_ == nullptr) {
                        throw std::logic_error("Flash-Next PLE replay records are unavailable");
                    }
                    Tensor records = ple_replay_records_->slice(2, 0, batch).slice(1, 0, width);
                    ops::flash_next_ple_replay_record(hyper, gathered, source.ple, *ple_state_,
                                                      valid, *active_linear_state_source_slots_,
                                                      width, batch, records, ple_output, work_,
                                                      stream, bf16_gemm_);
                } else {
                    ops::flash_next_ple_batch_update(hyper, gathered, source.ple, *ple_state_,
                                                     valid, *active_linear_state_source_slots_,
                                                     *active_linear_state_destination_slots_, width,
                                                     batch, ple_output, work_, stream, bf16_gemm_);
                }
            } else {
                Tensor ple_state =
                    ple_state_->slice(2, linear_state_destination_slot_, 1).view({10240, 9});
                ops::flash_next_ple(hyper, gathered, source.ple, ple_state, ple_output, work_,
                                    stream, bf16_gemm_);
            }
            ops::residual_add(ple_output, hyper, stream);
        }

        const ops::HyperConnectionWeights& attention_hc = source.attention_hc;
        if (pending) {
            ops::hyperconnection_combine_mix(hyper, block_output, injection, attention_hc,
                                             block_input, &injection, work_, stream);
        } else {
            ops::hyperconnection_mix(hyper, attention_hc, block_input, &injection, work_, stream);
        }
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(source)>,
                                     typename LoadedModelData::FullLayer>) {
            const auto cache =
                active_sequence_batch_ == 0
                    ? single_row_paged_kv_batch_view(
                          kv_.layer_view(static_cast<std::uint32_t>(full_index)))
                    : batch_text_kv_->batch_layer_view(static_cast<std::uint32_t>(full_index));
            ops::flash_next_qsa(block_input, positions, rope_positions, valid, table_rows,
                                source.projection, cache, *active_causal_attention_envelope_,
                                block_output, work_, stream, bf16_gemm_);
            ++full_index;
        } else {
            if (ph == Phase::Verify) {
                auto layer_state = state_.layer_view(static_cast<std::uint32_t>(gdn_index));
                if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
                    if (replay_records_ == nullptr) {
                        throw std::logic_error("Flash-Next GDN replay records are unavailable");
                    }
                    ops::flash_next_gdn_replay_record(
                        block_input, source.projection, layer_state.conv, layer_state.recurrent,
                        valid, *active_linear_state_source_slots_,
                        replay_records_->layer(static_cast<std::int32_t>(gdn_index), batch),
                        block_output, work_, stream);
                } else {
                    if (width != 1) {
                        throw std::logic_error(
                            "Flash-Next direct multi-token state update is unavailable");
                    }
                    ops::flash_next_gdn_batch_update(
                        block_input, source.projection, layer_state.conv, layer_state.recurrent,
                        *active_linear_state_source_slots_, *active_linear_state_destination_slots_,
                        block_output, work_, stream);
                }
            } else {
                Tensor conv_in       = state_.conv_slot(static_cast<std::uint32_t>(gdn_index),
                                                        linear_state_source_slot_);
                Tensor conv_out      = state_.conv_slot(static_cast<std::uint32_t>(gdn_index),
                                                        linear_state_destination_slot_);
                Tensor recurrent_in  = state_.recurrent_slot(static_cast<std::uint32_t>(gdn_index),
                                                             linear_state_source_slot_);
                Tensor recurrent_out = state_.recurrent_slot(static_cast<std::uint32_t>(gdn_index),
                                                             linear_state_destination_slot_);
                ops::flash_next_gdn(block_input, source.projection, conv_in, conv_out, recurrent_in,
                                    recurrent_out, block_output, work_, stream, bf16_gemm_);
            }
            ++gdn_index;
        }

        ops::hyperconnection_combine_mix(hyper, block_output, injection, source.mlp_hc, block_input,
                                         &injection, work_, stream);
        ops::flash_next_moe(block_input, source.post_mixer, block_output, work_, stream, bf16_gemm_,
                            active_causal_attention_envelope_->max_visible_keys >
                                ops::kFlashNextQsaSelectedTokens);
        pending = true;
    };
    for (int layer = 0; layer < TextConfig::layers; ++layer) {
        if (TextConfig::is_full_attention(layer)) {
            execute_layer(weights_.full_layers.at(full_index));
        } else {
            execute_layer(weights_.gdn_layers.at(gdn_index));
        }
    }
    if (pending) { ops::hyperconnection_combine(hyper, block_output, injection, stream); }
    if (flash_predictor_hidden_output_ != nullptr) {
        Tensor output = flash_predictor_hidden_output_->view({10240, tokens});
        CUDA_CHECK(cudaMemcpyAsync(output.data, hyper.data, hyper.bytes(), cudaMemcpyDeviceToDevice,
                                   stream));
    } else if (mtp_prefill_hidden_ != nullptr && ph == Phase::Prefill) {
        Tensor output = matrix_window(*mtp_prefill_hidden_, tokens);
        CUDA_CHECK(cudaMemcpyAsync(output.data, hyper.data, hyper.bytes(), cudaMemcpyDeviceToDevice,
                                   stream));
    }
    const ops::HyperConnectionWeights final_weights{
        .norm = weights_.final_hc.norm,
        .down = weights_.final_hc.down,
        .up   = weights_.final_hc.up,
    };
    ops::hyperconnection_mix(hyper, final_weights, x, nullptr, work_, stream);
#endif
}

void TextContext::run_layers(Tensor& x, Phase ph) {
    NullTap tap;
    run_layers(x, ph, tap);
}

template <class Tap>
PrefillChunkResult
TextContext::prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                          const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end) {
    runtime::ExecutionTimingRecorder timing;
    if (ids.empty()) { throw std::invalid_argument("TextContext::prefill requires tokens"); }
    if (ids.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill token count exceeds int32");
    }
    cudaStream_t s           = ctx_.stream;
    const int T              = static_cast<int>(ids.size());
    const int chunk          = static_cast<int>(prefill_chunk_);
    const std::uint32_t base = text_kv_base_;

    if (text_prefill != nullptr) {
        if (multimodal != nullptr || base != text_prefill->begin ||
            text_prefill->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("text prefill chunk does not match its full prompt");
        }
    }
    if (multimodal != nullptr) {
        if (base != multimodal->begin ||
            multimodal->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("multimodal prefill suffix does not match its cache base");
        }
        if (multimodal->positions.size() != 3 * multimodal->token_ids.size()) {
            throw std::invalid_argument("multimodal positions must have shape [3,T]");
        }
        if (multimodal->vision == nullptr) {
            throw std::invalid_argument("multimodal prefill requires a Vision session");
        }
        rope_delta_ = multimodal->rope_delta;
    } else if (text_kv_base_ == 0) {
        rope_delta_ = 0;
    }
    ops::set_i32_scalar(io_.rope_delta, rope_delta_, s);

    // Prefix-append prefill continues an existing cache: positions are absolute (start at the
    // resident length) and KV/GDN state is not reset. For a reset prefill base == 0.
    if (static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill absolute position exceeds int32");
    }
    const int base_i = static_cast<int>(base);

    const std::int64_t base64    = static_cast<std::int64_t>(base);
    const std::int64_t split_abs = prefill_split_frontier_;
    const bool has_split = split_abs > base64 && split_abs <= base64 + static_cast<std::int64_t>(T);
    const int split_rel  = has_split ? static_cast<int>(split_abs - base64) : -1;
    std::vector<qwen3_8_flash_next::PleIds> ple_ids;
    if constexpr (Variant::flash_next) {
        const std::span<const int> complete_ids = multimodal != nullptr ? multimodal->token_ids
                                                  : text_prefill != nullptr
                                                      ? text_prefill->token_ids
                                                      : ids;
        const std::size_t extent                = static_cast<std::size_t>(base) + ids.size();
        if (complete_ids.size() < extent) {
            throw std::logic_error("Flash-Next PLE history is incomplete");
        }
        std::vector<std::int32_t> represented(extent);
        std::copy_n(complete_ids.data(), extent, represented.data());
        ple_ids.resize(extent);
        qwen3_8_flash_next::compute_ple_ids(represented, ple_ids);
    }
    const bool prepare_mtp_prompt = mtp_enabled() && io_.mtp.has_value();
    if (prepare_mtp_prompt &&
        mtp_proposal_extent_ > static_cast<std::uint32_t>(io_.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP proposal extent exceeds the configured draft window");
    }
    int t0 = 0;
    for (; t0 < T;) {
        int len = std::min(chunk, T - t0);
        if (split_rel > 0 && t0 < split_rel && t0 + len > split_rel) { len = split_rel - t0; }
        work_.reset();

        VisionChunk vision_chunk;
        const std::uint32_t prompt_t0 = base + static_cast<std::uint32_t>(t0);
        if (multimodal != nullptr) {
            if (multimodal->vision == nullptr) {
                throw std::logic_error("multimodal prefill has no Vision session");
            }
            vision_chunk =
                multimodal->vision->prepare_chunk(prompt_t0, static_cast<std::uint32_t>(len));
            len = vision_chunk.length;
        }
        const bool is_last = finalize_at_end && (t0 + len == T);
        nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                      static_cast<std::uint64_t>(len));

        {
            std::vector<std::int32_t> local_scatter_indices;
            std::int32_t visual_begin = 0;
            if (vision_chunk.control != nullptr) {
                const auto scatter =
                    std::span<const std::int32_t>(vision_chunk.control->scatter_indices);
                const auto begin = std::lower_bound(scatter.begin(), scatter.end(), prompt_t0);
                const auto end   = std::lower_bound(begin, scatter.end(), prompt_t0 + len);
                const auto count = static_cast<std::int32_t>(end - begin);
                visual_begin     = static_cast<std::int32_t>(begin - scatter.begin());
                local_scatter_indices.resize(static_cast<std::size_t>(count));
                for (std::int32_t i = 0; i < count; ++i) {
                    local_scatter_indices[static_cast<std::size_t>(i)] =
                        begin[i] - static_cast<std::int32_t>(prompt_t0);
                }
            }

            const std::int32_t rope_axes = multimodal != nullptr ? 3 : (rope_delta_ != 0 ? 1 : 0);
            const auto roots             = workspace_recipe::text_prefill_roots<TextConfig>(
                work_, len, rope_axes, static_cast<std::int32_t>(local_scatter_indices.size()));
            Tensor ids_device = roots.ids;
            copy_i32(ids.data() + t0, ids_device, s);

            Tensor positions = roots.positions;
            ops::fill_i32_positions(positions, base_i + t0, s);

            Tensor rope_positions = positions;
            std::vector<std::int32_t> rope_positions_host;
            if (multimodal != nullptr) {
                rope_positions = roots.rope_positions;
                rope_positions_host.resize(static_cast<std::size_t>(3) * len);
                const std::size_t prompt_tokens = multimodal->token_ids.size();
                for (int axis = 0; axis < 3; ++axis) {
                    const auto* src = multimodal->positions.data() +
                                      static_cast<std::size_t>(axis) * prompt_tokens + prompt_t0;
                    std::copy_n(src, len,
                                rope_positions_host.data() + static_cast<std::size_t>(axis) * len);
                }
                copy_i32(rope_positions_host.data(), rope_positions, s);
            } else if (rope_delta_ != 0) {
                rope_positions = roots.rope_positions;
                ops::offset_i32_positions(positions, io_.rope_delta, rope_positions, s);
            }
            ScopedPositions scoped_cache(active_cache_positions_, positions);
            ScopedPositions scoped_rope(active_rope_positions_, rope_positions);
            const auto visible = static_cast<std::uint32_t>(base_i + t0 + len);
            const ops::CausalAttentionExecutionEnvelope chunk_envelope{visible, visible};
            ScopedEnvelope scoped_envelope(active_causal_attention_envelope_, chunk_envelope);

            Tensor x = roots.residual;
            ops::embedding(ids_device, *embed_, x, s);
            if (!local_scatter_indices.empty()) {
                Tensor indices_device = roots.scatter_indices;
                copy_i32(local_scatter_indices.data(), indices_device, s);
                Tensor embeddings = vision_chunk.embeddings.slice(
                    1, visual_begin, static_cast<std::int32_t>(local_scatter_indices.size()));
                ops::scatter(embeddings, indices_device, x, s);
            }
            if constexpr (Tap::enabled) { tap.begin(x); }
#ifdef NINFER_QWEN38_FLASH_NEXT
            const auto ple_element_bytes = dtype_size(weights_.ple_table_dtype);
            std::vector<std::byte> gathered(static_cast<std::size_t>(len) * 2560U *
                                            ple_element_bytes);
            qwen3_8_flash_next::gather_ple(
                *weights_.ple_table, weights_.ple_table_dtype,
                std::span<const qwen3_8_flash_next::PleIds>(ple_ids).subspan(prompt_t0, len),
                gathered, ple_gather_workers_);
            Tensor gathered_device = work_.alloc(weights_.ple_table_dtype, {2560, len});
            CUDA_CHECK(cudaMemcpyAsync(gathered_device.data, gathered.data(), gathered.size(),
                                       cudaMemcpyHostToDevice, s));
            ScopedPositions scoped_ple(active_ple_embeddings_, gathered_device);
            run_layers(x, Phase::Prefill, tap);
#else
            run_layers(x, Phase::Prefill, tap);
#endif
            if constexpr (requires { tap.capture_positions(positions, s); }) {
                tap.capture_positions(positions, s);
            }

            Tensor xf = prefill_hidden_.data != nullptr
                            ? matrix_window(prefill_hidden_, len)
                            : work_.alloc(DType::BF16, {kCfg.hidden, len});
            if constexpr (Variant::flash_next) {
                CUDA_CHECK(
                    cudaMemcpyAsync(xf.data, x.data, x.bytes(), cudaMemcpyDeviceToDevice, s));
            } else {
                ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, xf, s);
            }

            if (is_last) {
                Tensor last_xf = xf.slice(1, len - 1, 1);
                Tensor logits  = matrix_window(io_.logits, 1);
                ops::linear(last_xf, *lm_head_, logits, s, bf16_gemm_);
                // Set io_.pos to the bonus token's absolute position (base + T) before picking so
                // the sampler RNG is keyed by it (prefill purpose keeps it distinct from the first
                // decode step, which reuses the same io_.pos).
                ops::set_i32_scalar(io_.pos, base_i + T, s);
                ops::set_i32_scalar(io_.rope_pos, base_i + T + rope_delta_, s);
                if (sampling_config_ != nullptr) {
                    ops::sample(logits, io_.token, kCfg.token_domain, sampling_config_, io_.pos,
                                ops::kSamplePurposePrefill, work_, s);
                } else {
                    ops::argmax(logits, io_.token, kCfg.token_domain, s);
                }
            }

            if (prepare_mtp_prompt) {
                const std::uint32_t alignment_tokens =
                    multimodal != nullptr ? static_cast<std::uint32_t>(multimodal->token_ids.size())
                    : text_prefill != nullptr
                        ? static_cast<std::uint32_t>(text_prefill->token_ids.size())
                        : static_cast<std::uint32_t>(T);
                const std::uint32_t alignment_begin =
                    multimodal != nullptr || text_prefill != nullptr
                        ? prompt_t0
                        : static_cast<std::uint32_t>(t0);
                const qwen3_8_flash_next::MtpAlignmentWindow mtp_window =
                    qwen3_8_flash_next::plan_mtp_alignment_window(alignment_tokens, alignment_begin,
                                                                  static_cast<std::uint32_t>(len));
                const std::span<const int> alignment_ids =
                    multimodal != nullptr     ? multimodal->token_ids
                    : text_prefill != nullptr ? text_prefill->token_ids
                                              : ids;
                const int prompt_columns =
                    len - static_cast<int>(mtp_window.final_column_uses_generated_token);
                Tensor mtp_ids = work_.alloc(DType::I32, {len});
                if (prompt_columns != 0) {
                    Tensor prompt_mtp_ids = mtp_ids.slice(0, 0, prompt_columns);
                    copy_i32(alignment_ids.data() + mtp_window.shifted_embedding_begin,
                             prompt_mtp_ids, s);
                }
                if (mtp_window.final_column_uses_generated_token) {
                    Tensor generated_mtp_id = mtp_ids.slice(0, len - 1, 1);
                    CUDA_CHECK(cudaMemcpyAsync(generated_mtp_id.data, io_.token.data,
                                               sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
                }

                Tensor mtp_input_embeddings;
                const Tensor* mtp_input_embeddings_ptr = nullptr;
                if (multimodal != nullptr) {
                    mtp_input_embeddings = work_.alloc(DType::BF16, {kCfg.hidden, len});
                    ops::embedding(mtp_ids, *embed_, mtp_input_embeddings, s);
                    if (vision_chunk.control != nullptr) {
                        const qwen3_8_flash_next::MtpVisualOverlap overlap =
                            qwen3_8_flash_next::shifted_visual_overlap(
                                vision_chunk.control->scatter_indices, alignment_tokens,
                                mtp_window);
                        if (!overlap.empty()) {
                            Tensor shifted_indices = workspace_recipe::visual_scatter_indices(
                                work_, static_cast<std::int32_t>(overlap.size()));
                            qwen3_8_flash_next::detail::scatter_shifted_visual_embeddings(
                                mtp_input_embeddings, vision_chunk.embeddings, overlap,
                                shifted_indices, s);
                        }
                    }
                    mtp_input_embeddings_ptr = &mtp_input_embeddings;
                }
                if (is_last && mtp_proposal_extent_ != 0) {
                    Tensor logits     = matrix_window(io_.logits, 1);
                    Tensor draft0     = io_.mtp->draft_tokens.slice(0, 0, 1);
                    Tensor mtp_source = xf;
                    if constexpr (Variant::flash_next) {
                        if (mtp_prefill_hidden_ == nullptr) {
                            throw std::logic_error(
                                "Flash-Next prefill predictor hidden is unavailable");
                        }
                        mtp_source = matrix_window(*mtp_prefill_hidden_, len);
                    }
                    mtp_prefill_chunk(mtp_ids, mtp_source, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, true, &io_.mtp->ar_hidden,
                                      &logits, &draft0);

                    Tensor ar_position = io_.mtp->position.slice(0, 0, 1);
                    ops::set_i32_scalar(ar_position, base_i + T, s);
                    for (int i = 1; i < static_cast<int>(mtp_proposal_extent_); ++i) {
                        Tensor prev_token     = io_.mtp->draft_tokens.slice(0, i - 1, 1);
                        Tensor next_token     = io_.mtp->draft_tokens.slice(0, i, 1);
                        Tensor next_hidden    = work_.alloc(DType::BF16, {kCfg.hidden, 1});
                        const auto ar_visible = static_cast<std::uint32_t>(base_i + T + i);
                        const ops::CausalAttentionExecutionEnvelope ar_envelope{ar_visible,
                                                                                ar_visible};
                        if constexpr (Variant::flash_next) {
                            Tensor next_predictor = work_.alloc(DType::BF16, {4 * kCfg.hidden, 1});
                            mtp_forward_ar_step(prev_token, io_.mtp->ar_hidden, ar_position,
                                                ar_envelope, next_hidden, logits, next_token,
                                                &next_predictor);
                            CUDA_CHECK(cudaMemcpyAsync(io_.mtp->ar_hidden.data, next_predictor.data,
                                                       next_predictor.bytes(),
                                                       cudaMemcpyDeviceToDevice, s));
                        } else {
                            mtp_forward_ar_step(prev_token, io_.mtp->ar_hidden, ar_position,
                                                ar_envelope, next_hidden, logits, next_token);
                            CUDA_CHECK(cudaMemcpyAsync(io_.mtp->ar_hidden.data, next_hidden.data,
                                                       io_.mtp->ar_hidden.bytes(),
                                                       cudaMemcpyDeviceToDevice, s));
                        }
                        ops::increment_i32_scalar(ar_position, s);
                    }
                } else {
                    Tensor mtp_source = xf;
                    if constexpr (Variant::flash_next) {
                        if (mtp_prefill_hidden_ == nullptr) {
                            throw std::logic_error(
                                "Flash-Next prefill predictor hidden is unavailable");
                        }
                        mtp_source = matrix_window(*mtp_prefill_hidden_, len);
                    }
                    mtp_prefill_chunk(mtp_ids, mtp_source, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, false, nullptr, nullptr,
                                      nullptr);
                }
            }

            if (split_rel > 0 && t0 + len == split_rel &&
                rewrite_checkpoint_hidden_output_ != nullptr) {
                require_tensor_shape(*rewrite_checkpoint_hidden_output_, DType::BF16,
                                     {kCfg.hidden, 1}, "rewrite checkpoint hidden output");
                const Tensor checkpoint_hidden = xf.slice(1, len - 1, 1);
                CUDA_CHECK(cudaMemcpyAsync(rewrite_checkpoint_hidden_output_->data,
                                           checkpoint_hidden.data, checkpoint_hidden.bytes(),
                                           cudaMemcpyDeviceToDevice, s));
                if constexpr (Variant::flash_next) {
                    if (rewrite_checkpoint_mtp_hidden_output_ != nullptr &&
                        mtp_prefill_hidden_ == nullptr) {
                        throw std::logic_error(
                            "Flash-Next rewrite checkpoint has no predictor hidden output");
                    }
                    if (rewrite_checkpoint_mtp_hidden_output_ != nullptr) {
                        require_tensor_shape(*rewrite_checkpoint_mtp_hidden_output_, DType::BF16,
                                             {4 * kCfg.hidden, 1},
                                             "rewrite checkpoint MTP hidden output");
                        const Tensor checkpoint_mtp_hidden =
                            mtp_prefill_hidden_->slice(1, len - 1, 1);
                        CUDA_CHECK(cudaMemcpyAsync(
                            rewrite_checkpoint_mtp_hidden_output_->data, checkpoint_mtp_hidden.data,
                            checkpoint_mtp_hidden.bytes(), cudaMemcpyDeviceToDevice, s));
                    }
                }
            }
        }

        if constexpr (requires { tap.consume_prefill_chunk(len, false); }) {
            work_.reset();
            tap.consume_prefill_chunk(len, split_rel > 0 && t0 + len == split_rel);
        }

        t0 += len;
        break;
    }

    prefill_split_frontier_ = -1;

    timing.begin_wait();
    ctx_.synchronize();
    timing.end_wait();
    work_.reset();
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(t0),
                              .finalized        = finalize_at_end && t0 == T,
                              .timing           = timing.finish()};
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    NullTap tap;
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, tap,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end,
                                              DFlashFeatureSink& sink) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, sink,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(const qwen3_8_flash_next::PreparedPromptData& input,
                                              std::uint32_t begin, std::uint32_t nominal_length,
                                              VisionPrefillSession& vision, bool finalize_at_end) {
    if (begin >= input.token_ids.size() || nominal_length == 0 ||
        nominal_length > input.token_ids.size() - begin) {
        throw std::invalid_argument("multimodal prefill chunk is outside the prompt");
    }
    const std::span<const int> tokens(input.token_ids);
    const MultimodalPrefill multimodal{tokens, input.positions, &vision, begin, input.rope_delta};
    NullTap tap;
    return prefill_impl(tokens.subspan(begin, nominal_length), nullptr, &multimodal, tap,
                        finalize_at_end);
}

} // namespace
  // ninfer::models::qwen3_8_flash_next::detail::NINFER_QWEN38_FLASH_NEXT_RUNTIME_NS::schedule
