#pragma once

#include "core/device.h"

#include "models/qwen3_8_flash_next_125b_a6b/impl/config.h"
#include "models/qwen3_8_flash_next_125b_a6b/impl/model.h"
#include <ninfer/models/qwen3_8_flash_next_125b_a6b/package.h>
#include <ninfer/models/qwen3_8_flash_next/runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next_125b_a6b::detail {

using GraphExecutionProfile = qwen3_8_flash_next::GraphExecutionProfile;

struct Variant {
    using WeightsProfile                 = detail::WeightsProfile;
    using TextConfig                     = detail::TextConfig;
    using VisionConfig                   = detail::VisionConfig;
    using DFlashConfig                   = detail::DFlashConfig;
    using ModelView                      = RuntimeModelView;
    using FullAttentionProjectionWeights = ops::FlashNextQsaWeights;
    using GdnProjectionWeights           = qwen3_8_flash_next_125b_a6b::GdnWeights;
    using PostMixerWeights               = ops::FlashNextMoeWeights;
    using MtpAttentionProjectionWeights  = ops::FlashNextQsaWeights;
    using MtpPostMixerWeights            = ops::FlashNextMoeWeights;
    using VisionWeights                  = qwen3_8_flash_next::VisionWeights;
    using GraphExecutionProfile          = detail::GraphExecutionProfile;

    static constexpr bool flash_next                           = true;
    static constexpr float attention_scale                     = kAttentionScale;
    static constexpr float gdn_scale                           = kGdnScale;
    static constexpr std::uint32_t prefill_chunk_alignment     = kPrefillChunkAlignment;
    static constexpr std::uint32_t maximum_mtp_draft_tokens    = kMaximumMtpDraftTokens;
    static constexpr std::uint32_t maximum_dflash_draft_tokens = kMaximumDFlashDraftTokens;
    static constexpr std::uint32_t maximum_context             = kNativeContext;
    static constexpr bool supports_dflash                      = false;
    static constexpr std::int32_t draft_head_rows              = 147456;
    static constexpr std::int32_t draft_head_valid_rows        = 147456;

    static void attention_projection(const Tensor&, const FullAttentionProjectionWeights&, Tensor&,
                                     Tensor&, Tensor&, Tensor&, qwen3_8_flash_next::TextPhase,
                                     MemoryArena&, CudaExecutionView execution);
    static void attention_output_projection(const Tensor&, const Weight&, Tensor&,
                                            qwen3_8_flash_next::TextPhase, MemoryArena&,
                                            CudaExecutionView execution);
    static void mtp_attention_projection(const Tensor&, const MtpAttentionProjectionWeights&,
                                         Tensor&, Tensor&, Tensor&, Tensor&, MemoryArena&,
                                         CudaExecutionView execution);
    static void mtp_kv_projection(const Tensor&, const MtpAttentionProjectionWeights&, Tensor&,
                                  Tensor&, MemoryArena&, CudaExecutionView execution);
    static void mtp_q_gate_projection(const Tensor&, const MtpAttentionProjectionWeights&, Tensor&,
                                      Tensor&, MemoryArena&, CudaExecutionView execution);
    static void gdn_input_projection(const Tensor&, const GdnProjectionWeights&, Tensor&, Tensor&,
                                     qwen3_8_flash_next::TextPhase, MemoryArena&, CudaExecutionView execution);
    static void gdn_input_projection_snapshot(const Tensor&, const GdnProjectionWeights&,
                                              const Tensor&, Tensor&, const Tensor&, const Tensor&,
                                              const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                                              qwen3_8_flash_next::TextPhase, MemoryArena&,
                                              CudaExecutionView execution);
    static void gdn_input_projection_record(const Tensor&, const GdnProjectionWeights&,
                                            const Tensor&, const Tensor&, const Tensor&,
                                            const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                                            Tensor&, qwen3_8_flash_next::TextPhase, MemoryArena&,
                                            CudaExecutionView execution);
    static void gdn_output_projection(const Tensor&, const Weight&, Tensor&,
                                      qwen3_8_flash_next::TextPhase, MemoryArena&, CudaExecutionView execution);
    static void gdn_norm_control_projection(const Tensor&, const Tensor&, float,
                                            const GdnProjectionWeights&, Tensor&, Tensor&, Tensor&,
                                            MemoryArena&, CudaExecutionView);
    static void post_mixer(const Tensor&, const PostMixerWeights&, Tensor&,
                           qwen3_8_flash_next::TextPhase, MemoryArena&, CudaExecutionView execution);
    static void mtp_post_mixer(const Tensor&, const MtpPostMixerWeights&, Tensor&, MemoryArena&,
                               CudaExecutionView execution);

    static std::size_t mtp_attention_projection_workspace_capacity_bytes(std::int32_t,
                                                                         std::int32_t);
    static std::size_t mtp_kv_projection_workspace_capacity_bytes(std::int32_t, std::int32_t);
    static std::size_t mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t, std::int32_t);
    static std::size_t attention_projection_workspace_capacity_bytes(WeightsProfile,
                                                                     qwen3_8_flash_next::TextPhase,
                                                                     std::int32_t, std::int32_t);
    static std::size_t attention_output_projection_workspace_capacity_bytes(
        WeightsProfile, qwen3_8_flash_next::TextPhase, std::int32_t, std::int32_t);
    static std::size_t gdn_input_projection_workspace_capacity_bytes(WeightsProfile,
                                                                     qwen3_8_flash_next::TextPhase,
                                                                     std::int32_t, std::int32_t);
    static std::size_t gdn_input_projection_snapshot_workspace_capacity_bytes(
        WeightsProfile, qwen3_8_flash_next::TextPhase, std::int32_t, std::int32_t, std::int32_t);
    static std::size_t gdn_input_projection_record_workspace_capacity_bytes(
        WeightsProfile, qwen3_8_flash_next::TextPhase, std::int32_t, std::int32_t, std::int32_t);
    static std::size_t gdn_output_projection_workspace_capacity_bytes(WeightsProfile,
                                                                      qwen3_8_flash_next::TextPhase,
                                                                      std::int32_t, std::int32_t);
    static std::size_t gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t,
                                                                            std::int32_t);
    static std::size_t post_mixer_workspace_capacity_bytes(WeightsProfile,
                                                           qwen3_8_flash_next::TextPhase,
                                                           std::int32_t, std::int32_t);
    static std::size_t mtp_post_mixer_workspace_capacity_bytes(std::int32_t, std::int32_t);

    static std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t);
    static std::vector<GraphExecutionProfile> mtp_graph_profiles(std::uint32_t, std::uint32_t);
    static std::vector<GraphExecutionProfile> dflash_graph_profiles(std::uint32_t, std::uint32_t,
                                                                    std::uint32_t);
};

} // namespace ninfer::models::qwen3_8_flash_next_125b_a6b::detail
