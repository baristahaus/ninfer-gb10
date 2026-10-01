#pragma once

#ifndef NINFER_QWEN38_FLASH_NEXT_VARIANT
#    error "NINFER_QWEN38_FLASH_NEXT_VARIANT must name the complete exact Variant"
#endif
#ifndef NINFER_QWEN38_FLASH_NEXT_RUNTIME_NS
#    error "NINFER_QWEN38_FLASH_NEXT_RUNTIME_NS must be a unique identifier for this instantiation"
#endif

#include <ninfer/models/qwen3_8_flash_next/runtime.h>

namespace ninfer::models::qwen3_8_flash_next::detail::NINFER_QWEN38_FLASH_NEXT_RUNTIME_NS {

using Variant                        = NINFER_QWEN38_FLASH_NEXT_VARIANT;
using WeightsProfile                 = typename Variant::WeightsProfile;
using TextConfig                     = typename Variant::TextConfig;
using VisionConfig                   = typename Variant::VisionConfig;
using DFlashConfig                   = typename Variant::DFlashConfig;
using LoadedModelData                = typename Variant::ModelView;
using FullAttentionWeights           = typename LoadedModelData::FullLayer;
using GdnWeights                     = typename LoadedModelData::GdnLayer;
using MlpWeights                     = typename Variant::PostMixerWeights;
using MtpWeights                     = typename LoadedModelData::MtpLayer;
using DFlashWeights                  = typename LoadedModelData::DFlash;
using FullAttentionProjectionWeights = typename Variant::FullAttentionProjectionWeights;
using GdnProjectionWeights           = typename Variant::GdnProjectionWeights;
using VisionWeights                  = typename Variant::VisionWeights;
using GraphExecutionProfile          = typename Variant::GraphExecutionProfile;

using SequencePlan            = qwen3_8_flash_next::SequencePlan<Variant>;
using SequencePlanner         = qwen3_8_flash_next::SequencePlanner<Variant>;
using RequestBasePlan         = qwen3_8_flash_next::RequestBasePlan<Variant>;
using AdmissionCandidate      = qwen3_8_flash_next::AdmissionCandidate<Variant>;
using PressurePlanningSession = qwen3_8_flash_next::PressurePlanningSession<Variant>;
using PressureTargetHandle    = qwen3_8_flash_next::PressureTargetHandle;
using ResourcePlan            = qwen3_8_flash_next::ResourcePlan<Variant>;
using PersistentBackfillProof = qwen3_8_flash_next::PersistentBackfillProof<Variant>;
using SequenceHandle          = qwen3_8_flash_next::SequenceHandle<Variant>;
using ContinuationHandle      = qwen3_8_flash_next::ContinuationHandle<Variant>;
using SharedPrefixHandle      = qwen3_8_flash_next::SharedPrefixHandle<Variant>;
using CaptureOffer            = qwen3_8_flash_next::CaptureOffer<Variant>;
using CaptureAssessment       = qwen3_8_flash_next::CaptureAssessment;
using ActiveCaptureResult     = qwen3_8_flash_next::ActiveCaptureResult<Variant>;
using MaterializationResult   = qwen3_8_flash_next::MaterializationResult<Variant>;
using PendingBatch            = qwen3_8_flash_next::PendingBatch<Variant>;
using PrefillProgress         = qwen3_8_flash_next::PrefillProgress<Variant>;
using StartResult             = qwen3_8_flash_next::StartResult<Variant>;
using CommitResult            = qwen3_8_flash_next::CommitResult<Variant>;
using DiscardResult           = qwen3_8_flash_next::DiscardResult<Variant>;
using FinishResult            = qwen3_8_flash_next::FinishResult<Variant>;
using AbortResult             = qwen3_8_flash_next::AbortResult<Variant>;
using ReleaseResult           = qwen3_8_flash_next::ReleaseResult<Variant>;
using ContractAccess          = qwen3_8_flash_next::detail::RuntimeContractAccess<Variant>;
using Program                 = qwen3_8_flash_next::Program<Variant>;

inline constexpr float kAttentionScale                   = Variant::attention_scale;
inline constexpr float kGdnScale                         = Variant::gdn_scale;
inline constexpr std::uint32_t kPrefillChunkAlignment    = Variant::prefill_chunk_alignment;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = Variant::maximum_mtp_draft_tokens;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = Variant::maximum_dflash_draft_tokens;

inline std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity) {
    return Variant::ordinary_graph_profiles(capacity);
}

inline std::vector<GraphExecutionProfile> mtp_graph_profiles(std::uint32_t capacity,
                                                             std::uint32_t draft_window) {
    return Variant::mtp_graph_profiles(capacity, draft_window);
}

inline std::vector<GraphExecutionProfile> dflash_graph_profiles(std::uint32_t capacity,
                                                                std::uint32_t draft_window,
                                                                std::uint32_t batch_size) {
    return Variant::dflash_graph_profiles(capacity, draft_window, batch_size);
}

} // namespace ninfer::models::qwen3_8_flash_next::detail::NINFER_QWEN38_FLASH_NEXT_RUNTIME_NS
