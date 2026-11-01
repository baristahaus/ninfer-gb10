#pragma once

#include <ninfer/models/qwen3_8_flash_next/hybrid_topology.h>
#include <ninfer/models/qwen3_8_flash_next/vision.h>

#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next_125b_a6b::detail {

struct TextConfig {
    static constexpr int hidden                   = 2560;
    static constexpr int layers                   = 48;
    static constexpr int intermediate             = 640;
    static constexpr int output_rows              = 248320;
    static constexpr int token_domain             = 248320;
    static constexpr int gdn_conv_kernel          = 4;
    static constexpr int gdn_conv_state_width     = 3;
    static constexpr int gdn_key_heads            = 16;
    static constexpr int gdn_key_head_dim         = 128;
    static constexpr int gdn_value_heads          = 48;
    static constexpr int gdn_value_head_dim       = 128;
    static constexpr int query_heads              = 24;
    static constexpr int kv_heads                 = 2;
    static constexpr int head_dim                 = 256;
    static constexpr int rotary_dim               = 64;
    static constexpr int qsa_index_head_dim       = 128;
    static constexpr float rms_epsilon            = 1.0e-6F;
    static constexpr float rope_theta             = 1.0e7F;
    static constexpr int key_dim                  = gdn_key_heads * gdn_key_head_dim;
    static constexpr int value_dim                = gdn_value_heads * gdn_value_head_dim;
    static constexpr int convolution_dim          = 2 * key_dim + value_dim;
    static constexpr int query_size               = query_heads * head_dim;
    static constexpr int kv_size                  = kv_heads * head_dim;
    static constexpr int query_projection_rows    = 2 * query_size;
    static constexpr int mtp_layers               = 1;
    static constexpr int mtp_input_rows           = 2 * hidden;
    static constexpr int mtp_attention_input_rows = 2 * query_size + 2 * kv_size;
    static constexpr int mtp_mlp_gate_up_rows     = 2 * intermediate;

    static constexpr bool is_full_attention(int layer) {
        return qwen3_8_flash_next::is_full_attention_layer(layer);
    }

    static constexpr int full_attention_layers() { return 12; }

    static constexpr int gdn_layers() { return 36; }

    static constexpr int full_attention_index(int layer) { return layer / 4; }

    static constexpr int gdn_index(int layer) { return layer - layer / 4; }
};

struct VisionConfig : qwen3_8_flash_next::VisionBackboneConfig {
    static constexpr int output_hidden = TextConfig::hidden;
};

// The masked-draft companion's geometry. The 125B artifact's companion is the published
// DFlash2 draft: five sliding-attention layers at the text hidden, two-tap dynamic
// convolutions, a candidate selector over the target's tap layers 5/15/25/35/45.
struct DFlashConfig {
    static constexpr bool supported     = true;
    static constexpr int layers         = 5;
    static constexpr int local_layers   = 5;
    static constexpr int local_capacity = 2048;
    static constexpr int query_heads    = 32;
    static constexpr int kv_heads       = 8;
    static constexpr int head_dim       = 128;
    static constexpr int hidden         = 2560;
    static constexpr int intermediate   = 7680;
    static constexpr int query_size     = query_heads * head_dim;
    static constexpr int kv_size        = kv_heads * head_dim;
    static constexpr int groups         = hidden / 16;
    static constexpr int feature_rows   = 5 * hidden;
    static constexpr int mask_token     = 248077;
    static constexpr float rms_epsilon  = 1.0e-6F;
    static constexpr float rope_theta   = 1.0e7F;
    static constexpr float attention_scale = 1.0F / 11.3137085F;  // 1/sqrt(128)
    static constexpr std::array<int, 5> target_feature_layers = {5, 15, 25, 35, 45};
    static constexpr int selector_rank   = 256;
    static constexpr int selector_top_k  = 16;
};

// The published DFlash2 draft for Flash-Next (incoai/Qwen3.8-Flash-Next-DFlash2):
// five sliding-attention layers, block-diffusion drafting with a candidate path
// selector and two-tap dynamic convolutions, tapping the target's hidden states at
// layers 5/15/25/35/45. Load/bind support only; the draft execution and its
// schedule hooks are a separate work item.
struct DFlash2Config {
    static constexpr bool supported              = DFlashConfig::supported;
    static constexpr int layers                  = 5;
    static constexpr int hidden                  = 2560;
    static constexpr int intermediate            = 7680;
    static constexpr int query_heads             = 32;
    static constexpr int kv_heads                = 8;
    static constexpr int head_dim                = 128;
    static constexpr int query_size              = query_heads * head_dim;
    static constexpr int kv_size                 = kv_heads * head_dim;
    static constexpr int conv_kernel_size        = 2;
    static constexpr int conv_group_size         = 16;
    static constexpr int conv_projection_rows    = 2 * conv_kernel_size * (hidden / conv_group_size);
    static constexpr int selector_rank           = 256;
    static constexpr int selector_top_k          = 16;
    static constexpr int block_size              = 8;
    static constexpr int mask_token_id           = 248077;
    static constexpr std::size_t target_layers   = 5;
    static constexpr int feature_projection_cols = target_layers * hidden;
    static constexpr std::array<int, target_layers> target_layer_ids = {5, 15, 25, 35, 45};
};

inline constexpr float kAttentionScale                   = 0.0625F;
inline constexpr float kGdnScale                         = 0.08838834764831845F;
inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 3;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens  = 0;
inline constexpr std::uint32_t kMaximumDFlash2DraftTokens = DFlash2Config::block_size - 1;
inline constexpr std::uint32_t kNativeContext            = 262144;

} // namespace ninfer::models::qwen3_8_flash_next_125b_a6b::detail
