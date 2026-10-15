"""Closed persistent-object contract for the QAD-native Flash-Next 125B-A6B recipe.

The vendor recipe (``inventory.py``) stores attention/GDN/indexer/shared-expert
projections and the MTP drafter experts as BF16 and the PLE table as one FP8
object. The QAD checkpoint family declares different source formats for those
sites and the user decision for this recipe is native ingestion: the artifact
stores what the source declares.

Differences from the vendor recipe:

- Attention, GDN, QSA-indexer, and shared-expert projections are MXFP8 pairs,
  stored as ``dense_block_scale_k32_mxfp8_v1`` (A1/A2).
- The MTP drafter experts are NVFP4 W4A16 banks, stored in the existing
  ``expert_block_scale_k16_m128x4_v1`` layout with per-expert FP32 weight
  divisors and no activation divisor (B1/B2).
- The PLE n-gram table is NVFP4 group-16, stored as ``block_scale_k16_v1``
  with the source's global ``weight_scale_2`` word appended to the payload
  (E2a/E2b). The vendor FP8 table's separate ``(1,)`` scale object has no
  counterpart.
- Everything else (embeddings, norms, hyper-connections, routers, MTP trunk,
  vision, lm_head, the proposal shortlist) is BF16/F32 exactly as the vendor
  recipe.
"""

from __future__ import annotations

from tools.artifact.schema import TensorSpec

from . import inventory

MXFP8 = "mxfp8"
MXFP8_LAYOUT = "dense_block_scale_k32_mxfp8_v1"
PLE_TABLE_LAYOUT = "block_scale_k16_v1"
EXPERT_BANK_LAYOUT = inventory.EXPERT_NVFP4

WEIGHTS_ID = "qad_native"
OUTPUT_BASENAME = "qwen3_8_flash_next_125b_a6b_qad_native.ninfer"
RECIPE_ID = "qwen3_8_flash_next_125b_a6b_qad_native-v1"
SOURCE_REPOSITORY = "edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD"

_PLE_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
_PLE_TABLE = _PLE_PREFIX + "weight"
_PLE_TABLE_SCALE = _PLE_PREFIX + "weight_scale"
_MTP_MOE = "mtp.layers.0.mlp."

_PLE_METADATA = frozenset(
    {
        "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
    }
)

# MXFP8 projection sites by logical [N, K], under the base spec ids (which end
# in ``.weight``). The per-layer projection split is the vendor inventory's:
# 12 full-attention layers, 36 GDN layers, plus the shared expert in all 48.
_MXFP8_SITES: dict[str, tuple[int, int]] = {}
for _layer in inventory.LAYERS:
    _prefix = f"model.language_model.layers.{_layer}."
    if _layer in inventory.FULL_ATTENTION_LAYERS:
        _attn = _prefix + "self_attn."
        for _name, _n, _k in (
            ("q_proj.weight", 12288, 2560),
            ("k_proj.weight", 512, 2560),
            ("v_proj.weight", 512, 2560),
            ("o_proj.weight", 2560, 6144),
            ("indexer.index_qk_proj.weight", 640, 2560),
        ):
            _MXFP8_SITES[_attn + _name] = (_n, _k)
    else:
        _attn = _prefix + "linear_attn."
        for _name, _n, _k in (
            ("in_proj_a.weight", 48, 2560),
            ("in_proj_b.weight", 48, 2560),
            ("in_proj_qkv.weight", 10240, 2560),
            ("in_proj_z.weight", 6144, 2560),
            ("out_proj.weight", 2560, 6144),
        ):
            _MXFP8_SITES[_attn + _name] = (_n, _k)
    for _name, _n, _k in (
        ("shared_expert.gate_proj.weight", 640, 2560),
        ("shared_expert.up_proj.weight", 640, 2560),
        ("shared_expert.down_proj.weight", 2560, 640),
    ):
        _MXFP8_SITES[_prefix + "mlp." + _name] = (_n, _k)

_MTP_BANKS: dict[str, tuple[int, int, str]] = {
    _MTP_MOE + "experts.gate_up": (512, 1280, "gate_up"),
    _MTP_MOE + "experts.down": (512, 2560, "down"),
}


def _substitute(spec: TensorSpec) -> tuple[TensorSpec, ...]:
    if spec.id in _MXFP8_SITES:
        return (TensorSpec(spec.id, spec.shape, MXFP8, MXFP8_LAYOUT),)
    if spec.id == _PLE_TABLE:
        return (TensorSpec(spec.id, spec.shape, inventory.NVFP4, PLE_TABLE_LAYOUT),)
    if spec.id == _PLE_TABLE_SCALE:
        return ()
    if spec.id == _MTP_MOE + "experts.gate_up_proj":
        return (
            TensorSpec(
                _MTP_MOE + "experts.gate_up", (512, 1280, 2560),
                inventory.NVFP4, EXPERT_BANK_LAYOUT,
            ),
            inventory.direct(_MTP_MOE + "experts.gate_up_weight_divisors", (512,), inventory.FP32),
        )
    if spec.id == _MTP_MOE + "experts.down_proj":
        return (
            TensorSpec(
                _MTP_MOE + "experts.down", (512, 2560, 640),
                inventory.NVFP4, EXPERT_BANK_LAYOUT,
            ),
            inventory.direct(_MTP_MOE + "experts.down_weight_divisors", (512,), inventory.FP32),
        )
    return (spec,)


def _transform(specs: tuple[TensorSpec, ...]) -> tuple[TensorSpec, ...]:
    result: list[TensorSpec] = []
    for spec in specs:
        result.extend(_substitute(spec))
    return tuple(result)


TEXT_TENSOR_SPECS = _transform(inventory.TEXT_TENSOR_SPECS)
MTP_TENSOR_SPECS = _transform(inventory.MTP_TENSOR_SPECS)
VISION_TENSOR_SPECS = inventory.VISION_TENSOR_SPECS
TENSOR_SPECS = TEXT_TENSOR_SPECS + MTP_TENSOR_SPECS + VISION_TENSOR_SPECS
RESOURCE_SPECS = inventory.RESOURCE_SPECS
OBJECT_SPECS: tuple[str | TensorSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

MODEL_ID = inventory.MODEL_ID
TARGET_KEY = inventory.TARGET_KEY
LAYERS = inventory.LAYERS
FULL_ATTENTION_LAYERS = inventory.FULL_ATTENTION_LAYERS
GDN_LAYERS = inventory.GDN_LAYERS
EXPERTS = inventory.EXPERTS
BF16 = inventory.BF16
FP32 = inventory.FP32
FP8 = inventory.FP8
NVFP4 = inventory.NVFP4
I32 = inventory.I32
CONTIGUOUS = inventory.CONTIGUOUS
ROW_SPLIT = inventory.ROW_SPLIT
EXPERT_NVFP4 = inventory.EXPERT_NVFP4

__all__ = [
    "BF16",
    "EXPERT_NVFP4",
    "EXPERTS",
    "FULL_ATTENTION_LAYERS",
    "FP32",
    "FP8",
    "GDN_LAYERS",
    "I32",
    "LAYERS",
    "MODEL_ID",
    "MTP_TENSOR_SPECS",
    "MXFP8",
    "MXFP8_LAYOUT",
    "NVFP4",
    "OBJECT_SPECS",
    "OUTPUT_BASENAME",
    "PLE_TABLE",
    "PLE_TABLE_LAYOUT",
    "RECIPE_ID",
    "RESOURCE_SPECS",
    "SOURCE_REPOSITORY",
    "TARGET_KEY",
    "TENSOR_SPECS",
    "TEXT_TENSOR_SPECS",
    "VISION_TENSOR_SPECS",
    "WEIGHTS_ID",
]
