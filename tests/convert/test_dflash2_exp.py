"""The DFlash2 component of the Qwen4Exp route: config validation and the
draft parameter mapping (SP-7 lane split; the runtime consumer is a separate
package). The fixture mirrors the published incoai DFlash2 draft geometry.

The draft tensors come from a synthetic companion checkpoint with the real
names, shapes and dtypes; no network or local model directory is required.
"""

from __future__ import annotations

import torch
from safetensors.torch import save_file

from tools.convert.model import Model
from tools.convert.qwen3_5 import draft_config
from tools.convert.qwen4_exp import add_dflash2_draft
from tools.convert.sources.safetensors import SafetensorsSource

HIDDEN = 2560
VOCAB = 248320
TARGET_LAYERS = 48
DRAFT_LAYERS = 5
HEAD_DIM = 128
Q_ROWS = 32 * HEAD_DIM
KV_ROWS = 8 * HEAD_DIM
INTERMEDIATE = 7680
RANK = 256
TAPS = [5, 15, 25, 35, 45]


def _draft_config_raw() -> dict:
    return {
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "hidden_act": "silu",
        "attention_bias": False,
        "is_causal": False,
        "hidden_size": HIDDEN,
        "vocab_size": VOCAB,
        "num_target_layers": TARGET_LAYERS,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": HEAD_DIM,
        "num_hidden_layers": DRAFT_LAYERS,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
        "sliding_window": 2048,
        "use_sliding_window": True,
        "layer_types": ["sliding_attention"] * DRAFT_LAYERS,
        "dflash_config": {
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": 248077,
            "selector_rank": RANK,
            "selector_top_k": 16,
            "target_layer_ids": list(TAPS),
        },
    }


def _target_config() -> dict:
    return {
        "hidden_size": HIDDEN,
        "vocab_size": VOCAB,
        "num_hidden_layers": TARGET_LAYERS,
    }


def _companion(tmp_path) -> SafetensorsSource:
    generator = torch.Generator().manual_seed(20261101)

    def tensor(shape):
        return torch.empty(tuple(shape), dtype=torch.bfloat16).normal_(
            generator=generator
        )

    tensors = {
        "fc.weight": tensor((HIDDEN, len(TAPS) * HIDDEN)),
        "hidden_norm.weight": tensor((HIDDEN,)),
        "norm.weight": tensor((HIDDEN,)),
        "candidate_selector.hidden_projection.weight": tensor((RANK, HIDDEN)),
        "candidate_selector.predecessor_codebook.weight": tensor((VOCAB, RANK)),
        "candidate_selector.successor_codebook.weight": tensor((VOCAB, RANK)),
    }
    for i in range(DRAFT_LAYERS):
        p = f"layers.{i}."
        tensors.update(
            {
                p + "input_layernorm.weight": tensor((HIDDEN,)),
                p + "post_attention_layernorm.weight": tensor((HIDDEN,)),
                p + "self_attn.query_key_value.weight": tensor(
                    (Q_ROWS + 2 * KV_ROWS, HIDDEN)
                ),
                p + "self_attn.o_proj.weight": tensor((HIDDEN, Q_ROWS)),
                p + "self_attn.q_norm.weight": tensor((HEAD_DIM,)),
                p + "self_attn.k_norm.weight": tensor((HEAD_DIM,)),
                p + "attention_conv.base_kernel": tensor((2, 2, HIDDEN)),
                p + "attention_conv.kernel_projection.weight": tensor(
                    (4 * (HIDDEN // 16), HIDDEN)
                ),
                p + "mlp_conv.base_kernel": tensor((2, 2, HIDDEN)),
                p + "mlp_conv.kernel_projection.weight": tensor(
                    (4 * (HIDDEN // 16), HIDDEN)
                ),
                p + "mlp.gate_up_proj.weight": tensor((2 * INTERMEDIATE, HIDDEN)),
                p + "mlp.down_proj.weight": tensor((HIDDEN, INTERMEDIATE)),
            }
        )
    root = tmp_path / "dflash2"
    root.mkdir()
    save_file(tensors, root / "model.safetensors")
    (root / "config.json").write_text(json.dumps(_draft_config_raw()))
    return SafetensorsSource(root)


import json  # noqa: E402  (kept beside the fixture writers)


def test_draft_config_accepts_the_published_checkpoint() -> None:
    config = draft_config(_draft_config_raw(), _target_config(), "dflash2")
    dflash = config["dflash_config"]
    assert dflash["target_layer_ids"] == TAPS
    assert dflash["mask_token_id"] == 248077
    assert dflash["conv_kernel_size"] == 2
    assert dflash["conv_group_size"] == 16
    assert dflash["selector_rank"] == RANK
    assert dflash["selector_top_k"] == 16
    assert config["num_hidden_layers"] == DRAFT_LAYERS
    assert config["layer_types"] == ["sliding_attention"] * DRAFT_LAYERS


def test_draft_config_rejects_taps_outside_the_target() -> None:
    raw = _draft_config_raw()
    raw["dflash_config"]["target_layer_ids"] = [5, 15, 25, 35, 48]
    try:
        draft_config(raw, _target_config(), "dflash2")
    except ValueError as error:
        assert "invalid target_layer_ids" in str(error)
    else:
        raise AssertionError("out-of-range taps accepted")


def test_draft_parameters_map_the_companion_checkpoint(tmp_path) -> None:
    companion = _companion(tmp_path)
    model = Model({"text": {"config": _target_config()}})
    draft_cfg = draft_config(companion.config, _target_config(), "dflash2")
    add_dflash2_draft(model, companion, _target_config(), draft_cfg)

    assert len(model.parameters) == 3 + DRAFT_LAYERS * 14 + 3
    assert model.parameters["dflash2/feature_projection"].shape == (
        HIDDEN,
        len(TAPS) * HIDDEN,
    )
    assert model.parameters["dflash2/feature_projection"].inputs == (
        "dflash2/feature_projection/input",
    )
    assert model.parameters["dflash2/candidate_selector/hidden_projection"].inputs == (
        "dflash2/final_hidden",
    )
    for i in range(DRAFT_LAYERS):
        p = f"dflash2/layers/{i}/"
        assert model.parameters[p + "attention/query_key_value"].shape == (
            Q_ROWS + 2 * KV_ROWS,
            HIDDEN,
        )
        assert model.parameters[p + "attention/query_key_value"].inputs == (
            p + "attention/query_key_value/input",
        )
        assert model.parameters[p + "attention/context_key"].shape == (KV_ROWS, HIDDEN)
        assert model.parameters[p + "attention/context_key"].inputs == (
            p + "attention/context_key/input",
        )
        assert model.parameters[p + "attention_conv/base_kernel"].shape == (2, 2, HIDDEN)
        assert model.parameters[p + "mlp/gate_up"].shape == (2 * INTERMEDIATE, HIDDEN)
        assert model.parameters[p + "mlp/gate_up"].inputs == (p + "mlp/gate_up/input",)
        assert model.parameters[p + "mlp/down"].shape == (HIDDEN, INTERMEDIATE)
        assert model.parameters[p + "mlp/down"].inputs == (p + "mlp/down/input",)
    assert not model.packing_groups
    for name, parameter in model.parameters.items():
        assert parameter.residency == "dflash2", name
        assert parameter.direct_format == "bf16", name
