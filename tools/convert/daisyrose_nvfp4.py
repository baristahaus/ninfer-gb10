"""DaisyRose-27B nvfp4 recipe.

MLP projections below layer 56: preserve NVIDIA's NVFP4 codes from the
RadixArk ModelOpt checkpoint via the proven ModelOptNvfp4 adapter.
All other projections: row-FP8 requantized from the BF16 checkpoint
(fp8_row_maxabs); token embedding FP8; everything else BF16.
"""

from tools.convert.methods import fp8_row_maxabs, import_encoded
from tools.convert.official_recipes import _optional
from tools.convert.sources.modelopt_nvfp4 import ModelOptNvfp4

FP8 = "fp8_e4m3fn_row_bf16"


def _modelopt_prefix(name):
    # text/layers/L/mlp/gate -> model.language_model.layers.L.mlp.gate_proj
    parts = name.split("/")
    role = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}[parts[4]]
    return f"model.language_model.layers.{parts[2]}.mlp.{role}"


def configure(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    nvidia = ModelOptNvfp4(sources["nvidia"])
    recipe.assign("text/token_embedding", format=FP8, method=fp8_row_maxabs)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/") or name == "text/token_embedding":
            continue
        if not parameter.projection or name.endswith(
            ("/gdn/a_projection", "/gdn/b_projection")
        ):
            continue
        layer = int(name.split("/")[2]) if name.startswith("text/layers/") else -1
        if "/mlp/" in name and layer < 56:
            recipe.assign(
                name,
                format="nvfp4",
                method=import_encoded,
                source=nvidia.matrix(_modelopt_prefix(name), parameter.shape),
                activation_policy="AllowA4",
            )
        else:
            recipe.assign(name, format=FP8, method=fp8_row_maxabs)
