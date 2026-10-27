"""Flash-Next v3 component and logical-use description; no tensor dependencies."""

def describe(objects):
    components = {
        "text": {"config": {
            "architectures": ["Qwen3_8FlashNextForCausalLM"],
            "model_type": "qwen3_8_flash_next_text", "hidden_size": 2560,
            "num_hidden_layers": 48, "vocab_size": 248320, "num_experts": 512,
        }, "resources": {}, "proposal": {"domain": "indexed", "rows": 147456}},
        "mtp": {"config": {"architectures": ["Qwen3_8FlashNextMTP"]}, "target": "text"},
        "vision": {"config": {"model_type": "qwen3_8_flash_next_vision"}, "target": "text", "resources": {}},
    }
    bindings, uses = {}, []
    names = {o["id"] for o in objects}
    for obj in objects:
        name = obj["id"]
        if obj["kind"] == "resource":
            role = name.removeprefix("frontend/")
            component = "vision" if role in ("preprocessor_config.json", "video_preprocessor_config.json") else "text"
            components[component]["resources"][role] = name
            continue
        bindings[name] = {"object": name}
        if obj["layout"] == "expert_block_scale_k16_m128x4_v1":
            auxiliary = name + "_input_divisors"
            if auxiliary not in names:
                raise ValueError(f"{name}: missing activation divisors")
            uses.append({"parameter": name, "input": name + "/input", "activation_policy": "AllowA4",
                         "auxiliaries": {"activation_divisor": {"object": auxiliary}}})
        elif len(obj["shape"]) >= 2 and obj["format"] != "fp8_e4m3fn":
            uses.append({"parameter": name, "input": name + "/input", "activation_policy": "A16Only"})
    return {"components": components, "bindings": bindings, "uses": uses}
