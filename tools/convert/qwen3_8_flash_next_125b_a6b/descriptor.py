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
            input_divisors = name + "_input_divisors"
            weight_divisors = name + "_weight_divisors"
            if input_divisors in names:
                uses.append({"parameter": name, "input": name + "/input", "activation_policy": "AllowA4",
                             "auxiliaries": {"activation_divisor": {"object": input_divisors}}})
            elif weight_divisors in names:
                uses.append({"parameter": name, "input": name + "/input", "activation_policy": "AllowA16",
                             "auxiliaries": {"weight_divisor": {"object": weight_divisors}}})
            else:
                raise ValueError(f"{name}: missing bank divisors")
        elif len(obj["shape"]) >= 2 and obj["format"] != "fp8_e4m3fn" and obj["layout"] != "block_scale_k16_v1":
            uses.append({"parameter": name, "input": name + "/input", "activation_policy": "A16Only"})
    return {"components": components, "bindings": bindings, "uses": uses}
