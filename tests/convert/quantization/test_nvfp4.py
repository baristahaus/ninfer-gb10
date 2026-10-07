from __future__ import annotations

import torch

from tools.artifact.codecs.nvfp4 import unswizzle_nvfp4_scales
from tools.artifact.layouts import align_up
from tools.convert.qwen3_8_flash_next_125b_a6b import descriptor, inventory
from tools.convert.quantization.nvfp4 import quantize_expert_bank

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _decode_codes(packed: torch.Tensor) -> torch.Tensor:
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(packed.shape[0], -1).long()
    return _E2M1[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)


def test_bank_payload_is_nearest_nvfp4_rounding() -> None:
    experts, n, k = 2, 128, 64
    generator = torch.Generator().manual_seed(3)
    bank = (torch.randn(experts, n, k, generator=generator) *
            torch.logspace(-3, 1, k)).to(torch.bfloat16)
    bank[1, :, 16:32] = 0
    payload = b"".join(quantize_expert_bank(bank, torch.device("cpu")))
    code_bytes = experts * n * k // 2
    scale_offset = align_up(code_bytes, 256)
    assert len(payload) == scale_offset + experts * n * k // 16 + experts * 4
    raw = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    divisors = raw[-experts * 4:].view(torch.float32)
    for expert in range(experts):
        weight = bank[expert].double()
        global_scale = weight.abs().max() / (6 * 448)
        assert divisors[expert].item() == (1 / global_scale).float().item()
        packed = raw[expert * n * k // 2:(expert + 1) * n * k // 2].reshape(n, k // 2)
        plane = raw[scale_offset + expert * n * k // 16:scale_offset + (expert + 1) * n * k // 16]
        scales = unswizzle_nvfp4_scales(plane.clone(), (n, k)).view(torch.float8_e4m3fn)
        group = weight.reshape(n, k // 16, 16)
        expected = (group.abs().amax(dim=2) / 6 / global_scale).float().to(torch.float8_e4m3fn)
        assert torch.equal(scales.view(torch.uint8), expected.view(torch.uint8))
        step = scales.double().repeat_interleave(16, dim=1) / divisors[expert].double()
        decoded = _decode_codes(packed).double() * step
        # Every value takes a nearest representable code of its group scale.
        candidates = torch.cat((_E2M1, -_E2M1)).double()[None, None, :] * step[..., None]
        best = (candidates - weight[..., None]).abs().amin(dim=-1)
        assert torch.equal((decoded - weight).abs(), best)
        assert (decoded[weight == 0] == 0).all()


def test_e2m1_ties_round_to_even() -> None:
    bank = torch.zeros(1, 128, 64, dtype=torch.bfloat16)
    # Group amax 6 makes the group scale exactly 6/(6*global) = 448 and the step one global unit,
    # so the remaining values land exactly on E2M1 midpoints.
    row = torch.tensor([6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.75, -5.0, 0, 0, 0, 0, 0, 0])
    bank[0, 0, :16] = row.to(torch.bfloat16)
    payload = b"".join(quantize_expert_bank(bank, torch.device("cpu")))
    packed = torch.frombuffer(bytearray(payload[:8]), dtype=torch.uint8).reshape(1, 8)
    assert _decode_codes(packed)[0].tolist() == [
        6.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, -1.0, -4.0, 0, 0, 0, 0, 0, 0]


def test_nvfp4_mtp_experts_use_main_bank_contract() -> None:
    specs = inventory.object_specs(inventory.FP8, inventory.FP8_ROW, inventory.NVFP4)
    tensors = {spec.id: spec for spec in specs if isinstance(spec, inventory.TensorSpec)}
    prefix = "mtp.layers.0.mlp.experts."
    assert not set(inventory.MTP_EXPERT_BF16) & set(tensors)
    assert tensors[prefix + "gate_up"].format == inventory.NVFP4
    assert tensors[prefix + "down_input_divisors"].shape == (512,)
    objects = [{"id": spec.id, "kind": "tensor", "shape": list(spec.shape),
                "format": spec.format, "layout": spec.layout} for spec in tensors.values()]
    uses = {use["parameter"]: use for use in descriptor.describe(objects)["uses"]}
    assert uses[prefix + "gate_up"]["activation_policy"] == "AllowA4"
    assert uses[prefix + "down"]["auxiliaries"]["activation_divisor"]["object"] == (
        prefix + "down_input_divisors")
