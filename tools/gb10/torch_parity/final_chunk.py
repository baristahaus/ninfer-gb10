"""Flash-Next MTP end-to-end forward for the final prefill chunk (T=2).

Inputs: s1_parity (hyper after the stem of core-call 1), the engine's quantized
KV page (18 keys from the T=18 chunk) plus this call's own K/V appends at
positions 18,19. Every stage is the verified implementation; routing is this
model's own top-10. Deliverables: per-stage diffs vs the call-1 dumps and the
proposal token vs pr_out.
"""

import numpy as np
import torch
import sys

sys.path.insert(0, '/nfs/mtp-correction/torch_draft')
import draft_forward as DF

H, S, R = DF.H, DF.S, DF.R
T = 2
BASE_POS = 18


def rmsnorm_heads(xh, w):
    ms = (xh ** 2).mean(dim=1, keepdim=True)
    inv = 1.0 / torch.sqrt(ms + 1e-6)
    return DF.bf16(xh * inv * DF.bf16(w.view(1, 256, 1) + 1.0))


def rope(xh, positions):
    # positions: [T] int values; mrope plane-major [3, T] all equal
    half = 32
    out = xh.clone()
    theta = 1.0e7
    pos3 = positions.float().view(1, -1).repeat(3, 1)
    for p in range(half):
        axis = p % 3
        freq = theta ** (-2.0 * p / 64.0)
        ang = pos3[axis] * freq
        c, s = torch.cos(ang), torch.sin(ang)
        out[:, p, :] = DF.bf16(xh[:, p, :] * c - xh[:, p + half, :] * s)
        out[:, p + half, :] = DF.bf16(xh[:, p + half, :] * c + xh[:, p, :] * s)
    return out


def main():
    s1 = DF.load_act('s1_parity.bin', S * H)  # [S*H, T]
    attn = dict(
        norm=DF.load_bf16_weight('attn_hc_norm.bin', S * H),
        down=DF.decode_fp8_rows('fw_attn_hc_down_q.bin', R, S * H),
        up=DF.decode_fp8_rows('fw_attn_hc_up_q.bin', S * H, R),
        inject=DF.load_bf16_weight('fw_attn_hc_inject_q.bin', S * H),
    )
    mlp = dict(
        norm=DF.load_bf16_weight('mlp_hc_norm.bin', S * H),
        down=DF.decode_fp8_rows('fw_mlp_hc_down_q.bin', R, S * H),
        up=DF.decode_fp8_rows('fw_mlp_hc_up_q.bin', S * H, R),
        inject=DF.load_bf16_weight('fw_mlp_hc_inject_q.bin', S * H),
    )
    final = dict(
        norm=DF.load_bf16_weight('final_hc_norm.bin', S * H),
        down=DF.decode_fp8_rows('fw_final_hc_down_q.bin', R, S * H),
        up=DF.decode_fp8_rows('fw_final_hc_up_q.bin', S * H, R),
    )
    block1, inj1, _ = DF.hc_mix(s1, **attn, tokens=T)
    b1_ref = DF.load_act('s2_block1.bin', H)
    print('1 HC-mix   block diff %.5f (rel %.5f)' % (
        (DF.bf16(block1) - b1_ref).abs().max().item(),
        (DF.bf16(block1) - b1_ref).abs().max().item() / b1_ref.abs().max().item()))

    # QSA T=2
    qg = DF.decode_fp8_rows('w_query_gate_q.bin', 12288, 2560)
    query_w = torch.stack([qg[512 * h:512 * h + 256] for h in range(24)])
    gate_w = torch.stack([qg[512 * h + 256:512 * h + 512] for h in range(24)]).reshape(6144, 2560)
    key_w = DF.load_bf16_weight('w_key_q.bin', 2560).view(512, 2560).float()
    val_w = DF.load_bf16_weight('w_value_q.bin', 2560).view(512, 2560).float()
    out_w = DF.decode_fp8_rows('w_output_q.bin', 2560, 6144)
    qn_w = DF.load_bf16_flat('w_query_norm.bin')
    kn_w = DF.load_bf16_flat('w_key_norm.bin')
    xt = DF.bf16(block1)
    query = torch.stack([DF.bf16(query_w[h].float() @ xt) for h in range(24)])
    gate = DF.bf16(gate_w.float() @ xt).reshape(6144, T)
    key = DF.bf16(key_w @ xt).view(2, 256, T)
    value = DF.bf16(val_w @ xt).view(2, 256, T)
    nq = rmsnorm_heads(query, qn_w)
    nk = rmsnorm_heads(key, kn_w)
    positions = torch.tensor([float(BASE_POS + t) for t in range(T)])
    nq = rope(nq, positions)
    nk = rope(nk, positions)
    Hm = DF.hadamard_256().float()
    nqh = DF.bf16(torch.stack([Hm @ nq[h] for h in range(24)]))

    tab = DF.e4m3_table()
    k_codes = torch.from_numpy(np.fromfile(DF.D + '/qs_k_0.bin', dtype=np.uint8).copy()).long().view(2, 64, 256)
    k_scale = torch.from_numpy(np.fromfile(DF.D + '/qs_ks_0.bin', dtype=np.uint16).copy()).view(torch.float16).float().view(2, 64)
    v_codes = torch.from_numpy(np.fromfile(DF.D + '/qs_v_0.bin', dtype=np.uint8).copy()).long().view(2, 64, 256)
    v_scale = torch.from_numpy(np.fromfile(DF.D + '/qs_vs_0.bin', dtype=np.uint16).copy()).view(torch.float16).float().view(2, 64)
    khn = torch.stack([Hm @ nk[h] for h in range(2)])
    vhn = torch.stack([Hm @ value[h] for h in range(2)])
    knew = torch.zeros(2, T, 256)
    vnew = torch.zeros(2, T, 256)
    ksnew = torch.zeros(2, T)
    vsnew = torch.zeros(2, T)
    for t in range(T):
        for h in range(2):
            row = khn[h, :, t]
            am = row.abs().max().item()
            sc = float(np.float16(min(65504.0, max(2.0 ** -24, am / 448.0)))) if am > 0 else 0.0
            knew[h, t] = DF.e4m3_encode(row * (1.0 / sc)) if sc > 0 else 0
            ksnew[h, t] = sc
            row = vhn[h, :, t]
            am = row.abs().max().item()
            sc = float(np.float16(min(65504.0, max(2.0 ** -24, am / 448.0)))) if am > 0 else 0.0
            vnew[h, t] = DF.e4m3_encode(row * (1.0 / sc)) if sc > 0 else 0
            vsnew[h, t] = sc

    attention = torch.empty(24, 256, T)
    for t in range(T):
        seq = BASE_POS + t + 1
        for h in range(24):
            kv = h // 12
            keys = torch.cat([(tab[k_codes[kv, :BASE_POS]] * k_scale[kv, :BASE_POS].view(-1, 1)),
                              knew[kv, :t + 1] * ksnew[kv, :t + 1].view(-1, 1)]).to(torch.bfloat16).float()
            scores = (nqh[h, :, t].to(torch.bfloat16).float() @ keys.T) / 16.0
            p = torch.softmax(scores, dim=0)
            vals = torch.cat([(tab[v_codes[kv, :BASE_POS]] * v_scale[kv, :BASE_POS].view(-1, 1)),
                              vnew[kv, :t + 1] * vsnew[kv, :t + 1].view(-1, 1)]).to(torch.bfloat16).float()
            attention[h, :, t] = DF.bf16(p @ vals)
    gated = DF.bf16(torch.sigmoid(gate) * attention.reshape(6144, T))
    qsa_out = DF.bf16(out_w @ gated)
    qsa_ref = DF.load_act('t_dest.bin', H)
    print('2 QSA      out   diff %.5f (rel %.5f)' % (
        (qsa_out - qsa_ref).abs().max().item(),
        (qsa_out - qsa_ref).abs().max().item() / qsa_ref.abs().max().item()))

    gu_divs = torch.from_numpy(np.fromfile(DF.D + '/gu_weight_divs.bin', dtype=np.uint8).copy()).view(torch.float32)
    dn_divs = torch.from_numpy(np.fromfile(DF.D + '/down_weight_divs.bin', dtype=np.uint8).copy()).view(torch.float32)
    gu_in = torch.from_numpy(np.fromfile(DF.D + '/m_routed_gu_input_div.bin', dtype=np.uint8).copy()).view(torch.float32)
    dn_in = torch.from_numpy(np.fromfile(DF.D + '/m_routed_down_input_div.bin', dtype=np.uint8).copy()).view(torch.float32)
    moe_w = dict(
        router=DF.load_bf16_weight('fw_moe_router_q.bin', H),
        shared_packed=DF.decode_fp8_rows('fw_moe_shared_packed_q.bin', 1280, H),
        shared_down=DF.decode_fp8_rows('fw_moe_shared_down_q.bin', H, 640),
        shared_scale=DF.load_bf16_weight('fw_moe_shared_scale_q.bin', H),
        w13raw=DF.decode_bank_nodiv('m_routed_gu_codes.bin', 'm_routed_gu_scales.bin', DF.N_EXP, 1280, H),
        w2raw=DF.decode_bank_nodiv('m_routed_down_codes.bin', 'm_routed_down_scales.bin', DF.N_EXP, H, 640),
        gu_divs=gu_divs, dn_divs=dn_divs, gu_in=float(gu_in[0]), dn_in=dn_in,
    )
    moe_out = DF.moe(block1, moe_w)
    moe_ref = DF.load_act('s3_moe_out.bin', H)
    print('3 MoE      out   diff %.5f (rel %.5f)' % (
        (DF.bf16(moe_out) - moe_ref).abs().max().item(),
        (DF.bf16(moe_out) - moe_ref).abs().max().item() / moe_ref.abs().max().item()))

    folded = DF.combine_fold(s1, DF.bf16(moe_out), DF.bf16(inj1), tokens=T)
    block2, inj2, _ = DF.hc_mix(folded, **mlp, tokens=T)
    ff = DF.combine_fold(folded, DF.bf16(moe_out), DF.bf16(inj2), tokens=T)
    sample, _, _ = DF.hc_mix(ff, **final, tokens=T)
    sample_ref = DF.load_act('s4_sample.bin', H)
    print('4 final    sample diff %.5f (rel %.5f)' % (
        (DF.bf16(sample) - sample_ref).abs().max().item(),
        (DF.bf16(sample) - sample_ref).abs().max().item() / sample_ref.abs().max().item()))

    token = DF.propose(DF.bf16(sample[:, T - 1]), None)
    engine_token = int(torch.from_numpy(np.fromfile(DF.D + '/pr_out.bin', dtype=np.int32).copy())[0])
    print('5 proposal token mine %d engine %d match %s' % (token, engine_token, token == engine_token))


if __name__ == '__main__':
    main()
