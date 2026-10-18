"""Standalone Flash-Next MTP draft-forward in torch, composed from verified pieces.

Scope: one MTP-layer call (first mtp_forward_flash_next, T=18) starting from the
layer input s1_hyper_in. Every stage is the verified implementation:

  1. attention HC mix           (verified in the consolidated draft_forward)
  2. QSA and MoE in parallel    (parity_qsa / parity_moe_w4a4, verified)
  3. MLP HC mix + combine folds (verified in the consolidated draft_forward)
  4. final HC mix -> sample     (verified in the consolidated draft_forward)
  5. proposal head (Q4_G64) -> per-512-tile top-2 -> id map -> fp8 lm_head exact
     rescore -> argmax          (parity_proposal, verified)

Routing uses this model's own top-10 (no engine dumps). Weight divisors and the
per-expert input divisors come from the dumped static weight metadata.
"""

import struct
import sys

import numpy as np
import torch

sys.path.insert(0, '/nfs/mtp-correction/torch_draft')
torch.set_default_dtype(torch.float32)

D = '/nfs/mtp-correction/oracle_dumps4'
H, S, R, N_EXP, TOP, T = 2560, 4, 320, 512, 10, 18


def bf16(t):
    return t.to(torch.bfloat16).float()


def load_act(name, rows):
    a = np.fromfile(f'{D}/{name}', dtype=np.uint16)
    return torch.from_numpy(a.reshape(-1, rows).copy()).view(torch.bfloat16).float().T.contiguous()


def load_bf16_flat(name):
    a = np.fromfile(f'{D}/{name}', dtype=np.uint16)
    return torch.from_numpy(a.copy()).view(torch.bfloat16).float()


def load_bf16_weight(name, rows):
    a = np.fromfile(f'{D}/{name}', dtype=np.uint16)
    return torch.from_numpy(a.reshape(-1, rows).copy()).view(torch.bfloat16).float()


def decode_fp8_rows(name, n, k):
    payload = np.fromfile(f'{D}/{name}', dtype=np.uint8)
    codes = torch.from_numpy(payload[:n * k].copy()).view(torch.uint8)
    scales = torch.from_numpy(payload[n * k:].copy()).view(torch.uint16).view(torch.bfloat16).float()
    sign = torch.where((codes & 0x80) != 0, -1.0, 1.0)
    exp = ((codes >> 3) & 0xF).int()
    man = (codes & 7).float()
    val = torch.where(exp == 0, man * (2.0 ** -9), (1.0 + man * 0.125) * (2.0 ** (exp.float() - 7)))
    val = torch.nan_to_num(sign * val, nan=0.0)
    return val.view(n, k) * scales.view(n, 1)


def decode_bank_nodiv(code_name, scale_name, experts, rows, columns):
    codes = np.fromfile(f'{D}/{code_name}', dtype=np.uint8)
    scales_raw = np.fromfile(f'{D}/{scale_name}', dtype=np.uint8)
    c = torch.from_numpy(codes.copy())
    table = torch.tensor([0., 0.5, 1, 1.5, 2, 3, 4, 6, -0., -0.5, -1, -1.5, -2, -3, -4, -6])
    lo = (c & 0xF).long()
    hi = ((c >> 4) & 0xF).long()
    vals = torch.stack([table[lo], table[hi]], -1).reshape(experts, rows, columns)
    sw = torch.from_numpy(scales_raw.copy()).view(torch.uint8).long()
    gpr = columns // 16
    e_i = torch.arange(experts).view(experts, 1, 1)
    row = torch.arange(rows).view(1, rows, 1)
    g = torch.arange(gpr).view(1, 1, gpr)
    off = e_i * (rows * gpr) + ((row // 128) * (gpr // 4) + g // 4) * 512 + \
        ((row % 128) & 31) * 16 + (((row % 128) >> 5) * 4) + (g & 3)
    words = sw[off]
    s = torch.where((words & 0x80) != 0, -1.0, 1.0) * torch.where(
        ((words >> 3) & 0xF) == 0, (words & 7).float() * 2.0 ** -9,
        (1 + (words & 7).float() / 8) * 2.0 ** (((words >> 3) & 0xF).float() - 7))
    return (vals * s.repeat_interleave(16, dim=2)).to(torch.bfloat16).float()


_E4M3_TABLE = None


def e4m3_table():
    global _E4M3_TABLE
    if _E4M3_TABLE is None:
        vals = []
        for code in range(256):
            sg = -1.0 if (code & 0x80) else 1.0
            e = (code >> 3) & 0xF
            m = code & 7
            if e == 0:
                v = m * 2.0 ** -9
            else:
                v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
            vals.append(sg * v)
        _E4M3_TABLE = torch.tensor(vals)
    return _E4M3_TABLE


E2M1 = torch.tensor([0., 0.5, 1, 1.5, 2, 3, 4, 6])


def snap_e2m1(v):
    v = torch.clamp(v, 0.0, 6.0)
    idx = torch.clamp(torch.searchsorted(E2M1, v), 1, 7)
    lo_v = E2M1[idx - 1]
    hi_v = E2M1[idx]
    pick_hi = (v - lo_v) > (hi_v - v)
    tie = (v - lo_v) == (hi_v - v)
    pick_hi = pick_hi | (tie & ((idx % 2) == 0))
    return torch.where(pick_hi, hi_v, lo_v)


def e4m3_encode(x):
    x = torch.nan_to_num(x, nan=0.0)
    sign = torch.where(x < 0, -1.0, 1.0)
    ax = x.abs().clamp_min(1e-30)
    e = torch.floor(torch.log2(ax))
    m = ax / 2.0 ** e
    q = (m - 1.0) * 8.0
    fl = torch.floor(q)
    frac = q - fl
    mant = torch.where(frac > 0.5, fl + 1.0,
                       torch.where((frac == 0.5) & ((fl % 2) == 1), fl + 1.0, fl))
    carry = mant >= 8
    e = e + carry.float()
    mant = torch.where(carry, torch.zeros_like(mant), mant)
    sub = e < -6
    qs = ax / 2.0 ** -9
    fls = torch.floor(qs)
    fracs = qs - fls
    msub = torch.where(fracs > 0.5, fls + 1.0,
                       torch.where((fracs == 0.5) & ((fls % 2) == 1), fls + 1.0, fls))
    val = torch.where(sub, msub * 2.0 ** -9, (1 + mant / 8) * 2.0 ** e)
    return sign * torch.clamp(val, max=448.0)


def q_nvfp4(v, div):
    v = v.view(-1, 16)
    max_abs = v.abs().max(dim=1).values
    sc = e4m3_encode(div * max_abs / 6.0)
    ds = torch.where(sc == 0, torch.ones_like(sc), sc)
    mag = snap_e2m1(v.abs() * div / ds.view(-1, 1))
    q = torch.where(v < 0, -mag, mag)
    out = (q * ds.view(-1, 1)) / div
    return torch.where((sc == 0).view(-1, 1), torch.zeros_like(out), out).reshape(-1)


def hadamard_256():
    h = torch.ones(1, 1)
    while h.shape[0] < 256:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / 16.0


def rmsnorm_grouped(x, weight, tokens):
    v = x.float()
    var = (v * v).mean(1, keepdim=True)
    return bf16(v / torch.sqrt(var + 1e-6) * (1.0 + weight.view(S, H).unsqueeze(-1)))


def hc_mix(hyper, norm, down, up, inject=None, tokens=T):
    n = rmsnorm_grouped(hyper.view(S, H, tokens), norm, tokens)
    nf = n.reshape(S * H, tokens)
    v = bf16(down.float() @ nf)
    represented = bf16(v) * 0.25
    low = bf16(represented * torch.sigmoid(represented))
    gate = bf16(up.float() @ low).view(S, H, tokens)
    block = bf16((torch.sigmoid(gate) * n).sum(0) / S)
    injection = None
    if inject is not None:
        partials = torch.einsum('sdt,esd->set', n, inject.reshape(S, S, H))
        injection = bf16(partials.sum(0))
    return block, injection, n


def combine_fold(hyper, block, injection, tokens=T):
    scale = 2.0 / (1.0 + torch.exp(-injection.float() * 0.25))
    return bf16(hyper.view(S, H, tokens) + scale.unsqueeze(1) * block.unsqueeze(0)).reshape(S * H, tokens)


def qsa(block1, weights):
    tokens = block1.shape[1]
    qg = decode_fp8_rows('w_query_gate_q.bin', 12288, 2560)
    query_w = torch.stack([qg[512 * h:512 * h + 256] for h in range(24)])
    gate_w = torch.stack([qg[512 * h + 256:512 * h + 512] for h in range(24)]).reshape(6144, 2560)
    key_w = load_bf16_weight('w_key_q.bin', 2560).view(512, 2560).float()
    val_w = load_bf16_weight('w_value_q.bin', 2560).view(512, 2560).float()
    out_w = decode_fp8_rows('w_output_q.bin', 2560, 6144)
    qn_w = load_bf16_flat('w_query_norm.bin')
    kn_w = load_bf16_flat('w_key_norm.bin')
    mrope = torch.arange(tokens, dtype=torch.float32).view(1, -1).repeat(3, 1)

    xt = bf16(block1)
    query = torch.stack([bf16(query_w[h].float() @ xt) for h in range(24)])   # [24,256,T]
    gate = bf16(gate_w.float() @ xt).reshape(6144, tokens)
    key = bf16(key_w @ xt)
    value = bf16(val_w @ xt)

    def rmsnorm_heads(xh, w):
        ms = (xh ** 2).mean(dim=1, keepdim=True)
        inv = 1.0 / torch.sqrt(ms + 1.0e-6)
        return bf16(xh * inv * bf16(w.view(1, 256, 1) + 1.0))

    nq = rmsnorm_heads(query, qn_w)
    nk = rmsnorm_heads(key.view(2, 256, tokens), kn_w)

    theta = 1.0e7

    def rope(xh, pos):
        half = 32
        out = xh.clone()
        for p in range(half):
            axis = p % 3
            freq = theta ** (-2.0 * p / 64.0)
            ang = pos[axis] * freq
            c, s = torch.cos(ang), torch.sin(ang)
            first = xh[:, p, :]
            second = xh[:, p + half, :]
            out[:, p, :] = bf16(first * c - second * s)
            out[:, p + half, :] = bf16(second * c + first * s)
        return out

    nq = rope(nq, mrope)
    nk = rope(nk, mrope)
    Hm = hadamard_256().float()
    nq = bf16(torch.stack([Hm @ nq[h] for h in range(24)]))

    # self-contained cache: hadamard K/V in fp32, quantize e4m3 row-scaled
    kh = torch.stack([Hm @ nk[h] for h in range(2)])            # fp32 [2,256,T]
    vh = torch.stack([Hm @ value.view(2, 256, tokens)[h] for h in range(2)])

    def quant_row(row):
        absmax = row.abs().max().item()
        scale = float(np.float16(min(65504.0, max(2.0 ** -24, absmax / 448.0)))) if absmax > 0 else 0.0
        if scale == 0.0:
            return torch.zeros(256), 0.0
        return e4m3_encode(row * (1.0 / scale)), scale

    k_codes = torch.zeros(2, tokens, 256)
    k_scales = torch.zeros(2, tokens)
    v_codes = torch.zeros(2, tokens, 256)
    v_scales = torch.zeros(2, tokens)
    for t in range(tokens):
        for h in range(2):
            k_codes[h, t], k_scales[h, t] = quant_row(kh[h, :, t])
            v_codes[h, t], v_scales[h, t] = quant_row(vh[h, :, t])

    attention = torch.empty(24, 256, tokens)
    for t in range(tokens):
        seq = t + 1
        for h in range(24):
            kv = h // 12
            keys = (k_codes[kv, :seq] * k_scales[kv, :seq].view(-1, 1)).to(torch.bfloat16).float()
            scores = (nq[h, :, t].to(torch.bfloat16).float() @ keys.T) / 16.0
            p = torch.softmax(scores, dim=0)
            vals = (v_codes[kv, :seq] * v_scales[kv, :seq].view(-1, 1)).to(torch.bfloat16).float()
            attention[h, :, t] = bf16(p @ vals)
    att = attention.reshape(6144, tokens)
    gated = bf16(torch.sigmoid(gate) * att)
    return bf16(out_w.float() @ gated)


def moe(block1, weights):
    router = weights['router']
    shared_packed = weights['shared_packed']
    shared_down = weights['shared_down']
    shared_scale = weights['shared_scale']
    w13raw = weights['w13raw']
    w2raw = weights['w2raw']
    gu_divs = weights['gu_divs']
    dn_divs = weights['dn_divs']
    gu_in = float(weights['gu_in'])
    dn_in = weights['dn_in']
    tokens = block1.shape[1]
    out = torch.empty(H, tokens)
    for t in range(tokens):
        xt = bf16(block1[:, t])
        acc = torch.zeros(H)
        scores = bf16(router.float() @ xt)
        topv, topi = torch.topk(scores, TOP)
        alpha = torch.softmax(topv, dim=0)
        for rank in range(TOP):
            e = int(topi[rank])
            xq = bf16(q_nvfp4(xt, gu_in))
            g = bf16(w13raw[e, :640].float() @ xq) / float(gu_divs[e])
            u = bf16(w13raw[e, 640:].float() @ xq) / float(gu_divs[e])
            h = bf16(torch.nn.functional.silu(bf16(g)) * bf16(u))
            hq = bf16(q_nvfp4(h, float(dn_in[e])))
            dwn = bf16(w2raw[e].float() @ hq) / float(dn_divs[e]) / float(dn_in[e])
            acc = bf16(acc + bf16(alpha[rank]) * dwn)
        sa = torch.sigmoid(bf16(shared_scale[0].float() @ xt))
        sh = bf16(torch.nn.functional.silu(bf16(shared_packed[:640].float() @ xt)) *
                  bf16(shared_packed[640:].float() @ xt))
        acc = bf16(acc + bf16(sa) * bf16(shared_down.float() @ sh))
        out[:, t] = acc
    return out


def propose(sample_col, state):
    n, k = 147456, 2560
    payload = np.fromfile(f'{D}/pr_head_q.bin', dtype=np.uint8)
    codes = torch.from_numpy(payload[:n * 1280].copy()).view(torch.uint8).view(n, 1280)
    scales = torch.from_numpy(payload[n * 1280:].copy()).view(torch.uint16).view(torch.float16).float().view(n, 40)
    lo = (codes & 0xF).long()
    hi = ((codes >> 4) & 0xF).long()
    q = torch.stack([lo, hi], dim=-1).reshape(n, 40, 64)
    q = ((q ^ 8) - 8).float()
    head_w = (q * scales.view(n, 40, 1)).reshape(n, k)
    logits = bf16(head_w @ bf16(sample_col))
    topv, topi = torch.topk(logits.view(-1, 512), 2, dim=1)
    ids = torch.from_numpy(np.fromfile(f'{D}/pr_ids.bin', dtype=np.int32).copy())
    mapped = ids[(torch.arange(288).view(-1, 1) * 512 + topi).flatten()]
    pv = np.fromfile(f'{D}/fw_lm_head_q.bin', dtype=np.uint8)
    vocab = 248320
    lc = torch.from_numpy(pv[:vocab * 2560].copy()).view(torch.uint8)
    ls = torch.from_numpy(pv[vocab * 2560:].copy()).view(torch.uint16).view(torch.bfloat16).float()
    sign = torch.where((lc & 0x80) != 0, -1.0, 1.0)
    exp = ((lc >> 3) & 0xF).int()
    man = (lc & 7).float()
    val = torch.where(exp == 0, man * 2.0 ** -9, (1.0 + man * 0.125) * 2.0 ** (exp.float() - 7))
    lw = torch.nan_to_num(sign * val, nan=0.0).view(vocab, 2560)
    rows = mapped.long()
    scores = bf16(lw[rows] * ls[rows].view(-1, 1) @ bf16(sample_col))
    return rows[int(torch.argmax(scores))].item()


def main():
    attn = dict(
        norm=load_bf16_weight('attn_hc_norm.bin', S * H),
        down=decode_fp8_rows('fw_attn_hc_down_q.bin', R, S * H),
        up=decode_fp8_rows('fw_attn_hc_up_q.bin', S * H, R),
        inject=load_bf16_weight('fw_attn_hc_inject_q.bin', S * H),
    )
    mlp = dict(
        norm=load_bf16_weight('mlp_hc_norm.bin', S * H),
        down=decode_fp8_rows('fw_mlp_hc_down_q.bin', R, S * H),
        up=decode_fp8_rows('fw_mlp_hc_up_q.bin', S * H, R),
        inject=load_bf16_weight('fw_mlp_hc_inject_q.bin', S * H),
    )
    final = dict(
        norm=load_bf16_weight('final_hc_norm.bin', S * H),
        down=decode_fp8_rows('fw_final_hc_down_q.bin', R, S * H),
        up=decode_fp8_rows('fw_final_hc_up_q.bin', S * H, R),
    )
    gu_divs = torch.from_numpy(np.fromfile(f'{D}/gu_weight_divs.bin', dtype=np.uint8).copy()).view(torch.float32)
    dn_divs = torch.from_numpy(np.fromfile(f'{D}/down_weight_divs.bin', dtype=np.uint8).copy()).view(torch.float32)
    gu_in = torch.from_numpy(np.fromfile(f'{D}/m_routed_gu_input_div.bin', dtype=np.uint8).copy()).view(torch.float32)
    dn_in = torch.from_numpy(np.fromfile(f'{D}/m_routed_down_input_div.bin', dtype=np.uint8).copy()).view(torch.float32)
    moe_weights = dict(
        router=load_bf16_weight('fw_moe_router_q.bin', H),
        shared_packed=decode_fp8_rows('fw_moe_shared_packed_q.bin', 1280, H),
        shared_down=decode_fp8_rows('fw_moe_shared_down_q.bin', H, 640),
        shared_scale=load_bf16_weight('fw_moe_shared_scale_q.bin', H),
        w13raw=decode_bank_nodiv('m_routed_gu_codes.bin', 'm_routed_gu_scales.bin', N_EXP, 1280, H),
        w2raw=decode_bank_nodiv('m_routed_down_codes.bin', 'm_routed_down_scales.bin', N_EXP, H, 640),
        gu_divs=gu_divs, dn_divs=dn_divs, gu_in=float(gu_in[0]), dn_in=dn_in,
    )

    s1 = load_act('s1_hyper_in.bin', S * H)
    s2_block1_ref = load_act('s2_block1.bin', H)
    s3_moe_ref = load_act('s3_moe_out.bin', H)
    s3_block2_ref = load_act('s3_block2.bin', H)
    s4_sample_ref = load_act('s4_sample.bin', H)

    def rel(a, b):
        d = (a - b).abs().max().item()
        return d, d / (b.abs().max().item() + 1e-9)

    block1, inj1, _ = hc_mix(s1, **attn)
    d, r = rel(bf16(block1), s2_block1_ref)
    print(f'1 HC-mix   block max {d:.5f} (rel {r:.5f})')

    qsa_out = qsa(block1, attn)
    d, r = rel(qsa_out, load_act('t_dest.bin', H))
    print(f'2 QSA      out   max {d:.5f} (rel {r:.5f})')
    moe_out = moe(block1, moe_weights)
    d, r = rel(moe_out, s3_moe_ref)
    print(f'3 MoE      out   max {d:.5f} (rel {r:.5f})')

    folded = combine_fold(s1, bf16(moe_out), bf16(inj1))
    block2, inj2, _ = hc_mix(folded, **mlp)
    d, r = rel(bf16(block2), s3_block2_ref)
    print(f'4 MLP-mix  block max {d:.5f} (rel {r:.5f})')

    final_folded = combine_fold(folded, bf16(moe_out), bf16(inj2))
    sample, _, _ = hc_mix(final_folded, **final)
    d, r = rel(bf16(sample), s4_sample_ref)
    print(f'5 final    sample max {d:.5f} (rel {r:.5f})')

    token = propose(sample[:, -1], None)
    engine_token = int(torch.from_numpy(np.fromfile(f'{D}/pr_out.bin', dtype=np.int32).copy())[0])
    print(f'6 proposal token mine {token} engine {engine_token} match {token == engine_token}')


if __name__ == '__main__':
    main()
