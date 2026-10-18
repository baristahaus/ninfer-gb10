"""Flash-Next QSA parity vs engine dumps (first MTP prefill call, 18 tokens, one page).

Established dataflow (flash_next_qsa.cu):
  query, gate = split(packed fp8 row-scaled projection 12288x2560)   # 24 heads x [q256,gate256]
  key, value  = bf16 projections 512x2560
  rmsnorm per head: out = x/sqrt(mean(x^2)+1e-6) * (w + 1), w bf16 [256]
  rope: generic, axes=3, rotary_dim=64, theta=1e7; pair p in [0,32):
        axis=p%3, freq=theta^(-2p/64), rotate (d[p], d[p+32]); dims 64.. untouched
  fp8 cache: K and V rows are Hadamard-transformed (normalized H256/16) then
             row-quantized e4m3 with scale=fp16(448/absmax clamped), code=v/scale
  attention: qhat = H256/16 roped q; scores = qhat . khat / 16 (fp32 softmax);
             out = sum p * vhat (bf16 vhat, fp32 accum); heads 12 per kv head
  sigmoid_mul: out = sigmoid(gate) * attention
  destination = fp8 row-scaled output projection 2560x6144
"""

import numpy as np
import torch
import struct

torch.set_default_dtype(torch.float32)
D = '/nfs/mtp-correction/oracle_dumps4/'


def load(name, rows, dt=np.uint16, transpose=True):
    a = np.fromfile(D + name, dtype=dt)
    t = torch.from_numpy(a.reshape(-1, rows).copy())
    t = t.view(torch.bfloat16).float() if dt == np.uint16 else t.view(torch.float32) if dt == np.uint32 else t.float()
    return t.T.contiguous() if transpose else t


def load_bf16(name):
    return torch.from_numpy(np.fromfile(D + name, dtype=np.uint16).copy()).view(torch.bfloat16).float()


def decode_fp8_rows(name, n, k):
    payload = np.fromfile(D + name, dtype=np.uint8)
    codes = torch.from_numpy(payload[:n * k].copy()).view(torch.uint8)
    scales = torch.from_numpy(payload[n * k:].copy()).view(torch.uint16).view(torch.bfloat16).float()
    sign = torch.where((codes & 0x80) != 0, -1.0, 1.0)
    exp = ((codes >> 3) & 0xF).int()
    man = (codes & 7).float()
    val = torch.where(exp == 0, man * 2.0 ** -9, (1.0 + man * 0.125) * 2.0 ** (exp.float() - 7))
    return torch.nan_to_num(sign * val, nan=0.0).view(n, k) * scales.view(n, 1)


E4M3_MAX = 448.0
E4M3_TABLE = None


def e4m3_table():
    global E4M3_TABLE
    if E4M3_TABLE is None:
        vals = []
        for code in range(256):
            s = -1.0 if (code & 0x80) else 1.0
            e = (code >> 3) & 0xF
            m = code & 7
            if code == 0x80:
                v = float('nan')
            elif e == 0:
                v = m * 2.0 ** -9
            else:
                v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
            vals.append(s * v)
        E4M3_TABLE = torch.tensor(vals)
    return E4M3_TABLE


def e4m3_encode(x):
    x = torch.nan_to_num(x, nan=0.0)
    sign = torch.where(x < 0, -1.0, 1.0)
    ax = x.abs().clamp_min(1e-30)
    e = torch.floor(torch.log2(ax))
    m = ax / 2.0 ** e
    q = (m - 1.0) * 8.0
    fl = torch.floor(q)
    frac = q - fl
    # round half to even
    mant = torch.where(frac > 0.5, fl + 1.0,
                       torch.where((frac == 0.5) & ((fl % 2) == 1), fl + 1.0, fl))
    carry = mant >= 8
    e = e + carry.float()
    mant = torch.where(carry, torch.zeros_like(mant), mant)
    sub = e < -6
    qs = ax / 2.0 ** -9
    fls = torch.floor(qs); fracs = qs - fls
    msub = torch.where(fracs > 0.5, fls + 1.0,
                       torch.where((fracs == 0.5) & ((fls % 2) == 1), fls + 1.0, fls))
    val = torch.where(sub, msub * 2.0 ** -9, (1 + mant / 8) * 2.0 ** e)
    return sign * torch.clamp(val, max=448.0)


def bf16t(t):
    return t.to(torch.bfloat16).float()


def hadamard_256():
    # Sylvester H256 with H[i,j] = (-1)^popcount(i & j), normalized by 1/16.
    idx = torch.arange(256)
    bits = ((idx.view(-1, 1) & idx.view(1, -1)) != 0)
    bits = bits & (idx.view(-1, 1).ne(0))  # popcount via matrix is wrong; rebuild below
    # correct construction:
    h = torch.ones(256, 256)
    for i in range(256):
        pc = bin(i).count('1')
        for j in range(256):
            pc2 = bin(i & j).count('1')
            if (pc2 - pc) % 2:
                h[i, j] = -1.0
    # The butterfly produces H x with row bit-reversal? validated against the
    # engine by parity, sign convention fixed if needed.
    return h / 16.0


def hadamard_256_fast():
    n = 256
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / 16.0


def main():
    tokens = 18
    x = load('s2_block1.bin', 2560)          # [2560, tokens]
    qg = decode_fp8_rows('w_query_gate_q.bin', 12288, 2560)   # [12288, 2560]
    key_w = load('w_key_q.bin', 2560, transpose=False).view(512, 2560).float()
    val_w = load('w_value_q.bin', 2560, transpose=False).view(512, 2560).float()
    out_w = decode_fp8_rows('w_output_q.bin', 2560, 6144)     # [2560, 6144]
    qn_w = load_bf16('w_query_norm.bin')                      # [256]
    kn_w = load_bf16('w_key_norm.bin')

    # packed rows: head h occupies rows [512h, 512h+256) query, [512h+256, 512h+512) gate
    query_w = torch.stack([qg[512 * h:512 * h + 256] for h in range(24)])   # [24,256,2560]
    gate_w = torch.stack([qg[512 * h + 256:512 * h + 512] for h in range(24)])

    H = hadamard_256_fast().float()

    cache_pos = load('qs_cache_pos.bin', 1, np.int32).reshape(-1)  # [tokens]
    mrope = torch.from_numpy(np.fromfile(D + 'qs_mrope.bin', dtype=np.int32).copy()).view(3, -1)  # plane-major [3, tokens]
    t_dest = load('t_dest.bin', 2560)                              # [2560, tokens]
    t_att = load('t_attention.bin', 6144)

    theta = 1.0e7
    # projections (bf16 inputs/weights, fp32 accum)
    xt = bf16t(x)
    query = bf16t(torch.stack([bf16t(query_w[h].float() @ xt) for h in range(24)]))  # [24,256,T]
    gate = bf16t(torch.stack([bf16t(gate_w[h].float() @ xt) for h in range(24)]))
    key = bf16t(key_w @ xt)      # [512, T]
    value = bf16t(val_w @ xt)    # [512, T]

    # rmsnorm (unit offset)
    def rmsnorm_heads(xh, w):   # xh [H,256,T], w [256]
        ms = (xh ** 2).mean(dim=1, keepdim=True)
        inv = 1.0 / torch.sqrt(ms + 1.0e-6)
        return bf16t(xh * inv * bf16t(w.view(1, 256, 1) + 1.0))
    nq = rmsnorm_heads(query, qn_w)
    nk = rmsnorm_heads(key.view(2, 256, tokens), kn_w)  # [256,2,T]

    # rope
    def rope(xh, heads):
        half = 32
        out = xh.clone()
        for p in range(half):
            axis = p % 3
            freq = theta ** (-2.0 * p / 64.0)
            ang = mrope[axis].float() * freq
            c, s = torch.cos(ang), torch.sin(ang)
            first = xh[:, p, :]
            second = xh[:, p + half, :]
            out[:, p, :] = bf16t(first * c - second * s)
            out[:, p + half, :] = bf16t(second * c + first * s)
        return out
    nq = rope(nq, 24)
    nk = rope(nk, 2)

    # hadamard on query heads
    nq = bf16t(torch.einsum('ij,hjt->hit', H, nq))

    # cache: dequant dumped page 0 (K and V hadamarded at append)
    k_codes = np.fromfile(D + 'qs_k_0.bin', dtype=np.uint8)      # 256*64*2
    k_scale = torch.from_numpy(np.fromfile(D + 'qs_ks_0.bin', dtype=np.uint16).copy()).view(torch.float16).float()
    v_codes = np.fromfile(D + 'qs_v_0.bin', dtype=np.uint8)
    v_scale = torch.from_numpy(np.fromfile(D + 'qs_vs_0.bin', dtype=np.uint16).copy()).view(torch.float16).float()
    tab = e4m3_table()
    k_flat = tab[torch.from_numpy(k_codes.copy()).long()]
    v_flat = tab[torch.from_numpy(v_codes.copy()).long()]
    # layout: d + 256*(page_offset + 64*(head))  (page 0)
    k_cache = (k_flat.view(2, 64, 256).permute(2, 1, 0) * k_scale.view(2, 64).T).contiguous()
    v_cache = (v_flat.view(2, 64, 256).permute(2, 1, 0) * v_scale.view(2, 64).T).contiguous()

    attention = torch.empty(24, 256, tokens)
    for t in range(tokens):
        seq_len = int(cache_pos[t]) + 1
        for h in range(24):
            kv = h // 12
            qv = nq[h, :, t]                                     # [256]
            keys = k_cache[:, :seq_len, kv].to(torch.bfloat16).float()   # [256, seq]
            scores = (qv.to(torch.bfloat16).float() @ keys) / 16.0
            p = torch.softmax(scores, dim=0)
            vals = v_cache[:, :seq_len, kv].to(torch.bfloat16).float()
            attention[h, :, t] = bf16t(p @ vals.T)
    attention = attention.reshape(6144, tokens)

    gated = bf16t(torch.sigmoid(gate.reshape(6144, tokens)) * attention)
    att_ref = load('t_attention.bin', 6144)
    d1 = (attention.reshape(6144, tokens) - att_ref).abs()
    print('attention diff max %.5f  rel %.4f' % (d1.max().item(), d1.max().item() / att_ref.abs().max().item()))
    dest = bf16t(out_w @ gated)
    d2 = (dest - t_dest).abs()
    print('QSA dest diff max %.5f  rel %.4f' % (d2.max().item(), d2.max().item() / t_dest.abs().max().item()))


if __name__ == '__main__':
    main()
