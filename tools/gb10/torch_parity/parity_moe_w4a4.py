"""Flash-Next MoE parity, corrected dataflow + full NVFP4 W4A4 semantics.

Established formula (verified against engine dumps, first MTP prefill call,
tokens=18, assignments=180):

Layer dataflow (text_context_impl.h mtp body):
  s1 -> HC mix -> block1 (s2_block1, s2_inj1)
  QSA(block1) and MoE(block1) run IN PARALLEL on block1
  MoE out = s3_moe_out; HC combine_mix(s1, moe_out, s2_inj1) -> block2 (s3_block2)
  HC combine -> hyper; final HC mix -> s4_sample

Routed MoE (per assignment = (token, rank)):
  xhat = NVFP4(bf16 x, input divisor gu_in[e])   # gather_quantize_routes_kernel
  g,u  = (gu_codes*gu_scales @ xhat) / wdiv_gu[e]   # W4A4 mma, per-expert weight divisor
  h    = bf16(silu(g) * u)                          # GroupedSiluOutput
  hhat = NVFP4(h, input divisor dn_in[e])           # quantize_grouped_kernel
  down = (down_codes*down_scales @ hhat) / wdiv_dn[e] / dn_in[e]
  routed = sum_r bf16(alpha_r * down_r)             # reduce_grouped_kernel
  shared = sigmoid(ss@x) * (shared_down @ silu(shared_gate @ x, shared_up @ x))
  out    = routed + shared

NVFP4 quantize: per 16-group, scale = e4m3(div * maxabs / 6); code = e2m1(v*div/scale),
sign-magnitude, round-half-to-even; dequant operand = code * scale; the epilogue divides
the reduction by (weight divisor * input divisor).

Verification (oracle_dumps4):
  x codes:   my quantize == engine g_x_codes 100% (scales byte-exact)
  h codes:   99.3% (2.6% e4m3 scale flips from last-bit fp32 accumulation diffs)
  down+reduce+shared with engine h codes: max diff 0.0078 (0.9% rel, corr 1.0)
  gate chain fp64: h within 1-2 bf16 ULP of engine h (median 0.0039)
  self-contained (own quantize of own h): 0.13 rel 0.16 - limited by NVFP4
  decision-boundary flips from fp32 accumulation-order differences.

Root causes of the earlier mismatch: (1) wrong stage graph (fed s3_block2 into the
MoE; the MoE input is s2_block1), (2) single weight divisor instead of per-expert
arrays, (3) missing W4A4 activation quantization on both GEMMs.
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


def decode_bank_nodiv(code_name, scale_name, E, N, K):
    codes = np.fromfile(D + code_name, dtype=np.uint8)
    scales_raw = np.fromfile(D + scale_name, dtype=np.uint8)
    c = torch.from_numpy(codes.copy())
    table = torch.tensor([0., .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
    lo = (c & 0xF).long()
    hi = ((c >> 4) & 0xF).long()
    vals = torch.stack([table[lo], table[hi]], -1).reshape(E, N, K)
    sw = torch.from_numpy(scales_raw.copy()).view(torch.uint8).long()
    gpr = K // 16
    e_i = torch.arange(E).view(E, 1, 1)
    row = torch.arange(N).view(1, N, 1)
    g = torch.arange(gpr).view(1, 1, gpr)
    off = e_i * (N * gpr) + ((row // 128) * (gpr // 4) + g // 4) * 512 + \
        ((row % 128) & 31) * 16 + (((row % 128) >> 5) * 4) + (g & 3)
    words = sw[off]
    s = torch.where((words & 0x80) != 0, -1.0, 1.0) * torch.where(
        ((words >> 3) & 0xF) == 0,
        (words & 7).float() * 2.0 ** -9,
        (1 + (words & 7).float() / 8) * 2.0 ** (((words >> 3) & 0xF).float() - 7))
    return (vals * s.repeat_interleave(16, dim=2)).to(torch.bfloat16).float()


def decode_fp8_rows(name, n, k):
    payload = np.fromfile(D + name, dtype=np.uint8)
    codes = torch.from_numpy(payload[:n * k].copy()).view(torch.uint8)
    scales = torch.from_numpy(payload[n * k:].copy()).view(torch.uint16).view(torch.bfloat16).float()
    sign = torch.where((codes & 0x80) != 0, -1.0, 1.0)
    exp = ((codes >> 3) & 0xF).int()
    man = (codes & 7).float()
    val = torch.where(exp == 0, man * 2.0 ** -9, (1.0 + man * 0.125) * 2.0 ** (exp.float() - 7))
    return torch.nan_to_num(sign * val, nan=0.0).view(n, k) * scales.view(n, 1)


E2M1 = torch.tensor([0., .5, 1, 1.5, 2, 3, 4, 6])


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
    mant = torch.round((m - 1) * 8)
    carry = mant >= 8
    e = e + carry.float()
    mant = torch.where(carry, torch.zeros_like(mant), mant)
    sub = e < -6
    val = torch.where(sub, torch.round(ax / 2.0 ** -9) * 2.0 ** -9, (1 + mant / 8) * 2.0 ** e)
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


def main():
    gu_w = decode_bank_nodiv('m_routed_gu_codes.bin', 'm_routed_gu_scales.bin', 512, 1280, 2560).float()
    w2_w = decode_bank_nodiv('m_routed_down_codes.bin', 'm_routed_down_scales.bin', 512, 2560, 640).float()
    gu_divs = torch.from_numpy(np.fromfile(D + 'gu_weight_divs.bin', dtype=np.uint8).copy()).view(torch.float32)
    dn_divs = torch.from_numpy(np.fromfile(D + 'down_weight_divs.bin', dtype=np.uint8).copy()).view(torch.float32)
    gu_in = torch.from_numpy(np.fromfile(D + 'm_routed_gu_input_div.bin', dtype=np.uint8).copy()).view(torch.float32)
    dn_in = torch.from_numpy(np.fromfile(D + 'm_routed_down_input_div.bin', dtype=np.uint8).copy()).view(torch.float32)
    shared_packed = decode_fp8_rows('fw_moe_shared_packed_q.bin', 1280, 2560)
    shared_down = decode_fp8_rows('fw_moe_shared_down_q.bin', 2560, 640)
    ss = load('fw_moe_shared_scale_q.bin', 2560, transpose=False).view(1, 2560)
    inp = load('r0_input.bin', 2560)
    ids = load('r0_ids.bin', 10, np.int32)
    alpha = load('r0_alpha.bin', 10, np.uint32).view(torch.float32)
    s3_moe = load('s3_moe_out.bin', 2560)

    def bf16t(t):
        return t.to(torch.bfloat16).float()

    out = torch.empty(2560, 18)
    for t in range(18):
        xt = bf16t(inp[:, t])
        acc = torch.zeros(2560)
        for rank in range(10):
            e = int(ids[rank, t])
            xq = bf16t(q_nvfp4(xt, float(gu_in[e])))
            g = bf16t(gu_w[e, :640] @ xq) / float(gu_divs[e])
            u = bf16t(gu_w[e, 640:] @ xq) / float(gu_divs[e])
            h = bf16t(torch.nn.functional.silu(bf16t(g)) * bf16t(u))
            hq = bf16t(q_nvfp4(h, float(dn_in[e])))
            dwn = bf16t(w2_w[e] @ hq) / float(dn_divs[e]) / float(dn_in[e])
            acc = bf16t(acc + bf16t(alpha[rank, t]) * dwn)
        sa = torch.sigmoid(bf16t(ss[0] @ xt))
        sh = bf16t(torch.nn.functional.silu(bf16t(shared_packed[:640] @ xt)) *
                   bf16t(shared_packed[640:] @ xt))
        acc = bf16t(acc + bf16t(sa) * bf16t(shared_down @ sh))
        out[:, t] = acc
    d = (out - s3_moe).abs()
    print('self-contained MoE diff max %.5f (rel %.4f), engine absmax %.4f' %
          (d.max().item(), d.max().item() / s3_moe.abs().max().item(), s3_moe.abs().max().item()))
    print('per-token max:', [round(d[:, t].max().item(), 4) for t in range(18)])
    for t in range(6):
        print('token %d corr %.4f' % (t, float(torch.corrcoef(torch.stack([out[:, t], s3_moe[:, t]]))[0, 1])))


if __name__ == '__main__':
    main()
