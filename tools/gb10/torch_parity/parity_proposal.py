"""Draft-head shortlist chain verification vs engine dumps (first proposal call, T=1).

Chain (text_context_impl.h proposal_argmax + ops/launcher/argmax.cu):
  proposal_logits = linear(hidden, proposal_head)          # engine's pr_logits used directly
  per 512-row tile of the shortlist: top-2 (value, shortlist row) -> candidate pair,
  ids remapped through proposal_head_ids (shortlist row -> vocab row)
  exact rescore: score = lm_head fp8 row (vocab row) . hidden (fp32 accumulation,
  code * row_scale), fp8_e4m3fn_row_bf16 exact head
  proposal token = candidate with the max exact score
"""

import numpy as np
import torch

torch.set_default_dtype(torch.float32)
D = '/nfs/mtp-correction/oracle_dumps4/'


def main():
    hidden = torch.from_numpy(np.fromfile(D + 'pr_hidden.bin', dtype=np.uint16).copy()) \
        .view(torch.bfloat16).float()                     # [2560]
    logits = torch.from_numpy(np.fromfile(D + 'pr_logits.bin', dtype=np.uint16).copy()) \
        .view(torch.bfloat16).float()                     # [147456]
    ids = torch.from_numpy(np.fromfile(D + 'pr_ids.bin', dtype=np.int32).copy())   # [147456]
    cand_ids = torch.from_numpy(np.fromfile(D + 'pr_cand_ids.bin', dtype=np.int32).copy())
    cand_scores = torch.from_numpy(np.fromfile(D + 'pr_cand_scores.bin', dtype=np.uint32).copy()) \
        .view(torch.float32)
    out_tok = int(torch.from_numpy(np.fromfile(D + 'pr_out.bin', dtype=np.int32).copy())[0])

    # stage 1: per 512-row tile top-2
    tiles = logits.view(-1, 512)
    topv, topi = torch.topk(tiles, 2, dim=1)
    my_pairs = topi.flatten()
    my_vals = topv.flatten()
    print('candidate id match:', bool((my_pairs == cand_ids).all()),
          'mismatches', int((my_pairs != cand_ids).sum()))
    # tie handling check: compare values
    dv = (my_vals - torch.topk(tiles, 2, dim=1).values.flatten()).abs().max()

    # stage 2: exact rescore with lm_head fp8 rows
    payload = np.fromfile(D + 'fw_lm_head_q.bin', dtype=np.uint8)
    vocab = 248320
    codes = torch.from_numpy(payload[:vocab * 2560].copy()).view(torch.uint8)
    scales = torch.from_numpy(payload[vocab * 2560:].copy()).view(torch.uint16).view(torch.bfloat16).float()
    tab = torch.tensor([0., 0.5, 1, 1.5, 2, 3, 4, 6, -0., -0.5, -1, -1.5, -2, -3, -4, -6])  # placeholder
    sign = torch.where((codes & 0x80) != 0, -1.0, 1.0)
    exp = ((codes >> 3) & 0xF).int()
    man = (codes & 7).float()
    val = torch.where(exp == 0, man * 2.0 ** -9, (1.0 + man * 0.125) * 2.0 ** (exp.float() - 7))
    w = torch.nan_to_num(sign * val, nan=0.0).view(vocab, 2560)

    rows = cand_ids.long()
    sel = w[rows] * scales[rows].view(-1, 1)              # candidate lm_head rows dequantized
    scores = bf16t(sel @ bf16t(hidden))
    print('score max diff vs engine:', (scores - cand_scores).abs().max().item())
    best = int(torch.argmax(scores))
    print('my proposal token', rows[best].item(), 'engine', out_tok,
          'match', rows[best].item() == out_tok)


def bf16t(t):
    return t.to(torch.bfloat16).float()


if __name__ == '__main__':
    main()
