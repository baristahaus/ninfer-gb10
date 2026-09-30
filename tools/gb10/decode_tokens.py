#!/usr/bin/env python3
"""Decode NInfer real-test token prefixes to text with the Flash-Next BPE tokenizer.

Usage: decode_tokens.py LABEL=TOKEN TOKEN [TOKEN ...]  (one label per token list)
Reads the tokenizer from the local HF cache (NFS); CPU-only.
"""
import sys

SNAP = "/nfs/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540"

def main() -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAP)
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    for i, arg in enumerate(args):
        if "=" not in arg:
            ids = [int(t) for t in arg.split()]
            label = f"case{i}"
        else:
            label, rest = arg.split("=", 1)
            ids = [int(t) for t in rest.replace("=", " ").split()]
        print(f"{label}: {ids}\n  {tok.decode(ids)!r}\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
