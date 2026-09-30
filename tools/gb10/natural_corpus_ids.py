#!/usr/bin/env python3
"""Write a non-repeating ninfer_bench corpus from the perplexity corpus.

Usage: natural_corpus_ids.py TOKENIZER OUT_IDS [--tokens 262144]

`bench/fixtures/qwen3_8_flash_next_context.ids` tiles one 3,442-token text, so every prompt
longer than that ends inside a repetition and MTP drafts the continuation almost perfectly
(accepted length 3.96 of 4 at K=3 in the 7d gate). This corpus concatenates the perplexity
corpus streams in manifest order, each tokenized once, so decode continues natural text.

TOKENIZER is a Flash-Next checkpoint or tokenizer directory (or a tokenizer.json). Uses the
`tokenizers` package when present, otherwise `transformers`. Writes OUT_IDS (whitespace-separated
ids, the fixture format) and OUT_IDS with .manifest.json beside it.
"""
import argparse
import hashlib
import json
import pathlib

MANIFEST = "eval/corpora/perplexity-1m/manifest.json"


def encoder(path):
    p = pathlib.Path(path)
    tokenizer_json = p if p.suffix == ".json" else p / "tokenizer.json"
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(tokenizer_json))
        return lambda text: tok.encode(text, add_special_tokens=False).ids
    except ImportError:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(p if p.is_dir() else p.parent),
                                            local_files_only=True)
        return lambda text: tok.encode(text, add_special_tokens=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer")
    ap.add_argument("out_ids")
    ap.add_argument("--tokens", type=int, default=262144)
    args = ap.parse_args()
    encode = encoder(args.tokenizer)
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    ids, used = [], []
    for stream in manifest["streams"]:
        with open(f"eval/corpora/perplexity-1m/{stream['path']}", encoding="utf-8") as f:
            ids.extend(encode(f.read()))
        used.append(stream["id"])
        if len(ids) >= args.tokens:
            break
    if len(ids) < args.tokens:
        raise SystemExit(f"corpus holds only {len(ids)} tokens")
    ids = ids[: args.tokens]
    text = " ".join(map(str, ids))
    out = pathlib.Path(args.out_ids)
    out.write_text(text + "\n")
    out.with_suffix(".manifest.json").write_text(json.dumps({
        "artifact_type": "ninfer_bench_corpus",
        "schema_version": 1,
        "tokenizer_source": "local_hf",
        "add_special_tokens": False,
        "chat_template": False,
        "tokens": len(ids),
        "token_count": len(ids),
        "ids_sha256": hashlib.sha256((text + "\n").encode()).hexdigest(),
        "source": "perplexity-1m streams, concatenated in manifest order, not tiled",
        "source_files": used,
    }, indent=2) + "\n")
    print(f"wrote {len(ids)} ids from {len(used)} streams to {out}")


if __name__ == "__main__":
    main()
