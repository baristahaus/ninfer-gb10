#!/usr/bin/env python3
"""W2.1 self-trace sampler: 250x2048 English sysadmin/agent rows in sc_trace format.

Seeds come from real box material (TEB tool-call transcripts, bash history, ops
dialogs); continuations are sampled from the resident pool at the deployment
profile. Rows are tokenized with the Qwen3.8 tokenizer and packed as
cal_trace.safetensors {"tokens": int32 [250, 2048]} plus a manifest.
"""

import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path("/home/barista/zmodern/ninfer-gb10")
OUT = REPO / "profiles" / "cal-trace"
ROWS, ROW_TOKENS = 250, 2048
POOL = "http://172.30.30.16:8000"
TOKENIZER_PATH = str(sorted(Path("/home/barista/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots").glob("*/"))[0])

TEB_REPORTS = sorted(Path("/tmp/runs/2026/11").glob("*.md"))


def bash_history_seeds(count):
    lines = [l.rstrip() for l in open(Path.home() / ".bash_history", encoding="utf-8", errors="replace") if l.strip() and not l.startswith("#")]
    rng = random.Random(41)
    seeds = []
    tasks = [
        "Continue this terminal session. The user is debugging on their Arch workstation:\n",
        "You are an ops agent assisting on this box. Continue the working session:\n",
        "Continue the sysadmin session transcript:\n",
    ]
    for i in range(count):
        start = rng.randrange(0, max(1, len(lines) - 12))
        chunk = "\n".join(lines[start:start + 10])
        seeds.append(rng.choice(tasks) + f"```\n{chunk}\n```\nContinue with the next logical commands and explain what you observe.")
    return seeds


def teb_seeds(count):
    seeds = []
    for report in TEB_REPORTS:
        text = report.read_text(encoding="utf-8", errors="replace")
        # Strip the score header; keep the transcript body as the session context.
        body = re.sub(r"^#.*?\n(- .*\n)*", "", text, count=1).strip()
        if len(body) > 200:
            seeds.append("Here is a transcript of a tool-calling agent session on a GPU inference workstation.\n\n"
                         + body[:4000] + "\n\nContinue the session: issue the next tool calls and narrate the reasoning.")
    rng = random.Random(97)
    out = []
    while len(out) < count and seeds:
        out.append(seeds[rng.randrange(len(seeds))])
    return out


def ops_dialog_seeds(count):
    topics = [
        ("a CUDA build failure", "cmake --build build -j exits with 'error: parameter packs not nested deep enough' in a templated kernel header"),
        ("a GPU OOM at model load", "the engine fails planning with 'Device budget cannot hold model weights and required execution resources' on a 32 GiB card"),
        ("a hung HTTP endpoint", "curl to the inference server's /health returns after 30s while the GPU shows 100% utilization"),
        ("disk tiering misbehavior", "the tiered model load spills expert weights to host RAM even though VRAM has 6 GiB free"),
        ("Prometheus metrics drift", "the /metrics scrape shows request counters advancing but zero histogram observations for prefill time"),
        ("a kernel regression", "decode throughput dropped from 460 to 41 tok/s after a router change; nsys shows host occupancy 57%"),
        ("CUDA graph capture", "graph replay swallows per-round NVTX ranges, so the attribution tool finds no ranges in the trace"),
        ("a stuck systemd service", "the pool service was killed by a wrapper timeout and now nothing holds the GPU allocation"),
        ("rsync to a remote node", "shipping a 54 GiB checkpoint over a 10GbE link to the GB10 lane machine; rsync stalls at 2 GiB"),
        ("kernel module mismatch", "nvidia-smi reports 'API mismatch' after a partial driver upgrade on Arch"),
        ("OOM killer investigation", "dmesg shows the OOM killer reaping the serve process during a 125B tiered load"),
        ("venv dependency skew", "the installed exllamav3 resolves to 1.5.1 while the pipeline pins 1.6.1; pip check fails"),
        ("NVLink/topology debug", "nvidia-smi topo -m shows unexpected PIX paths between the two GPUs; verify the firmware"),
        ("log rotation", "journald fills /var at 4 GB/day from a noisy inference server; design a rotation policy"),
        ("zombie processes", "pgrep shows stale ninfer-serve processes after an aborted bench driver; they hold shm segments"),
    ]
    rng = random.Random(2027)
    out = []
    for i in range(count):
        topic, detail = topics[i % len(topics)]
        out.append(
            f"You are an expert Linux/CUDA ops agent working interactively on an Arch workstation "
            f"with an RTX 5090 (32 GiB VRAM, 123 GiB host RAM, CUDA 13.4). The current problem: "
            f"{topic} — {detail}.\n\nWork the problem: state the diagnostic commands you run, show "
            f"their realistic output, interpret it, and continue until you either fix the issue or "
            f"have a concrete next hypothesis. Be terse and technical."
        )
    return out


def build_seeds():
    seeds = []
    seeds += bash_history_seeds(70)
    seeds += teb_seeds(80)
    seeds += ops_dialog_seeds(100)
    rng = random.Random(11)
    rng.shuffle(seeds)
    return seeds[:ROWS]


def chat(messages):
    body = json.dumps({
        "model": "qwen3.8-27b",
        "messages": messages,
        "max_tokens": 700,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.0,
        "frequency_penalty": 0.0,
    }).encode()
    req = urllib.request.Request(POOL + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        message = json.load(resp)["choices"][0]["message"]
    # Thinking mode: the reasoning text is genuine on-domain model output; with a
    # bounded max_tokens the visible content can be empty while thinking continues.
    return ((message.get("reasoning_content") or "") + "\n" +
            (message.get("content") or "")).strip()


def sample_continuation(seed_text, tokenizer):
    """Multi-turn continuation until 2048 tokens of real content accumulate.

    A single chat turn ends at <|im_end|>; with ignore_eos the serve would just loop
    the eos token. Instead the assistant's text is appended and the model is asked
    to continue, so every token in the row is genuine domain content.
    """

    messages = [{"role": "user", "content": seed_text}]
    texts = []
    token_count = 0
    for turn in range(8):
        reply = chat(messages)
        if not reply or reply.strip() == "":
            break
        texts.append(reply)
        token_count += len(tokenizer.encode(reply))
        if token_count >= ROW_TOKENS:
            break
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": "Continue exactly where you left off."})
    return "\n".join(texts)


def main():
    from transformers import AutoTokenizer
    import safetensors.torch as st
    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    seeds = build_seeds()
    print(f"{len(seeds)} seeds", flush=True)

    manifest = {"rows": ROWS, "row_tokens": ROW_TOKENS, "profile": "deployment-default",
                "source": "W2.1 self-trace", "seeds": [None] * len(seeds)}
    rows: dict[int, list] = {}
    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor
    import hashlib
    def work(i):
        text = seeds[i] + "\n\n" + sample_continuation(seeds[i], tokenizer)
        ids = tokenizer.encode(text)[:ROW_TOKENS]
        if len(ids) < ROW_TOKENS:
            ids = ids + [tokenizer.eos_token_id] * (ROW_TOKENS - len(ids))
        manifest["seeds"][i] = {"row": i, "chars": len(seeds[i]),
                                "seed_sha256": hashlib.sha256(seeds[i].encode()).hexdigest()}
        rows[i] = ids
        done = len(rows)
        if done % 10 == 0 or done == len(seeds):
            el = time.time() - t0
            print(f"row {done}/{len(seeds)} elapsed {el:.0f}s eta {el/done*(len(seeds)-done):.0f}s", flush=True)
        if done % 25 == 0:
            ordered = [rows[i] for i in sorted(rows)]
            st.save_file({"input_ids": torch.tensor(ordered, dtype=torch.long),
                          "lengths": torch.tensor([ROW_TOKENS] * len(ordered), dtype=torch.long)},
                         str(OUT / "cal_trace.partial.safetensors"))
            json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(work, range(len(seeds))))

    packed = torch.tensor([rows[i] for i in sorted(rows)], dtype=torch.long)
    lengths = torch.tensor([ROW_TOKENS] * len(rows), dtype=torch.long)
    st.save_file({"input_ids": packed, "lengths": lengths},
                 str(OUT / "cal_trace.safetensors"))
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)
    print(f"done: {OUT / 'cal_trace.safetensors'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
