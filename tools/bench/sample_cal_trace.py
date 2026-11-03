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


def sample_continuation(seed_text):
    body = json.dumps({
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": seed_text}],
        "max_tokens": 2600,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.0,
        "frequency_penalty": 0.0,
        "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(POOL + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


def main():
    from transformers import AutoTokenizer
    import safetensors.torch as st
    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    seeds = build_seeds()
    print(f"{len(seeds)} seeds", flush=True)

    rows = []
    manifest = {"rows": ROWS, "row_tokens": ROW_TOKENS, "profile": "deployment-default",
                "source": "W2.1 self-trace", "seeds": []}
    t0 = time.time()
    for i, seed in enumerate(seeds):
        text = seed + "\n\n" + sample_continuation(seed)
        ids = tokenizer.encode(text)
        ids = ids[:ROW_TOKENS]
        if len(ids) < ROW_TOKENS:
            ids = ids + [tokenizer.eos_token_id] * (ROW_TOKENS - len(ids))
        rows.append(ids)
        import hashlib
        manifest["seeds"].append({"row": i, "chars": len(seed),
                                  "seed_sha256": hashlib.sha256(seed.encode()).hexdigest()})
        if (i + 1) % 10 == 0 or i == len(seeds) - 1:
            el = time.time() - t0
            print(f"row {i+1}/{len(seeds)} elapsed {el:.0f}s eta {el/(i+1)*(len(seeds)-i-1):.0f}s", flush=True)
        if (i + 1) % 25 == 0:  # incremental save so the run is resumable
            st.save_file({"input_ids": torch.tensor(rows, dtype=torch.long),
                          "lengths": torch.tensor([ROW_TOKENS] * len(rows), dtype=torch.long)},
                         str(OUT / "cal_trace.partial.safetensors"))
            json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)

    packed = torch.tensor(rows, dtype=torch.long)
    lengths = torch.tensor([ROW_TOKENS] * len(rows), dtype=torch.long)
    st.save_file({"input_ids": packed, "lengths": lengths},
                 str(OUT / "cal_trace.safetensors"))
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)
    print(f"done: {OUT / 'cal_trace.safetensors'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
