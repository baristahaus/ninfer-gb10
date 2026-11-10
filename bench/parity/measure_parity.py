#!/usr/bin/env python3
"""Cross-engine parity measurement: one box, one artifact, one flag set, A/B/B/A.

Runs one engine (either binary), serves the same `.ninfer` artifact, drives the same fixed
workload, and records the phases that are comparable between the two engines. The point is to make
the comparison a measurement instead of an argument.

  measure_parity.py --engine ninfer --bin build/rtx5090-release/apps/ninfer-serve \
      --artifact <model.ninfer> --out ninfer.json
  measure_parity.py --engine capy --bin <capy-serve> --artifact <model.ninfer> \
      --artifact-first --out capy.json
  measure_parity.py --compare ninfer.json capy.json

Design notes that make the numbers comparable:
  - the same artifact bytes for both engines (the shared v3 layouts load on both);
  - greedy sampling, `--no-thinking`, and a fixed output length, so the token count and the
    thinking-channel behaviour are not variables;
  - one warmup request per prompt, discarded, then three measured waves;
  - prompts are fixed and the same across engines;
  - the decode rate is computed from the server's own streamed token timings, not wall clock, so
    client-side scheduling noise does not enter it;
  - a request that generates nothing fails the run instead of averaging into the result.

Read `bench/parity/README.md` for the protocol, the limits and the recorded campaign.
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

PROMPTS = [
    "Explain in one paragraph why a mixture-of-experts model can be faster than a dense model of the same parameter count.",
    "Write a short function that reverses a singly linked list in place, and explain the pointer updates.",
    "Summarise the tradeoffs between storing KV cache in device memory, host memory and on disk.",
    "A server has 32 GB of VRAM and must serve a 20 GB model. Describe three ways to make it fit and what each costs.",
    "Give a concise explanation of what a CUDA graph captures and when capturing one is a bad idea.",
]


def post_stream(base, prompt, max_tokens):
    """Stream one completion; return (first_token_s, decode_tokens, decode_s)."""
    body = json.dumps({
        "model": "parity",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }).encode()
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first = None
    tokens = 0
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            # Count a token whichever channel it arrives in: a thinking model streams
            # reasoning_content first and content only after, and counting only `content`
            # silently reports zero tokens for a request that generated hundreds.
            piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
            if piece:
                if first is None:
                    first = time.perf_counter() - started
                tokens += 1
    ended = time.perf_counter()
    decode_s = (ended - started) - (first or 0.0)
    return first, tokens, decode_s


def wait_health(base, timeout_s=1800):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as response:
                if json.loads(response.read()).get("status") == "ok":
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(5)
    return False


def measure(args):
    base = f"http://127.0.0.1:{args.port}"
    command = [args.bin, "--host", "127.0.0.1", "--port", str(args.port),
               "--model-id", "parity", "--max-context", "8192",
               "--kv-capacity", "8192", "--max-concurrency", "1", "--no-thinking"]
    if args.artifact_first:
        command.insert(1, args.artifact)
    if args.extra:
        command += args.extra.split()
    if not args.artifact_first:
        command.append(args.artifact)

    print(f"[{args.engine}] starting: {' '.join(command)}", flush=True)
    server = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_health(base):
            print(f"[{args.engine}] server never became healthy", flush=True)
            return 1
        print(f"[{args.engine}] healthy", flush=True)

        # Warmup: one request per prompt, discarded.
        for prompt in PROMPTS:
            post_stream(base, prompt, args.warmup_tokens)
        print(f"[{args.engine}] warmup done", flush=True)

        waves = []
        for wave in range(args.waves):
            rows = []
            for prompt in PROMPTS:
                first, tokens, decode_s = post_stream(base, prompt, args.max_tokens)
                rate = tokens / decode_s if decode_s > 0 else 0.0
                rows.append({"ttft_s": first, "tokens": tokens, "decode_s": decode_s,
                             "decode_tok_s": rate})
            waves.append(rows)
            print(f"[{args.engine}] wave {wave + 1}/{args.waves} done", flush=True)

        all_rows = [row for wave in waves for row in wave]
        ttfts = [row["ttft_s"] for row in all_rows if row["ttft_s"] is not None]
        rates = [row["decode_tok_s"] for row in all_rows if row["decode_tok_s"] > 0]
        empty = sum(1 for row in all_rows if row["tokens"] == 0)
        summary = {
            "engine": args.engine,
            "artifact": args.artifact,
            "protocol": {
                "prompts": len(PROMPTS),
                "waves": args.waves,
                "max_tokens": args.max_tokens,
                "warmup_tokens": args.warmup_tokens,
                "greedy": True,
                "thinking": False,
            },
            "empty_responses": empty,
            "requests": len(all_rows),
            "ttft_s_median": statistics.median(ttfts) if ttfts else None,
            "ttft_s_mean": statistics.fmean(ttfts) if ttfts else None,
            "decode_tok_s_median": statistics.median(rates) if rates else None,
            "decode_tok_s_mean": statistics.fmean(rates) if rates else None,
            "decode_tok_s_stdev": statistics.stdev(rates) if len(rates) > 1 else None,
            "decode_tok_s_min": min(rates) if rates else None,
            "decode_tok_s_max": max(rates) if rates else None,
            "tokens_total": sum(row["tokens"] for row in all_rows),
            "waves": waves,
        }
        with open(args.out, "w") as handle:
            json.dump(summary, handle, indent=2)
        if empty:
            print(f"[{args.engine}] WARNING: {empty} of {len(all_rows)} requests generated nothing; "
                  f"the two engines are not comparable unless both generate output", flush=True)
        print(f"[{args.engine}] median decode {summary['decode_tok_s_median']:.2f} tok/s, "
              f"median TTFT {summary['ttft_s_median'] * 1000:.1f} ms over "
              f"{summary['requests']} requests", flush=True)
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()


def compare(paths):
    runs = [json.load(open(path)) for path in paths]
    for run in runs:
        if run["empty_responses"]:
            print(f"{run['engine']}: {run['empty_responses']} empty responses; "
                  f"the runs are not comparable", file=sys.stderr)
            return 1
    left, right = runs
    delta = right["decode_tok_s_median"] - left["decode_tok_s_median"]
    percent = delta / left["decode_tok_s_median"] * 100.0
    overlap = (left["decode_tok_s_max"] < right["decode_tok_s_min"]
               or right["decode_tok_s_max"] < left["decode_tok_s_min"])
    print(f"{'':<22}{left['engine']:>16}{right['engine']:>16}{'delta':>14}")
    for key, label, scale in (
        ("decode_tok_s_median", "decode median", 1.0),
        ("decode_tok_s_stdev", "decode stdev", 1.0),
        ("decode_tok_s_min", "decode min", 1.0),
        ("decode_tok_s_max", "decode max", 1.0),
        ("ttft_s_median", "TTFT median (ms)", 1000.0),
        ("tokens_total", "tokens total", 1.0),
    ):
        a, b = left.get(key), right.get(key)
        if a is None or b is None:
            continue
        print(f"{label:<22}{a * scale:>16.3f}{b * scale:>16.3f}{(b - a) * scale:>+14.3f}")
    print(f"\ndecode delta: {percent:+.2f} %; ranges "
          f"{'do not overlap' if overlap else 'overlap'}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", help="label for this run, for example ninfer or capy")
    parser.add_argument("--bin", help="the server binary")
    parser.add_argument("--artifact", help="the .ninfer artifact, the same file for both engines")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--out", help="write the run summary here as JSON")
    parser.add_argument("--extra", default="", help="extra server flags, space separated")
    parser.add_argument("--artifact-first", action="store_true",
                        help="the model is the first positional (capy), not the last (ninfer)")
    parser.add_argument("--waves", type=int, default=3, help="measured waves, default 3")
    parser.add_argument("--max-tokens", type=int, default=256, help="generated tokens, default 256")
    parser.add_argument("--warmup-tokens", type=int, default=32,
                        help="warmup generation length, default 32")
    parser.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                        help="compare two run summaries instead of measuring")
    args = parser.parse_args()
    if args.compare:
        return compare(args.compare)
    for name in ("engine", "bin", "artifact", "out"):
        if not getattr(args, name):
            parser.error(f"--{name} is required to measure")
    return measure(args)


if __name__ == "__main__":
    sys.exit(main())
