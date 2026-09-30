#!/usr/bin/env python3
"""Aggregate and per-request throughput at N concurrent requests against a running ninfer-serve.

Usage: concurrency_sweep.py BASE_URL OUT_JSON --n 1,2,4,8 [--max-tokens 512] [--prompt-chars 8000]
                            [--shared]

For each N, fires N greedy chat requests at once (one thread each) and waits for all of them.
Prompts are the first PROMPT_CHARS characters of distinct perplexity-corpus streams, so no two
requests share a prefix; --shared sends the same stream N times instead (the prefix-reuse
regime). Thinking is disabled so every request decodes a comparable continuation.

Per N it records each request's wall seconds and completion tokens, the batch wall time from the
first send to the last response, and aggregate tok/s (sum of completion tokens over the batch
wall). Decode-only rates and per-round timings come from the server's request log
(request_log_summary.py).

Per-request HTTP errors are recorded in the result JSON instead of aborting the batch;
the sweep exits nonzero only when every request in a batch failed.
"""
import argparse
import json
import threading
import time
import urllib.request

MANIFEST = "eval/corpora/perplexity-1m/manifest.json"
CONTINUE = "\n\nContinue the text above. Write only the continuation."


def post(base_url, payload):
    req = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.load(resp)


def prompts(count, chars, shared):
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    streams = manifest["streams"]
    if not shared and count > len(streams):
        raise SystemExit(f"only {len(streams)} distinct corpus streams")
    out = []
    for i in range(count):
        stream = streams[0 if shared else i]
        with open(f"eval/corpora/perplexity-1m/{stream['path']}", encoding="utf-8") as f:
            out.append((stream["id"], f.read()[:chars] + CONTINUE))
    return out


def run_batch(base_url, model, batch, max_tokens):
    results = [None] * len(batch)
    start = threading.Barrier(len(batch))

    def worker(i, stream_id, text):
        payload = {
            "model": model,
            "stream": False,
            "temperature": 0,
            "max_tokens": max_tokens,
            "reasoning_effort": "none",
            "messages": [{"role": "user", "content": text}],
        }
        start.wait()
        t0 = time.time()
        try:
            resp = post(base_url, payload)
        except Exception as error:
            t1 = time.time()
            results[i] = {"stream": stream_id, "start": t0, "end": t1, "seconds": t1 - t0,
                          "error": str(error)}
            return
        t1 = time.time()
        results[i] = {"stream": stream_id, "start": t0, "end": t1, "seconds": t1 - t0,
                      "completion_tokens": resp["usage"]["completion_tokens"],
                      "finish_reason": resp["choices"][0]["finish_reason"]}

    threads = [threading.Thread(target=worker, args=(i, sid, text))
               for i, (sid, text) in enumerate(batch)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ok = [r for r in results if "error" not in r]
    if not ok:
        first = next(r["error"] for r in results if "error" in r)
        raise SystemExit(f"all {len(batch)} requests failed; first error: {first}")
    wall = max(r["end"] for r in ok) - min(r["start"] for r in results)
    tokens = sum(r["completion_tokens"] for r in ok)
    return {"n": len(batch), "failed": len(batch) - len(ok), "wall_seconds": wall,
            "completion_tokens": tokens,
            "aggregate_tok_s": tokens / wall,
            "per_request_tok_s": [r["completion_tokens"] / r["seconds"] for r in ok],
            "requests": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("out_json")
    ap.add_argument("--n", default="1,2,4,8")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--prompt-chars", type=int, default=8000)
    ap.add_argument("--shared", action="store_true")
    args = ap.parse_args()
    with urllib.request.urlopen(args.base_url + "/v1/models", timeout=60) as resp:
        model = json.load(resp)["data"][0]["id"]
    runs = []
    for n in (int(x) for x in args.n.split(",")):
        batch = prompts(n, args.prompt_chars, args.shared)
        run = run_batch(args.base_url, model, batch, args.max_tokens)
        runs.append(run)
        per = run["per_request_tok_s"]
        print(f"N={n}: aggregate {run['aggregate_tok_s']:.1f} tok/s"
              + (f", {run['failed']} failed" if run["failed"] else "")
              + f", per request min {min(per):.1f} / max {max(per):.1f} tok/s, "
              f"wall {run['wall_seconds']:.1f} s",
              flush=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"shared": args.shared, "max_tokens": args.max_tokens, "runs": runs}, f,
                  indent=1)


if __name__ == "__main__":
    main()
