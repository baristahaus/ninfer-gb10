#!/usr/bin/env python3
"""Admission-stall repro against ninfer-serve (C-DECODE setup).

Phase 1: one request (corpus stream 0). Phase 2: once it finishes, 8 requests at once,
with stream 0 (the response-replay prompt) placed second so the replay is not admitted
first. Records per-request timing for correlation with the server's throughput lines.

Usage: pr36fix_c4_stall.py BASE_URL OUT_JSON
"""
import json
import sys
import threading
import time
import urllib.request

ROOT = "/home/apollo11/ninfer-gb10/eval/corpora/perplexity-1m"
CONTINUE = "\n\nContinue the text above. Write only the continuation."
MAX_TOKENS = 512


def post(base_url, model, text):
    payload = {
        "model": model,
        "stream": False,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "reasoning_effort": "none",
        "messages": [{"role": "user", "content": text}],
    }
    req = urllib.request.Request(base_url + "/v1/chat/completions",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        body = json.load(resp)
    return {"seconds": time.time() - t0, "t0": t0, "t1": time.time(),
            "completion_tokens": body["usage"]["completion_tokens"],
            "finish_reason": body["choices"][0]["finish_reason"]}


def main():
    base_url, out = sys.argv[1], sys.argv[2]
    with urllib.request.urlopen(base_url + "/v1/models", timeout=60) as resp:
        model = json.load(resp)["data"][0]["id"]
    manifest = json.load(open(ROOT + "/manifest.json"))
    streams = manifest["streams"][:8]

    def prompt(i):
        with open(f"{ROOT}/{streams[i]['path']}", encoding="utf-8") as f:
            return f.read()[:8000] + CONTINUE

    log = {"phase1": None, "phase2_start_unix_ms": None, "phase2": None}

    print(f"[{time.strftime('%FT%TZ', time.gmtime())}] phase 1: single request "
          f"({streams[0]['id']})", flush=True)
    log["phase1"] = {"stream": streams[0]["id"], **post(base_url, model, prompt(0))}
    log["phase1"]["t1_unix_ms"] = int(log["phase1"]["t1"] * 1000)
    print(f"[{time.strftime('%FT%TZ', time.gmtime())}] phase 1 done "
          f"({log['phase1']['completion_tokens']} tokens, "
          f"{log['phase1']['seconds']:.1f} s); firing 8 at once (replay second)", flush=True)

    log["phase2_start_unix_ms"] = int(time.time() * 1000)
    order = [1, 0, 2, 3, 4, 5, 6, 7]  # stream 0 (the replay) placed second
    results = [None] * 8

    def worker(slot):
        i = order[slot]
        results[slot] = {"stream": streams[i]["id"], **post(base_url, model, prompt(i))}

    threads = [threading.Thread(target=worker, args=(s,)) for s in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log["phase2"] = results
    for r in results:
        print(f"[{time.strftime('%FT%TZ', time.gmtime())}] {r['stream']}: "
              f"{r['completion_tokens']} tokens in {r['seconds']:.1f} s "
              f"({r['finish_reason']})", flush=True)
    with open(out, "w") as f:
        json.dump(log, f, indent=1)
    print(f"results: {out}", flush=True)


if __name__ == "__main__":
    main()
