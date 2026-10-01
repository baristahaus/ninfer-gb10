#!/usr/bin/env python3
"""K4 discard-path check.

Batched short requests (max_tokens / stop strings) at C4 must produce exactly the
same output as the same requests run solo (serial loop). The engine's discard path
(K4b discard_successor_rows) runs when a request leaves the MTP cohort while the
others keep decoding.
"""
import hashlib
import http.client
import json
import sys
import threading

HOST = "127.0.0.1"
PORT = int(sys.argv[1])
OUT = sys.argv[2]

# Distinct prompts from the project fabric corpus (prose + code) so no two
# requests share a prefix-cache path.
P0 = ("Write a short paragraph about the first lighthouse built on a granite "
      "headland, and why its keepers kept the log in triplicate.")
P1 = "Reply with exactly: STOPWORD"
P2 = ("Write a Python function that merges two sorted lists into one sorted list "
      "without using the built-in sorted function.")
P3 = ("Explain how a steam engine converts pressure into rotation, naming each "
      "part in order as the cycle runs.")

REQS = [
    {"name": "maxtok10", "prompt": P0, "max_tokens": 10, "stop": None},
    {"name": "stopword", "prompt": P1, "max_tokens": 128, "stop": ["STOPWORD"]},
    {"name": "maxtok16", "prompt": P2, "max_tokens": 16, "stop": None},
    {"name": "survivor", "prompt": P3, "max_tokens": 256, "stop": None},
]


def served_model():
    c = http.client.HTTPConnection(HOST, PORT, timeout=30)
    c.request("GET", "/v1/models")
    data = json.loads(c.getresponse().read())
    return data["data"][0]["id"]


def one(model, r, mode):
    body = {"model": model, "messages": [{"role": "user", "content": r["prompt"]}],
            "max_tokens": r["max_tokens"], "stream": False, "temperature": 0}
    if r["stop"]:
        body["stop"] = r["stop"]
    c = http.client.HTTPConnection(HOST, PORT, timeout=600)
    c.request("POST", "/v1/chat/completions", json.dumps(body))
    resp = c.getresponse()
    data = json.loads(resp.read())
    if "choices" not in data:
        return {"name": r["name"], "mode": mode, "error": json.dumps(data)[:300]}
    choice = data["choices"][0]
    msg = choice["message"]
    reasoning = msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    text = reasoning + "\x00" + content
    return {"name": r["name"], "mode": mode, "text": text,
            "finish": choice.get("finish_reason"),
            "tokens": data.get("usage", {}).get("completion_tokens"),
            "prompt_sha": hashlib.sha256(r["prompt"].encode()).hexdigest()[:16],
            "max_tokens": r["max_tokens"], "stop": r["stop"]}


def run(reqs, model, mode):
    if mode == "batch":
        results = [None] * len(reqs)

        def worker(i):
            results[i] = one(model, reqs[i], mode)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(reqs))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        results = [one(model, r, mode) for r in reqs]
    with open(OUT, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    for r in results:
        if r.get("error"):
            print(f"  {r['name']}: ERROR {r['error']}")
        else:
            print(f"  {r['name']:10s} tokens={r['tokens']} finish={r['finish']} "
                  f"len={len(r['text'])} sha={hashlib.sha256(r['text'].encode()).hexdigest()[:16]}")


def main():
    model = served_model()
    mode = sys.argv[3] if len(sys.argv) > 3 else "batch"
    only = sys.argv[4].split(",") if len(sys.argv) > 4 else None
    reqs = [r for r in REQS if only is None or r["name"] in only]
    run(reqs, model, mode)


if __name__ == "__main__":
    main()
