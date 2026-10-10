#!/usr/bin/env python3
"""Batch invariance, one round: the first prompt of each of the prose/code/
json/math classes, replayed as one C4 batch (4 concurrent requests, 256
tokens, greedy, thinking off).

One round per invocation. batch_invariance.sh runs two rounds with a fresh
server and dropped caches between them and compares the outputs exactly.
"""
import concurrent.futures
import importlib.util
import json
import os
import sys
import time
import urllib.request

SERVE_LOAD = "/home/apollo11/dgpp/scripts/serve_load.py"
CLASSES_ORDER = ("prose", "code", "json", "math")


def load_classes():
    # serve_load imports bench_stream from its own directory; make that
    # resolvable when it is imported as a module instead of run as a script.
    spec_dir = os.path.dirname(SERVE_LOAD)
    if spec_dir not in sys.path:
        sys.path.insert(0, spec_dir)
    spec = importlib.util.spec_from_file_location("serve_load", SERVE_LOAD)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # safe: main() is behind the __main__ guard
    return m.CLASSES


def one(host, port, name, prompt):
    body = json.dumps({
        "model": "qwen3.8-flash-next-125b-a6b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0,
        # Thinking off, as the parity protocol: lk's Flash-Next template rejects
        # reasoning_effort whenever thinking is disabled ("reasoning effort cannot
        # be combined with disabled thinking"), so the template kwarg is the knob.
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.load(resp)
    ch = data["choices"][0]
    return {
        "class": name,
        "finish": ch["finish_reason"],
        "chars": len(ch["message"]["content"]),
        "wall_s": round(time.time() - t0, 2),
        "text": ch["message"]["content"],
    }


def main():
    host, port, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    classes = load_classes()
    reqs = [(n, classes[n][0]) for n in CLASSES_ORDER]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(one, host, port, n, p) for n, p in reqs]
        results = [f.result() for f in futs]
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    for r in results:
        print(f"{r['class']}: {r['chars']} chars, {r['wall_s']}s, finish={r['finish']}")


if __name__ == "__main__":
    main()
