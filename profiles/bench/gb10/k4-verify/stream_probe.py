#!/usr/bin/env python3
"""Shutdown drain test client.

Starts a streaming chat completion (long), then records what happens when the
serve is signalled externally. Usage:
  stream_probe.py <port> <max_tokens> <out.json>
The test driver sends the signals; this process only reports what the stream
saw, with timestamps relative to its own start.
"""
import http.client
import json
import sys
import time

PORT = int(sys.argv[1])
MAX_TOKENS = int(sys.argv[2])
OUT = sys.argv[3]

PROMPT = ("Explain, in detail, how a hydraulic press converts pump pressure into "
          "tonnage, naming each component in the order the cycle runs it, and give "
          "the pressure ratios of a typical twin-ram design.")

t0 = time.time()
c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=600)
c.request("GET", "/v1/models")
model = json.loads(c.getresponse().read())["data"][0]["id"]
body = {"model": model, "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS, "stream": True, "temperature": 0}
c.request("POST", "/v1/chat/completions", json.dumps(body))
resp = c.getresponse()
result = {"status_line": f"{resp.status} {resp.reason}", "t_first_token": None,
          "t_done": None, "tokens": 0, "finish": None, "error_event": None}
if resp.status != 200:
    result["body"] = resp.read().decode()[:300]
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result))
    sys.exit(0)

buf = ""
while True:
    chunk = resp.read(1024)
    if not chunk:
        break
    buf += chunk.decode(errors="replace")
    while "\n" in buf:
        line, buf = buf.split("\n", 1)
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            result["t_done"] = time.time() - t0
            break
        try:
            ev = json.loads(data)
        except ValueError:
            continue
        if ev.get("error"):
            result["error_event"] = ev["error"]
            result["t_done"] = time.time() - t0
            break
        ch = ev.get("choices") or []
        if ch:
            delta = ch[0].get("delta") or {}
            if delta.get("reasoning_content") or delta.get("content"):
                if result["t_first_token"] is None:
                    result["t_first_token"] = time.time() - t0
                result["tokens"] += 1
            if ch[0].get("finish_reason"):
                result["finish"] = ch[0]["finish_reason"]
if result["t_done"] is None and not result["error_event"]:
    result["error_event"] = "stream closed without [DONE] or finish"
    result["t_done"] = time.time() - t0
result["elapsed"] = time.time() - t0
with open(OUT, "w") as f:
    json.dump(result, f, indent=1)
print(json.dumps(result))
