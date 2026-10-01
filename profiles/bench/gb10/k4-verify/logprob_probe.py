#!/usr/bin/env python3
"""Logprobs probe: cold first 256-token request with top-2 logprobs per position.

Usage: logprob_probe.py <port> <out.json>
Captures the message text (reasoning + content, discard_load.py convention) plus the
per-position logprob and top-2 logprobs, so the output can be compared byte-for-byte
with the plain runs and the divergence point can be inspected.
"""
import hashlib
import http.client
import json
import sys

PORT = int(sys.argv[1])
OUT = sys.argv[2]
PROMPT = ("Explain how a steam engine converts pressure into rotation, naming each "
          "part in order as the cycle runs.")

c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=600)
c.request("GET", "/v1/models")
model = json.loads(c.getresponse().read())["data"][0]["id"]
body = {"model": model, "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 256, "stream": False, "temperature": 0,
        "logprobs": True, "top_logprobs": 2}
c.request("POST", "/v1/chat/completions", json.dumps(body))
resp = c.getresponse()
data = json.loads(resp.read())
if "choices" not in data:
    print("ERROR:", json.dumps(data)[:400])
    sys.exit(1)
choice = data["choices"][0]
msg = choice["message"]
reasoning = msg.get("reasoning_content") or ""
content = msg.get("content") or ""
text = reasoning + "\x00" + content
lp = (choice.get("logprobs") or {}).get("content") or []
out = {"finish": choice.get("finish_reason"),
       "tokens": data.get("usage", {}).get("completion_tokens"),
       "text_len": len(text),
       "text_sha": hashlib.sha256(text.encode()).hexdigest()[:16],
       "prompt_sha": hashlib.sha256(PROMPT.encode()).hexdigest()[:16],
       "n_positions": len(lp),
       "logprobs": [{"tok": e.get("token"), "logprob": e.get("logprob"),
                     "top2": [{"tok": t.get("token"), "logprob": t.get("logprob")}
                              for t in (e.get("top_logprobs") or [])]}
                    for e in lp]}
with open(OUT, "w") as f:
    json.dump(out, f)
print(f"positions={len(lp)} finish={out['finish']} tokens={out['tokens']} "
      f"len={out['text_len']} sha={out['text_sha']}")
