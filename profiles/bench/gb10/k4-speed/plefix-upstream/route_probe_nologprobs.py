#!/usr/bin/env python3
"""C1/C4 identity probe for the upstream serve, which has no token-logprobs
surface (the fork's --token-logprobs flag and request-body logprobs are absent
from upstream master). Same 73-token prompt, greedy, 256 tokens as
route_probe.py; identity is judged on the reasoning+content text.
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
        "max_tokens": 256, "stream": False, "temperature": 0}
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
usage = data.get("usage", {})
out = {"finish": choice.get("finish_reason"),
       "completion_tokens": usage.get("completion_tokens"),
       "prompt_tokens": usage.get("prompt_tokens"),
       "text_len": len(text),
       "text_sha": hashlib.sha256(text.encode()).hexdigest()[:16],
       "prompt_sha": hashlib.sha256(PROMPT.encode()).hexdigest()[:16]}
with open(OUT, "w") as f:
    json.dump(out, f)
print(f"prompt_tokens={out['prompt_tokens']} completion={out['completion_tokens']} "
      f"finish={out['finish']} len={out['text_len']} sha={out['text_sha']}")
