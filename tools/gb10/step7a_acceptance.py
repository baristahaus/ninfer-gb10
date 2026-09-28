#!/usr/bin/env python3
"""Step 7a MTP acceptance baseline.

Sends one greedy continuation request per perplexity corpus stream to the
serving endpoint, then reads the server request log for the speculative
acceptance counters. Writes acceptance.json next to the request log.
"""
import json
import sys
import time
import urllib.request

MANIFEST = "eval/corpora/perplexity-1m/manifest.json"
PROMPT_CHARS = 8000
MAX_TOKENS = 1024
CONTINUE = "\n\nContinue the text above. Write only the continuation."


def get(base_url, path):
    with urllib.request.urlopen(base_url + path, timeout=60) as resp:
        return json.load(resp)


def post(base_url, payload, timeout=1800):
    req = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def main():
    base_url, req_jsonl, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    model = get(base_url, "/v1/models")["data"][0]["id"]
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    records = []
    for stream in manifest["streams"]:
        with open(f"eval/corpora/perplexity-1m/{stream['path']}", encoding="utf-8") as f:
            text = f.read()
        payload = {
            "model": model,
            "stream": False,
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "user", "content": text[:PROMPT_CHARS] + CONTINUE}
            ],
        }
        t0 = time.time()
        resp = post(base_url, payload)
        record = {
            "stream": stream["id"],
            "domain": stream["domain"],
            "seconds": round(time.time() - t0, 2),
            "generated_tokens": resp["usage"]["completion_tokens"],
        }
        records.append(record)
        print(f"{stream['id']}: {record['generated_tokens']} tokens in "
              f"{record['seconds']}s", flush=True)

    time.sleep(3)  # let the server flush the request log
    rounds = drafted = accepted = fallback = 0
    per_position = {}
    n = 0
    with open(req_jsonl, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            spec = rec.get("speculative")
            if not isinstance(spec, dict):
                continue
            n += 1
            rounds += spec.get("rounds", 0)
            drafted += spec.get("drafted_tokens", 0)
            accepted += spec.get("accepted_tokens", 0)
            fallback += spec.get("fallback_steps", 0)
            for pos, count in enumerate(spec.get("accepted_per_position", [])):
                per_position[pos] = per_position.get(pos, 0) + count
    summary = {
        "requests": n,
        "rounds": rounds,
        "drafted_tokens": drafted,
        "accepted_tokens": accepted,
        "acceptance_ratio": round(accepted / drafted, 6) if drafted else None,
        "fallback_steps": fallback,
        # Per-round acceptance at draft position p (pos 0 is attempted every
        # round; pos 1 only when pos 0 is accepted, so pos 1 is a lower bound
        # on its per-attempt rate).
        "accepted_per_position_per_round": {
            str(k): round(v / rounds, 6) for k, v in sorted(per_position.items())
        } if rounds else {},
        "streams": records,
    }
    with open(f"{out_dir}/acceptance.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"requests={n} rounds={rounds} drafted={drafted} accepted={accepted} "
          f"acceptance={summary['acceptance_ratio']} "
          f"per-position-per-round={summary['accepted_per_position_per_round']}",
          flush=True)


if __name__ == "__main__":
    main()
