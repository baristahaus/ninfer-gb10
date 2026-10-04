#!/usr/bin/env python3
"""Long-prompt operations workload against an OpenAI-compatible server (NInfer or DGPP).

Usage: long_context.py BASE_URL OUT_JSON [--tokens 15000,30000,60000] [--tasks script,rca]
                       [--reps 2] [--max-tokens 1536] [--no-followup] [--no-interference]
                       [--no-yield] [--short-tokens 2000] [--thinking]

Prompts are synthetic incident bundles (`ops_corpus.py`) sized to the requested prompt tokens,
followed by one of two operations tasks: write test/review/log scripts, or gather evidence for a
root-cause analysis and triage. Every (size, task, rep) uses its own bundle, so first turns never
reuse a prefix. Measured per request, from the client's stream:
- TTFT: send to the first streamed content or reasoning delta (queue, prefill, first round);
- prefill rate: prompt tokens over TTFT, first turns only (a lower bound on the server's rate);
- decode rate: completion tokens after the first, over first delta to last;
- the largest gap between streamed deltas;
- which planted incident facts the answer names (a smoke signal that the context was used).

With follow-ups on, each first turn is followed by a second turn that appends the answer and a
follow-up question: its TTFT measures prefix reuse (the server's request log says how many prompt
tokens it recomputed). With interference on, one request decodes on the smallest size while a
request on the largest size arrives; the decoding stream's delta gaps before and during that
prefill show how long a new long prompt stalls a running one (needs a server with two lanes).
With the yield probe on, a short prompt (--short-tokens, one prefill chunk or less) arrives 3 s into
the largest prompt's prefill: its TTFT shows whether it went ahead of the long prefill, and the long
request's TTFT what that cost it.

Sizing: the first call sends a calibration bundle with max_tokens 1 and reads usage.prompt_tokens
to set characters per token for this tokenizer; sizes are then exact to about 1%. Thinking is off
unless --thinking (both `reasoning_effort` and `chat_template_kwargs.enable_thinking` are sent,
so either engine honors it). Server-side phase times and MTP acceptance come from NInfer's request
log (`request_log_summary.py`).
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request

import ops_corpus

CALIBRATION_SEED = 999_999


def model_id(base_url):
    with urllib.request.urlopen(base_url + "/v1/models", timeout=60) as resp:
        return json.load(resp)["data"][0]["id"]


def payload(model, messages, max_tokens, thinking, stream):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "stream": stream, "chat_template_kwargs": {"enable_thinking": thinking}}
    if not thinking:
        body["reasoning_effort"] = "none"
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


def post_json(base_url, body):
    req = urllib.request.Request(base_url + "/v1/chat/completions",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as resp:
        return json.load(resp)


def stream_chat(base_url, body, on_delta=None):
    """Send one streamed request; return timings, text and usage."""
    req = urllib.request.Request(base_url + "/v1/chat/completions",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json",
                                          "Accept": "text/event-stream"})
    t0 = time.time()
    first = None
    deltas = []
    text = []
    usage = None
    finish = None
    try:
        with urllib.request.urlopen(req, timeout=7200) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                    if piece:
                        now = time.time()
                        first = first or now
                        deltas.append(now)
                        text.append(delta.get("content") or "")
                        if on_delta:
                            on_delta(now)
                    finish = choice.get("finish_reason") or finish
    except Exception as error:  # recorded, not fatal: one failed request must not end a campaign
        return {"error": str(error), "start": t0, "end": time.time()}
    t1 = time.time()
    return {"start": t0, "first": first, "end": t1, "deltas": deltas, "text": "".join(text),
            "usage": usage, "finish": finish}


def summarize(result, cold):
    if "error" in result or result.get("first") is None:
        return {"error": result.get("error", "no streamed delta")}
    usage = result.get("usage") or {}
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens", len(result["deltas"]))
    ttft = result["first"] - result["start"]
    span = result["end"] - result["first"]
    gaps = [b - a for a, b in zip(result["deltas"], result["deltas"][1:])]
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "ttft_s": round(ttft, 3),
        "prefill_tok_s": round(prompt / ttft, 1) if cold and prompt else None,
        "decode_tok_s": round((completion - 1) / span, 2) if span > 0 and completion > 1 else None,
        "max_gap_s": round(max(gaps), 3) if gaps else None,
        "finish": result.get("finish"),
    }


def calibrate(base_url, model, thinking):
    chars = 60_000
    bundle = ops_corpus.build(chars, CALIBRATION_SEED)
    body = payload(model, [{"role": "user", "content": bundle.text + "\n\n" + ops_corpus.TASKS["rca"]}],
                   1, thinking, False)
    resp = post_json(base_url, body)
    tokens = resp["usage"]["prompt_tokens"]
    return len(bundle.text) / tokens, tokens


def interference(base_url, model, small, large, cpt, max_tokens, thinking, seed):
    """Decode a small-prompt request; start a large-prompt request once it streams steadily."""
    small_bundle = ops_corpus.build(int(small * cpt), seed)
    large_bundle = ops_corpus.build(int(large * cpt), seed + 1)
    started = {}
    trigger = threading.Event()

    def on_delta(now):
        started.setdefault("first", now)
        if now - started["first"] > 3.0:
            trigger.set()

    out = {}

    def decoder():
        body = payload(model, [{"role": "user", "content": small_bundle.text + "\n\n" +
                                ops_corpus.TASKS["script"]}], max_tokens, thinking, True)
        out["decoder"] = stream_chat(base_url, body, on_delta)
        trigger.set()

    thread = threading.Thread(target=decoder)
    thread.start()
    trigger.wait()
    arrival = time.time()
    body = payload(model, [{"role": "user", "content": large_bundle.text + "\n\n" +
                            ops_corpus.TASKS["rca"]}], 64, thinking, True)
    out["arrival"] = stream_chat(base_url, body)
    thread.join()

    dec = out["decoder"]
    if "error" in dec or "error" in out["arrival"] or out["arrival"].get("first") is None:
        return {"error": dec.get("error") or out["arrival"].get("error") or "no delta"}
    prefill_end = out["arrival"]["first"]
    before = [b - a for a, b in zip(dec["deltas"], dec["deltas"][1:]) if b <= arrival]
    during = [b - a for a, b in zip(dec["deltas"], dec["deltas"][1:]) if arrival < b <= prefill_end + 0.5]
    after = [b - a for a, b in zip(dec["deltas"], dec["deltas"][1:]) if b > prefill_end + 0.5]
    return {
        "small_tokens": small, "large_tokens": large,
        "arrival_ttft_s": round(prefill_end - arrival, 3),
        "decoder_gap_median_before_s": round(statistics.median(before), 4) if before else None,
        "decoder_gap_max_during_s": round(max(during), 3) if during else None,
        "decoder_stall_s": round(sum(g for g in during if g > 0.25), 3) if during else None,
        "decoder_gap_median_after_s": round(statistics.median(after), 4) if after else None,
        "decoder_finished_before_arrival_prefill": dec["end"] <= prefill_end,
    }


def yield_probe(base_url, model, short, large, cpt, thinking, seed):
    """Start a large-prompt prefill; send a short prompt 3 s later; time both first tokens."""
    large_bundle = ops_corpus.build(int(large * cpt), seed)
    short_bundle = ops_corpus.build(int(short * cpt), seed + 1)
    out = {}

    def long_request():
        body = payload(model, [{"role": "user", "content": large_bundle.text + "\n\n" +
                                ops_corpus.TASKS["rca"]}], 64, thinking, True)
        out["long"] = stream_chat(base_url, body)

    thread = threading.Thread(target=long_request)
    thread.start()
    time.sleep(3.0)
    body = payload(model, [{"role": "user", "content": short_bundle.text + "\n\n" +
                            ops_corpus.TASKS["script"]}], 64, thinking, True)
    out["short"] = stream_chat(base_url, body)
    thread.join()
    if any("error" in out[k] or out[k].get("first") is None for k in ("long", "short")):
        return {"error": out["long"].get("error") or out["short"].get("error") or "no delta"}
    return {
        "short_tokens": short, "large_tokens": large,
        "short_ttft_s": round(out["short"]["first"] - out["short"]["start"], 3),
        "long_ttft_s": round(out["long"]["first"] - out["long"]["start"], 3),
        "short_first_before_long_first": out["short"]["first"] < out["long"]["first"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base_url")
    ap.add_argument("out_json")
    ap.add_argument("--model", default="")
    ap.add_argument("--tokens", default="15000,30000,60000")
    ap.add_argument("--tasks", default="script,rca")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--no-followup", action="store_true")
    ap.add_argument("--no-interference", action="store_true")
    ap.add_argument("--no-yield", action="store_true")
    ap.add_argument("--short-tokens", type=int, default=2000)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--seed-base", type=int, default=1000)
    args = ap.parse_args()

    model = args.model or model_id(args.base_url)
    sizes = [int(t) for t in args.tokens.split(",")]
    tasks = args.tasks.split(",")
    cpt, calibration_tokens = calibrate(args.base_url, model, args.thinking)
    record = {"model": model, "chars_per_token": round(cpt, 4), "calibration_prompt_tokens":
              calibration_tokens, "thinking": args.thinking, "max_tokens": args.max_tokens,
              "requests": [], "interference": None, "yield": None}
    print(f"model {model}; {cpt:.3f} characters per token", flush=True)

    seed = args.seed_base
    for size in sizes:
        for task in tasks:
            for rep in range(args.reps):
                seed += 1
                bundle = ops_corpus.build(int(size * cpt), seed)
                messages = [{"role": "user", "content": bundle.text + "\n\n" + ops_corpus.TASKS[task]}]
                first = stream_chat(args.base_url,
                                    payload(model, messages, args.max_tokens, args.thinking, True))
                row = {"size": size, "task": task, "rep": rep, "seed": seed, "turn": 1,
                       **summarize(first, cold=True)}
                if "text" in first:
                    row["facts"] = ops_corpus.mentions(first["text"], bundle.facts)
                record["requests"].append(row)
                print(json.dumps(row), flush=True)
                if args.no_followup or "text" not in first:
                    continue
                messages += [{"role": "assistant", "content": first["text"]},
                             {"role": "user", "content": ops_corpus.FOLLOWUPS[task]}]
                second = stream_chat(args.base_url,
                                     payload(model, messages, args.max_tokens // 2, args.thinking, True))
                row2 = {"size": size, "task": task, "rep": rep, "seed": seed, "turn": 2,
                        **summarize(second, cold=False)}
                record["requests"].append(row2)
                print(json.dumps(row2), flush=True)

    if not args.no_interference and len(sizes) > 1:
        seed += 10
        record["interference"] = interference(args.base_url, model, min(sizes), max(sizes), cpt,
                                              args.max_tokens, args.thinking, seed)
        print(json.dumps({"interference": record["interference"]}), flush=True)

    if not args.no_yield:
        seed += 10
        record["yield"] = yield_probe(args.base_url, model, args.short_tokens, max(sizes), cpt,
                                      args.thinking, seed)
        print(json.dumps({"yield": record["yield"]}), flush=True)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)

    print("\n| tokens | task | turn | n | TTFT s (median) | prefill tok/s | decode tok/s | max gap s "
          "| facts named per rep (of 4) |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for size in sizes:
        for task in tasks:
            for turn in (1, 2):
                rows = [r for r in record["requests"] if r["size"] == size and r["task"] == task
                        and r["turn"] == turn and "error" not in r]
                if not rows:
                    continue
                med = lambda key: statistics.median([r[key] for r in rows if r.get(key) is not None]) \
                    if any(r.get(key) is not None for r in rows) else float("nan")
                facts = ""
                if turn == 1:
                    hits = [sum(r["facts"].values()) for r in rows if "facts" in r]
                    facts = ", ".join(str(h) for h in hits)
                print(f"| {size} | {task} | {turn} | {len(rows)} | {med('ttft_s'):.2f} | "
                      f"{med('prefill_tok_s'):.0f} | {med('decode_tok_s'):.1f} | "
                      f"{med('max_gap_s'):.2f} | {facts} |")
    errors = [r for r in record["requests"] if "error" in r]
    if errors:
        print(f"\n{len(errors)} request(s) failed; see {args.out_json}")


if __name__ == "__main__":
    main()
