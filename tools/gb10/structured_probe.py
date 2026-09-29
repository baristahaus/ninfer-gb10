#!/usr/bin/env python3
"""How often structured output holds without constrained decoding (plan step 4 evidence).

Usage: structured_probe.py BASE_URL OUT_JSON [--samples 2]

Against a running ninfer-serve, sends:
  * `response_format: json_schema` requests over three schemas (flat, enum/number, nested array),
    ten prompts each;
  * `tool_choice: "required"` requests over two tools, fifteen prompts, five of which do not
    obviously need either tool (the case a prompt-level directive handles worst);
each with thinking disabled and enabled, `--samples` times under the model's default sampling.

Classifies every response: valid, invalid JSON, schema violation (and which), empty; for tools:
call present, arguments parse, required arguments and enum values hold. Prints a table per
mode and writes every classified response (no generated text beyond the parsed object) to
OUT_JSON.
"""
import argparse
import json
import time
import urllib.error
import urllib.request

SCHEMAS = {
    "person": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "skills": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name", "age", "skills"],
        "additionalProperties": False,
    },
    "review": {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "confidence": {"type": "number"},
            "summary": {"type": "string"},
        },
        "required": ["sentiment", "confidence", "summary"],
        "additionalProperties": False,
    },
    "order": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"sku": {"type": "string"}, "quantity": {"type": "integer"}},
                    "required": ["sku", "quantity"],
                    "additionalProperties": False,
                },
            },
            "total": {"type": "number"},
        },
        "required": ["items", "total"],
        "additionalProperties": False,
    },
}

PEOPLE = [
    "Maria Chen, 34, is a backend engineer who writes Go and Rust and mentors juniors.",
    "Our new hire Tomás (29) handles payroll, Excel modelling and Spanish translation.",
    "Dr. Aisha Bello is 51; she does cardiology, clinical trials and public speaking.",
    "Kenji is a 19-year-old student into robotics, C++ and competitive chess.",
    "Priya Nair, aged 42, leads marketing: SEO, copywriting, brand strategy.",
    "The retired pilot Hans Müller (67) now teaches navigation and restores gliders.",
    "Lena, 25, freelance illustrator: watercolor, Procreate, children's books.",
    "Omar Haddad is thirty-eight and works as an electrician and solar installer.",
    "Sofia Rossi (31) is a sommelier who also speaks French and Japanese.",
    "Grace O'Neill, 58, a nurse practitioner specialising in geriatrics and wound care.",
]
REVIEWS = [
    "The blender died after two weeks and support never answered.",
    "Absolutely love this jacket, warm and the zippers feel solid.",
    "It's a phone case. It fits. Nothing more to say.",
    "Delivery was late but the coffee itself is outstanding.",
    "Worst hotel stay of my life: noisy, dirty, rude staff.",
    "Decent headphones for the price, bass is a bit muddy.",
    "The book started slow but the ending made me cry, highly recommended.",
    "Software update broke my printer's Wi-Fi. Again.",
    "Fine for a weekend trip, I would not take it hiking.",
    "Five stars. My kids play with this puzzle every day.",
]
ORDERS = [
    "Two of SKU A-100 and one B-220, total came to 57.40.",
    "I need 12 units of widget W-9 at 3.25 each, so 39.",
    "Order: 1x LAMP-01, 4x BULB-60W; invoice total 88.",
    "Please send five T-SHIRT-M and five T-SHIRT-L, 150 dollars total.",
    "Just one KEYBOARD-K2 for 129.99.",
    "Three boxes of PAPER-A4 (item P-A4), total 21.",
    "Items: CAB-HDMI x2, ADPT-USBC x1. Grand total 34.50.",
    "6 MUG-WHT, 6 MUG-BLK; the bill was 72.",
    "One tent TENT-4P and two SLEEP-BAG-R, 410 in total.",
    "Seven FILTER-X7 cartridges, 63 total.",
]
PROMPTS = {
    "person": [f"Extract the person described here as JSON.\n\n{t}" for t in PEOPLE],
    "review": [f"Classify this product review as JSON.\n\n{t}" for t in REVIEWS],
    "order": [f"Extract the order as JSON.\n\n{t}" for t in ORDERS],
}

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["city", "unit"]}}},
    {"type": "function", "function": {
        "name": "search_web",
        "description": "Search the web and return the top results.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"}},
            "required": ["query"]}}},
]
TOOL_PROMPTS = [
    "What's the weather in Lisbon right now, in celsius?",
    "Is it hot in Phoenix today? Use fahrenheit.",
    "Weather for Oslo please.",
    "Find recent news about fusion energy startups.",
    "Search for the best vegan ramen recipe, top 3 results.",
    "Look up who won the 2026 Tour de France.",
    "Should I bring an umbrella in Tokyo today?",
    "Find papers on speculative decoding for LLM inference.",
    "What's the temperature in Nairobi?",
    "Search: GB10 unified memory bandwidth benchmarks.",
    # No tool is obviously needed below; `required` still demands one.
    "What is 17 times 23?",
    "Write a haiku about autumn.",
    "Translate 'good morning' into German.",
    "Explain what a hash map is in one sentence.",
    "Say hello.",
]


def valid(value, schema, path="$"):
    """Minimal JSON Schema check for the keywords the probe schemas use. Returns an error or None."""
    kind = schema.get("type")
    if "enum" in schema and value not in schema["enum"]:
        return f"{path}: not in enum"
    if kind == "object":
        if not isinstance(value, dict):
            return f"{path}: not an object"
        for key in schema.get("required", []):
            if key not in value:
                return f"{path}.{key}: missing"
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = [k for k in value if k not in props]
            if extra:
                return f"{path}: extra {extra}"
        for key, sub in props.items():
            if key in value:
                err = valid(value[key], sub, f"{path}.{key}")
                if err:
                    return err
    elif kind == "array":
        if not isinstance(value, list):
            return f"{path}: not an array"
        for i, item in enumerate(value):
            err = valid(item, schema["items"], f"{path}[{i}]")
            if err:
                return err
    elif kind == "string" and not isinstance(value, str):
        return f"{path}: not a string"
    elif kind == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        return f"{path}: not an integer"
    elif kind == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        return f"{path}: not a number"
    return None


def post(base_url, payload):
    req = urllib.request.Request(
        base_url + "/v1/chat/completions", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            return json.load(resp), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}"


def classify_json(resp, schema):
    content = resp["choices"][0]["message"].get("content") or ""
    if not content.strip():
        return "empty", None
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return "invalid_json", None
    err = valid(value, schema)
    return ("valid", None) if err is None else ("schema_violation", err)


def classify_tool(resp):
    calls = resp["choices"][0]["message"].get("tool_calls") or []
    if not calls:
        return "no_call", None
    fn = calls[0]["function"]
    tool = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}.get(fn["name"])
    if tool is None:
        return "unknown_tool", fn["name"]
    try:
        args = json.loads(fn["arguments"])
    except json.JSONDecodeError:
        return "invalid_arguments", None
    err = valid(args, {**tool, "additionalProperties": False})
    return ("valid", None) if err is None else ("argument_violation", err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("out_json")
    ap.add_argument("--samples", type=int, default=2)
    args = ap.parse_args()
    with urllib.request.urlopen(args.base_url + "/v1/models", timeout=60) as resp:
        model = json.load(resp)["data"][0]["id"]
    records = []
    for thinking in ("none", "default"):
        extra = {"reasoning_effort": "none"} if thinking == "none" else {}
        for sample in range(args.samples):
            for name, schema in SCHEMAS.items():
                for i, text in enumerate(PROMPTS[name]):
                    payload = {"model": model, "max_tokens": 4096, **extra,
                               "messages": [{"role": "user", "content": text}],
                               "response_format": {"type": "json_schema", "json_schema": {
                                   "name": name, "schema": schema, "strict": True}}}
                    t0 = time.time()
                    resp, error = post(args.base_url, payload)
                    outcome, detail = ("http_error", error) if resp is None else \
                        classify_json(resp, schema)
                    records.append({"kind": "json_schema", "case": name, "prompt": i,
                                    "thinking": thinking, "sample": sample, "outcome": outcome,
                                    "detail": detail, "seconds": round(time.time() - t0, 2)})
            for i, text in enumerate(TOOL_PROMPTS):
                payload = {"model": model, "max_tokens": 4096, **extra, "tools": TOOLS,
                           "tool_choice": "required",
                           "messages": [{"role": "user", "content": text}]}
                t0 = time.time()
                resp, error = post(args.base_url, payload)
                outcome, detail = ("http_error", error) if resp is None else classify_tool(resp)
                records.append({"kind": "tool_required",
                                "case": "no_tool_needed" if i >= 10 else "tool_needed",
                                "prompt": i, "thinking": thinking, "sample": sample,
                                "outcome": outcome, "detail": detail,
                                "seconds": round(time.time() - t0, 2)})
            print(f"thinking={thinking} sample={sample}: {len(records)} records", flush=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=1)
    print("\n| kind | case | thinking | n | valid | outcomes |\n|---|---|---|---:|---:|---|")
    groups = {}
    for r in records:
        groups.setdefault((r["kind"], r["case"], r["thinking"]), []).append(r["outcome"])
    for (kind, case, thinking), outs in sorted(groups.items()):
        counts = {o: outs.count(o) for o in sorted(set(outs))}
        print(f"| {kind} | {case} | {thinking} | {len(outs)} | {outs.count('valid')} | {counts} |")


if __name__ == "__main__":
    main()
