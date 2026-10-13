"""Exercise the implemented NInfer HTTP product contract with the standard library."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

# A one-pixel PNG. The target frontend performs its normal resize/patch expansion.
_IMAGE_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
    "AScY42YAAAAASUVORK5CYII="
)


class ContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class Response:
    status: int
    content_type: str
    body: bytes


def request(
    base_url: str, method: str, path: str, payload: Any | None = None
) -> Response:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        base_url + path, data=body, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            return Response(
                status=response.status,
                content_type=response.headers.get_content_type(),
                body=response.read(),
            )
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ContractError(
            f"{method} {path} returned HTTP {error.code}: {detail}"
        ) from error


def json_response(
    base_url: str, method: str, path: str, payload: Any | None = None
) -> dict[str, Any]:
    response = request(base_url, method, path, payload)
    if response.status != 200 or response.content_type != "application/json":
        raise ContractError(
            f"{method} {path} returned status={response.status} content-type={response.content_type}"
        )
    try:
        value = json.loads(response.body)
    except json.JSONDecodeError as error:
        raise ContractError(f"{method} {path} returned invalid JSON") from error
    if not isinstance(value, dict):
        raise ContractError(f"{method} {path} did not return a JSON object")
    return value


def wait_for_health(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if json_response(base_url, "GET", "/health") == {"status": "ok"}:
                return
        except (ContractError, urllib.error.URLError) as error:
            last_error = error
        time.sleep(0.25)
    raise ContractError(
        f"server did not become healthy within {timeout:g}s: {last_error}"
    )


def fetch_metrics(base_url: str) -> dict[str, float]:
    request = urllib.request.Request(f"{base_url}/metrics")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as error:
        raise ContractError(f"GET /metrics failed: {error}") from error
    metrics: dict[str, float] = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        try:
            metrics[name] = float(value)
        except ValueError as error:
            raise ContractError(f"GET /metrics has a malformed line: {line!r}") from error
    if not metrics:
        raise ContractError("GET /metrics rendered no metric lines")
    return metrics


def check_metrics(base_url: str, model: str) -> dict[str, Any]:
    metrics = fetch_metrics(base_url)

    def require(name: str) -> float:
        if name not in metrics:
            raise ContractError(f"GET /metrics is missing {name}")
        return metrics[name]

    requests_total = require("ninfer:requests_total")
    if requests_total < 1:
        raise ContractError("ninfer:requests_total did not count the smoke's requests")
    for name in (
        "ninfer:prompt_tokens_total",
        "ninfer:generation_tokens_total",
        "ninfer:cached_tokens_total",
        "ninfer:computed_prefill_tokens_total",
        "ninfer:mtp_drafted_total",
        "ninfer:mtp_accepted_total",
        "ninfer:request_latency_seconds_count",
        "ninfer:request_latency_seconds_sum",
        "ninfer:time_to_first_token_seconds_sum",
        "ninfer:prefill_seconds_sum",
        "ninfer:decode_seconds_sum",
    ):
        require(name)
    if metrics["ninfer:request_latency_seconds_count"] != requests_total:
        raise ContractError(
            "ninfer:request_latency_seconds_count does not match requests_total"
        )
    if metrics["ninfer:prompt_tokens_total"] < 1 or metrics["ninfer:generation_tokens_total"] < 1:
        raise ContractError("token counters did not accumulate over the smoke's requests")
    for gauge in (
        "ninfer:pool_tokens",
        "ninfer:pool_free_tokens",
        "ninfer:kept_prompts",
        "ninfer:context_length",
    ):
        require(gauge)
    if metrics["ninfer:pool_tokens"] < 1 or metrics["ninfer:context_length"] < 1:
        raise ContractError("pool or context gauges are unset")
    model_info = f'ninfer:model_info{{model_name="{model}"}}'
    if model_info not in metrics:
        raise ContractError(f"GET /metrics does not declare the model identity {model_info!r}")
    return {
        "requests_total": int(requests_total),
        "prompt_tokens_total": int(metrics["ninfer:prompt_tokens_total"]),
        "generation_tokens_total": int(metrics["ninfer:generation_tokens_total"]),
        "cached_tokens_total": int(metrics["ninfer:cached_tokens_total"]),
        "mtp_accepted_total": int(metrics["ninfer:mtp_accepted_total"]),
        "pool_tokens": int(metrics["ninfer:pool_tokens"]),
        "kept_prompts": int(metrics["ninfer:kept_prompts"]),
    }


def require_usage(usage: Any, prompt_key: str, completion_key: str) -> tuple[int, int]:
    if not isinstance(usage, dict):
        raise ContractError("response usage is not an object")
    prompt = usage.get(prompt_key)
    completion = usage.get(completion_key)
    if not isinstance(prompt, int) or prompt <= 0:
        raise ContractError(f"invalid {prompt_key}: {prompt!r}")
    if not isinstance(completion, int) or completion < 0:
        raise ContractError(f"invalid {completion_key}: {completion!r}")
    return prompt, completion


def openai_nonstream(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    stop: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": 0,
    }
    if stop is not None:
        payload["stop"] = stop
    response = json_response(base_url, "POST", "/v1/chat/completions", payload)
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ContractError("OpenAI response must contain exactly one choice")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ContractError("OpenAI response is missing assistant content")
    if choice.get("finish_reason") not in {"stop", "length", "tool_calls"}:
        raise ContractError(
            f"invalid OpenAI finish_reason: {choice.get('finish_reason')!r}"
        )
    prompt, completion = require_usage(
        response.get("usage"), "prompt_tokens", "completion_tokens"
    )
    if completion > max_tokens:
        raise ContractError("OpenAI output exceeds the requested token budget")
    usage = response["usage"]
    if usage.get("total_tokens") != prompt + completion:
        raise ContractError("OpenAI total_tokens does not equal prompt + completion")
    return response


def parse_openai_stream(response: Response) -> tuple[str, str, str, dict[str, Any]]:
    if response.status != 200 or response.content_type != "text/event-stream":
        raise ContractError(
            f"OpenAI stream returned status={response.status} content-type={response.content_type}"
        )
    content: list[str] = []
    reasoning: list[str] = []
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    saw_role = False
    saw_done = False
    for block in response.body.decode("utf-8").replace("\r\n", "\n").split("\n\n"):
        if not block:
            continue
        lines = [line[6:] for line in block.splitlines() if line.startswith("data: ")]
        if len(lines) != 1:
            raise ContractError(f"malformed OpenAI SSE block: {block!r}")
        if lines[0] == "[DONE]":
            if saw_done:
                raise ContractError("OpenAI stream emitted [DONE] more than once")
            saw_done = True
            continue
        if saw_done:
            raise ContractError("OpenAI stream emitted data after [DONE]")
        try:
            event = json.loads(lines[0])
        except json.JSONDecodeError as error:
            raise ContractError("OpenAI stream contained invalid JSON") from error
        choices = event.get("choices")
        if not isinstance(choices, list):
            raise ContractError("OpenAI stream event is missing choices")
        event_usage = event.get("usage")
        if event_usage is not None:
            if choices:
                raise ContractError("OpenAI usage event unexpectedly contains choices")
            if usage is not None:
                raise ContractError("OpenAI stream emitted usage more than once")
            if finish_reason is None:
                raise ContractError(
                    "OpenAI stream emitted usage before its finish event"
                )
            usage = event_usage
            continue
        if usage is not None:
            raise ContractError("OpenAI stream emitted an event after usage")
        if len(choices) != 1:
            raise ContractError("ordinary OpenAI stream event must contain one choice")
        if finish_reason is not None:
            raise ContractError("OpenAI stream emitted an event after its finish event")
        choice = choices[0]
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            raise ContractError("OpenAI stream choice is missing delta")
        if "role" in delta:
            if saw_role or delta.get("role") != "assistant":
                raise ContractError("invalid or duplicate assistant-role event")
            saw_role = True
        if "content" in delta:
            if not isinstance(delta["content"], str):
                raise ContractError("OpenAI content delta is not a string")
            content.append(delta["content"])
        if "reasoning_content" in delta:
            if not isinstance(delta["reasoning_content"], str):
                raise ContractError("OpenAI reasoning delta is not a string")
            reasoning.append(delta["reasoning_content"])
        reason = choice.get("finish_reason")
        if reason is not None:
            if finish_reason is not None or reason not in {
                "stop",
                "length",
                "tool_calls",
            }:
                raise ContractError(
                    f"invalid or duplicate OpenAI finish reason: {reason!r}"
                )
            finish_reason = reason
    if not saw_role or not saw_done or finish_reason is None or usage is None:
        raise ContractError(
            "OpenAI stream did not complete its role/finish/usage/[DONE] contract"
        )
    prompt, completion = require_usage(usage, "prompt_tokens", "completion_tokens")
    if usage.get("total_tokens") != prompt + completion:
        raise ContractError(
            "OpenAI streamed total_tokens does not equal prompt + completion"
        )
    return "".join(content), "".join(reasoning), finish_reason, usage


def response_text(response: dict[str, Any]) -> tuple[str, str]:
    output = response.get("output")
    if not isinstance(output, list):
        raise ContractError("Responses output is not an array")
    content: list[str] = []
    reasoning: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            raise ContractError("Responses output contains a non-object Item")
        item_type = item.get("type")
        parts = item.get("content", [])
        if item_type in {"message", "reasoning"} and not isinstance(parts, list):
            raise ContractError(f"Responses {item_type} content is not an array")
        for part in parts:
            if not isinstance(part, dict) or not isinstance(part.get("text"), str):
                raise ContractError("Responses text content part has the wrong shape")
            if part.get("type") == "output_text":
                content.append(part["text"])
            elif part.get("type") == "reasoning_text":
                reasoning.append(part["text"])
            else:
                raise ContractError(
                    f"unexpected Responses content type: {part.get('type')!r}"
                )
    return "".join(content), "".join(reasoning)


def require_responses_usage(usage: Any) -> tuple[int, int]:
    prompt, completion = require_usage(usage, "input_tokens", "output_tokens")
    if usage.get("total_tokens") != prompt + completion:
        raise ContractError("Responses total_tokens does not equal input + output")
    input_details = usage.get("input_tokens_details")
    output_details = usage.get("output_tokens_details")
    if not isinstance(input_details, dict) or not isinstance(
        input_details.get("cached_tokens"), int
    ):
        raise ContractError("Responses cached-token usage is missing")
    if not isinstance(output_details, dict) or not isinstance(
        output_details.get("reasoning_tokens"), int
    ):
        raise ContractError("Responses reasoning-token usage is missing")
    return prompt, completion


def responses_nonstream(
    base_url: str,
    model: str,
    input_value: Any,
    *,
    store: bool,
    previous_response_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": input_value,
        "max_output_tokens": 16,
        "temperature": 0,
        "store": store,
    }
    if previous_response_id is not None:
        payload["previous_response_id"] = previous_response_id
    response = json_response(base_url, "POST", "/v1/responses", payload)
    if response.get("object") != "response" or response.get("status") not in {
        "completed",
        "incomplete",
    }:
        raise ContractError("Responses non-streaming envelope has the wrong shape")
    if not isinstance(response.get("id"), str) or not response["id"].startswith(
        "resp_"
    ):
        raise ContractError("Responses non-streaming envelope has an invalid id")
    if response.get("model") != model or response.get("store") is not store:
        raise ContractError("Responses non-streaming request fields were not echoed")
    _, completion = require_responses_usage(response.get("usage"))
    if completion > payload["max_output_tokens"]:
        raise ContractError("Responses output exceeds the requested token budget")
    response_text(response)
    return response


def parse_responses_stream(response: Response) -> tuple[str, str, dict[str, Any]]:
    if response.status != 200 or response.content_type != "text/event-stream":
        raise ContractError(
            "Responses stream returned "
            f"status={response.status} content-type={response.content_type}"
        )
    content: list[str] = []
    reasoning: list[str] = []
    terminal: dict[str, Any] | None = None
    expected_sequence = 0
    for block in response.body.decode("utf-8").replace("\r\n", "\n").split("\n\n"):
        if not block:
            continue
        event_lines = [
            line[7:] for line in block.splitlines() if line.startswith("event: ")
        ]
        data_lines = [
            line[6:] for line in block.splitlines() if line.startswith("data: ")
        ]
        if len(event_lines) != 1 or len(data_lines) != 1 or data_lines[0] == "[DONE]":
            raise ContractError(f"malformed Responses SSE block: {block!r}")
        try:
            event = json.loads(data_lines[0])
        except json.JSONDecodeError as error:
            raise ContractError("Responses stream contained invalid JSON") from error
        event_type = event.get("type")
        if event_type != event_lines[0]:
            raise ContractError("Responses SSE event name differs from JSON type")
        if event.get("sequence_number") != expected_sequence:
            raise ContractError("Responses sequence_number is not contiguous")
        expected_sequence += 1
        if terminal is not None:
            raise ContractError(
                "Responses stream emitted data after its terminal event"
            )
        if event_type == "response.output_text.delta":
            if not isinstance(event.get("delta"), str):
                raise ContractError("Responses output_text delta is not a string")
            content.append(event["delta"])
        elif event_type == "response.reasoning_text.delta":
            if not isinstance(event.get("delta"), str):
                raise ContractError("Responses reasoning delta is not a string")
            reasoning.append(event["delta"])
        elif event_type in {
            "response.completed",
            "response.incomplete",
            "response.failed",
        }:
            value = event.get("response")
            if not isinstance(value, dict):
                raise ContractError("Responses terminal event is missing its Response")
            terminal = value
    if terminal is None or terminal.get("status") not in {"completed", "incomplete"}:
        raise ContractError("Responses stream did not terminate successfully")
    terminal_content, terminal_reasoning = response_text(terminal)
    if "".join(content) != terminal_content or "".join(reasoning) != terminal_reasoning:
        raise ContractError("Responses deltas do not reconstruct terminal output Items")
    require_responses_usage(terminal.get("usage"))
    return terminal_content, terminal_reasoning, terminal


def expect_error(
    base_url: str, path: str, payload: dict[str, Any], code: str | None
) -> dict[str, Any]:
    """POST a request that must fail with HTTP 400; OpenAI errors must carry `code`."""
    req = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            raise ContractError(f"POST {path} succeeded with HTTP {response.status}; expected 400")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        if error.code != 400:
            raise ContractError(f"POST {path} returned HTTP {error.code}; expected 400: {body}")
    value = json.loads(body)
    if code is not None and value.get("error", {}).get("code") != code:
        raise ContractError(f"POST {path} returned {body}; expected error code {code!r}")
    if code is None and value.get("error", {}).get("type") != "invalid_request_error":
        raise ContractError(f"POST {path} returned {body}; expected invalid_request_error")
    return value


def require_constraint(value: Any, branch: str) -> None:
    if (
        not isinstance(value, dict)
        or value.get("branch") != branch
        or value.get("complete") is not True
        or value.get("terminated") is not True
        or value.get("cache") not in {"hit", "built", "waited"}
        or not isinstance(value.get("mask_positions"), int)
        or value["mask_positions"] <= 0
    ):
        raise ContractError(f"constraint observation is not a terminated {branch} result: {value!r}")


_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}
_YES_NO_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["answer"],
    "additionalProperties": False,
}


def chat_payload(model: str, content: str, **fields: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_completion_tokens": 96,
        "temperature": 0,
        "enable_thinking": False,
    }
    payload.update(fields)
    return payload


def exercise_constraints_enforced(base_url: str, model: str) -> dict[str, Any]:
    """Qwen3.5 family: output and tool constraints are enforced and observed."""
    chat = "/v1/chat/completions"
    schema_format = {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": _YES_NO_SCHEMA, "strict": True},
    }
    answered = json_response(
        base_url, "POST", chat,
        chat_payload(model, "Is water wet? Answer in JSON.", response_format=schema_format),
    )
    choice = answered["choices"][0]
    if choice["finish_reason"] != "stop":
        raise ContractError(f"json_schema answer did not stop: {choice['finish_reason']!r}")
    if json.loads(choice["message"]["content"]).get("answer") not in {"yes", "no"}:
        raise ContractError("json_schema answer does not satisfy its schema")
    require_constraint(answered.get("constraint"), "content")

    choices = ["alpha", "beta", "gamma"]
    chosen = json_response(
        base_url, "POST", chat,
        chat_payload(model, "Pick a Greek letter name.", structured_outputs={"choice": choices}),
    )
    if chosen["choices"][0]["message"]["content"] not in choices:
        raise ContractError("structured_outputs.choice returned a value outside the list")
    require_constraint(chosen.get("constraint"), "content")

    pattern = "(BUG|TASK)-[0-9]{4}"
    matched = json_response(
        base_url, "POST", chat,
        chat_payload(model, "Name a ticket.", structured_outputs={"regex": pattern}),
    )
    if not re.fullmatch(pattern, matched["choices"][0]["message"]["content"]):
        raise ContractError("structured_outputs.regex returned a non-matching value")

    stream_payload = chat_payload(
        model, "Describe a cat as a JSON object.",
        response_format={"type": "json_object"}, stream=True,
        stream_options={"include_usage": True}, max_completion_tokens=128,
    )
    streamed = request(base_url, "POST", chat, stream_payload)
    content, _, stream_finish, _ = parse_openai_stream(streamed)
    stream_constraint = None
    for block in streamed.body.decode("utf-8").split("\n\n"):
        if block.startswith("data: {"):
            event = json.loads(block[6:])
            if "constraint" in event:
                if stream_constraint is not None:
                    raise ContractError("OpenAI stream sent the constraint observation twice")
                stream_constraint = event["constraint"]
    if stream_constraint is None:
        raise ContractError("OpenAI stream omitted the constraint observation")
    if stream_finish == "stop":
        if not isinstance(json.loads(content), dict):
            raise ContractError("streamed json_object answer is not an object")
        require_constraint(stream_constraint, "content")

    tools = [_WEATHER_TOOL]
    for label, fields in (
        ("required", {"tool_choice": "required"}),
        (
            "named",
            {
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
                "parallel_tool_calls": False,
            },
        ),
    ):
        called = json_response(
            base_url, "POST", chat,
            chat_payload(model, "Hello there.", tools=tools, **fields),
        )
        choice = called["choices"][0]
        calls = choice["message"].get("tool_calls") or []
        if choice["finish_reason"] != "tool_calls" or not calls:
            raise ContractError(f"tool_choice {label} did not produce a tool call")
        if label == "named" and len(calls) != 1:
            raise ContractError("named tool choice without parallel calls produced several calls")
        for call in calls:
            arguments = json.loads(call["function"]["arguments"])
            if call["function"]["name"] != "get_weather" or set(arguments) != {"city"} or (
                not isinstance(arguments["city"], str)
            ):
                raise ContractError(f"strict tool call violates its schema: {call!r}")
        require_constraint(called.get("constraint"), "tools")

    anthropic = json_response(
        base_url, "POST", "/v1/messages",
        {
            "model": model,
            "max_tokens": 96,
            "messages": [{"role": "user", "content": "Hello there."}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get the current weather for a city.",
                    "input_schema": _WEATHER_TOOL["function"]["parameters"],
                    "strict": True,
                }
            ],
            "tool_choice": {"type": "any"},
        },
    )
    uses = [block for block in anthropic.get("content", []) if block.get("type") == "tool_use"]
    if anthropic.get("stop_reason") != "tool_use" or not uses:
        raise ContractError("Anthropic tool_choice any did not produce tool_use")
    if any(set(block.get("input", {})) != {"city"} for block in uses):
        raise ContractError("Anthropic strict tool input violates its schema")

    formatted = json_response(
        base_url, "POST", "/v1/responses",
        {
            "model": model,
            "input": "Is water wet? Answer in JSON.",
            "max_output_tokens": 96,
            "temperature": 0,
            "store": False,
            "text": {"format": {"type": "json_schema", "name": "answer", "schema": _YES_NO_SCHEMA}},
            "reasoning": {"effort": "none"},
        },
    )
    text, _ = response_text(formatted)
    if formatted.get("status") != "completed" or json.loads(text).get("answer") not in {"yes", "no"}:
        raise ContractError("Responses text.format json_schema answer does not satisfy its schema")

    expect_error(base_url, chat, chat_payload(model, "x", structured_outputs={"regex": "(?=a)"}),
                 "invalid_regex")
    expect_error(
        base_url, chat,
        chat_payload(model, "x", response_format={
            "type": "json_schema",
            "json_schema": {"name": "a", "schema": {"type": "string", "unknownKeyword": 1}},
        }),
        "unsupported_json_schema",
    )
    expect_error(base_url, chat, chat_payload(model, "x", guided_json={"type": "object"}), None)
    expect_error(
        base_url, chat,
        chat_payload(model, "x", response_format={"type": "json_object"}, stop=["END"]),
        None,
    )
    return {
        "json_schema_answer": choice_content(answered),
        "stream_finish_reason": stream_finish,
        "tool_calls": len(calls),
    }


def choice_content(response: dict[str, Any]) -> str:
    return response["choices"][0]["message"]["content"]


def exercise_constraints_refused(base_url: str, model: str) -> dict[str, Any]:
    """Qwen3.8 Flash-Next: no token-mask execution, so enforceable requests are refused."""
    chat = "/v1/chat/completions"
    expect_error(
        base_url, chat,
        chat_payload(model, "x", response_format={"type": "json_object"}),
        "unsupported_json_schema",
    )
    expect_error(base_url, chat, chat_payload(model, "x", structured_outputs={"choice": ["a"]}),
                 "invalid_choice")
    expect_error(base_url, chat, chat_payload(model, "x", tools=[_WEATHER_TOOL]),
                 "invalid_tool_constraint")
    relaxed = dict(_WEATHER_TOOL, function=dict(_WEATHER_TOOL["function"], strict=False))
    expect_error(
        base_url, chat,
        chat_payload(model, "x", tools=[relaxed], tool_choice="required"),
        "invalid_tool_constraint",
    )
    free = json_response(base_url, "POST", chat,
                         chat_payload(model, "Hello there.", tools=[relaxed], max_completion_tokens=16))
    if "constraint" in free:
        raise ContractError("free tool generation reported a constraint observation")
    return {"refused": 4}


def check_constraint_metrics(base_url: str, enforced: bool) -> dict[str, Any]:
    metrics = fetch_metrics(base_url)
    terminated = metrics.get('ninfer:constraint_requests_total{outcome="terminated"}')
    positions = metrics.get("ninfer:constraint_mask_positions_total")
    uploaded = metrics.get("ninfer:constraint_mask_upload_bytes_total")
    if terminated is None or positions is None or uploaded is None:
        raise ContractError("GET /metrics is missing the constraint counters")
    if enforced and (terminated < 1 or positions < 1 or uploaded < 1):
        raise ContractError("constraint counters did not move for enforced requests")
    if not enforced and (terminated != 0 or positions != 0):
        raise ContractError("constraint counters moved on a family without constraints")
    return {
        "terminated": int(terminated),
        "tools_branch": int(metrics.get('ninfer:constraint_branch_total{branch="tools"}', 0)),
        "mask_positions": int(positions),
        "mask_upload_bytes": int(uploaded),
    }


def exercise(base_url: str, model: str, constraints: str) -> dict[str, Any]:
    models = json_response(base_url, "GET", "/v1/models")
    entries = models.get("data")
    if (
        models.get("object") != "list"
        or not isinstance(entries, list)
        or len(entries) != 1
    ):
        raise ContractError("model-list response has the wrong shape")
    if entries[0].get("id") != model or entries[0].get("owned_by") != "ninfer":
        raise ContractError(
            "model-list response does not identify the configured NInfer model"
        )
    single_model = json_response(base_url, "GET", f"/v1/models/{model}")
    if single_model.get("id") != model:
        raise ContractError("single-model response has the wrong id")

    anthropic_prompt = {
        "model": model,
        "max_tokens": 4,
        "messages": [{"role": "user", "content": "Reply briefly."}],
    }
    counted = json_response(
        base_url, "POST", "/v1/messages/count_tokens", anthropic_prompt
    )
    input_tokens = counted.get("input_tokens")
    if not isinstance(input_tokens, int) or input_tokens <= 0:
        raise ContractError("count_tokens returned a non-positive input_tokens value")

    messages = [{"role": "user", "content": "Reply with a single short word."}]
    stops = ["__NINFER_SMOKE_UNLIKELY_STOP__"]
    nonstream = openai_nonstream(base_url, model, messages, max_tokens=4, stop=stops)
    message = nonstream["choices"][0]["message"]
    if nonstream["usage"]["completion_tokens"] <= 0:
        raise ContractError("OpenAI request completed without producing a token")
    if not message.get("content") and not message.get("reasoning_content"):
        raise ContractError("OpenAI request completed without publishing output bytes")
    stream_payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": 4,
        "temperature": 0,
        "stop": stops,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    streamed = request(base_url, "POST", "/v1/chat/completions", stream_payload)
    content, reasoning, stream_finish, stream_usage = parse_openai_stream(streamed)
    if not content and not reasoning:
        raise ContractError("OpenAI stream completed without publishing output bytes")
    if (
        not 0
        < stream_usage["completion_tokens"]
        <= stream_payload["max_completion_tokens"]
    ):
        raise ContractError("OpenAI stream violated its output token budget")
    if stream_usage["prompt_tokens"] != nonstream["usage"]["prompt_tokens"]:
        raise ContractError("OpenAI input token count changed for the same messages")

    responses_input = "Reply with a single short word."
    response_count = json_response(
        base_url,
        "POST",
        "/v1/responses/input_tokens",
        {"model": model, "input": responses_input},
    )
    if response_count.get("object") != "response.input_tokens" or not isinstance(
        response_count.get("input_tokens"), int
    ):
        raise ContractError("Responses input_tokens returned the wrong shape")
    response_sync = responses_nonstream(base_url, model, responses_input, store=False)
    response_prompt_tokens, response_output_tokens = require_responses_usage(
        response_sync.get("usage")
    )
    if response_prompt_tokens != response_count["input_tokens"]:
        raise ContractError("Responses input_tokens differs from generation usage")
    response_stream_payload = {
        "model": model,
        "input": responses_input,
        "max_output_tokens": 16,
        "temperature": 0,
        "store": False,
        "stream": True,
    }
    response_stream = request(
        base_url, "POST", "/v1/responses", response_stream_payload
    )
    _, _, response_stream_terminal = parse_responses_stream(response_stream)
    streamed_input_tokens, streamed_output_tokens = require_responses_usage(
        response_stream_terminal.get("usage")
    )
    if streamed_output_tokens > response_stream_payload["max_output_tokens"]:
        raise ContractError("Responses stream violated its output token budget")
    if streamed_input_tokens != response_count["input_tokens"]:
        raise ContractError("Responses stream input count differs from input_tokens")

    stored_response = responses_nonstream(
        base_url, model, "Remember the code word ORCHID. Reply briefly.", store=True
    )
    stored_id = stored_response.get("id")
    if not isinstance(stored_id, str) or not stored_id.startswith("resp_"):
        raise ContractError("stored Response has an invalid id")
    retrieved = json_response(base_url, "GET", f"/v1/responses/{stored_id}")
    if retrieved != stored_response:
        raise ContractError("retrieved Response differs from the created Response")
    input_items = json_response(
        base_url, "GET", f"/v1/responses/{stored_id}/input_items?order=asc"
    )
    if input_items.get("object") != "list" or len(input_items.get("data", [])) != 1:
        raise ContractError("Responses input_items list has the wrong shape")
    continuation_input = "What code word was given? Reply with only that word."
    continuation = responses_nonstream(
        base_url,
        model,
        continuation_input,
        store=True,
        previous_response_id=stored_id,
    )
    standalone_continuation_count = json_response(
        base_url,
        "POST",
        "/v1/responses/input_tokens",
        {"model": model, "input": continuation_input},
    )["input_tokens"]
    continuation_prompt_tokens, _ = require_responses_usage(continuation.get("usage"))
    if continuation.get("previous_response_id") != stored_id or (
        continuation_prompt_tokens <= standalone_continuation_count
    ):
        raise ContractError("previous_response_id did not reconstruct stored context")
    continuation_id = continuation.get("id")
    for response_id in (continuation_id, stored_id):
        deleted = json_response(base_url, "DELETE", f"/v1/responses/{response_id}")
        if deleted != {
            "id": response_id,
            "object": "response.deleted",
            "deleted": True,
        }:
            raise ContractError("Responses delete returned the wrong object")

    image_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _IMAGE_DATA_URI}},
                {"type": "text", "text": "What is visible? Answer briefly."},
            ],
        }
    ]
    image_response = openai_nonstream(base_url, model, image_messages, max_tokens=2)
    image_prompt_tokens, _ = require_usage(
        image_response.get("usage"), "prompt_tokens", "completion_tokens"
    )
    if image_prompt_tokens <= input_tokens:
        raise ContractError(
            "image request did not expand the prompt through the Vision frontend"
        )

    anthropic = json_response(base_url, "POST", "/v1/messages", anthropic_prompt)
    if anthropic.get("type") != "message" or anthropic.get("role") != "assistant":
        raise ContractError("Anthropic response has the wrong envelope")
    if anthropic.get("stop_reason") not in {"end_turn", "max_tokens", "tool_use"}:
        raise ContractError(
            f"invalid Anthropic stop_reason: {anthropic.get('stop_reason')!r}"
        )
    blocks = anthropic.get("content")
    if not isinstance(blocks, list) or not blocks:
        raise ContractError("Anthropic response content is empty")
    anthropic_usage = anthropic.get("usage")
    anthropic_input_tokens, _ = require_usage(
        anthropic_usage, "input_tokens", "output_tokens"
    )
    cache_read = anthropic_usage.get("cache_read_input_tokens") or 0
    if not isinstance(cache_read, int) or cache_read < 0:
        raise ContractError("Anthropic usage cache_read_input_tokens is invalid")
    if anthropic_input_tokens + cache_read != input_tokens:
        raise ContractError(
            "Anthropic usage prompt accounting differs from count_tokens: "
            f"input={anthropic_input_tokens} cache_read={cache_read} count={input_tokens}"
        )

    return {
        "format": "ninfer_serve_contract_v3",
        "model": model,
        "count_tokens": input_tokens,
        "openai_finish_reason": stream_finish,
        "openai_completion_tokens": stream_usage["completion_tokens"],
        "responses_input_tokens": response_prompt_tokens,
        "responses_output_tokens": response_output_tokens,
        "image_prompt_tokens": image_prompt_tokens,
        "anthropic_stop_reason": anthropic["stop_reason"],
        "constraints": (
            exercise_constraints_enforced(base_url, model)
            if constraints == "enforced"
            else exercise_constraints_refused(base_url, model)
        ),
        "constraint_metrics": check_constraint_metrics(base_url, constraints == "enforced"),
        "metrics": check_metrics(base_url, model),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--health-timeout", type=float, default=300.0)
    parser.add_argument(
        "--constraints",
        choices=("enforced", "refused"),
        default="enforced",
        help="enforced for the Qwen3.5 family; refused for Qwen3.8 Flash-Next",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    wait_for_health(base_url, args.health_timeout)
    print(
        json.dumps(
            exercise(base_url, args.model, args.constraints), ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
