**Title:** fix(flash-next): accept reasoning effort none with disabled thinking

**Target:** `lkarlslund/ninfer6000` `master` (`8f574ee4`).
**Head:** `baristahaus/ninfer-gb10` branch `upstream/ninfer6000/reasoning-effort-none` (one commit).
Independent of the other PRs in this set.

## Problem and scope

Related Issue: none

An OpenAI chat request with `"reasoning_effort": "none"` fails with HTTP 400:

```text
reasoning effort cannot be combined with disabled thinking
```

`translate.cpp` handles the value correctly as far as it goes. It maps `none` to
`enable_thinking = false`, and it keeps `ReasoningEffort::None` in the render options. The Flash-Next
template's `resolve_reasoning_instructions` then rejects any set effort while thinking is disabled,
including `None`, the one effort that means "thinking off". Every client that asks for no reasoning
the OpenAI way is refused, and has to send `chat_template_kwargs: {"enable_thinking": false}`
instead. A toggle-only template has the same check, so `none` fails there too.

The Qwen3.5 frontend passes `none` through to its Jinja template, which accepts it with thinking off,
so this is specific to the Flash-Next template.

## Implementation

`resolve_reasoning_instructions` treats `ReasoningEffort::None` like an unset effort wherever thinking
is off, on both template kinds. Any other effort without thinking is still rejected, and effort with
thinking on is unchanged.

## Verification

- `tests/models/qwen3_8_flash_next/test_frontend.cpp` gains a case: thinking off with
  `ReasoningEffort::None` prepares a prompt that starts in content, not reasoning. The test needs
  `NINFER_QWEN38_FLASH_NEXT_WEIGHTS`, as before.
- Compile-checked on `master` (`sm_121a`): the template and the test.
- Found driving a `master` build with the DGPP load client on NVIDIA GB10: `"reasoning_effort":
  "none"` returned 400, and the `chat_template_kwargs` form returned 200.
- Not yet run end-to-end with this commit; see the validation run in this set's README.
