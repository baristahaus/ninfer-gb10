# Draft upstream notes (2026-10-29)

Postable findings for `lkarlslund/quench6k` and `Neroued/ninfer`. Not an authority; remove
after posting.

## Note 1, for quench6k: streaming tool-call publication on the OpenAI route

When a model emits tool calls while the client streams, an OpenAI-compatible server has two
sensible publication contracts, and mixing them silently breaks clients:

1. The event contract: the server emits incremental `tool_calls` deltas (name first, then
   argument fragments) while the model generates.
2. The progress contract: arguments stay private while generating; the stream carries a
   nonstandard liveness observation (we emit `{"tool_call_progress": {"bytes": N}}` chunks),
   and each finished call arrives as one complete `tool_calls` delta before the terminal
   chunk.

Both end with the normal `finish_reason: "tool_calls"` chunk. A client that only reads
terminal tool calls works with either. We serve the same Qwen3.8 Flash-Next 125B model with
two engine routes on the C++ side: one family streams events, the other publishes progress
plus terminal calls, because its frontend deliberately keeps parameters private during
generation.

The failure we hit, which your `96347d5` fix may relate to: a stream that emits neither
deltas nor a progress observation, then carries finished tool calls only in the terminal
state, fails any encoder that verifies "streamed tools agree with terminal calls". The
verified repair is to adopt the terminal calls as complete deltas at finish time when nothing
was streamed. Worth pinning with a regression test either way.

## Note 2, for Neroued/ninfer: the two-generation include hazard

A layered refactor that moves kernel implementations into a new owning directory while old
call sites keep including the old paths leaves two byte-identical or near-identical headers
in the tree. They compile, they pass tests, and they are a latent trap: the two copies define
the same symbols in the same namespace with bodies that drift, and any translation unit that
ends up including both fails to link or, worse, silently uses one definition per TU.

What worked for us, in order of safety:

1. Delete copies that no build rule compiles and nothing includes (verify with a
   glob-aware dead-file analysis; `file(GLOB)` compilation is easy to miss).
2. For byte-identical clusters, keep one canonical path and turn the other paths into
   one-line include shims, so every unit converges on one definition.
3. Leave same-name-different-body pairs in place and record them; resolve them by ownership
   (pick the layer that owns the format, port the call sites, delete the shadow) rather than
   by shimming, because a shim across divergent bodies changes semantics.

The trigger that exposed our third category: a cross-lane merge advanced one copy to a
three-format enum while a shared shim header still had the two-format version, and the merged
tree stopped compiling on the lane the merge did not touch.
