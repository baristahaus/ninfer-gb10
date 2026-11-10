# Cross-engine parity harness

`measure_parity.py` measures one `.ninfer` artifact on two engines on the same box, so that a
comparison between engines is a measurement instead of an argument.

## What it measures

- time to first token (TTFT) and decode tokens per second for each request;
- the median, mean, standard deviation and range over all measured requests;
- the total generated tokens, so a run that generated less than it should is visible.

The rates come from the server's own streamed token timings, not from wall-clock time around the
request, so client-side scheduling noise does not enter them.

## Run

Build the engine under test first. For this repository:

```bash
cmake --preset rtx5090-release
cmake --build build/rtx5090-release -j --target ninfer-serve
```

Measure this engine:

```bash
python3 bench/parity/measure_parity.py --engine ninfer \
  --bin build/rtx5090-release/apps/ninfer-serve \
  --artifact /path/to/model.ninfer \
  --out /tmp/ninfer.json
```

Measure the other engine. capy-flash takes the artifact as its first positional argument:

```bash
python3 bench/parity/measure_parity.py --engine capy \
  --bin /path/to/capy-flash/ninfer-serve \
  --artifact /path/to/model.ninfer \
  --artifact-first \
  --out /tmp/capy.json
```

Both runs use the same artifact bytes. Run the two engines one after the other: this is a GPU job,
and a second device-allocating process must not run at the same time.

## Compare

```bash
python3 bench/parity/measure_parity.py --compare /tmp/ninfer.json /tmp/capy.json
```

The comparison prints the medians, the standard deviations, the ranges, the TTFT medians and the
token totals for both runs, and states whether the decode ranges overlap. It refuses to compare two
runs when either one produced empty responses.

## Protocol

| | |
|---|---|
| prompts | 5 fixed prompts in the script, the same for both engines |
| sampling | greedy (`temperature 0`), `--no-thinking` on both engines |
| output length | 256 tokens per request by default (`--max-tokens`) |
| warmup | one request per prompt, discarded |
| measured waves | 3 waves of 5 requests by default (`--waves`) |
| requests | 15 per engine |
| server flags | `--max-context 8192 --kv-capacity 8192 --max-concurrency 1`, plus `--extra` |

The token count is fixed and the sampling is greedy, so the number of generated tokens is not a
variable. The thinking channel is off, because a thinking model streams `reasoning_content` first
and the first *content* token arrives after all the reasoning; a first-token timestamp taken from
`content` alone is not a TTFT measurement. The script counts a token from any non-empty delta
channel and fails the comparison when a run produced nothing.

## Limits

A parity number invites over-reading. This protocol establishes the following, and nothing else:

- **tp 1 only.** The two-device lane is not measured.
- **One artifact.** The measurement is for one dense 27B artifact. A 4-bit MoE artifact such as
  Flash-Next 125B exercises different kernels.
- **No speculation.** Both engines run without MTP or DFlash2, so the drafter paths are unmeasured.
- **Short prompts.** A TTFT of about 52 ms is dominated by request setup, not by prefill. This
  workload says nothing about prefill throughput on long contexts.
- **Decode is the memory-bound regime.** Near-parity there is expected when both engines run on the
  same silicon, and it is not evidence that the compute-bound paths (prefill, MoE, attention) are
  equally matched.

The measurement that would discriminate further is prefill on a long prompt, and decode under MTP.

## Recorded campaign

The first run with this protocol, its conditions, its result and its methodological failure are
recorded in [Cross-engine parity](../../docs/performance/cross-engine-parity.md).
