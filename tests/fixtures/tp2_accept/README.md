# tests/fixtures/tp2_accept

The two-GPU acceptance test's prompt and golden chains (`tests/models/qwen3_5/test_engine_tp2_accept_real.cpp`).

## Why the prompt is a fixture and not a string

The chain this test compares is only as good an oracle as the prompt that produces it. The 6-token
prompt the test used first, `{9707, 11, 1879, 13, 5765, 374}`, makes the reference lane repeat its
own tail: the 32-token greedy continuation is

```
11 1879 13 5765 374  11 1879 13 5765 374  11 1879 13 5765 374  11 1879 13 5765 374
11 1879 13 5765 374  11 1879 13 5765 374  11 1879
```

— a period-5 cycle over five distinct tokens, so any implementation that reproduces the loop passes,
including wrong ones. The prompt is therefore stored as TOKEN IDS: both arms read the same list and
neither tokenizes anything, so two hosts cannot disagree about what was prefilled.

## The files

| File | What |
|---|---|
| `prompt.ids` | 32 token ids: the `reasoning-short` prompt of [`../prompts/corpus.json`](../prompts/corpus.json), tokenized once with the artifact's own tokenizer (`Engine::tokenize_text`) and checked in as ids |
| `golden_plain.ids` | the width-1 greedy chain of the plain lane (no speculative backend) |
| `golden_mtp.ids` | the width-1 greedy chain of the MTP lane (`NINFER_TP2_SPEC=mtp`) |

The prompt text, for the record: *"A box has 3 red and 2 blue balls. One ball is drawn without
looking. What is the probability it is red? Reply as a fraction."* It was chosen because its greedy
continuation is 32 tokens with 23 distinct ids and no adjacent repeats, and because 32 prefilled
tokens cross the A8 activation threshold (17), so the round exercises the quantized prefill path as
well as the decode path.

## Provenance of the goldens

- Host: trueno, one RTX 5090 (32 GB), `sm_120a`.
- Artifact: `/home/barista/ninfer/models/qwen3_8_27b_nvfp4.v3.ninfer`. The v2 sibling
  (`qwen3_8_27b_nvfp4.ninfer`) is rejected by the engine, which requires v3. **A comparison against
  a chain captured here needs this same artifact**: a different quantization or a different 27B
  artifact produces a different chain.
- Options: greedy (`temperature` 0, `top_k` 1), 32 requested output tokens, `max_context` 2048,
  `kv_capacity` 2048, CUDA graph off, context cache off.
- Stability: the plain chain was captured twice and reproduced on two further runs, identical each
  time. The MTP chain was captured twice, identical both times, and equals the plain chain: a greedy
  MTP round verifies and commits the same continuation the plain lane produces.

## Regenerating

```
NINFER_TP2_ACCEPT=1 NINFER_TP2_ARM=1 NINFER_TP2_CAPTURE=1 \
NINFER_TEST_ARTIFACT=/path/to/qwen3_8_27b_nvfp4.v3.ninfer \
  ./build/rtx5090-release/tests/ninfer_qwen3_5_tp2_accept_real_test
```

`NINFER_TP2_CAPTURE=1` prints the chain instead of comparing it. Add `NINFER_TP2_SPEC=mtp` for the
MTP lane. Regenerate only when the reference lane itself changed: a mismatch between hosts is a
result to report, not a reason to re-record.

## The MTP golden, and the binding defect it exposed

`NINFER_TP2_SPEC=mtp` at width 1 used to throw before producing a token:

```
mtp: native input requires one contiguous parent region
     (parts=4, part 1 begins 0 but part 0 ends 31457280, different parents)
```

The artifact binds the MTP block's four attention roles as contiguous ranges of ONE object
(`mtp/layer/attention/query_key_gate_value`), but the load path materializes each parameter's
binding as its own request, so the four views arrived in four parents and the MTP Program's fused
projection could not take them. The MTP block's dense MLP had the same shape: the artifact binds
gate|up as ranges of one object, and the Q8 role shape `N17408K5120` is not in the Q8 table.

The fix is at the load site (`src/models/qwen3_5/load.cpp`, `materialize_model`): the MTP block's
four attention roles and its gate|up pair are fused groups, so the gather puts each group back into
one parent with four (or two) contiguous views — the same mechanism the GDN q|k|v role already
uses. The chain below was captured with that fix; no other file changed.
