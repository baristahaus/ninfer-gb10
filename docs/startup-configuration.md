# Startup configuration

`ninfer` and `ninfer-serve` share the TOML 1.0 configuration, the command-line overrides and the startup parameter display.
This document maintains the fields, the default values, the source rules and the resource semantics. Request and protocol behavior appear in
[CLI](cli.md) and [Serving](serving.md) respectively.

The [ninfer.example.toml](../ninfer.example.toml) in the project root is the complete reference. Its comments are in English,
and the comments list the alternative fields of mutually exclusive options. Git ignores other local TOML files:

```bash
cp ninfer.example.toml ninfer.local.toml
# Edit model.path and the desired settings.
./build/rtx5090-release/apps/ninfer --config ninfer.local.toml
./build/rtx5090-release/apps/ninfer-serve --config ninfer.local.toml --port 8081
```

Select one configuration file explicitly. You can also start entirely from the command line. By default the startup
shows the effective configuration and the sources of the active entry at INFO level, and fills in the actual resource values after
loading completes. There is no separate configuration print switch.

## 1. Configuration sources, types and resolution order

### 1.1 Sources and default values

```text
Shared defaults and explicit entry defaults
  → explicit configuration in the TOML file
  → explicit configuration on the command line
  → the owning module resolves model/device-related defaults
```

The CLI generation parameters form this request. The serve generation parameters are the defaults for HTTP requests:

```text
Model and final thinking mode default sampling values
  → generation defaults formed by the startup configuration
  → explicit fields in the current HTTP request
```

An omitted sampling field stays unset. A startup cannot override the model defaults with fixed values. Zero, false and the empty
list are actual values. The default seed keeps the existing entry behavior: 0 for the CLI and random per request for serve. The explicit
`random` value works on both entries, and the random seed is generated when the request is formed.

| Item | CLI entry default | Serve entry default |
|---|---|---|
| max_context | 8192 | 8192 |
| kv_capacity | final max_context | final max_context |
| max_concurrency | 1 | 1 |
| Output limit | 128 | 8192, the HTTP request can override |
| prefix_reuse | false | true |
| extra_state_slots | 0 | final max_concurrency |
| Host cache | 0 B | RTX 5090: 8 GiB + 8 native StateImage; Spark: 0 B |
| seed | 0 | random per request |

An entry default applies only to a field the user did not set or explicitly reset to the default. CLI `main()` uses the resolved resource settings
directly. It keeps the concurrency resources or caches that the configuration explicitly requires, and the CLI still submits only its own single request.
The public EngineOptions and the product's shared engine defaults use the same source. Their default context
and their explicit KV base value are both 8192. The product's rule for an omitted KV resolves after all overrides complete. A
KvCapacityPolicy supplied by a direct C++ caller still carries clear explicit/automatic semantics.

### 1.2 Value types and restoring defaults

In the tables of this document, `default`, `model`, `none`, `random`, `auto` and `context` are strings that the specified field
allows. They are not whole-file replacement rules:

- `default`: restores the entry/engine default policy of that field. It applies to the fields the table explicitly allows.
- `model`: removes the corresponding sampling or template override and leaves it to the model and the final prompt mode.
- `none`: explicitly disables a budget, backend or constraint. The field decides the exact meaning.
- `random`: generates the seed when the request is formed. It does not take one seed while reading the file for all requests to share.
- `auto`: uses the existing automatic selection policy for the KV capacity and the media preprocessing thread count.
- `context`: the KV capacity follows the final max_context and does not multiply by the concurrency.

A patch distinguishes an absent field, an explicit restore to the default, and a concrete value. A restore to the default is not treated as "continue inheriting the lower-priority
configuration". The same text in an ordinary string field is still text. For example, `"none"` in a regex is regex content.

A count is a non-negative or positive integer. Each field states its range. A floating-point parameter must be finite. A native TOML integer covers its representable
range. A uint64 seed above INT64_MAX uses a decimal string and keeps the full uint64 contract.

A capacity accepts a non-negative integer (bytes) or a string with a unit `B/KiB/MiB/GiB`. A duration accepts a non-negative integer
(milliseconds) or an `ms/s` string. Spaces may appear between the number and the unit. A fraction is converted by integer decimal arithmetic, and the
result must be a whole number of bytes/milliseconds that does not overflow the target type. A budget does not round up. Formatted display may use
readable units, and the structured record also keeps the exact integer.

### 1.3 Command line, lists and mode groups

- `--config` appears at most once. The relative position of the file and argv does not affect the source priority.
- The model positional argument appears at most once and overrides `model.path`. It can interleave with options. `--` ends option parsing.
- For an ordinary scalar from the same source, the last occurrence wins. The TOML syntax rejects a duplicate TOML key.
- A boolean item uses positive and negative switches and also accepts the positive form `--flag=true|false`. An optional boolean item
  that the table allows to restore its default also accepts `--flag=default` or `--flag=model`.
- The first occurrence of a list parameter at this layer replaces the lower-priority list. Later parameters with the same name append and keep their order.
  `--clear-stops` empties the three groups of CLI stop conditions. A stop parameter that appears afterwards rebuilds the list.
- `--prompt` / `--messages` form the input choice. One source can select only one. A higher-priority choice replaces
  the lower-priority input, and a prompt on the command line is not rejected just because the file has messages.
- A constraint is one choice with a kind. A command-line constraint replaces the file's whole constraint. Selecting
  different constraints in the same source reports an error. `--constraint none` clears the file constraint. Other kinds use the existing dedicated parameters.
- `--spec none` clears the lower-priority drafts/proposal settings and resolves to None/0/Full. The same source
  that explicitly declares none and also gives non-zero drafts or an optimized head reports an error. Enabling a backend is checked against the final K.
- thinking and CLI effort resolve together by source: the effort at this layer decides the thinking at this layer. A separate
  thinking choice overrides the lower-priority effort. An explicit conflict in the same layer reports an error. Request overrides on serve follow the protocol rules.

`--help` works at any option position. It prints and exits first, without reading the configuration or initializing the Engine.
A `--help` that appears as the value of a parameter such as `--prompt` is still content.

### 1.4 Paths and entry scope

A relative file path in the configuration is relative to the configuration file directory. An argv path is relative to the working directory
at startup. An empty string clears an optional file only where the table explicitly allows it. A string does not expand environment variables, shell syntax or expressions.

So that a message file starts stably with the configuration, a local relative media path inside CLI messages is uniformly
relative to the messages file directory. `send_to_serve` uses the same path rule.
HTTP(S), data URI and absolute paths keep their original meaning. The tool behavior of an explicit `--media-root` continues to override explicitly.
Ordinary strings in prompt, regex, grammar or JSON Schema text are not rewritten.

One TOML file can hold both CLI and serve configuration. Every field receives syntax, type and range checks. Required input,
mutual exclusion and resource consumption are checked for the active entry. A path of an unused entry is not opened. An unknown section/key reports an error,
and the error names the file, the field and the position.

## 2. All file fields and their command-line mapping

In the tables below, "default" means the behavior when the field is omitted. An explicit value in an example may override the entry default.
Unless marked otherwise, a field applies to both CLI and serve.

### 2.1 Model and engine

| TOML field | Command line | Type / default | Meaning and behavior |
|---|---|---|---|
| `model.path` | model positional argument | path; required | v3 `.ninfer` artifact |
| `model.chat_template` | `--chat-template` | path / `""` | a non-empty value overrides the artifact template; an empty value restores the artifact template |
| `model.vision` | `--vision` / `--no-vision` | bool / false | fixes the Vision weights and execution resources at startup; an HTTP request cannot enable it later |
| `engine.device` | `--device` | non-negative int / 0 | the local CUDA ordinal of the current single rank |
| `engine.max_context` | `--max-context` | positive uint32 / 8192 | the logical context limit of one request, including the processed input and the later generation |
| `engine.kv_dtype` | `--kv-dtype` | enum / `bf16` | `bf16/int8/fp8/nvfp4/k8v4`; the consumer decides which representation it supports |
| `engine.kv_capacity` | `--kv-capacity` | positive integer, `context`, `auto` / `context` | the token capacity of the shared Main KV; the model plan derives the other pools. Resource planning checks the context, the concurrency, the page alignment and the budget relation |
| `engine.prefill_chunk` | `--prefill-chunk` | a positive integer multiple of 128 or `default` / `default` | default is 4096 for Qwen4Exp/CPU expert offload and 1024 otherwise; the actual window is also limited by the context; `auto` and 0 are not accepted |
| `engine.cuda_graph` | `--cuda-graph` / `--no-cuda-graph` | bool / true | selects the existing Graph execution path; the native execution plan decides the Graph granularity |
| `engine.max_concurrency` | `--max-concurrency` | 1..8 / 1 | fixes the resident execution lanes at startup; decides the base capacity of the active State |
| `engine.max_pending_requests` | `--max-pending-requests` | positive uint32 / 16 | the waiting request allowance; together with the resident lanes it forms the outstanding limit |
| `engine.pending_timeout` | `--pending-timeout` | positive millisecond duration / `30s` | the waiting deadline before a request enters execution, following the existing admission/preparation/queue deadline; it does not limit the total generation duration |

The Engine/planner provides the exact automatic KV policy, and the configuration layer keeps that request.
The parser does not build a model capability table for the availability of Vision, speculative or a specific format.

### 2.2 Context and other memory

| TOML field | Command line | Type / default | Meaning and behavior |
|---|---|---|---|
| `context_cache.prefix_reuse` | `--prefix-reuse` / `--no-prefix-reuse` | bool or `default` / entry default | controls the reading and retention of history across requests; the pause/Replay mechanism stays available |
| `context_cache.extra_state_slots` | **`--extra-state-slots`** | non-negative uint32 or `default` / CLI 0, serve C | extra complete StateImage slots beyond the C active StateImages |
| `context_cache.host_capacity` | **`--host-cache-capacity`** | capacity or `default` / entry and target default | the shared total budget of the Host State/KV, the pause snapshots and the in-flight destination region; 0 B does not establish that backing |
| `context_cache.cost_presets` | `--context-cost-presets` | path / `""` | the existing context-cost JSON presets; a matching item overrides the existing cost parameter, and a missing item keeps the original value |
| `media.cache_capacity` | `--media-cache-capacity` | capacity / `1GiB` | the retention allowance for prepared media; 0 B disables retained media reuse |
| `media.live_capacity` | `--media-live-capacity` | positive capacity / `2GiB` | the allowance for all live BF16 patch payloads, including the retained cache; it is not added to the cache capacity again |
| `media.preprocess_threads` | `--media-preprocess-threads` | 1..64 or `auto` / `auto` | automatic selection by Host concurrency, at most 16; it keeps the internal 0 automatic encoding |

The specific rules of `context_cache.host_capacity`:

- The CLI default is 0 B. The RTX 5090 default of serve is
  `8 GiB + 8 * native_state_image_stride`, and its exact value requires the model geometry to be loaded.
- Spark uses the Device context, and its default is 0 B. The Context resource preparation reports an explicit non-zero value
  as unsupported, and the product parser checks only the value and the source.
- The budget covers the Host StateImage, the Host KV, the pause snapshots and the in-flight migration destination region. Native image/page alignment
  constrains the objects that actually fit, and the capacity cannot exceed the requested limit upward because of alignment.
- It does not limit the total process RAM. The CPU expert backing, the PLE preparation, the media, the grammar, the network and other
  pinned staging are each planned by their own owner.
- `prefix_reuse=false` does not mean these resources are released. The retained allowance is still available for request pause and resume.
- Setting the Host capacity to 0 keeps the Device checkpoint and the feasible Replay path. It does not disable
  CPU expert offload on that basis.

The specific rules of `extra_state_slots`:

- The total slot count of the StateImage main pool is `max_concurrency + extra_state_slots`. It is not one more per request.
- For example, with C=4 and extra=2 there are 6 complete images in total. With extra=0 there are still 4 active images.
- The model and the future ParallelPlan decide the fields and the byte count of one image. The GDN records, the workspace,
  the KV pages and the backend temporary region are charged independently and are not summarised by this integer.
- The Engine interface is `ContextCacheOptions::extra_state_slots`. The statistic fields that describe the physical total slot count and
  the current occupancy each keep their own meaning.

The configuration file covers all user parameters of CLI/serve. The
`purpose`, the startup callback, the model expert-priority array, the grammar cache capacity and
the MoE CPU team override inside the Engine keep their original call/model interfaces. These internal options are not part of the file protocol.

### 2.3 Speculative

| TOML field | Command line | Type / default | Meaning and behavior |
|---|---|---|---|
| `speculative.backend` | `--spec` | `none/mtp/dflash/dflash2` / `none` | selects one backend fixed at startup |
| `speculative.draft_tokens` | `--draft-tokens` | non-negative integer / 0 | 0 for none; 1..5 for MTP; 1..15 for DFlash/DFlash2 |
| `speculative.proposal_head` | `--proposal-head` | `full/optimized` / `full` | selects the proposal head; the parameter binding and the execution preparation check which representation actually works |

Switching a backend uses the final K, and a legal lower-priority K can stay. An explicit disable clears the
dependencies as described in section 1.3. The selected backend, K and proposal head all take part in the startup resource planning.

### 2.4 Generation defaults

| TOML field | Command line | Type / default | Meaning and behavior |
|---|---|---|---|
| `generation.max_tokens` | `--max-tokens` | positive uint32 or `default` / CLI 128, serve 8192 | the output budget of this CLI run; the client default of serve. It includes the inserted control tokens and is limited by the effective context headroom |
| `generation.thinking` | `--thinking` / `--no-thinking` | bool or `model` / `model` | the template thinking default; a serve request can override it, and section 1.3 describes the CLI effort relation |
| `generation.preserve_thinking` | `--preserve-thinking` / `--no-preserve-thinking` | bool or `model` / `model` | whether a later prompt keeps the reasoning of closed turns; the template interprets it |
| `generation.thinking_budget` | `--thinking-budget` | positive uint32 or `none` / `none` | the upper limit of the model-origin thinking tokens when thinking is on; the final stop and control suffix rules follow the Engine |
| `generation.sampling.temperature` | `--temperature` | [0,2] or `model` / `model` | 0 is greedy; a positive value follows the existing random sampling contract |
| `generation.sampling.top_k` | `--top-k` | integer [0,20] or `model` / `model` | 0 keeps the existing meaning and resolves to the current Top-20 limit |
| `generation.sampling.top_p` | `--top-p` | [0,1] or `model` / `model` | the cumulative probability cut of the global candidates |
| `generation.sampling.min_p` | `--min-p` | [0,1] or `model` / `model` | the filter threshold relative to the maximum candidate probability |
| `generation.sampling.presence_penalty` | `--presence-penalty` | [-2,2] or `model` / `model` | the penalty for whether a generated token has appeared |
| `generation.sampling.frequency_penalty` | `--frequency-penalty` | [-2,2] or `model` / `model` | the penalty for the occurrence count of a generated token |
| `generation.sampling.seed` | `--seed` | uint64, decimal string, `default`, `random` / `default` | the full uint64 range; the CLI default is 0 and the serve default is random per request |

`--temperature 0` selects greedy. The startup temperature of serve is the request default, and an explicit HTTP temperature takes priority.
HTTP parameters keep the value contract of each protocol. For example, an OpenAI request can set the output limit to 0.

The service default thinking budget can coexist with the disabled default thinking, for later requests that
enable thinking. A request that finally disables thinking does not apply the default budget. A CLI request that explicitly gives
a disabled thinking and a positive budget still reports a conflict. The configuration print does not render the prompt or the selected model sampling values early.

### 2.5 CLI input and output

The following fields apply only to `ninfer` and never become server-level injection into an HTTP request.

| TOML field | Command line | Type / default | Meaning and behavior |
|---|---|---|---|
| `cli.prompt` | `--prompt` | non-empty string / unset | one of prompt and messages; keeps the text as is |
| `cli.messages` | `--messages` | path / unset | the existing messages JSON array or an object with tools; section 1.4 resolves the media paths |
| `cli.reasoning_effort` | `--reasoning-effort` | `model/none/minimal/low/medium/high/xhigh/max` / `model` | template effort; none also disables thinking |
| `cli.raw_output` | `--raw-output` / `--no-raw-output` | bool / false | outputs the raw representation; its relation to the constraint follows the existing contract |
| `cli.print_token_ids` | `--print-token-ids` / `--no-print-token-ids` | bool / false | lists the generated IDs in the diagnostic output; it does not mix into the answer stdout |
| `cli.stop_token_ids` | `--stop-token-id`, repeatable | non-negative I32 array / [] | the model checks the actual token domain; it keeps the existing policy of the model default stop |
| `cli.stop` | `--stop`, repeatable | non-empty string array / [] | the answer/content stop string |
| `cli.reasoning_stop` | `--reasoning-stop`, repeatable | non-empty string array / [] | the reasoning stop string |

The constraint fields live in `[cli.constraint]`:

| Field | Command line | Type / default | Meaning and behavior |
|---|---|---|---|
| `kind` | `--constraint none` or the choices below | `none/grammar/json_object/json_schema/regex/choice` / `none` | one mutually exclusive choice; the dedicated fields of another kind must not appear at the same time |
| `grammar_file` | `--grammar-file` | non-empty path | required for kind=grammar; follows the GBNF/root contract |
| `json_schema_file` | `--json-schema-file` | non-empty path | required for kind=json_schema |
| No extra field | `--json-object` | kind=json_object | the constraint is a JSON object root |
| `regex` | `--regex` | string | required for kind=regex; the empty string is a valid empty-content regex and is not treated as unset |
| `choices` | `--choice`, repeatable | non-empty string array | required for kind=choice; an array element can be an empty string, and the text and the order are kept |

The constraint file is read when the CLI request is actually built. The existing request/Frontend contract verifies how custom stop conditions, raw output, tools and constraints are
compatible. The configuration layer does not write a second set of grammar capability checks.

### 2.6 Service and logging

| TOML field | Command line | Type / default | Meaning and behavior |
|---|---|---|---|
| `server.host` | `--host` | non-empty string / `127.0.0.1` | the listen address, serve-only |
| `server.port` | `--port` | [1,65535] / 8080 | the listen port, serve-only |
| `server.api_key` | `--api-key` | string / `""` | an empty value does not enable authentication; every configuration display and log is redacted |
| `server.model_id` | `--model-id` | string / `""` | an empty value uses the artifact public name; a non-empty value only changes the HTTP model alias |
| `server.cors` | `--cors` / `--no-cors` | bool / false | the existing permissive CORS behavior |
| `server.max_request_size` | `--max-request-size` | positive capacity / `384MiB` | the request body limit before JSON parsing |
| `server.responses.max_records` | `--response-store-max-records` | positive integer / 1024 | the record limit of the process-local Responses store |
| `server.responses.max_size` | `--response-store-max-size` | positive capacity / `256MiB` | the memory limit of the Responses store; it is unrelated to the context cache |
| `logging.level` | `--log-level` | `trace/debug/info/warning/error/critical/off` / `info` | the product diagnostic log level |
| `logging.stats_interval` | `--stats-interval` | millisecond duration / `5s` | the serve throughput statistic period; 0 disables the periodic output |
| `logging.request_jsonl` | `--request-log-jsonl` | path / `""` | the serve structured startup/request log; an empty value disables it |

### 2.7 Process control and environment

| Control item | Behavior |
|---|---|
| `--config FILE` | reads one TOML file; it is not a field of the file itself |
| `--help` / `-h` | prints the help grouped by function and exits |
| `--clear-stops` | empties the three groups of CLI stop conditions from the inheritance and this layer, then allows later stop parameters to build a list |
| `NINFER_VRAM_BUDGET_MIB` | keeps the existing positive-integer MiB VRAM budget limit; it is not a TOML or ordinary CLI field |
| `NINFER_CUDA_SYNC` | keeps `spin/blocking/yield/auto` with a default of spin; it is read before the DeviceContext initialization |

The configuration check verifies the types, ranges and source relations it owns and requires the required items of the active entry to be complete.
Environment controls take effect under the existing contract of their own owner. The display records their original setting and does not mix it into
the implicit override layer of the TOML.

## 3. The configuration display at the start of startup

### 3.1 The default startup display

A shared `StartupConfigurationView` and renderer group by model, engine, cache, speculative,
generation, media, server/CLI and logging. The fields print in a fixed order with aligned widths and human-readable
units. A long path wraps and is not truncated. Only a TTY uses color, and redirected text has no ANSI/progress control characters.

```text
configuration  entry              serve
               source             /home/neroued/ninfer/configs/flash-next.toml
model          artifact           /home/neroued/ninfer/out/qwen3_8_flash_next.ninfer
               vision             enabled
engine         context            32,768 tokens
               concurrency        2
               prefill chunk      4,096 tokens
               KV                 fp8 / 65,536 tokens
context cache  prefix reuse       enabled
               extra state slots  2  (4 total)
               host capacity      default (target/model)
speculative    backend            mtp / 3 drafts / full head
generation     temperature        model default
               seed               random per request
server         listen             127.0.0.1:8080
               authentication     disabled
```

The block above is only a layout example. The actual output covers every effective setting of the active entry and does not print only the non-default fields.
The source information distinguishes the default, the configuration file and the command line. The file path and the command-line override fields are clearly visible. Details of an unrelated entry
and of a disabled mode are not displayed as values in effect.

- On every startup: after a successful parse, initialize logging, print the INFO-level configuration first, then read the request resources,
  bind the service and create the Engine. The CLI uses stderr and serve uses the existing log output. The log levels
  warning/error/off can suppress the automatic INFO display.
- The API key shows only disabled/configured. The exact argv and the structured configuration are redacted in the same way.

### 3.2 Requested values and actual values

Before startup you can determine the sources, the merged values, the paths, the explicit capacities and the entry defaults. A derived value that needs no device can be
computed immediately, for example a total slot count of 4 for extra slots=2 and C=2.

Facts that remain undecided stay marked, such as `prefill_chunk=default`, `kv_capacity=auto`, the model sampling defaults, the native StateImage byte count,
the Host default and the Graph/weight capacity. After the Engine preparation completes, it consumes the actual
EngineOptions, LoadSummary and MemorySummary and prints the resolved summary, showing the actual chunk,
the KV/State/Host capacities and the resource planning result.

### 3.3 Structured logging

When request_jsonl is enabled, a normal startup writes one redacted `startup_config`
record before the heavy resource initialization. It holds the entry, the configuration path, the merged parameters, the sources and the relevant environment settings. After success, `server_start`
keeps the actual model and resource data. Update the version and the readers under the log schema convention.

Do not record only the argv of `--config FILE` and miss the actual values. The configuration layer keeps the requested values and the Engine
returns the actual values. The two are recorded separately, and an actual value is never written back to pollute the user configuration or the model default override state.

## 4. Implementation and ownership

`src/product/config/` maintains the shared parameters. `schema.*` defines the fields, the types, the ranges and the help.
`file.cpp` is the only implementation that contains the TOML parser. `options.cpp` handles argv, the layered overrides and the
typed startup object. `configuration.cpp` builds the read-only display view from the current object.

`LaunchOptions` holds one `EngineOptions`, the `GenerationDefaults`, the logging settings and the sources.
`CliOptions` adds the input, the output and the constraints. `ServeOptions` adds the HTTP/Responses settings.
The CLI and the GenerationService pass `launch.engine` directly to the Engine and bind their own startup observer.
The model and the resource planning continue to own the actual representation support, the device capacity and the model-related defaults. The product layer does not copy the capability checks.

`src/product/logging/configuration.*` owns the readable display, and the serve request log owns the JSONL.
`startup_config` stores the startup requested values and the sources, and `server_start` stores the actual facts after preparation.
The current request log schema is 31. `engine.context_cache.extra_state_slots` means the extra slots and
`total_device_state_slots` means the total slots. The throughput occupancy field stays independent.

The parser is fixed at [toml++ v3.4.0](https://github.com/marzer/tomlplusplus/releases/tag/v3.4.0),
MIT license. `third_party/tomlplusplus/` holds the unmodified upstream single header and the license.
`ninfer_product_config` privately depends on `ninfer::toml`, and only the product build uses it. The configuration headers, the Engine,
the models and the Ops do not expose TOML types, and the CMake configure does not download dependencies.
