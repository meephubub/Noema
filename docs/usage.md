# Noema — usage guide

Noema is the AI agent layer of Agora: a Rust-native runtime that coordinates
local models (Gemma 4 for reasoning, Needle 2 for routing and tool
formatting), tools, human approval, memory, and escalation behind one small
API. This guide covers how to build it, run it, and extend it.

---

## 1. What Noema is

Noema is built around a simple division of labour:

- **Gemma 4 thinks.** It is the reasoning model: complex requests,
  conversation, images, audio, and interpreting tool results. It runs
  locally through LiteRT-LM (the `litert-lm-rust` bindings are vendored in
  this workspace).
- **Needle 2 formats.** It is a 45M-parameter on-device model for routing and
  structured tool calls. One physical model is exposed as many *logical
  agents*: the text router, plus one formatter per tool.
- **Noema coordinates.** It owns sessions, events, routing, tool
  registration, risk evaluation, approval, escalation, and the agent loop.
- **Tools act.** Tools are independent Rust crates that implement one trait
  and register with the runtime. Nothing in the core changes when a tool is
  added.

Everything is local by default: Gemma runs on CPU, Needle is a self-contained
engine, and no cloud calls are made unless escalation is configured.

---

## 2. Prerequisites

- A Rust toolchain (stable, edition 2021, MSRV 1.75).
- **Gemma 4** — a `.litertlm` model file. By default Noema looks in
  `models/` for `gemma-4-E2B-it.litertlm`; override with the `NOEMA_GEMMA_MODEL`
  environment variable or an explicit builder path. *Only needed for the
  `gemma-example`, `router-example`, and `filesearch-example` (which degrade
  gracefully without it).*
- **Native binaries** — `noema-gemma` (via `litert-lm-rust`) and
  `noema-needle` automatically download a prebuilt archive from GitHub on
  first build and cache it at `~/.noema/prebuilt/`. No manual setup is
  needed for the default case (Windows x86_64 or macOS arm64). Override
  the cache location with `NOEMA_PREBUILT_DIR`.
- **Needle 2** — the engine is downloaded automatically. At runtime, the
  `noema-needle` crate searches `NEEDLE_LIB_PATH`, then the shared
  `~/.cache/cactus-needle/` cache, then `~/.noema/prebuilt/needle/`.
- **LiteRT-LM** — downloaded automatically. The build script links the
  platform-appropriate libraries (`.dll`/`.if.lib` on Windows, `.dylib` on
  macOS). On Windows, `noema-native` stages the DLLs next to executables;
  on macOS, an rpath is embedded so no `DYLD_LIBRARY_PATH` is needed.
- **Platform notes**:
  - Windows: builds with `x86_64-pc-windows-msvc`.
  - macOS: Apple Silicon (arm64) supported. Needle 2 runs via the
    `noema-needle-static` crate (links `libneedle.a` at build time).
    LiteRT-LM runs via `noema-gemma` with Metal acceleration.

---

## 3. Building and testing

```sh
cargo build --workspace
cargo test --workspace          # all unit + integration tests
cargo clippy --workspace --all-targets
```

Real-inference tests are `#[ignore]`d by default (they need the engines and
take real time):

```sh
cargo test -p noema-gemma --test gemma -- --ignored      # 8 tests vs. real Gemma 4
cargo test -p noema-router --test router_real -- --ignored
cargo test -p noema-router --test tool_real -- --ignored
```

---

## 4. The examples

Each example is a runnable binary that exercises a slice of the system
against the real engines (or degrades gracefully without them):

| Example | Command | What it shows |
| --- | --- | --- |
| Basic | `cargo run -p basic-example` | Session + model + streaming events |
| Gemma | `cargo run -p gemma-example` | Real multi-turn Gemma conversation: streaming, memory, usage, image/audio turns |
| Needle | `cargo run -p needle-example` | Real Needle 2 structured tool calls |
| Router | `cargo run -p router-example` | Needle routes simple actions, Gemma handles the rest |
| Tools | `cargo run -p tools-example` | Register tools; Gemma-facing summaries vs Needle schemas; format + execute |
| Filesearch | `cargo run -p filesearch-example` | Phase 7 chain: Gemma semantic request → Needle format → filesystem → result → Gemma |
| Approval | `cargo run -p approval-example` | Phase 8: risky call pauses for approve / reject / expire |
| Agent | `cargo run -p agent-example` | Phase 10: the full agent loop inside one `session.send` |
| Escalation | `cargo run -p escalation-example` | Phase 11: config-driven cloud provider (model/URL/key); graceful failure without a key |
| Multimodal | `cargo run -p multimodal-example` | Phase 12: mixed text/image, mixed text/audio, and image reasoning → tools on the real engine |

`gemma-example`, `router-example`, `filesearch-example`, and `agent-example`
load the 2.5 GB Gemma model and take a while to start. The other examples
need only the Needle engine (or nothing at all).

---

## 5. Core concepts

### 5.1 The runtime

Everything starts with a `Noema` runtime, built through a builder:

```rust
use noema_core::{init_logging, LogLevel, Noema};

let noema = Noema::builder()
    .with_model(gemma_model)            // reasoning model (e.g. GemmaModel)
    .with_router(router)                // initial text router (NeedleRouter)
    .with_tools(registry)               // registered tools
    .with_tool_formatter_for("search_files", formatter)  // per-tool Needle agent
    .with_approval_policy(policy)       // risk/approval gating
    .build()
    .await?;
```

### 5.2 Sessions, events, and conversation state

Sessions own the ephemeral conversation: each `send` appends the user
message, runs the agent loop, and commits the turn (including any tool
steps) back into the session's history. Models are **request-driven** — every
call carries the full transcript, so multi-turn memory works for any model
backend, not just Gemma. (The rig adapter forwards full chat history by
default for the same reason.)

A session is an ephemeral unit of agent state (conversation, pending work).
Sessions are created and closed on the runtime; everything a session does is
published as an event:

```rust
let session = noema.create_session().await?;
let mut events = session.events();

session.send(Message::text(Role::User, "Open my flashcards")).await?;

while let Some(event) = events.next().await {
    match event {
        Event::RoutingCompleted { .. } => { /* handled by Needle, model never ran */ }
        Event::ModelDelta { delta, .. } => { /* stream a token */ }
        Event::ToolApprovalRequired { .. } => { /* ask the user */ }
        _ => {}
    }
}
```

The frontend subscribes and reacts; it never polls for agent state. The full
event vocabulary is in `crates/noema-events` (session lifecycle, routing,
model, tools, approvals, memory, escalation, errors, and the content-free
`ModelMetrics` / `ToolMetrics` / `EscalationMetrics` observability events).

`send` returns a `SendOutcome`:

- `SendOutcome::Routed(action)` — the router handled it; the model never ran.
- `SendOutcome::Model(response)` — the request reached the model
  (escalated or not routed), streamed through the bus, and drained into a
  complete `ModelResponse`.

### 5.3 The agent loop

`send` runs the full agent loop (Phase 10) automatically:

```text
user message
    ↓
model turn (streamed: ModelStarted / ModelDelta / ModelCompleted)
    ├── reply names a registered tool → ToolRequested
    │     → per-tool Needle formatter → ToolFormatted
    │     → risk gate / approval → ToolStarted → ToolCompleted
    │     → result fed back (Role::Tool message) → next model turn
    └── no tool mentioned → the reply is the final answer
```

- **Tool-intent detection** is deliberately simple: a reply naming a
  registered tool (or its crate's short name, e.g. `filesearch` for
  `noema-filesearch`) as a whole word is treated as a semantic request.
- The **dynamic Gemma tool summaries** are injected as the request's system
  prompt while tools are registered, so the model knows what it can call.
- The loop is bounded by `LimitsConfig::max_agent_iterations` (default 20)
  and `LimitsConfig::max_tool_calls` (default 50).
- **Failure recovery**: if the formatter cannot serve the reply (e.g. the
  model named a tool while declining), the model's reply becomes the final
  answer and an `Error` event is published — the send does not fail.
  Rejected approvals and genuine tool failures abort the send.

### 5.4 Models

All models implement the `Model` trait in `noema-core` (text, images, audio,
streaming, system prompts, cancellation, usage, escalation requests). Gemma
sits behind `noema-gemma`; a `Model for Arc<M>` blanket impl lets you
register a shared handle. Any Noema model can also drive a rig agent through
the `CompletionModel` adapter in `noema-rig`.

### 5.5 Routing

Every plain-text user request is offered to the registered router first
(`Router` trait). `NeedleRouter` (in `noema-router`) uses Needle 2 with the
six default Agora actions; it acts only at or above a confidence threshold
(default 0.6) and escalates everything else to the reasoning model.
Multimodal and non-user turns always skip routing.

### 5.6 Escalation

When the router escalates — or a model requests escalation — an
`EscalationPolicy` decides what happens: escalate to the local model
(`Local`), escalate to a cloud provider (`Cloud`), or deny. The default
policy escalates locally and never to the cloud; `offline_mode` always wins.

Cloud escalation (Phase 11) is a general abstraction. Providers implement
`ModelProvider` and are registered on the runtime; the core never
hard-codes one:

```rust
use noema_api::prelude::*;

// Model name, base URL, and API key — the whole provider configuration.
let provider = OpenAICompatibleProvider::new(
    "openai",                             // provider id (policy's preferred_provider)
    "gemini-2.5-pro",                     // model name
    "https://generativelanguage.googleapis.com/v1beta/openai", // base URL
    std::env::var("GEMINI_API_KEY").ok(), // API key (None for local endpoints)
);

let noema = Noema::builder()
    .with_provider(provider)
    .with_escalation_policy(EscalationPolicy {
        allow_local: false,
        allow_cloud: true,
        preferred_provider: Some("openai".into()),
        maximum_latency: Some(std::time::Duration::from_secs(30)),
        ..EscalationPolicy::default()
    })
    .build()
    .await?;
```

With several providers registered, the policy's `preferred_provider` picks
one; a single provider is used automatically. When the policy chooses
`Cloud`, the session resolves the provider, enforces the per-request budget
(`LimitsConfig::max_cloud_escalations`) and `maximum_latency`, streams the
provider's turn under its own model id (`ModelStarted` / `ModelDelta` /
`ModelCompleted`), and feeds the answer back so the **local agent
continues** — for both router escalations (Needle → cloud) and mid-loop
model escalations (Gemma → cloud). The same three fields live in
`NoemaConfig::cloud` (`model`, `base_url`, `api_key`) for
configuration-file-driven setups; `noema-api` re-exports
`OpenAICompatibleProvider`.

`OpenAICompatibleProvider` speaks the OpenAI chat-completions protocol, so
it reaches Gemini (OpenAI-compatible endpoint), OpenAI, Ollama, vLLM,
LocalAI, and friends; `with_streaming(true)` switches on SSE streaming, and
every request honours the session's cancellation token. Cost limits
(`maximum_cost`) remain policy fields awaiting provider-reported pricing;
latency limits are enforced as a timeout. Without an API key, `examples/escalation`
demonstrates the wiring and fails gracefully with a clear escalation error
(no real-endpoint test ships).

### 5.7 Tools

A tool is any type implementing `NoemaTool`:

```rust
#[async_trait]
impl NoemaTool for MyTool {
    fn metadata(&self) -> ToolMetadata { /* name, crate, description, risk */ }
    fn schema(&self) -> ToolSchema { /* the full Needle-facing schema */ }
    async fn execute(&self, call: ToolCall) -> Result<ToolResult> { /* do it */ }
}
```

Register it and the registry derives everything else:

```rust
let mut registry = ToolRegistry::new();
registry.register(MyTool)?;

registry.gemma_tool_section();   // what Gemma sees: name + crate + one-liner, no schema
registry.needle_tools_json();    // what the tool Needle agents bind to: full schemas
registry.validate_call(&call)?;  // schema validation before anything runs
registry.execute(call).await?;   // validated execution
```

The registry keeps three views of each tool so the reasoning model's context
stays small while Needle gets the complete schemas.

### 5.8 Tool formatting (tool-specific Needle agents)

The reasoning model never produces the final structured schema — that is the
tool's logical Needle agent. `NeedleToolFormatter` (in `noema-router`) binds
one engine instance per tool (its schema plus any instructions) and turns a
semantic request into a validated `ToolCall`:

```rust
let formatter = NeedleToolFormatter::from_tool(&schema, None)?;
let noema = Noema::builder()
    .with_tool_formatter_for("search_files", formatter)
    .build().await?;

let call = session.format_tool(schema, "search for notes.txt").await?;
// → ToolCall { tool: "search_files", arguments: { "query": "notes.txt" } }
let result = session.execute_tool(call).await?;
```

Two engine behaviours worth knowing (verified against the real model):

- The engine's schema parser is **key-order sensitive** — `name` must
  serialize before `description` in tool JSON. The registry and schema
  already handle this (`serde_json` is built with `preserve_order`).
- The formatter's default confidence gate (0.15) is much lower than the
  router's (0.6): the engine's confidence head is routing-tuned, so absolute
  values for formatting tasks run far lower, and the call is schema-validated
  before execution anyway. A refusal (empty call list) is the engine's
  escalation signal.

### 5.9 Human approval

Each tool declares a `RiskLevel` (`None < Low < Medium < High < Critical`).
The session's `ApprovalPolicy` decides when a call must pause:

- `require_approval_above: Some(RiskLevel::High)` (the default) → calls at
  `High` and above pause.
- `None` → only `Critical` always pauses.
- `Critical` always requires approval regardless of the threshold.
- The approval flow can be disabled entirely (`NoemaConfig::approval.enabled
  = false`).

When a call requires approval, `execute_tool` publishes
`ToolApprovalRequired` and waits. The frontend sees the full proposal via
`session.pending_approvals()` — tool, description, arguments, risk, expiry —
and answers:

```rust
let pending = session.pending_approvals();
let id = pending[0].id.clone();
session.approve_tool(id)?;   // or session.reject_tool(id)?;
```

Approved calls execute; rejected calls error with an approval error and
never run; undecided calls expire after the policy timeout (the request is
removed from the store so it can't be approved late).

The approval vocabulary is re-exported through `noema-api`, so the frontend
only ever imports that one crate: `ApprovalId`, `ApprovalRequest`,
`ApprovalPolicy`, `ApprovalDecision`, plus the tool types
(`ToolCall`, `ToolResult`, `ToolRegistry`, `NoemaTool`, …) for driving
`format_tool` / `execute_tool` directly.

### 5.10 Configuration

`NoemaConfig` is strongly typed with sensible defaults (see
`crates/noema-core/src/config.rs`): gemma/needle/cloud/memory/tools/risk/
approval/limits/logging/streaming sections plus `offline_mode`. The `cloud`
section carries `enabled`, `preferred_provider`, and the provider's three
fields — `model`, `base_url`, `api_key` — plus `maximum_cost` and
`maximum_latency_ms` (enforced as the escalation timeout). Override via
`Noema::builder().with_config(..)` or the convenience setters
(`with_logging`, `with_approval_policy`, `with_escalation_policy`,
`with_provider`, …).

### 5.11 Observability

Every runtime keeps content-free metrics (Phase 13). A point-in-time
snapshot is available at any time:

```rust
use noema_api::prelude::*;

// After some activity:
let metrics: MetricsSnapshot = noema.metrics();
println!("model turns: {}", metrics.total_model_turns());
println!("input tokens: {}", metrics.total_input_tokens());
println!("tool calls: {} ({} failed)", metrics.total_tool_calls(), metrics.total_tool_failures());
println!("cloud escalations: {}", metrics.cloud_escalations());
// Per-model / per-tool breakdowns:
// metrics.models.get("gemma")  → ModelMetrics { turns, input_tokens, output_tokens, latency_ms }
// metrics.tools.get("search_files") → ToolMetrics { calls, failures, latency_ms }
```

The same numbers stream live on the event bus as `ModelMetrics`,
`ToolMetrics`, and `EscalationMetrics` events (per model turn, per tool
call, per escalation), and every record emits a content-free `tracing`
debug line. Observability is **privacy-aware by design**: metrics and logs
carry identifiers, counts, latencies, and token totals — never message
content.

### 5.12 Multimodal input

Text, image, and audio travel as one ordered message — `ContentPart::Text`
/ `Image` / `Audio`, in any combination — and multimodal user turns skip
the text router, going straight to Gemma:

```rust
use noema_api::prelude::*;

let message = Message::new(Role::User, vec![
    ContentPart::text("What color is this image?"),
    ContentPart::image(png_bytes, "image/png"),
]);
// ...also ContentPart::audio(wav_bytes, "audio/wav"), or all three.
```

The agent loop treats multimodal turns like any other: an image turn's
reasoning can name a tool, which is then formatted and executed inside the
same `session.send`. The current Gemma 4 E2B checkpoint has a working
vision channel but no audio channel: image turns answer directly, while
audio turns are accepted and declined gracefully. `examples/multimodal`
shows all three paths on the real engine.

### 5.13 Production hardening

Phase 14 tightened the runtime; the knobs live in `LimitsConfig` and
`NoemaConfig`:

- **Resource limits.** `max_agent_iterations` and `max_tool_calls` bound
  the agent loop; `max_cloud_escalations` bounds cloud calls per request;
  `max_response_tokens` is forwarded to the model as `max_tokens`;
  `max_context_tokens` trims the oldest transcript messages before every
  model request (estimated ~4 chars/token; the current turn is always
  kept); `max_tool_execution_seconds` times out runaway tools; and
  `max_concurrent_tools` caps concurrent tool execution via a semaphore.
  `max_tool_call_depth` covers nested tool calls, which the current
  sequential loop does not produce.
- **Robust cancellation.** `session.cancel()` cancels the in-flight model
  turn (propagated to models and cloud providers); timed-out tool calls are
  dropped, aborting their async work. Every limit above surfaces a
  strongly-typed error rather than hanging.
- **Concurrency controls.** Concurrent `session.send` calls on one session
  are serialized (the transcript and cancellation token are shared state),
  so sends queue instead of racing.
- **Schema validation.** Every model-generated tool call is validated
  against the registered tool's schema before execution — never execute an
  unvalidated call (see §5.7).
- **Prompt-injection defences.** Tool results are fed back delimited
  (`<tool_result>…</tool_result>`) and explicitly framed as *data, not
  instructions*; the agent system prompt and the cloud escalation prompt
  both state the trust boundary (user content, tool output, and memory are
  data to reason about, never instructions to follow).
- **Privacy-aware by design.** Local-first by default; cloud escalation is
  opt-in and `offline_mode` always wins; telemetry never records message
  content (§5.11).
- **API stability.** The public surface (`noema-api`) is additive: new
  events and metrics are added as new variants/fields rather than changing
  existing ones, so consumers can match with `..`/`_`.

Memory policy (Mnemo) is intentionally deferred: nothing is persisted
until Mnemo lands, so there is no memory to leak or review yet. When Mnemo
arrives, its extraction policy belongs to that crate.

---

## 5.13 The Needle→Gemma bridge

`noema-bridge` provides a two-tier inference session: Needle 2 with 5 stub
tools runs first for fast, deterministic tool dispatch; when confidence is
below a configurable threshold (default 0.6) or when Needle refuses, the
same prompt is forwarded to Gemma 4 for full reasoning.

```rust
use std::sync::Arc;
use noema_api::prelude::*;

let noema = Noema::builder()
    .with_model(gemma_model)
    .build()
    .await?;
let session = noema.create_session().await?;

// Or use the bridge directly:
let bridge = noema_bridge::BridgeSession::from_default(
    noema_bridge::BridgeConfig {
        min_confidence: 0.6,
        needle_max_tokens: 256,
        ..Default::default()
    },
)?
.with_gemma(Arc::new(gemma_model));

let outcome = bridge
    .send(
        Message::text(Role::User, "search for rust docs"),
        CancellationToken::new(),
    )
    .await?;
```

The 5 stub tools are: `search`, `calculate`, `translate`, `summarize`,
and `navigate`. They return basic canned results and can be replaced with
real implementations via `BridgeSession::with_tools()`.

`BridgeSession`, `BridgeConfig`, and `stub_registry` are re-exported
through `noema_api::prelude`.

---

## 6. Building, publishing, and using the crates

See [`docs/publishing.md`](publishing.md) for the complete guide: how to
build the workspace, publish the crates to crates.io (bottom-up order,
`--dry-run` checks, native-artifact caveats), and depend on them from your
own projects — either the quick `noema-api` path or the full local-engine
setup.

---

## 7. Implementing your own tool (step by step)

1. **Create a crate.** `noema-<tool>` (e.g. `noema-filesearch`), depending
   on `noema-tools` (+ `async-trait`).
2. **Implement `NoemaTool`.** Provide `metadata` (name, crate, description,
   risk), `schema` (the full JSON-Schema `parameters` object), optional
   `needle_instructions`, and `execute`.
3. **Register it.** `registry.register(MyTool)?` — at app startup, alongside
   any other tools. No core changes.
4. **Bind its Needle agent** (optional but recommended): one
   `NeedleToolFormatter` per tool, registered with
   `with_tool_formatter_for(tool_name, formatter)`.
5. **Use it.** Gemma sees the one-line summary in `gemma_tool_section()`;
   the tool's formatter turns semantic requests into validated calls;
   `session.execute_tool` gates on risk, asks for approval when needed, and
   streams `ToolStarted`/`ToolCompleted`/`ToolFailed` events.

`examples/filesearch` and `crates/noema-filesearch` are the reference
implementations; `examples/tools` shows the registration views.

---

## 8. Testing your changes

- **Unit tests** live next to the code (e.g. `crates/noema-tools`, the
  session tests in `crates/noema-core`). They use fake models, fake engines,
  and temp directories — no engines needed.
- **Real-inference tests** are `#[ignore]`d integration tests against the
  actual Needle engine / Gemma model (see section 3). Model output is
  probabilistic, so they assert structure (right tool name, required
  arguments present, refusal on unsupported input), not exact text.
- New tools should ship with unit tests covering: valid calls, invalid calls
  (missing required args), no-match results, and the result cap.

---

## 9. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `NeedleError::EngineNotFound` | No `prebuilt/needle/<platform>/` library. Check `NEEDLE_LIB_PATH` or the cactus cache. |
| Linker can't find `litert-lm.lib` | `prebuilt/litert-lm.if.lib` must exist; `noema-native` stages the DLLs next to the exe. |
| Gemma image turn: "Vision executor should not be null" | Set `vision_backend: Some(Backend::Cpu)` in `GemmaOptions` (the default). |
| `gemma-example` slow to start | It loads the 2.5 GB model; that is expected. |
| macOS: dylib not found at runtime | Ensure all dylibs are in the same directory as the executable, or that `DYLD_LIBRARY_PATH` includes `prebuilt/macos/`. The build script embeds `@executable_path` rpath automatically. |
| macOS: `litert-lm.dylib` not found at link time | Download the C API from Google's [LiteRT-LM release](https://github.com/google-ai-edge/LiteRT-LM/releases) and place in `prebuilt/macos/`. |
| Router/formatter returns "low confidence" | The engine is uncertain. Raise confidence by using the recommended phrasings in the examples, or tune `with_min_confidence`. |
| Tool call formats with wrong arguments | The engine only uses values evidenced in the input; rephrase the request to include the exact value. Absolute `path` arguments are extracted unreliably by the base engine — prefer the default search root. |
| Tests hang | A previous run left a test exe running (`taskkill //F //IM noema_core-*.exe`); event streams only end when their bus closes or you break on a terminating event. |

---

## 10. Layout

```
crates/
  noema-core/        runtime, sessions, model trait, router trait, tooling, approval wiring
  noema-api/         frontend-facing API
  noema-rig/         rig adapters (CompletionModel for any Noema model)
  noema-gemma/       Gemma 4 via vendored litert-lm-rust (multimodal, streaming, usage)
  noema-needle/      Needle 2 via its official C API (DylibEngine, CliEngine)
  noema-needle-static/  Needle 2 statically linked (StaticEngine, macOS/Linux)
  noema-router/      initial text router + tool-specific Needle formatter
  noema-tools/       NoemaTool trait, registry, schemas, metadata, risk levels
  noema-filesearch/  the reference tool (read-only filesystem search)
  noema-approval/    approval requests, policy, store
  noema-events/      event definitions + broadcast bus
  noema-native/      build helper that stages LiteRT-LM DLLs
  litert-lm-rust/    vendored Gemma 4 runtime bindings
examples/
  basic, gemma, needle, router, tools, filesearch, approval, agent
prebuilt/            native engines (Needle 2, LiteRT-LM DLLs)
models/              Gemma model file (gitignored)
```

See `plan.md` for the full architecture and the phase roadmap (through
phase 14: observability, hardening).
