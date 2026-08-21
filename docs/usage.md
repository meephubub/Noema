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
- **Needle 2** — the engine lives in `prebuilt/needle/<platform>/`
  (`libneedle.*` + `needle.exe`). The `noema-needle` crate also searches
  `NEEDLE_LIB_PATH` and the shared `~/.cache/cactus-needle/` cache.
- **LiteRT-LM DLLs** — in `prebuilt/` (e.g. `litert-lm.dll`,
  `litert-lm.if.lib`). The `noema-native` build helper stages them next to
  every built executable automatically, so no `PATH` changes are needed.
- Windows only: `dumpbin` (from the MSVC toolchain) for inspecting the DLLs;
  the build works with MSVC (`x86_64-pc-windows-msvc`).

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
model, tools, approvals, memory, escalation, errors).

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
Cloud escalation is the phase-11 milestone (the `ModelProvider` abstraction
exists; no provider is wired yet).

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
approval/limits/logging/streaming sections plus `offline_mode`. Override via
`Noema::builder().with_config(..)` or the convenience setters
(`with_logging`, `with_approval_policy`, `with_escalation_policy`, …).

---

## 6. Implementing your own tool (step by step)

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

## 7. Testing your changes

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

## 8. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `NeedleError::EngineNotFound` | No `prebuilt/needle/<platform>/` library. Check `NEEDLE_LIB_PATH` or the cactus cache. |
| Linker can't find `litert-lm.lib` | `prebuilt/litert-lm.if.lib` must exist; `noema-native` stages the DLLs next to the exe. |
| Gemma image turn: "Vision executor should not be null" | Set `vision_backend: Some(Backend::Cpu)` in `GemmaOptions` (the default). |
| `gemma-example` slow to start | It loads the 2.5 GB model; that is expected. |
| Router/formatter returns "low confidence" | The engine is uncertain. Raise confidence by using the recommended phrasings in the examples, or tune `with_min_confidence`. |
| Tool call formats with wrong arguments | The engine only uses values evidenced in the input; rephrase the request to include the exact value. Absolute `path` arguments are extracted unreliably by the base engine — prefer the default search root. |
| Tests hang | A previous run left a test exe running (`taskkill //F //IM noema_core-*.exe`); event streams only end when their bus closes or you break on a terminating event. |

---

## 9. Layout

```
crates/
  noema-core/        runtime, sessions, model trait, router trait, tooling, approval wiring
  noema-api/         frontend-facing API
  noema-rig/         rig adapters (CompletionModel for any Noema model)
  noema-gemma/       Gemma 4 via vendored litert-lm-rust (multimodal, streaming, usage)
  noema-needle/      Needle 2 via its official C API
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
