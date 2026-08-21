# Noema

The AI agent layer of Agora.

Noema is a Rust-native AI agent runtime built around Gemma 4, Needle 2, Rig, and Mnemo.

It provides Agora with an intelligent interface capable of understanding requests, reasoning over context, using tools, interacting with persistent memory, processing multimodal input, and escalating difficult tasks to larger models.

**Documentation: [Usage guide](docs/usage.md)** — build, run, extend, and troubleshoot Noema.

Architecture

                    ┌──────────────┐
                    │    Agora     │
                    │   Frontend   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │    Noema     │
                    │ AI Agent     │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
           Noema        Noema        Mnemo
           Needle       Gemma        Memory
              │            │
              ▼            ▼
           Needle 2     Gemma 4
                         │
                         ▼
                    Cloud Models

The Models

Gemma 4

Gemma 4 is Noema’s primary reasoning model.

It handles:

* Complex requests
* Reasoning
* Conversation
* Images
* Audio
* Tool selection
* Tool results
* Multistep tasks
* Cloud escalation

Gemma 4 runs locally through litert-lm-rust⁠￼.

Needle 2

Needle 2 is Noema’s lightweight model for tasks where a full reasoning model isn’t necessary.

For text requests, Needle first determines whether the request is a simple Agora action:

"Open my flashcards"
        ↓
     Needle 2
        ↓
  OpenFlashcards

If it cannot handle the request, it escalates to Gemma 4.

Needle is also responsible for turning Gemma’s semantic tool requests into validated structured tool calls.

Gemma:
    "find the file abc.exe"
        ↓
Needle:
    search_files({
        query: "abc.exe"
    })
        ↓
Noema:
    validate → approve → execute

Only one Needle model exists. Noema creates logical tool-specific Needle agents, each with the schema and instructions for its respective tool.

Tools

Noema uses a modular crate-based tool architecture.

Tools are independent Rust crates named:

noema-<tool>

Examples:

noema-filesearch
noema-flashcards
noema-pdf
noema-notes

A tool crate provides:

* Tool metadata
* Tool description
* Needle schema
* Risk level
* Execution implementation

Gemma receives only lightweight information about available tools:

filesearch — noema-filesearch
flashcards — noema-flashcards
pdf — noema-pdf

The complete schemas are given only to the corresponding Needle tool agent. This keeps Gemma’s context small while still allowing precise structured tool execution.

Adding a tool should not require modifying Noema’s core agent loop.

Human Approval

Tools can declare their risk level in their schema.

For higher-risk operations, Noema pauses execution and sends the complete proposed tool call to Agora:

Gemma
  ↓
Needle
  ↓
Tool Call
  ↓
Risk Check
  ↓
User Approval
  ↓
Execution

The user can see exactly what will happen before approving it.

Memory

Noema does not implement its own persistent memory.

Instead, it uses Mnemo as Agora’s memory layer.

Noema
  │
  ├── Retrieve relevant memories
  │
  └── Store useful new memories
             │
             ▼
           Mnemo

Noema manages ephemeral agent and session state; Mnemo manages persistent knowledge and memory.

Multimodal

Noema is designed around Gemma 4’s multimodal capabilities.

Audio

In voice mode, audio is provided directly to Gemma 4 rather than passing through a separate transcription layer.

System Prompt
      ↓
Context / Text
      ↓
Audio
      ↓
Gemma 4

Images

Images can be provided directly to Gemma alongside text and context.

This enables workflows such as:

"What does this diagram mean?"
        +
      Image
        ↓
     Gemma 4

Multimodal requests can still lead to tool calls when necessary.

Model Escalation

Noema supports escalation from both local models to larger cloud models.

Local Model
    ↓
Task too difficult
    ↓
Cloud Provider
    ↓
Larger Model
    ↓
Local agent continues

The cloud layer is provider-agnostic: the runtime registers abstract
`ModelProvider`s (`Noema::builder().with_provider(..)`) and the escalation
policy decides where a request goes (`Local` / `Cloud` / `Denied`). The
bundled `OpenAICompatibleProvider` (`crates/noema-provider-http`) reaches
Gemini, OpenAI, Ollama, vLLM, and any OpenAI-compatible endpoint from just
three fields — model name, base URL, and API key — which also live in
`NoemaConfig::cloud`. Escalations are bounded by a per-request budget and
the policy's latency limit, and the cloud answer is fed back so the local
agent continues.

Cloud escalation can be disabled for fully local/offline operation
(`NoemaConfig::offline_mode` / `EscalationPolicy::offline_only`).

Rust API

Noema exposes an ergonomic Rust API for Agora.

Conceptually:

let noema = Noema::builder()
    .with_gemma(gemma)
    .with_needle(needle)
    .with_memory(mnemo)
    .build()
    .await?;
let session = noema.create_session().await?;
session.send(message).await?;

Agent activity is exposed through an event stream:

let mut events = session.events().await?;
while let Some(event) = events.next().await {
    // Handle agent events
}

Events can represent:

* Model generation
* Tool requests
* Tool execution
* Approval requests
* Memory operations
* Escalation
* Streaming responses
* Errors

This allows Agora to present the agent’s state in real time without polling.

Design Philosophy

Noema is built around a simple separation of responsibilities:

Gemma thinks.
Needle formats.
Noema coordinates.
Mnemo remembers.
Tools act.

The result is an agent architecture that keeps expensive reasoning focused on the tasks that actually require it, while using a lightweight model for routing and precise tool execution.

Status

Noema is under active development. Phases 1–8 and 10 of the plan are implemented (Phase 9, Mnemo, is deferred until Mnemo itself is ready):

* Phase 1 — workspace foundation, core crates, sessions, events.
* Phase 2 — model abstractions: messages, multimodal content parts, streaming
  responses, cancellation, provider abstraction.
* Phase 3 — Gemma 4 through a vendored litert-lm-rust (`crates/litert-lm-rust`):
  token streaming, multi-turn memory, usage metadata, system prompts, and
  image/audio input (`crates/noema-gemma`). Images are verified end-to-end;
  audio flows through but the current checkpoint has no audio channel.
* Phase 4 — Needle 2 through its official C API (`crates/noema-needle`):
  structured tool calls, multi-turn conversation, refusal, and a CLI fallback.
* Phase 5 — the initial text router (`crates/noema-router`): every plain-text
  request is offered to Needle 2 first; simple application actions are routed
  (the model never runs), anything else escalates to the model, and the
  runtime emits `RoutingStarted`/`RoutingCompleted`/`RoutingEscalated` events.
* Phase 6 — tool infrastructure (`crates/noema-tools` + `crates/noema-router`):
  the `NoemaTool` trait, `ToolRegistry`, `ToolSchema`/`ToolMetadata`/risk
  levels, dynamic Gemma tool summaries (schema-free), complete Needle schemas,
  per-tool logical Needle agents (`NeedleToolFormatter`), and session
  `format_tool`/`execute_tool` with tool events. A third-party `noema-*`
  crate can register a tool without touching the core agent loop.
* Phase 7 — the first real tool (`crates/noema-filesearch`): a bounded,
  read-only `search_files` tool with schema, risk classification, and
  execution, demonstrated end-to-end in `examples/filesearch` (semantic
  request → Needle format → filesystem → result → Gemma).
* Phase 8 — human approval (`crates/noema-approval`): risk-gated execution
  with `ToolApprovalRequired` events, `session.approve_tool`/`reject_tool`,
  and timeouts. `examples/approval` shows approve, reject, and expire paths.
* Phase 10 — the full agent loop inside `session.send`: model turn → tool
  intent → Needle formatting → risk/approval → execution → result fed back,
  bounded by iteration/tool-call limits, with the dynamic Gemma tool
  summaries injected as the system prompt. Sessions own the conversation,
  so multi-turn memory works for any model. `examples/agent` runs it against
  the real engines.
* Phase 11 — cloud escalation as a general abstraction: the runtime
  registers abstract `ModelProvider`s and `EscalationDecision::Cloud` is
  fully wired (budget + latency limits, streamed provider events, result
  fed back for the local agent to continue). `crates/noema-provider-http`
  ships `OpenAICompatibleProvider` — configured by model name, base URL,
  and API key (also in `NoemaConfig::cloud`) — with optional SSE streaming
  and cancellation. `examples/escalation` shows the wiring and fails
  gracefully without an API key.
* Phase 12 — the multimodal agent: text, image, and audio travel as one
  ordered message; multimodal turns skip the router and flow straight to
  Gemma, whose reasoning can drive tool calls inside the loop
  (`examples/multimodal` runs mixed text/image and text/audio on the real
  engine).
* Phase 13 — observability: `Noema::metrics()` returns a content-free
  `MetricsSnapshot` (per-model turns/tokens/latency, per-tool
  calls/failures/latency, escalation counts), and the same numbers stream
  as `ModelMetrics` / `ToolMetrics` / `EscalationMetrics` events. Telemetry
  never records message content.
* Phase 14 — production hardening: all resource limits enforced (context
  trimming, response-token clamp, tool execution timeouts, tool
  concurrency cap, loop/cloud budgets), serialized per-session sends,
  prompt-injection defences (delimited tool results framed as data, explicit
  trust boundaries), content-free telemetry, and opt-in cloud with
  `offline_mode` always winning.
* Rig integration (`crates/noema-rig`) — any Noema model drives rig agents
  through a `CompletionModel` adapter (full chat history forwarded by
  default).

Running locally:

```sh
# Gemma 4 (needs models/gemma-4-E2B-it.litertlm; DLLs are staged automatically)
cargo run -p gemma-example       # streaming, memory, usage, image + audio turns

# Needle 2 (engine lives in prebuilt/needle/)
cargo run -p needle-example

# Initial text router: Needle routes simple actions, Gemma handles the rest
cargo run -p router-example

# Tool infrastructure: register tools, Needle formats, Noema executes
cargo run -p tools-example

# First real tool: Gemma → Needle format → filesystem → result → Gemma
cargo run -p filesearch-example

# Human approval: risky tool calls pause for approve / reject / expire
cargo run -p approval-example

# Full agent loop: model → tool → result → answer inside one session.send
cargo run -p agent-example

# Cloud escalation: router → provider (configured by model, URL, API key)
cargo run -p escalation-example

# Multimodal: mixed text/image, mixed text/audio, image reasoning → tools
cargo run -p multimodal-example
```

Both engines are local: Gemma runs on CPU via LiteRT-LM, Needle is a
self-contained 45M-parameter engine. No cloud calls are made unless cloud
escalation is explicitly enabled and a provider is registered (the default
policy is local-only, and `offline_mode` always wins).

### macOS (Apple Silicon)

Noema runs on macOS arm64. The build script automatically links the correct
platform libraries and embeds an rpath so dylibs are found at runtime.

**Gemma**: place the macOS LiteRT dylibs in `prebuilt/macos/` and the main
C API library (`litert-lm.dylib`) alongside them. The C API library is
available from Google's [LiteRT-LM v0.16.0+ release](https://github.com/google-ai-edge/LiteRT-LM/releases)
in `litert_lm_c_api-0.1.0.zip` or `CLiteRTLM_mac.xcframework.zip`.
The model file (`models/gemma-4-E2B-it.litertlm`) is platform-independent.

**Needle 2**: the `noema-needle-static` crate links `libneedle.a` at build
-time (Cactus Compute ships static libraries for macOS arm64 on HuggingFace).
Place `libneedle.a` and `needle.h` in `prebuilt/needle/macos-arm64/`.

Building, publishing, and consuming the crates is covered in
[`docs/publishing.md`](docs/publishing.md): build the workspace, publish
bottom-up to crates.io, or depend on `noema-api` from your own projects.

The project is being built entirely in Rust with a focus on:

* Local inference
* Low latency
* Minimal context usage
* Modular tools
* Multimodal interaction
* Persistent memory
* Human-in-the-loop execution
* Model-agnostic escalation
* A clean frontend API

Project Structure

noema/
├── crates/
│   ├── noema-core/
│   ├── noema-api/
│   ├── noema-rig/
│   ├── noema-gemma/
│   ├── noema-needle/         # Needle 2 via its official C API (DylibEngine)
│   ├── noema-needle-static/  # Needle 2 statically linked (StaticEngine)
│   ├── noema-router/        # initial text router + tool-specific Needle agents
│   ├── noema-filesearch/    # the reference tool (read-only file search)
│   ├── noema-approval/      # risk-gated human approval
│   ├── noema-provider-http/ # OpenAI-compatible cloud provider (model, URL, key)
│   ├── noema-native/        # stages LiteRT-LM DLLs next to executables
│   ├── litert-lm-rust/      # vendored Gemma 4 runtime binding
│   ├── noema-memory/
│   ├── noema-context/
│   ├── noema-events/
│   ├── noema-approval/
│   └── noema-tools/
│
├── examples/
│   ├── basic/
│   ├── gemma/               # real local Gemma 4 conversation + rig path
│   ├── needle/              # real Needle 2 tool calls
│   ├── router/              # Needle routes actions, Gemma handles the rest
│   ├── tools/               # register tools, Needle formats, Noema executes
│   ├── filesearch/          # Gemma → Needle format → filesystem → Gemma
│   ├── approval/            # approve / reject / expire risky tool calls
│   └── agent/               # the full agent loop inside session.send
├── prebuilt/                # native engines (Needle 2, LiteRT-LM DLLs/dylibs)
│   ├── macos/               # macOS arm64 LiteRT dylibs (user-provided)
├── models/                  # Gemma 4 model file (gitignored)
├── tests/
├── Cargo.toml
├── README.md
└── plan.md

License

License information will be added as the project matures.