# Noema — Implementation Roadmap

This roadmap is derived from `plan.md` (excluding the Development Phases section). It translates the goals, architecture, flows, and acceptance criteria in the plan into an ordered, milestone-based implementation plan with concrete deliverables and completion checks.

## Guiding Principle

> **Gemma decides what should happen. Needle makes it executable. Noema coordinates everything.**

Every milestone below must preserve the core architectural separation:

| Component | Role |
|---|---|
| **Agora** | The environment / frontend |
| **Noema** | The intelligence and orchestration layer |
| **Mnemo** | The persistent memory |
| **Gemma 4** | The primary reasoning model |
| **Needle 2** | The efficient router and structured tool-call formatter |
| **Rig** | The model/agent middle layer |
| **Tools** | Independent Rust crates |

## Cross-Cutting Requirements (apply to all milestones)

- **Everything in Rust.** Noema is written entirely in Rust; no foreign runtime components.
- **Model-agnostic core.** No inference backend types may leak into the core agent architecture; model access goes through internal traits only.
- **Ephemeral vs. persistent state.** Sessions, agent loops, pending calls, and approvals are ephemeral inside Noema. Persistent memory belongs exclusively to Mnemo — Noema never implements a second memory system.
- **Token efficiency.** Minimize model context everywhere: lightweight tool summaries to Gemma, full schemas only to Needle, trimmed context, deduplicated tool results.
- **Extensibility.** Adding a tool must never require modifying the core agent loop.
- **Security by default.** Risk levels, human approval, validation, sandboxing, timeouts, cancellation, and privacy-aware cloud escalation are built in from the start, not bolted on at the end.
- **Testing alongside implementation.** Each milestone ships with its unit, integration, and (where applicable) end-to-end tests.

---

## Milestone 1 — Workspace Foundation

**Objective:** Stand up the Cargo workspace and the crate skeleton such that the project compiles, is configurable, and can start and create a session.

### Scope
1. **Workspace layout** per the plan's structure:
   - `crates/`: `noema-core`, `noema-api`, `noema-rig`, `noema-gemma`, `noema-needle`, `noema-memory`, `noema-tools`, `noema-approval`, `noema-context`, `noema-events`
   - `examples/`: `basic`, `tools`, `multimodal`, `approval`
   - `tests/`: `integration`, `routing`, `tools`, `escalation`
   - Crate boundaries may be adjusted during implementation, but responsibilities must stay separated.
2. **Crate responsibilities** defined per the plan:
   - `noema-core`: agent runtime, session management, model routing, tool orchestration, agent state, core traits, configuration.
   - `noema-api`: public frontend-facing API — sessions, messages, events, approval API, streaming interface.
   - `noema-rig`: Rig adapters, agent/model/provider integration.
   - `noema-gemma`: litert-lm-rust integration, multimodal requests, streaming, Gemma-specific message handling.
   - `noema-needle`: Needle Rust binding integration, inference interface, request/response types (the binding itself is an external project).
   - `noema-memory`: Mnemo integration — retrieval, insertion, context conversion, memory policies.
   - `noema-context`: context construction and optimization — conversation, memory, tool summaries, prompt construction, trimming.
   - `noema-events`: event definitions and streaming infrastructure.
   - `noema-approval`: approval requests, approval state, risk policies, approval lifecycle.
   - `noema-tools`: tool traits, registry, schemas, metadata, risk levels.
3. **Strongly typed errors** for the major categories: `ModelError`, `NeedleError`, `ToolError`, `MemoryError`, `ApprovalError`, `ContextError`, `EscalationError`, `SessionError`, `ConfigurationError`. Errors must carry enough detail for logging/debugging without leaking sensitive information to the frontend.
4. **Strongly typed configuration** covering: Gemma model, Needle model, cloud providers, memory, tool set, risk policies, approval policies, iteration limits, context limits, logging, streaming, offline mode.
5. **Configurable logging levels**: `off`, `error`, `warn`, `info`, `debug`, `trace`.
6. **Basic session abstraction** — enough to create and close a session with no model attached.

### Completion Criteria
- `cargo build` succeeds across the workspace; all crates have declared, sensible responsibilities.
- A `Noema` instance can be constructed from configuration and can create/close a session.
- Error, configuration, and logging types are in place and unit-tested.

---

## Milestone 2 — Core Abstractions

**Objective:** Define the abstractions every other milestone builds on: models, messages, events, sessions, and cancellation.

### Scope
1. **Model trait** around the capabilities Noema actually requires:
   ```rust
   trait Model {
       async fn generate(&self, request: ModelRequest) -> Result<ModelResponse>;
   }
   ```
   The request/response types must support: text, images, audio, streaming, tool-related messages, system prompts, model escalation, cancellation, usage metadata, and errors.
2. **ModelProvider abstraction** so Gemma 4 local, Needle 2 local, cloud models, future local models, and future cloud models can all be plugged in without changing the core agent architecture.
3. **Ordered multimodal message abstraction.** Messages must support ordered content parts (text, image, audio) rather than assuming plain text — required for mixed requests like *"Summarise this page" + image + text*.
4. **Event type definitions** covering the full event set: `SessionStarted`, `UserMessageReceived`, `RoutingStarted`, `RoutingCompleted`, `RoutingEscalated`, `ModelStarted`, `ModelDelta`, `ModelCompleted`, `ToolRequested`, `ToolFormatted`, `ToolApprovalRequired`, `ToolApproved`, `ToolRejected`, `ToolStarted`, `ToolProgress`, `ToolCompleted`, `ToolFailed`, `MemoryRetrieved`, `MemoryWritten`, `EscalationStarted`, `EscalationCompleted`, `AssistantStarted`, `AssistantDelta`, `AssistantCompleted`, `Error`, `SessionCompleted`.
5. **Session abstraction** encapsulating ephemeral state: id, conversation, state, pending approvals — supporting create, send message, send multimodal message, subscribe to events, approve/reject tool call, cancel, close.
6. **Cancellation** design: a cancellation mechanism that propagates through Frontend → Noema → Rig → Model, and Noema → Tool execution.

### Completion Criteria
- Core traits compile and are documented; a no-op/test model implementation can be swapped in.
- Events, sessions, and cancellation are unit-tested at the abstraction level.
- The API crate exposes the session lifecycle without leaking model internals.

---

## Milestone 3 — Model Integrations (Gemma, Needle, Rig, Cloud)

**Objective:** Implement the concrete model adapters behind the abstractions, each isolated in its own crate.

### Scope
1. **noema-gemma — Gemma 4 integration** via `litert-lm-rust`, kept behind a Gemma abstraction so LiteRT-specific types never spread through the codebase:
   - Text input, image input, audio input.
   - Streaming output.
   - System prompts and conversation context.
   - Tool-intent generation (semantic tool requests, not final schemas).
   - Escalation decision output.
2. **noema-needle — Needle 2 integration** through its dedicated Rust binding crate (an external project; consuming it as a normal dependency):
   - Needle inference interface with Needle-specific request/response types.
   - Structured output handling and error handling.
   - One physical Needle model, exposed as multiple logical agents later (see Milestone 5).
3. **noema-rig — Rig integration** as the middle layer:
   - Reuse Rig for agents, model interactions, tool interfaces where appropriate, message handling, streaming, and provider abstraction.
   - Noema-specific orchestration sits *above* Rig; do not duplicate what Rig already provides reliably.
4. **Cloud provider abstraction** (escalation-ready, not hard-coded):
   ```rust
   trait ModelProvider {
       async fn complete(&self, request: ModelRequest) -> Result<ModelResponse>;
   }
   ```
   Supporting Gemini, OpenAI, other providers, and future providers interchangeably.
5. **Gemma abstraction layering** as specified: Noema → Gemma model abstraction → litert-lm-rust → Gemma 4, and Noema → Needle Rust crate → Needle 2 C API → Needle 2.

### Completion Criteria
- Noema can hold a conversation with a text-only Gemma 4 locally and run Needle 2 inference.
- Model adapters can stream tokens and report usage metadata/errors through the core trait.
- A stub cloud provider can be registered and invoked through the same interface.

---

## Milestone 4 — Initial Text Routing

**Objective:** Implement the cheap-path Needle router that handles simple application actions without invoking Gemma.

### Scope
1. **Needle router** responsible only for simple application-level requests, e.g. *"Open my flashcards"*, *"Show me my notes"*, *"Open my PDFs"*, *"Go to settings"*, *"Start a revision session"*, *"Open the last document"*.
2. **Router protocol**: return a structured result — either `Handled { action }` (e.g. `OpenFlashcards`) or `Escalate`. The router must **not** attempt to answer arbitrary questions.
3. **Action registry**: a typed set of known application actions; unknown/unsupported input escalates to Gemma.
4. **Frontend event** emission for handled actions so Agora can act on them (e.g. `OpenFlashcards` → Agora opens flashcards).
5. **Text request flow** wired so the initial request goes Needle-first and only escalates to Gemma when unrecognized.

### Completion Criteria
- "Open flashcards" → Needle → `OpenFlashcards` event; complex/unsupported request → Needle → escalate to Gemma.
- Gemma is never invoked on the simple path.
- Unit tests cover handled, unknown, and ambiguous router inputs; integration tests cover both paths.

---

## Milestone 5 — Tool Infrastructure

**Objective:** Build the tool abstraction, registry, schemas, and discovery so third-party tool crates can register without touching the core agent loop.

### Scope
1. **Tool trait** standardized across all tool crates:
   ```rust
   pub trait NoemaTool {
       fn metadata(&self) -> ToolMetadata;
       fn schema(&self) -> ToolSchema;
       async fn execute(&self, call: ToolCall) -> Result<ToolResult>;
   }
   ```
   Every tool must expose: tool name, crate name, description, input schema, risk level, execution handler, Needle instructions/schema, output description, and optional capabilities/metadata.
2. **Tool schema** format primarily for Needle — e.g. `{ name, description, risk, parameters: { ... } }` with required/optional and typed parameters. Gemma never receives full schemas.
3. **Risk levels** standardized before implementation: `None`, `Low`, `Medium`, `High`, `Critical` (final set to be confirmed). Risk is evaluated by Noema, not the frontend.
4. **Tool Registry** with: tool discovery, tool metadata, schema retrieval, risk metadata, Needle-agent creation, execution, and tool lifecycle management.
5. **Tool discovery on init**: installed crates → registration → registry → Gemma tool summary (lightweight descriptions only) + Needle schemas (complete).
6. **Tool metadata separation**: Gemma-facing metadata vs. Needle-facing schema vs. execution metadata vs. risk metadata, to minimize token usage without sacrificing structured execution.
7. **Registration API**: `noema.register_tool(filesearch)`; tools installable as Cargo dependencies; default Agora builds include the standard Noema tool set.
8. **Tool-specific Needle agents**: one physical Needle 2 model exposed as multiple logical agents — each with a dedicated system prompt, tool schema, and instructions, sharing the same runtime.
9. **Reference tool crate** (`noema-filesearch` or similar) implementing the full contract end-to-end: schema, Needle instructions, risk classification, execution, results, and Gemma summary.
10. **Dynamic Gemma tool section** in the system prompt built from the registry, e.g.:
    ```
    Available tools:
    filesearch   (crate: noema-filesearch)
    flashcards   (crate: noema-flashcards)
    pdf          (crate: noema-pdf)
    notes        (crate: noema-notes)
    ```

### Completion Criteria
- A third-party `noema-*` crate can register a tool with no modification to the core agent loop.
- Gemma receives only name + lightweight description + crate ownership; Needle receives complete schemas.
- Full path works: Gemma semantic request → tool-specific Needle → structured call → filesystem → result → Gemma.
- The reference tool is covered by unit and integration tests.

---

## Milestone 6 — Tool Execution, Approval, and Results

**Objective:** Implement the execution pipeline: validation, risk evaluation, human approval, deterministic execution, result handling, and multi-tool orchestration.

### Scope
1. **Semantic tool requests**: Gemma emits a semantic request (e.g. `find the file "abc.exe"`) referencing a capability + crate; Noema routes it to the corresponding tool-specific Needle agent.
2. **Structured call production**: the tool-specific Needle converts the semantic request into the exact structured tool call.
3. **Independent schema validation** of every Needle-generated call, separate from the model: Parse → Schema validation → Risk evaluation → Approval → Execution. **Never execute an unvalidated model-generated tool call.**
4. **Risk evaluation & approval gating**:
   - Tool request → risk evaluation → approval required? → No: execute / Yes: frontend approval.
   - No tool with a required approval level may execute before approval is received.
5. **Approval system** (`noema-approval`):
   - Approval requests carrying the complete tool call: tool name, tool description, arguments, risk level, intended action, potential consequences where available.
   - Approval lifecycle and state; frontend responds `Approve` or `Reject`.
   - Unique approval IDs tied to the exact pending request; configurable expiry.
   - Public API: `session.approve_tool(request_id)` / `session.reject_tool(request_id)`.
6. **Deterministic tool execution**: models never directly invoke Rust functions. Sole path: Model → structured request → Noema validation → Tool Registry → Rust tool.
7. **Tool results** with a standardized representation:
   ```rust
   struct ToolResult { success: bool, content: ToolContent, metadata: ToolMetadata }
   ```
   Supporting text, structured data, files, images, audio, errors, and metadata; converted into a representation appropriate for Gemma and returned to it for interpretation and follow-up decisions.
8. **Multiple tool calls**: sequential execution (Tool A → result → Tool B → result) and parallel execution where dependencies permit; Noema determines dependencies before running parallel requests.
9. **Tool routing flow** fully wired: Gemma → Noema Tool Router → tool-specific Needle → structured call → risk evaluation → execution → result → Gemma.

### Completion Criteria
- High-risk tool: Needle → Noema → frontend → user approval → execution; rejection cancels cleanly.
- Unvalidated or schema-invalid calls are rejected before execution.
- Sequential and parallel multi-tool flows tested; dependency detection verified.
- Approval expiry, unique IDs, and the frontend approval API are covered by tests.

---

## Milestone 7 — Agent Loop and Context

**Objective:** Implement the full reasoning loop, context assembly, and resource limits that keep the agent from running away.

### Scope
1. **Full agent loop**:
   User request → Needle Router → (simple action: execute) / (escalate: context build → Gemma) → Gemma decides: respond, or tool intent → tool-specific Needle → structured call → risk check → approved? execute / approval → frontend → result → Gemma → continue or respond.
2. **Context assembly** (`noema-context`): a Context Builder that packages conversation history, relevant Mnemo memories, current application state, available tools (lightweight summaries), tool results, previous agent decisions, relevant documents, and user preferences from Mnemo — then minimizes it before model inference.
3. **Context efficiency**: minimize tool schemas sent to Gemma, unnecessary conversation history, duplicate tool results, irrelevant memories, and repeated system instructions. The key optimization: Gemma gets `tool name + lightweight semantic description`; Needle gets the full schema.
4. **Resource limits** (configurable): maximum agent iterations, maximum tool calls, maximum tool-call depth, maximum context size, maximum response length, maximum tool execution time, maximum cloud escalation count, maximum concurrent tools — to prevent runaway loops.
5. **Prompt architecture**:
   - Versioned, separated from implementation logic.
   - Minimum prompt set: Gemma system prompt, Needle router prompt, Needle tool prompt, escalation prompt, multimodal prompt.
   - Tool crates provide their tool-specific Needle instructions; Noema dynamically constructs Gemma's available-tool section.
6. **Gemma system prompt** defining: Noema's role, user context, how tools work, available tools + crate ownership, how to issue semantic requests, how to handle tool results, when to clarify, when to escalate, multimodal behavior, and safety/approval behavior — without unnecessary schema information.
7. **Failure recovery**: typed handling of model, tool, memory, and escalation failures within the loop, with bounded retries where safe.

### Completion Criteria
- Gemma completes complex multi-step tool tasks (e.g. *"Find the document I was studying yesterday and make me five flashcards from the inflation section"* → filesearch → pdf → flashcards → final response).
- Loop, context-size, and tool-call limits demonstrably stop runaway behavior.
- Context trimming verified to reduce token usage without breaking the task.

---

## Milestone 8 — Mnemo Memory Integration

**Objective:** Connect persistent memory through Mnemo behind a dedicated abstraction, keeping memory policy out of the core agent.

### Scope
1. **Memory abstraction** (`noema-memory`) — the only interface between Noema and Mnemo: retrieval, insertion, context conversion, memory policies.
2. **Memory retrieval before complex requests**: user request → relevant Mnemo retrieval → context → Gemma.
3. **Memory creation after useful interactions**: conversation → memory extraction → Mnemo. Extraction must not be indiscriminate — avoid storing temporary tool results, irrelevant conversation, redundant information, and sensitive information without appropriate handling.
4. **Session/memory separation**:
   - Persistent (Mnemo): user knowledge, preferences, long-term learning context, relevant historical information.
   - Ephemeral (Noema): current agent loop, current conversation, pending tool calls, pending approvals, current model context, active multimodal inputs, current tool execution state.
5. **Memory events**: `MemoryRetrieved` / `MemoryWritten` emitted into the event stream.

### Completion Criteria
- Noema remembers relevant information across sessions via Mnemo; retrieved memories appear in context and in events.
- No persistent state is stored inside Noema; the ephemeral/persistent boundary is enforced and tested.
- Memory extraction selectivity verified: irrelevant or sensitive content is not written.

---

## Milestone 9 — Cloud Escalation and Offline Mode

**Objective:** Implement escalation from both Gemma and Needle to abstract cloud providers, governed by policy and privacy controls.

### Scope
1. **Escalation interface**: models request escalation with structured metadata, not free-text — e.g. `Escalate { reason, context }`. Noema decides whether and how to escalate.
2. **Escalation policy** enforced by Noema (a model's request never bypasses user configuration): `allow_cloud_escalation`, `preferred_provider`, `maximum_cost`, `maximum_latency`, `privacy_policy`, `offline_only`.
3. **Both escalation paths**:
   - Local Gemma → difficulty detected → escalation request → provider → larger cloud model → result → local agent continues.
   - Needle → escalation for routing/tool tasks beyond its capability → provider (or Gemma, per the router flow).
4. **Cloud providers abstract** behind the `ModelProvider` trait; Rig provides the middle layer where appropriate.
5. **Offline mode**: operate entirely locally with Needle 2 + Gemma 4 + Mnemo + local tools; no cloud escalation when enabled.
6. **Privacy**: local processing by default; Noema knows when data leaves the machine; sensitive context is not sent to a cloud provider unless permitted by configuration/policy.
7. **Escalation events**: `EscalationStarted` / `EscalationCompleted` in the stream.

### Completion Criteria
- Local model escalates a difficult task to a stub cloud provider and continues the agent loop with the result.
- Escalation blocked/rerouted correctly under offline mode and policy constraints (cost, latency, privacy).
- Needle escalation path covered by integration tests.

---

## Milestone 10 — Multimodal Agent

**Objective:** Full text, image, and audio support, including mixed-content requests and multimodal-driven tool use.

### Scope
1. **Audio mode**: audio sent directly to Gemma 4's native audio modality — system prompt → conversation/context → audio → Gemma. No intermediate ASR stage. System prompt tells Gemma how to interpret the audio (e.g. *"Answer the user's question using the audio file provided below."*). Gemma handles speech understanding, audio interpretation, reasoning, and response generation. Needle text routing is not required for audio-mode requests.
2. **Image mode**: images passed directly to Gemma 4's multimodal interface — system prompt → user text → image → Gemma — for questions, homework, diagrams, documents, screenshots, educational material, and visual explanations.
3. **Mixed multimodal requests** via the ordered message abstraction: text+image, text+audio, text+image+audio, and any combination.
4. **Multimodal tool workflows**: images/audio can lead to tool calls (e.g. reasoning about a screenshot, then invoking a tool); results return to Gemma.
5. **Multimodal prompts** and per-modality system-prompt guidance.

### Completion Criteria
- Audio → Gemma 4 → reasoning → response; image → Gemma 4 → reasoning/tool use → response; mixed-content requests work end-to-end.
- Multimodal inputs respected in context assembly, event stream, and cancellation.
- Covered by model tests for valid/ambiguous multimodal behavior.

---

## Milestone 11 — Prompt Integrity and Security Hardening

**Objective:** Harden the system against prompt injection and enforce the full security posture.

### Scope
1. **Prompt boundaries**: distinguish system instructions, agent instructions, user content, retrieved memory, tool output, and external documents; tool outputs and user documents must never override system instructions. Preserve these boundaries during prompt construction.
2. **Security controls** (finalize and verify all): tool risk levels, human approval, explicit tool registration, input validation, schema validation, sandboxing where appropriate, capability restrictions, timeouts, cancellation, output size limits, cloud escalation privacy controls.
3. **Deterministic execution guarantee**: the only path from model to code is Model → structured request → Noema validation → Tool Registry → Rust tool.
4. **Robust cancellation** throughout the stack, including long-running tools where possible.
5. **Concurrency controls** for parallel tools and shared state (registry, sessions, approvals).
6. **Memory policy review**: confirm Mnemo policy surfaces handle sensitive data per configuration.

### Completion Criteria
- Injection attempts (malicious tool output, documents, or user content) fail to alter system behavior; tested with adversarial fixtures.
- Every security control is implemented, documented, and covered by tests.
- Cancellation and concurrency verified under load.

---

## Milestone 12 — Observability

**Objective:** Structured, privacy-aware observability across the whole runtime.

### Scope
1. **Structured events** recording: `session_id`, `model`, `model_latency`, `token_usage`, `tool`, `tool_latency`, `approval_latency`, `escalation`, `error`.
2. **Model and tool metrics**: latency, token usage, escalation tracking, approval tracking.
3. **Privacy-aware telemetry**: no sensitive user content logged by default; configurable logging levels `off` → `trace`.
4. **Streaming state visibility**: the frontend never polls; it observes agent state through the event stream (text generation, tool-call generation, tool execution state, approval state, escalation, progress).

### Completion Criteria
- Metrics appear in the event/telemetry stream for a sample text, tool, approval, and escalation flow.
- Default logging contains no user content; tests assert sensitive content is not emitted.
- Logging-level configuration works end-to-end.

---

## Milestone 13 — Public API Polish and Streaming

**Objective:** Finalize the ergonomic, stable public Rust API for the Agora frontend.

### Scope
1. **Builder API**:
   ```rust
   let noema = Noema::builder()
       .with_gemma(gemma)
       .with_needle(needle)
       .with_memory(mnemo)
       .build()
       .await?;
   ```
2. **Session API**:
   ```rust
   let session = noema.create_session(...).await?;
   session.send(message).await?;
   session.events().await?;
   ```
   Supporting create session, send message, send multimodal message, subscribe to events, approve/reject tool call, cancel operation, close session.
3. **Streaming/event-driven interface** so the frontend subscribes and consumes events without polling:
   ```rust
   let mut events = noema.subscribe(session_id).await?;
   while let Some(event) = events.next().await { ... }
   ```
   Streaming covers text generation, tool-call generation, tool execution state, approval state, model escalation, and progress information.
4. **Frontend encapsulation**: the frontend must not need to know which model was selected, how tools are routed, how schemas work, how Needle is configured, how memory retrieval works, or how escalation works.
5. **Examples** for `basic`, `tools`, `multimodal`, and `approval`; documented API surface.

### Completion Criteria
- All example programs run against the reference tool set.
- API surface is stable, documented, and covered by integration tests simulating a frontend client.

---

## Milestone 14 — Testing, Quality, and Definition of Done

**Objective:** Comprehensive automated coverage and final validation against the plan's Definition of Done.

### Scope
1. **Unit tests**: context assembly, tool registration, schema handling, risk evaluation, approval state, model routing, escalation logic, event generation, session state, cancellation.
2. **Integration tests**: Frontend → Noema → Gemma; Frontend → Noema → Needle; Gemma → Needle → Tool; Tool → Gemma; Gemma → Cloud; Needle → Cloud.
3. **End-to-end tests** for complete user workflows: open flashcards, search for a file, read a PDF, create flashcards, modify a note, execute a high-risk tool, reject a tool, escalate a difficult query, use audio, use image.
4. **Model testing** (probabilistic, so structurally repeated): valid tool requests, invalid tool requests, ambiguous requests, tool selection accuracy, schema formatting accuracy, escalation behavior, prompt-injection resistance, multimodal behavior. **Needle is tested especially heavily** — its output directly determines executable tool calls.
5. **Schema validation tests**: every Needle-produced call is parsed and validated independently of the model; invalid calls never execute.
6. **Definition of Done checklist** (from the plan) as the final gate — Noema is production-ready when it can:
   - Run entirely from Rust; run Gemma 4 via litert-lm-rust; run Needle 2 via its Rust binding.
   - Use Rig as the orchestration layer and Mnemo for persistent memory.
   - Route simple text actions through Needle; escalate unsupported text requests to Gemma.
   - Accept text, image, and audio input; pass audio to Gemma after system/text content.
   - Expose tools to Gemma without full schemas; route semantic requests to the right Needle instance; give Needle complete schemas.
   - Validate Needle-generated tool calls; enforce risk levels; require frontend approval for risky calls; execute approved tools; return results to Gemma.
   - Support multiple tool calls and multi-step agent loops; support cloud escalation from both Gemma and Needle; keep cloud providers abstract.
   - Allow tools as crates and allow adding tools without changing the core agent loop.
   - Provide a simple Rust frontend API, streaming events, and cancellation.
   - Maintain ephemeral session state; keep persistent memory in Mnemo.
   - Provide structured observability and comprehensive unit, integration, and end-to-end tests.
7. **Future-proofing check**: architecture must accommodate Gemma 5, other local models, other small routers, other cloud models, and specialized multimodal models without rewriting Noema.

### Completion Criteria
- Full test suite green (unit, integration, e2e, model).
- Definition of Done checklist verified item-by-item with evidence from tests and running examples.

---

## Dependency Overview

```
M1 Workspace ──► M2 Core Abstractions ──► M3 Model Integrations ──► M4 Text Routing
                                              │                        │
                                              ▼                        ▼
                                         M5 Tool Infrastructure ──► M6 Execution & Approval
                                              │                        │
                                              ▼                        ▼
                                         M7 Agent Loop & Context ◄────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
             M8 Mnemo Memory            M9 Escalation             M10 Multimodal
                    └─────────────────────────┼─────────────────────────┘
                                              ▼
                                M11 Security Hardening
                                              ▼
                                    M12 Observability
                                              ▼
                              M13 Public API & Streaming
                                              ▼
                            M14 Testing & Definition of Done
```

Milestones 8–10 can proceed in parallel once Milestone 7 lands; M11, M12, and M13 harden and polish the accumulated system; M14 is the final gate.

## Definition of Done (Final Gate)

The project is production-ready only when every item in Milestone 14's checklist (mirroring the plan's Definition of Done) passes — with special emphasis on:

- **Schema validation before any execution.**
- **Approval gating for risky tool calls.**
- **Ephemeral session state vs. Mnemo persistent memory.**
- **Privacy-aware cloud escalation with offline mode.**
- **Comprehensive model, integration, and end-to-end tests.**
