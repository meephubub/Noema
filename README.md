# Noema

The AI agent layer of Agora.

Noema is a Rust-native AI agent runtime built around Gemma 4, Needle 2, Rig, and Mnemo.

It provides Agora with an intelligent interface capable of understanding requests, reasoning over context, using tools, interacting with persistent memory, processing multimodal input, and escalating difficult tasks to larger models.

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

The cloud layer is provider-agnostic and can support different providers through a common abstraction.

Cloud escalation can be disabled for fully local/offline operation.

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

Noema is currently under active development.

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
│   ├── noema-needle/
│   ├── noema-memory/
│   ├── noema-context/
│   ├── noema-events/
│   ├── noema-approval/
│   └── noema-tools/
│
├── examples/
├── tests/
├── Cargo.toml
├── README.md
└── plan.md

License

License information will be added as the project matures.