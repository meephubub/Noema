Noema

The AI agent layer of Agora.

Noema is a fully Rust-based, model-agnostic AI agent runtime designed for Agora. It provides the intelligent interface between the frontend, the user’s learning environment, application tools, Mnemo memory, and external models.

Noema uses Gemma 4 as its primary intelligent model, Needle 2 as a lightweight routing and tool-formatting model, Rig as the agent/model orchestration layer, and Mnemo as its persistent memory system.

The system is designed around a principle:

Gemma decides what should happen. Needle makes it executable. Noema coordinates everything.

⸻

1. Goals

Noema must:

* Be written entirely in Rust.
* Use Gemma 4 through litert-lm-rust.
* Use Needle 2 through its dedicated Rust binding crate.
* Use mnemo for persistent memory.
* Use Rig as the middle/orchestration layer.
* Provide a simple Rust API for the Agora frontend.
* Support text, image, and audio input.
* Use Gemma 4’s native audio modality for voice interactions.
* Use Gemma 4’s image modality for image interactions.
* Use Needle 2 for initial text routing.
* Use Needle 2 for all tool-call formatting.
* Allow Gemma 4 and Needle 2 to escalate difficult tasks to larger cloud models.
* Make tools easy to create and install.
* Allow tools to be distributed as independent Rust crates.
* Support human approval for risky tool calls.
* Provide streaming/event-driven execution.
* Keep persistent memory inside Mnemo.
* Keep Noema’s session/agent state ephemeral.
* Minimize model context and token usage wherever possible.
* Keep the architecture extensible enough to support future models and tools.

⸻

2. Non-Goals

Noema should not:

* Implement persistent memory itself.
* Become a database.
* Implement the Needle 2 model/binding itself.
* Implement the Gemma model runtime itself.
* Become the frontend.
* Contain application-specific tool implementations.
* Hard-code individual tools into the core agent.
* Require Gemma 4 to understand complex tool schemas.
* Require every tool to be implemented inside the Noema repository.

Noema should provide the infrastructure that allows these components to work together.

⸻

3. High-Level Architecture

                         ┌─────────────────────┐
                         │        Agora        │
                         │      Frontend       │
                         └──────────┬──────────┘
                                    │
                              Rust API
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       Noema         │
                         │      noema-core     │
                         │                     │
                         │  Session Manager    │
                         │  Event System       │
                         │  Agent Runtime      │
                         │  Tool Registry      │
                         │  Approval System    │
                         │  Context Builder    │
                         │  Model Router       │
                         └───────┬─────┬───────┘
                                 │     │
                 ┌───────────────┘     └────────────────┐
                 ▼                                        ▼
          ┌──────────────┐                        ┌──────────────┐
          │    Rig       │                        │    Mnemo     │
          │ Orchestration│                        │   Memory     │
          └──────┬───────┘                        └──────────────┘
                 │
        ┌────────┼───────────────┐
        ▼        ▼               ▼
    Needle 2  Gemma 4       Cloud Model
       │         │
       │         │
       │         └── litert-lm-rust
       │
       └── Needle Rust binding

⸻

4. Core Architecture Principles

4.1 Gemma Is the Intelligent Agent

Gemma 4 is responsible for:

* Understanding complex user requests.
* Reasoning about context.
* Understanding documents and images.
* Understanding audio.
* Deciding which tool or tools are required.
* Determining the semantic parameters required by a tool.
* Deciding when a task should be escalated.
* Interpreting tool results.
* Producing the final response.

Gemma should not be responsible for producing the final structured schema of a tool call.

⸻

4.2 Needle Is the Execution Interface

Needle 2 is responsible for:

* Initial lightweight text routing.
* Converting semantic tool requests into structured tool calls.
* Following individual tool schemas.
* Validating/formatting tool arguments.
* Producing the exact tool call that Noema can execute.
* Determining whether simple routing/tool tasks are within its capabilities.
* Escalating difficult tasks where appropriate.

Needle should not replace Gemma’s reasoning.

⸻

4.3 Noema Is the Coordinator

Noema is responsible for:

* Selecting the appropriate model.
* Managing sessions.
* Building context.
* Managing memory retrieval.
* Managing memory writes.
* Managing tools.
* Routing tool requests to Needle.
* Enforcing schemas.
* Enforcing risk policies.
* Requesting human approval.
* Executing approved tools.
* Handling tool results.
* Managing model escalation.
* Streaming events.
* Handling failures.
* Providing the frontend API.

⸻

5. Workspace Structure

The Noema repository should be a Rust workspace.

noema/
├── Cargo.toml
├── README.md
├── LICENSE
├── plan.md
│
├── crates/
│   ├── noema-core/
│   ├── noema-api/
│   ├── noema-rig/
│   ├── noema-gemma/
│   ├── noema-needle/
│   ├── noema-memory/
│   ├── noema-tools/
│   ├── noema-approval/
│   ├── noema-context/
│   └── noema-events/
│
├── examples/
│   ├── basic/
│   ├── tools/
│   ├── multimodal/
│   └── approval/
│
└── tests/
    ├── integration/
    ├── routing/
    ├── tools/
    └── escalation/

The exact crate boundaries may be adjusted during implementation, but responsibilities should remain separated.

⸻

6. External Dependencies

Noema will depend on several external Rust crates/projects.

Required

litert-lm-rust

Provides the Rust interface to Gemma 4.

Noema should depend on it through an abstraction rather than spreading LiteRT-specific types throughout the entire codebase.

Noema
  ↓
Gemma model abstraction
  ↓
litert-lm-rust
  ↓
Gemma 4

⸻

Needle Rust Binding

A separate Rust project will expose the Needle 2 C API.

Noema will consume this as a normal Rust dependency.

Noema
  ↓
Needle Rust crate
  ↓
Needle 2 C API
  ↓
Needle 2

The implementation of the binding is explicitly outside the scope of this repository.

⸻

mnemo

Noema uses Mnemo for persistent memory.

Noema should never implement a second persistent memory system.

⸻

Rig

Rig is the middle layer connecting Noema’s agent architecture to models and tools.

Noema-specific orchestration should sit above Rig.

⸻

7. Model Abstraction

Noema should not make the rest of the application dependent on a specific inference backend.

Define internal model traits around the capabilities Noema actually requires.

Conceptually:

trait Model {
    async fn generate(
        &self,
        request: ModelRequest,
    ) -> Result<ModelResponse>;
}

The actual API should support:

* Text
* Images
* Audio
* Streaming
* Tool-related messages
* System prompts
* Model escalation
* Cancellation
* Usage metadata
* Errors

The model abstraction should allow:

Gemma 4 local
Needle 2 local
Cloud model
Future local model
Future cloud model

without changing the core agent architecture.

⸻

8. Gemma 4 Integration

Gemma 4 is the primary reasoning model.

The integration should be implemented in a dedicated crate/layer.

noema-gemma
    ↓
litert-lm-rust
    ↓
Gemma 4

Gemma must support:

* Text input
* Image input
* Audio input
* Streaming output
* System prompts
* Conversation context
* Tool intent generation
* Escalation decisions

⸻

9. Text Request Flow

Text requests have a special optimization path.

The initial request should not immediately invoke Gemma 4.

Instead:

User
  │
  ▼
Needle Router
  │
  ├── Simple application action
  │        │
  │        ▼
  │     Execute
  │
  └── Not recognised
           │
           ▼
        Gemma 4

The purpose is to handle cheap, deterministic/simple requests without invoking the larger Gemma model.

⸻

10. Initial Needle Router

The initial Needle instance is responsible only for simple application-level requests.

Examples:

"Open my flashcards"
"Show me my notes"
"Open my PDFs"
"Go to settings"
"Start a revision session"
"Open the last document"

The router should return a structured result such as:

Handled
    action = OpenFlashcards
or
Escalate

It must not attempt to answer arbitrary questions.

If Needle determines that the request is outside its supported routing capabilities:

Needle Router
      ↓
Escalate
      ↓
Gemma 4

⸻

11. Gemma Tool Architecture

Gemma receives a list of available tools through its system prompt.

It should see:

Available tools:
filesearch
  crate: noema-filesearch
flashcards
  crate: noema-flashcards
pdf
  crate: noema-pdf
notes
  crate: noema-notes

Gemma should not receive the full JSON/schema definitions.

This reduces context size and token usage.

Gemma only needs to understand:

1. What a capability is.
2. Which crate owns it.
3. What it can semantically request.

⸻

12. Semantic Tool Requests

When Gemma wants to use a tool, it should produce a semantic request rather than the final structured schema.

Example:

find the file "abc.exe"

Gemma knows that this belongs to:

filesearch

Noema then routes the request to the Needle instance associated with that tool.

⸻

13. Tool-Specific Needle Instances

There will be one physical Needle 2 model.

However, Noema will expose multiple logical Needle agents.

                         Needle 2
                            │
            ┌───────────────┼────────────────┐
            │               │                │
            ▼               ▼                ▼
      Filesearch        Flashcards         PDF
       Needle            Needle           Needle
            │               │                │
            ▼               ▼                ▼
       Filesearch        Flashcard          PDF
        schema            schema            schema

Each logical Needle agent has:

* A dedicated system prompt.
* A dedicated tool schema.
* A dedicated set of instructions.
* Access to the same underlying Needle model runtime.

The model itself is not duplicated.

⸻

14. Tool Crates

Tools should be distributed as independent Rust crates.

Examples:

noema-filesearch
noema-flashcards
noema-pdf
noema-notes
noema-browser
noema-system

These crates should be installable as dependencies and included with Noema by default where appropriate.

⸻

15. Tool Crate Contract

Every tool crate must expose a standard interface.

Conceptually:

pub trait NoemaTool {
    fn metadata(&self) -> ToolMetadata;
    fn schema(&self) -> ToolSchema;
    async fn execute(
        &self,
        call: ToolCall,
    ) -> Result<ToolResult>;
}

The exact trait should be designed during implementation.

A tool must expose:

* Tool name.
* Crate name.
* Description.
* Input schema.
* Risk level.
* Execution handler.
* Needle instructions/schema.
* Output description.
* Optional capabilities/metadata.

⸻

16. Tool Schema

The tool schema is primarily for Needle.

For example:

{
  "name": "search_files",
  "description": "Search for files on the local system",
  "risk": "low",
  "parameters": {
    "query": {
      "type": "string",
      "required": true
    },
    "path": {
      "type": "string",
      "required": false
    }
  }
}

Gemma does not receive this full schema.

Needle does.

⸻

17. Tool Risk

Risk must be defined inside the tool schema.

Possible levels:

None
Low
Medium
High
Critical

The final set of levels should be standardized before implementation.

Risk is evaluated by Noema, not by the frontend.

Noema should enforce:

Tool request
    ↓
Risk evaluation
    ↓
Approval required?
 ┌──┴──┐
No     Yes
│       │
▼       ▼
Execute  Frontend approval
             │
       ┌─────┴─────┐
       ▼           ▼
    Approved     Rejected
       │           │
       ▼           ▼
    Execute       Cancel

⸻

18. Human Tool Approval

Needle creates the final structured tool call but does not execute it.

Example:

Gemma
  ↓
"Delete the file abc.exe"
  ↓
Needle
  ↓
{
    tool: delete_file,
    path: "abc.exe"
}
  ↓
Noema
  ↓
Risk evaluation
  ↓
Approval required
  ↓
Frontend

The frontend should receive the complete tool call.

The user must be able to see:

* Tool name.
* Tool description.
* Arguments.
* Risk level.
* Intended action.
* Potential consequences where available.

The frontend then responds with:

Approve

or:

Reject

No tool with a required approval level may execute before approval is received.

⸻

19. Tool Execution

Once approved:

Approval
   ↓
Noema
   ↓
Tool Registry
   ↓
Tool
   ↓
Result
   ↓
Noema
   ↓
Gemma

Tool results are returned to Gemma so that it can:

* Interpret them.
* Decide whether another tool is required.
* Answer the user.
* Escalate if necessary.

⸻

20. Complete Tool Flow

Example:

User:
"Find abc.exe"
        │
        ▼
Gemma 4:
"filesearch → find the file abc.exe"
        │
        ▼
Noema Tool Router
        │
        ▼
Filesearch Needle
        │
        ▼
Structured tool call:
search_files({
    query: "abc.exe"
})
        │
        ▼
Risk evaluation
        │
        ▼
Filesearch execution
        │
        ▼
Result:
/Users/example/Downloads/abc.exe
        │
        ▼
Gemma 4
        │
        ▼
"Found abc.exe in Downloads."

⸻

21. Multiple Tool Calls

Gemma must be able to request multiple tools.

Support:

Sequential execution

Tool A
  ↓
Result
  ↓
Tool B
  ↓
Result

Parallel execution

       ┌── Tool A
Gemma ─┤
       └── Tool B

Parallel execution should only be used where tool dependencies permit it.

Noema should determine dependencies before executing parallel requests.

⸻

22. Tool Result Handling

Tool results must have a standardized representation.

Conceptually:

struct ToolResult {
    success: bool,
    content: ToolContent,
    metadata: ToolMetadata,
}

Results should support:

* Text
* Structured data
* Files
* Images
* Audio
* Errors
* Metadata

The result should be converted into a representation appropriate for Gemma.

⸻

23. Multimodal Input

Noema must support:

Text
Image
Audio

and combinations of them.

⸻

24. Audio Mode

When Noema is in voice/audio mode, the audio should be sent directly to Gemma 4.

The audio must be provided after the system prompt and any preceding text content.

Conceptually:

System prompt
      ↓
Conversation/context
      ↓
Audio input
      ↓
Gemma 4

The system prompt should explicitly tell Gemma how to interpret the audio.

For example:

Answer the user's question using the audio file provided below.

Noema should not introduce an unnecessary ASR stage.

Gemma handles:

* Speech understanding.
* Audio interpretation.
* Reasoning.
* Response generation.

⸻

25. Image Mode

Images should be passed directly to Gemma 4’s multimodal interface.

Example:

System prompt
      ↓
User text
      ↓
Image
      ↓
Gemma 4

Images can be used for:

* Questions.
* Homework.
* Diagrams.
* Documents.
* Screenshots.
* Educational material.
* Visual explanations.

⸻

26. Mixed Multimodal Requests

Noema should support combinations such as:

"Explain this"
+
Image

or:

"Answer the question in this recording"
+
Audio

or:

"Summarise this page"
+
Image
+
Text

The message abstraction must therefore support ordered multimodal content rather than assuming every message is plain text.

⸻

27. Context Assembly

Before invoking Gemma, Noema should build a context package.

                Context Builder
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
 Conversation      Mnemo         Tools
        │             │             │
        └─────────────┼─────────────┘
                      ▼
                 Model Context

Context may contain:

* Current user message.
* Conversation history.
* Relevant Mnemo memories.
* Current application state.
* Available tools.
* Tool results.
* Previous agent decisions.
* Relevant documents.
* User preferences from Mnemo.

Context should be minimized before model inference.

⸻

28. Mnemo Integration

Noema uses Mnemo for persistent memory.

Noema should interact with Mnemo through a dedicated abstraction.

Responsibilities include:

Memory retrieval

Before complex requests:

User request
   ↓
Relevant Mnemo retrieval
   ↓
Context
   ↓
Gemma

Memory creation

After useful interactions:

Conversation
   ↓
Memory extraction
   ↓
Mnemo

Memory extraction should not happen indiscriminately.

Noema should avoid storing:

* Temporary tool results.
* Irrelevant conversation.
* Redundant information.
* Sensitive information without appropriate handling.

The exact memory policy belongs to Mnemo’s API and configuration.

⸻

29. Conversation State

Noema should distinguish:

Persistent state

Handled by Mnemo.

Examples:

* User knowledge.
* Preferences.
* Long-term learning context.
* Relevant historical information.

Ephemeral state

Handled by Noema.

Examples:

* Current agent loop.
* Current conversation.
* Pending tool calls.
* Pending approvals.
* Current model context.
* Active multimodal inputs.
* Current tool execution state.

⸻

30. Model Escalation

Both Needle and Gemma must be capable of escalating difficult tasks.

Noema should provide a general model escalation abstraction.

Local Model
    ↓
Difficulty detected
    ↓
Escalation request
    ↓
Model Provider
    ↓
Larger Cloud Model

The cloud model must not be hard-coded.

The architecture should support:

CloudProvider
    ├── Gemini
    ├── OpenAI
    ├── Other provider
    └── Future provider

Rig should provide the middle layer where appropriate.

⸻

31. Escalation Interface

Conceptually:

trait ModelProvider {
    async fn complete(
        &self,
        request: ModelRequest,
    ) -> Result<ModelResponse>;
}

A model should be able to request escalation using structured metadata rather than merely writing a natural-language statement.

Example:

Escalate {
    reason: "Requires substantially larger reasoning capacity",
    context: ...
}

Noema then decides whether and how to perform the escalation.

⸻

32. Escalation Policy

Noema should be responsible for enforcing global escalation policy.

Possible configuration:

allow_cloud_escalation
preferred_provider
maximum_cost
maximum_latency
privacy_policy
offline_only

A model’s request to escalate should not automatically bypass user configuration.

⸻

33. Rig Integration

Rig acts as the middle layer between Noema and model/tool abstractions.

Conceptually:

Noema
   ↓
Rig
   ├── Gemma
   ├── Needle
   ├── Cloud Models
   └── Agent primitives

Noema-specific components should remain above Rig.

Rig should provide/reuse functionality for:

* Agents.
* Model interactions.
* Tool interfaces where appropriate.
* Message handling.
* Streaming.
* Provider abstraction.

Noema should avoid duplicating functionality already provided reliably by Rig.

⸻

34. Event System

Noema should be event-driven.

The frontend should subscribe to an event stream.

Example:

let mut events = noema.subscribe(session_id).await?;
while let Some(event) = events.next().await {
    // Handle event
}

Events should include at minimum:

SessionStarted
UserMessageReceived
RoutingStarted
RoutingCompleted
RoutingEscalated
ModelStarted
ModelDelta
ModelCompleted
ToolRequested
ToolFormatted
ToolApprovalRequired
ToolApproved
ToolRejected
ToolStarted
ToolProgress
ToolCompleted
ToolFailed
MemoryRetrieved
MemoryWritten
EscalationStarted
EscalationCompleted
AssistantStarted
AssistantDelta
AssistantCompleted
Error
SessionCompleted

⸻

35. Streaming

Responses should be streamed whenever possible.

Streaming should support:

* Text generation.
* Tool-call generation.
* Tool execution state.
* Approval state.
* Model escalation.
* Progress information.

The frontend should never need to poll Noema for agent state.

⸻

36. Public Rust API

The frontend must have a simple and ergonomic Rust API.

The exact API should be finalized during implementation, but the architecture should resemble:

let noema = Noema::builder()
    .with_gemma(gemma)
    .with_needle(needle)
    .with_memory(mnemo)
    .build()
    .await?;

Then:

let session = noema
    .create_session(...)
    .await?;

And:

session.send(message).await?;

With events:

session.events().await?;

The frontend should not need to know:

* Which model was selected.
* How tools are routed.
* How schemas work.
* How Needle is configured.
* How memory retrieval works.
* How escalation works.

⸻

37. Session API

Sessions should encapsulate ephemeral agent state.

Conceptually:

Session {
    id,
    conversation,
    state,
    pending_approvals,
}

The API should support:

Create session
Send message
Send multimodal message
Subscribe to events
Approve tool call
Reject tool call
Cancel operation
Close session

⸻

38. Tool Approval API

The frontend should have a direct method to respond to approval requests.

Conceptually:

session.approve_tool(request_id).await?;

and:

session.reject_tool(request_id).await?;

Approval IDs must be unique and tied to the exact pending tool request.

Approvals should expire if configured to do so.

⸻

39. Cancellation

Noema should support cancellation throughout the stack.

Cancellation must be propagated through:

Frontend
  ↓
Noema
  ↓
Rig
  ↓
Model

and:

Noema
  ↓
Tool execution

Long-running tools should support cancellation where possible.

⸻

40. Error Handling

Errors should be strongly typed.

Major categories:

ModelError
NeedleError
ToolError
MemoryError
ApprovalError
ContextError
EscalationError
SessionError
ConfigurationError

Errors should contain enough information for logging/debugging without leaking sensitive information to the frontend.

⸻

41. Tool Registry

Noema needs a central tool registry.

Tool Registry
├── filesearch
├── flashcards
├── pdf
├── notes
└── ...

The registry should provide:

* Tool discovery.
* Tool metadata.
* Schema retrieval.
* Risk metadata.
* Needle-agent creation.
* Execution.
* Tool lifecycle management.

Adding a tool should not require modifying the central agent loop.

⸻

42. Tool Installation

Tools should be installable through Cargo dependencies.

Example:

[dependencies]
noema-core = "..."
noema-filesearch = "..."
noema-flashcards = "..."
noema-pdf = "..."

Noema should provide a standard registration mechanism.

Conceptually:

noema.register_tool(filesearch);
noema.register_tool(flashcards);

Default Agora builds should include the standard Noema tool set.

⸻

43. Tool Discovery

When Noema initializes:

Installed crates
      ↓
Tool registration
      ↓
Tool registry
      ↓
Gemma tool summary
      +
Needle schemas

Gemma receives only the lightweight descriptions.

Needle receives the complete schemas.

⸻

44. Tool Metadata

Tool metadata should distinguish between:

Gemma-facing metadata
Needle-facing schema
Execution metadata
Risk metadata

This allows the system to minimize token usage without sacrificing structured tool execution.

⸻

45. Prompt Architecture

Prompts should be versioned and separated from implementation logic.

At minimum:

Gemma system prompt
Needle router prompt
Needle tool prompt
Escalation prompt
Multimodal prompt

Tool crates should be able to provide the tool-specific Needle instructions.

Noema should dynamically construct Gemma’s available-tool section.

⸻

46. Gemma System Prompt

The Gemma system prompt should define:

* Noema’s role.
* User context.
* How tools work.
* Available tools.
* Tool crate ownership.
* How to issue semantic tool requests.
* How to handle tool results.
* When to ask for clarification.
* When to escalate.
* Multimodal behavior.
* Safety/approval behavior.

The prompt should avoid including unnecessary schema information.

⸻

47. Needle Tool Prompt

Each logical Needle agent should receive:

* Its tool’s complete schema.
* Tool-specific formatting instructions.
* Valid argument requirements.
* Output format.
* Error-handling rules.

For example:

You are the filesearch tool formatter.
Convert the requested file-search operation into the following schema:
...

Needle then converts:

find the file "abc.exe"

into the structured tool call.

⸻

48. Prompt Injection Considerations

Tool outputs and user-provided documents must not automatically override Noema’s system instructions.

Noema should distinguish:

System instructions
Agent instructions
User content
Retrieved memory
Tool output
External documents

Prompt construction should preserve these boundaries.

⸻

49. Security

Noema should assume tools can have significant system access.

Security controls should include:

* Tool risk levels.
* Human approval.
* Explicit tool registration.
* Input validation.
* Schema validation.
* Sandboxing where appropriate.
* Capability restrictions.
* Timeouts.
* Cancellation.
* Output size limits.
* Cloud escalation privacy controls.

⸻

50. Resource Limits

Noema should support configurable limits for:

* Maximum agent iterations.
* Maximum tool calls.
* Maximum tool-call depth.
* Maximum context size.
* Maximum response length.
* Maximum tool execution time.
* Maximum cloud escalation count.
* Maximum concurrent tools.

This prevents runaway agent loops.

⸻

51. Agent Loop

The general Gemma agent loop should resemble:

                  ┌──────────────┐
                  │ User Request │
                  └──────┬───────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Needle Router   │
                └───────┬─────────┘
                        │
             ┌──────────┴──────────┐
             │                     │
        Simple action          Escalate
             │                     │
             ▼                     ▼
         Execute              Context Build
                                   │
                                   ▼
                                Gemma 4
                                   │
                         ┌─────────┴─────────┐
                         │                   │
                      Respond             Tool intent
                                             │
                                             ▼
                                       Tool-specific
                                        Needle
                                             │
                                             ▼
                                      Structured call
                                             │
                                             ▼
                                       Risk check
                                             │
                                  ┌──────────┴──────────┐
                                  │                     │
                               Approved              Approval
                                  │                     │
                                  ▼                     ▼
                                Execute              Frontend
                                  │                     │
                                  ▼                     │
                               Result ◄─────────────────┘
                                  │
                                  ▼
                                Gemma
                                  │
                         ┌────────┴────────┐
                         │                 │
                      Continue          Respond

⸻

52. Simple Request Path

Example:

User:
"Open my flashcards"
        ↓
Needle Router
        ↓
OpenFlashcards
        ↓
Noema API event
        ↓
Agora opens flashcards

Gemma is never invoked.

⸻

53. Complex Request Path

Example:

User:
"Find the document I was studying yesterday and make
me five flashcards from the section about inflation."
        ↓
Needle Router
        ↓
Escalate
        ↓
Gemma 4
        ↓
filesearch
"find the document studied yesterday"
        ↓
Filesearch Needle
        ↓
Tool call
        ↓
Result
        ↓
Gemma
        ↓
pdf
"extract the relevant inflation section"
        ↓
PDF Needle
        ↓
Tool call
        ↓
Result
        ↓
Gemma
        ↓
flashcards
"create five flashcards from this section"
        ↓
Flashcards Needle
        ↓
Tool call
        ↓
Approval if required
        ↓
Flashcards created
        ↓
Gemma
        ↓
Final response

⸻

54. Audio Request Path

User
  │
  ▼
Audio input
  │
  ▼
Noema
  │
  ├── System prompt
  ├── Context
  └── Audio
          │
          ▼
       Gemma 4
          │
          ▼
       Response

Needle’s text routing path is not required for audio-mode requests.

⸻

55. Image Request Path

User
  │
  ├── Text
  └── Image
       │
       ▼
     Noema
       │
       ▼
    Gemma 4
       │
       ▼
   Reasoning/tool use

Images can subsequently lead to tool calls.

⸻

56. Context Efficiency

Token efficiency is a major architectural requirement.

Noema should minimize:

* Tool schemas sent to Gemma.
* Unnecessary conversation history.
* Duplicate tool results.
* Irrelevant memories.
* Repeated system instructions.

Needle exists partly to solve this problem.

The key optimization is:

Gemma:
tool name + lightweight semantic description
Needle:
full schema

rather than:

Gemma:
full schema for every installed tool

⸻

57. Observability

Noema should provide structured observability.

Record events such as:

session_id
model
model_latency
token_usage
tool
tool_latency
approval_latency
escalation
error

Observability must avoid logging sensitive user content by default.

Provide configurable logging levels:

off
error
warn
info
debug
trace

⸻

58. Testing Strategy

Noema should have comprehensive automated tests.

Unit Tests

Test:

* Context assembly.
* Tool registration.
* Schema handling.
* Risk evaluation.
* Approval state.
* Model routing.
* Escalation logic.
* Event generation.
* Session state.
* Cancellation.

Integration Tests

Test:

Frontend → Noema → Gemma
Frontend → Noema → Needle
Gemma → Needle → Tool
Tool → Gemma
Gemma → Cloud
Needle → Cloud

End-to-End Tests

Test complete user workflows.

Examples:

Open flashcards
Search for a file
Read a PDF
Create flashcards
Modify a note
Execute high-risk tool
Reject a tool
Escalate difficult query
Use audio
Use image

⸻

59. Model Testing

Because model output is probabilistic, model tests should include:

* Valid tool requests.
* Invalid tool requests.
* Ambiguous requests.
* Tool selection accuracy.
* Schema formatting accuracy.
* Escalation behavior.
* Prompt injection resistance.
* Multimodal behavior.

Needle should be tested especially heavily because its output directly determines executable tool calls.

⸻

60. Tool Schema Validation

Every tool call produced by Needle must be validated independently of the model.

Needle output
    ↓
Parse
    ↓
Schema validation
    ↓
Risk evaluation
    ↓
Approval
    ↓
Execution

Never execute an unvalidated model-generated tool call.

⸻

61. Deterministic Tool Execution

Tool execution should be deterministic given:

Tool
+
Validated arguments
+
Execution environment

Models should never directly invoke Rust functions.

The only path should be:

Model
→ structured request
→ Noema validation
→ Tool registry
→ Rust tool

⸻

62. Configuration

Noema should expose configuration for:

Gemma model
Needle model
Cloud providers
Memory
Tool set
Risk policies
Approval policies
Iteration limits
Context limits
Logging
Streaming
Offline mode

Configuration should be strongly typed.

⸻

63. Offline Mode

Noema should support operating entirely locally when possible.

Offline:
Needle 2
   +
Gemma 4
   +
Mnemo
   +
Local tools

No cloud escalation should occur when offline mode is enabled.

⸻

64. Privacy

The default architecture should favor local processing.

Cloud escalation must be explicit and configurable.

Noema should know when data leaves the local machine.

Potentially sensitive context should not be sent to a cloud provider unless permitted by configuration/policy.

⸻

65. Extensibility

Adding a new tool should require approximately:

1. Create noema-<tool> crate.
2. Implement NoemaTool.
3. Define schema.
4. Define Needle instructions.
5. Define risk level.
6. Implement execution.
7. Register crate.

No modifications to the core agent loop should be necessary.

⸻

66. Future Model Support

Although Gemma 4 and Needle 2 are the initial models, the architecture should not assume they are permanent.

Future support should be possible for:

Gemma 5
Other local models
Other small routers
Other cloud models
Specialized multimodal models

without rewriting Noema.

⸻

67. Suggested Crate Responsibilities

noema-core

The central runtime.

Contains:

* Agent runtime.
* Session management.
* Model routing.
* Tool orchestration.
* Agent state.
* Core traits.
* Configuration.

noema-api

Public frontend-facing Rust API.

Contains:

* Sessions.
* Messages.
* Events.
* Approval API.
* Streaming interface.

noema-rig

Integration between Noema and Rig.

Contains:

* Rig adapters.
* Agent integration.
* Model adapters.
* Provider integration.

noema-gemma

Gemma 4 integration.

Contains:

* litert-lm-rust integration.
* Multimodal requests.
* Streaming.
* Gemma-specific message handling.

noema-needle

Needle 2 integration.

Contains:

* Needle Rust binding integration.
* Needle inference interface.
* Needle-specific request/response types.

The actual Needle binding remains an external project.

noema-memory

Mnemo integration.

Contains:

* Memory retrieval.
* Memory insertion.
* Context conversion.
* Memory policies.

noema-context

Context construction and optimization.

Contains:

* Conversation context.
* Memory context.
* Tool summaries.
* Prompt construction.
* Context trimming.

noema-events

Event definitions and streaming infrastructure.

noema-approval

Human approval infrastructure.

Contains:

* Approval requests.
* Approval state.
* Risk policies.
* Approval lifecycle.

noema-tools

Common tool infrastructure.

Contains:

* Tool traits.
* Registry.
* Schemas.
* Metadata.
* Risk levels.

⸻

68. Development Phases

Implementation status:

* Phase 1 (Workspace Foundation) — done.
* Phase 2 (Model Abstractions) — done.
* Phase 3 (Gemma 4) — done. litert-lm-rust is vendored at `crates/litert-lm-rust`;
  the Gemma adapter (`crates/noema-gemma`) streams tokens, keeps multi-turn
  memory by replaying a Rust-side history as the native conversation preface,
  reports usage, supports system prompts, text, and image/audio content parts,
  and honours cancellation. The LiteRT-LM DLLs live in `prebuilt/` and are
  staged next to every executable at build time (`crates/noema-native`); the
  model file lives in `models/` (overridable with `NOEMA_GEMMA_MODEL`).
* Phase 4 (Needle 2) — done. The Needle engine (`prebuilt/needle/`) is driven
  through the official C API (`crates/noema-needle`): structured tool calls,
  multi-turn conversations, refusal/escalation, and a CLI fallback.
* Phase 5 (Initial Text Router) — done. `crates/noema-router` routes every
  plain-text user request through Needle 2 (`NeedleRouter`) before the
  reasoning model runs: the default registry covers the six simple Agora
  actions, the router acts only at or above a confidence threshold
  (escalating low-confidence calls), and the runtime publishes
  `RoutingStarted` / `RoutingCompleted` / `RoutingEscalated` events while
  handled requests never invoke the model.
* Phase 6 (Tool Infrastructure) — done. `crates/noema-tools` defines the
  tool contract (`NoemaTool` trait, `ToolMetadata`/`ToolSummary`,
  `ToolSchema` with lightweight `required` validation, `ToolCall`/
  `ToolResult`, and the five `RiskLevel`s) and the central `ToolRegistry`
  with three views of every tool: `gemma_tool_section()` builds the
  schema-free "available tools" block for the Gemma system prompt,
  `needle_tools_json()`/`tool_needle_json()` emit the complete schemas the
  tool Needle agents bind to, and `execute()` validates then runs a call.
  `noema-router` adds `NeedleToolFormatter`, the per-tool logical Needle
  agent: one physical model, one agent per tool, each bound to its own
  schema and instructions, turning a semantic request into a validated
  `ToolCall` (with a confidence gate tuned lower than the router's, since
  the call is schema-validated before execution). The runtime and sessions
  accept a `ToolRegistry` plus default and per-tool formatters
  (`with_tools`/`with_tool`/`with_tool_formatter`/`with_tool_formatter_for`),
  and sessions expose `format_tool` and `execute_tool` (streaming
  `ToolStarted`/`ToolCompleted`/`ToolFailed` events). `examples/tools` runs
  the whole flow against the real Needle engine: register tools, show both
  views, format "store the note…" and "retrieve the note" into structured
  calls, execute them, and watch unsupported requests get refused.
  Registration is pure addition — no core loop changes, per the deliverable.
* Phase 7 (First Tool) — done. `crates/noema-filesearch` is the reference
  tool crate: a bounded, read-only `search_files(query, path?)` tool with
  schema, risk classification (Low), and a case-insensitive recursive
  filesystem walk (skipping `target`/`.git`/`node_modules`, capped at 25
  results). `examples/filesearch` walks the full plan chain against the
  real engines: user request → Gemma 4 semantic request (best-effort — the
  small E2B checkpoint is not reliably agentic, so it falls back to the
  user's words) → filesearch Needle agent → `search_files` call → execution
  → result back to Gemma for the final answer.
* Phase 8 (Human Approval) — done. `crates/noema-approval` implements the
  approval lifecycle: `ApprovalPolicy` (risk threshold + timeout, with
  `Critical` always requiring approval), `ApprovalRequest` (the complete
  proposal the frontend reviews), and `ApprovalStore` (pending requests
  resolved via one-shot channels). `Session::execute_tool` gates on risk —
  `ToolApprovalRequired` is published and execution waits — and
  `approve_tool`/`reject_tool` answer pending requests; undecided requests
  expire and are removed. `examples/approval` demonstrates approve, reject,
  and expire paths with a simulated destructive tool.
* Phase 10 (Full Agent Loop) — done. `Session::send` now runs the whole
  loop: the session owns the conversation (models are request-driven —
  `GemmaModel` seeds each turn from `request.messages`, and the rig adapter
  forwards full history by default), and `send` iterates model turn → tool
  intent detection (a reply naming a registered tool or its crate's short
  name) → `ToolRequested` → per-tool Needle formatting (`ToolFormatted`) →
  risk gate / approval → `ToolStarted`/`ToolCompleted` → result fed back as
  a `Role::Tool` message → next turn, bounded by
  `LimitsConfig::max_agent_iterations` and `max_tool_calls`. The dynamic
  Gemma tool summaries are injected as the request system prompt inside the
  loop. Failure recovery: when formatting fails (e.g. the model named a tool
  while declining), the model's reply becomes the final answer with an
  `Error` event rather than aborting the send; rejected approvals and tool
  failures abort. Multi-turn memory now works at the session level for any
  model. `examples/agent` runs the loop against the real engines.
* Phase 11 (Cloud Escalation) — done. Escalation is now a general,
  configurable abstraction. The runtime holds a registry of
  [`ModelProvider`]s (keyed by provider id; `NoemaBuilder::with_provider`
  registers one, `Noema::providers()` exposes them, and sessions resolve by
  the policy's `preferred_provider` or fall back to a sole registration).
  `EscalationDecision::Cloud` is fully wired: `Session::start_escalation`
  resolves the provider, enforces the per-request budget
  (`LimitsConfig::max_cloud_escalations`) and the policy's
  `maximum_latency` (a tokio timeout), runs the provider streaming
  `ModelStarted` / `ModelDelta` / `ModelCompleted` under the provider's id,
  and feeds the answer back so the local agent continues — for both
  router escalations (Needle → cloud) and mid-loop model escalations
  (Gemma → cloud). `crates/noema-provider-http` ships the general
  [`OpenAICompatibleProvider`], which speaks the OpenAI chat-completions
  protocol to Gemini, OpenAI, Ollama, vLLM, and friends from just a model
  name, base URL, and API key (the same three fields live in
  [`NoemaConfig::cloud`]), with optional SSE streaming and cancellation.
  `examples/escalation` shows the config + policy + provider wiring and
  fails gracefully without a key (no real-endpoint test ships, per the
  no-key constraint). `noema-api` re-exports the provider. Cost limits stay
  policy fields awaiting provider-reported pricing.
* Phase 12 (Multimodal Agent) — done. Text, image, and audio flow through
  one ordered [`ContentPart`] message abstraction (`ContentPart::Text` /
  `Image` / `Audio`, in any combination), multimodal user turns skip the
  text router and go straight to Gemma, and the agent loop treats them like
  any other turn — an image turn's reasoning can drive a tool call
  (`multimodal_turn_can_drive_tool_use` proves image → reasoning → tool →
  response in the loop). `examples/multimodal` runs the plan's paths on the
  real engine: mixed text/image (verified: the E2B checkpoint describes
  `red.png`), mixed text/audio (accepted and declined gracefully — the
  checkpoint has no audio channel; a future audio-capable checkpoint
  answers directly with no code changes), and an image turn whose reasoning
  attempts a filesearch tool call (best-effort: the small checkpoint is not
  reliably agentic, so the loop answers directly when it does not name the
  tool).
* Phase 13 (Observability) — done. `crates/noema-core/src/metrics.rs`
  adds a content-free [`MetricsCollector`] per runtime: every model turn
  (per-model turns, input/output tokens, latency), tool call (per-tool
  calls, failures, latency), and escalation (counts, cloud counts,
  provider latency) is aggregated and surfaced as a [`MetricsSnapshot`]
  through [`Noema::metrics`] / [`Session::metrics`]. The same numbers
  stream live on the event bus as `ModelMetrics` / `ToolMetrics` /
  `EscalationMetrics` events, and every record emits a content-free
  `tracing::debug!` line (model id, latency, tokens — never message
  content). Telemetry is privacy-aware by design: nothing the user said or
  a tool returned is recorded or logged.
* Phase 14 (Production Hardening) — done. Resource limits are now
  enforced end-to-end: `max_response_tokens` is forwarded to models as
  `max_tokens`, `max_context_tokens` trims the oldest transcript messages
  before every request (~4 chars/token estimate; the current turn is always
  kept), `max_tool_execution_seconds` times out runaway tools, and
  `max_concurrent_tools` caps concurrent execution via a semaphore (the
  loop's iteration/tool/cloud budgets were already enforced). Robust
  cancellation: `session.cancel()` propagates to models and providers, and
  timed-out tool futures are dropped, aborting their async work. Concurrency
  controls: concurrent `send` calls on one session are serialized by a
  per-session lock so the transcript and cancellation token can never race.
  Prompt-injection defences: tool results are fed back delimited
  (`<tool_result>…</tool_result>`) and framed as data-not-instructions, and
  both the agent system prompt and the cloud escalation prompt state the
  trust boundary explicitly. Schema validation of model-generated calls was
  already in place and is unchanged. Telemetry stays content-free (§13),
  cloud escalation stays opt-in with `offline_mode` always winning, and the
  public `noema-api` surface remains additive. `docs/publishing.md` covers
  building, publishing (bottom-up crates.io order, dry-run checks, native
  artifact caveats), and consuming the crates from other projects. Memory
  policy review: nothing is persisted until Mnemo lands, so there is no
  memory to leak or review yet (deferred with Phase 9).
* Phase 9 (Mnemo) — deferred (Mnemo is not yet complete).
* Phase 15 (Needle→Gemma Bridge) — done. `crates/noema-bridge` provides a
  two-tier inference session: Needle 2 with 5 stub tools runs first for
  fast, deterministic tool dispatch; when confidence is below a configurable
  threshold (default 0.6), or when Needle refuses, the same prompt is
  forwarded to Gemma 4 for full reasoning. `BridgeSession` exposes a
  `send(Message, CancellationToken)` API returning `SendOutcome`, with an
  optional `.with_gemma()` for the escalation target. `stub_registry()`
  provides the 5 placeholder tools (search, calculate, translate, summarize,
  navigate) that return basic canned results. The bridge is re-exported
  through `noema-api` (`BridgeSession`, `BridgeConfig`, `stub_registry`).

⸻

Phase 1 — Workspace Foundation

Implement:

* Cargo workspace.
* Core crates.
* Error types.
* Configuration.
* Logging.
* Basic public API.
* Session abstraction.

Deliverable:

Noema can start and create a session.

⸻

Phase 2 — Model Abstractions

Implement:

* Generic model interfaces.
* Streaming responses.
* Multimodal message types.
* Model provider abstraction.
* Cancellation.

Deliverable:

Noema can communicate with an abstract model.

⸻

Phase 3 — Gemma 4

Implement:

* litert-lm-rust integration.
* Gemma 4 request conversion.
* Streaming.
* System prompts.
* Text.
* Image.
* Audio.

Deliverable:

Noema can hold a multimodal conversation with Gemma 4.

⸻

Phase 4 — Needle 2

Implement:

* Needle Rust crate integration.
* Inference interface.
* Streaming if supported.
* Structured output handling.
* Error handling.

Deliverable:

Noema can execute Needle 2 inference.

⸻

Phase 5 — Initial Text Router

Implement:

* Router prompt.
* Simple action schema.
* Action registry.
* Escalation response.
* Frontend event.

Deliverable:

"Open flashcards"
→ Needle
→ OpenFlashcards
Complex request
→ Needle
→ Gemma

⸻

Phase 6 — Tool Infrastructure

Implement:

* Tool trait.
* Tool registry.
* Tool metadata.
* Tool schemas.
* Risk metadata.
* Tool-specific Needle agents.
* Dynamic Gemma tool summaries.

Deliverable:

A third-party noema-* crate can register a tool
without modifying the core agent loop.

⸻

Phase 7 — First Tool

Build a simple reference tool such as:

noema-filesearch

Implement:

* Schema.
* Needle instructions.
* Risk classification.
* Execution.
* Results.
* Gemma integration.

Deliverable:

Gemma
→ filesearch semantic request
→ Filesearch Needle
→ structured call
→ filesystem
→ result
→ Gemma

⸻

Phase 8 — Human Approval

Implement:

* Risk levels.
* Approval requests.
* Event streaming.
* Frontend approval API.
* Approval timeout.
* Approval rejection.
* Execution gating.

Deliverable:

High-risk tool
→ Needle
→ Noema
→ frontend
→ user approval
→ execution

⸻

Phase 9 — Mnemo

Implement:

* Mnemo adapter.
* Relevant memory retrieval.
* Memory context injection.
* Memory writing.
* Session/memory separation.

Deliverable:

Noema remembers relevant information through Mnemo.

⸻

Phase 10 — Full Agent Loop

Implement:

* Multi-step reasoning.
* Sequential tools.
* Parallel tools.
* Tool result processing.
* Loop limits.
* Failure recovery.
* Cancellation.

Deliverable:

Gemma can complete complex multi-tool tasks.

⸻

Phase 11 — Cloud Escalation

Implement:

* Generic cloud provider interface.
* Rig integration.
* Provider configuration.
* Gemma escalation.
* Needle escalation.
* Cost/latency policies.
* Privacy policies.
* Offline mode.

Deliverable:

Local model
→ determines task is too difficult
→ cloud model
→ result
→ local agent continues

⸻

Phase 12 — Multimodal Agent

Implement:

* Audio mode.
* Image mode.
* Mixed text/image input.
* Mixed text/audio input.
* Multimodal tool workflows.

Deliverable:

Audio/image
→ Gemma 4
→ reasoning
→ tools
→ response

⸻

Phase 13 — Observability

Implement:

* Structured events.
* Model metrics.
* Tool metrics.
* Latency.
* Token usage.
* Escalation tracking.
* Debug logging.
* Privacy-aware telemetry.

⸻

Phase 14 — Production Hardening

Implement:

* Resource limits.
* Robust cancellation.
* Error recovery.
* Schema validation.
* Prompt-injection defenses.
* Concurrency controls.
* Security review.
* Memory policy review.
* API stability.
* Documentation.

⸻

69. Definition of Done

Noema is considered production-ready when it can:

* Run entirely from Rust.
* Run Gemma 4 through litert-lm-rust.
* Run Needle 2 through its Rust binding.
* Use Rig as the orchestration layer.
* Use Mnemo for persistent memory.
* Route simple text actions through Needle.
* Escalate unsupported text requests to Gemma.
* Accept text input.
* Accept image input.
* Accept audio input.
* Pass audio to Gemma after system/text content.
* Expose available tools to Gemma without full schemas.
* Route semantic tool requests to the appropriate Needle instance.
* Give Needle the complete tool schema.
* Validate Needle-generated tool calls.
* Enforce tool risk levels.
* Require frontend approval for risky calls.
* Execute approved tools.
* Return tool results to Gemma.
* Support multiple tool calls.
* Support multi-step agent loops.
* Support cloud escalation from Gemma.
* Support cloud escalation from Needle.
* Keep cloud providers abstract.
* Allow tools to be distributed as crates.
* Allow tools to be added without changing the core agent loop.
* Provide a simple Rust frontend API.
* Provide streaming events.
* Support cancellation.
* Maintain ephemeral session state.
* Keep persistent memory in Mnemo.
* Provide structured observability.
* Have comprehensive unit, integration, and end-to-end tests.

⸻

70. Final Architecture

The completed architecture should ultimately resemble:

                                  AGORA
                                    │
                                    │ Rust API
                                    ▼
                              ┌───────────┐
                              │  NOEMA    │
                              │           │
                              │ API       │
                              │ Sessions  │
                              │ Events    │
                              │ Runtime   │
                              │ Context   │
                              │ Registry  │
                              │ Approval  │
                              └─────┬─────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         ▼                     ▼
                    ┌─────────┐           ┌─────────┐
                    │  Rig    │           │ Mnemo   │
                    │         │           │ Memory  │
                    └────┬────┘           └─────────┘
                         │
              ┌──────────┼──────────────┐
              │          │              │
              ▼          ▼              ▼
          ┌───────┐  ┌────────┐   ┌────────────┐
          │Needle │  │ Gemma 4│   │Cloud Model │
          │  2    │  │        │   │  Provider  │
          └───┬───┘  └───┬────┘   └────────────┘
              │          │
              │          │
              │          │
              │     ┌────┴─────────────┐
              │     │                  │
              │     ▼                  ▼
              │  Text reasoning    Audio/Image
              │
              ▼
       ┌─────────────────────┐
       │ Logical Needle      │
       │ Tool Agents         │
       │                     │
       │ Filesearch          │
       │ Flashcards          │
       │ PDF                 │
       │ Notes               │
       │ ...                 │
       └──────────┬──────────┘
                  │
                  ▼
           ┌──────────────┐
           │ Tool Schema  │
           │ + Risk       │
           └──────┬───────┘
                  │
                  ▼
           ┌──────────────┐
           │   Noema      │
           │ Validation  │
           └──────┬───────┘
                  │
          ┌───────┴────────┐
          │                │
       No Approval      Approval
          │                │
          │                ▼
          │           ┌─────────┐
          │           │ Agora   │
          │           │  User   │
          │           └────┬────┘
          │                │
          └────────┬───────┘
                   ▼
             Tool Execution
                   │
                   ▼
              Tool Result
                   │
                   ▼
                Gemma 4
                   │
                   ▼
             Final Response

The core architectural distinction should remain clear throughout implementation:

Agora is the environment.

Noema is the intelligence and orchestration layer.

Mnemo is the persistent memory.

Gemma 4 is the primary reasoning model.

Needle 2 is the efficient router and structured tool-call formatter.

Rig is the model/agent middle layer.

Tools are independent Rust crates.

This separation is what allows Noema to remain small, extensible, efficient, and model-agnostic while still being deeply integrated with Agora.