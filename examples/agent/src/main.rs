//! Phase 10 demo: the full agent loop.
//!
//! A single `session.send(...)` now runs the whole loop from the plan:
//!
//! ```text
//! user message
//!     ↓
//! Gemma 4 turn (streamed)
//!     ├── reply names a registered tool → ToolRequested
//!     │     → Needle formatter → ToolFormatted
//!     │     → risk gate / approval → ToolStarted → ToolCompleted
//!     │     → result fed back → next Gemma turn
//!     └── otherwise → the reply is the final answer
//! ```
//!
//! The session owns the conversation, so multi-turn memory and tool results
//! survive across loop iterations and across `send` calls — for any model.
//!
//! Usage:
//!
//! ```sh
//! cargo run -p agent-example
//! ```
//!
//! Needs the Needle engine (`prebuilt/needle/`) and the Gemma model
//! (`models/`). The small E2B checkpoint is not reliably agentic, so it may
//! simply answer instead of naming a tool — the loop handles both.

use noema_core::{init_logging, LogLevel, Message, ModelResponse, Noema, Role, SendOutcome};
use noema_events::Event;
use noema_filesearch::{Filesearch, TOOL_NAME};
use noema_gemma::GemmaModel;
use noema_router::NeedleToolFormatter;
use noema_tools::ToolRegistry;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging(LogLevel::Info)?;

    // A small, self-contained study directory under the current directory
    // (the tool's default search root).
    let study_dir = std::env::current_dir().expect("cwd").join(".noema-demo");
    std::fs::create_dir_all(&study_dir)?;
    for name in ["notes.txt", "exam-schedule.txt"] {
        let path = study_dir.join(name);
        if !path.exists() {
            std::fs::write(path, "demo content")?;
        }
    }

    // Register the tool and its logical Needle agent.
    let mut registry = ToolRegistry::new();
    registry.register(Filesearch::default())?;
    let schema = registry.get(TOOL_NAME).expect(TOOL_NAME).schema();

    let noema = Noema::builder()
        .with_model(GemmaModel::from_default()?)
        .with_tools(registry.clone())
        .with_tool_formatter_for(TOOL_NAME, NeedleToolFormatter::from_tool(&schema, None)?)
        .build()
        .await?;
    let session = noema.create_session().await?;
    let mut events = session.events();

    println!(">>> user: search for notes.txt\n");
    let outcome = session
        .send(Message::text(Role::User, "search for notes.txt"))
        .await?;

    let mut tool_steps = 0usize;
    println!("--- event stream ---");
    // `send` has already finished by the time we drain, so a short quiet
    // window means the stream is exhausted (the bus stays open for the
    // session's lifetime).
    while let Ok(Some(event)) =
        tokio::time::timeout(std::time::Duration::from_millis(250), events.next()).await
    {
        match event {
            Event::ModelStarted { .. } => println!("model: started"),
            Event::ModelDelta { delta, .. } => print!("{delta}"),
            Event::ModelCompleted { .. } => println!(),
            Event::ToolRequested { .. } => {
                tool_steps += 1;
                println!("tool: semantic request detected");
            }
            Event::ToolFormatted { .. } => println!("tool: formatted into a structured call"),
            Event::ToolStarted { .. } => println!("tool: executing…"),
            Event::ToolCompleted { .. } => println!("tool: completed"),
            Event::ToolFailed { .. } => println!("tool: FAILED"),
            Event::ToolApprovalRequired { .. } => println!("tool: approval required"),
            Event::Error { error, .. } => println!("error: {error}"),
            _ => {}
        }
    }

    match outcome {
        SendOutcome::Routed(action) => println!("\n<<< routed to {} (no model needed)", action.id),
        SendOutcome::Model(ModelResponse::Text { content, .. }) => {
            println!("\n<<< final answer: {content}");
        }
        SendOutcome::Model(ModelResponse::Escalate(request)) => {
            println!("\n<<< escalated: {}", request.reason);
        }
        SendOutcome::Model(other) => println!("\n<<< {other:?}"),
    }
    if tool_steps > 0 {
        println!("\nThe agent loop ran {tool_steps} tool step(s) before answering.");
    }

    session.close().await?;
    std::fs::remove_dir_all(&study_dir).ok();
    Ok(())
}
