//! Phase 7 end-to-end: the first real tool.
//!
//! Walks the full chain from the plan:
//!
//! ```text
//! user request
//!     ↓
//! Gemma 4  →  semantic request ("find the file notes.txt")
//!     ↓
//! Filesearch Needle agent  →  search_files({ query: "notes.txt" })
//!     ↓
//! Filesearch execution  →  matching paths
//!     ↓
//! Gemma 4  →  final answer
//! ```
//!
//! `noema-filesearch` is a read-only, Low-risk tool, so it executes without
//! approval (the approval flow is demonstrated by `examples/approval`).
//!
//! Usage:
//!
//! ```sh
//! cargo run -p filesearch-example
//! ```
//!
//! Needs the Needle engine (`prebuilt/needle/`); the Gemma model (`models/`)
//! is used when present, otherwise the user request doubles as the semantic
//! request so the formatting + execution half of the chain still runs.

use noema_core::{
    init_logging, LogLevel, Message, Model, ModelRequest, ModelResponse, Noema, Result as NoemaResult,
    Role,
};
use noema_filesearch::{Filesearch, TOOL_NAME};
use noema_gemma::GemmaModel;
use noema_router::NeedleToolFormatter;
use noema_tools::{ToolRegistry, ToolResult};
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging(LogLevel::Info)?;

    // A small, self-contained study directory to search. It lives under the
    // current directory because the engine extracts the `path` argument
    // unreliably (a base-model quirk with absolute paths) — the tool's
    // default root is the current directory, which finds these files.
    let study_dir = std::env::current_dir().expect("cwd").join(".noema-demo");
    std::fs::create_dir_all(study_dir.join("inflation")).ok();
    for name in ["notes.txt", "exam-schedule.txt", "inflation/summary.txt", "recipes.md"] {
        let path = study_dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        if !path.exists() {
            std::fs::write(path, "demo content").ok();
        }
    }

    // Register the tool and its logical Needle agent.
    let mut registry = ToolRegistry::new();
    registry.register(Filesearch::default())?;
    let schema = registry.get(TOOL_NAME).expect(TOOL_NAME).schema();
    let noema = Noema::builder()
        .with_tools(registry.clone())
        .with_tool_formatter_for(TOOL_NAME, NeedleToolFormatter::from_tool(&schema, None)?)
        .build()
        .await?;
    let session = noema.create_session().await?;

    // Gemma 4 produces the semantic request (optional: falls back to the
    // user's own words when the model is unavailable). The system prompt
    // carries the dynamic Gemma tool summary — name, crate, and a one-line
    // description, never the schema — so Gemma can issue a semantic request
    // that the filesearch Needle agent will format.
    let gemma = GemmaModel::from_default().ok();
    let user_request = "search for notes.txt";
    let gemma_system = format!(
        "You are a tool dispatcher inside Noema. You never answer the user \
         directly and you never run tools yourself. The tools available are:\n\
         {}\n\
         Reply with exactly one short sentence describing the tool call you \
         want to make, using the exact tool name, for example: \"search for \
         the file X using search_files\".",
        registry.gemma_tool_section()
    );
    let mut semantic = user_request.to_string();
    if let Some(model) = &gemma {
        println!(">>> user: {user_request}");
        println!("... Gemma 4 decides what to do");
        let response = model
            .generate(
                ModelRequest::new(vec![Message::text(
                    Role::User,
                    format!("{user_request}. The files are under {}.", study_dir.display()),
                )])
                .with_system(gemma_system),
                Default::default(),
            )
            .await?;
        let content = drain_response(response).await?;
        println!("<<< Gemma: {}", content.lines().next().unwrap_or(""));
        // The small local model is not reliably agentic; if its answer does
        // not survive formatting, fall back to the user's own words below.
        semantic = content;
    } else {
        println!("(no Gemma model found — using the user request as the semantic request)");
    }

    // Filesearch Needle agent: semantic request → structured call. If the
    // model's answer does not format (a refusal, or an off-topic reply),
    // retry with the user's original request.
    let call = match session.format_tool(schema.clone(), &semantic).await {
        Ok(call) => call,
        Err(first_error) => {
            println!("... Gemma's answer did not format ({first_error}); retrying with the user request");
            session.format_tool(schema.clone(), user_request).await?
        }
    };
    println!("\n>>> formatted call: {} {:?}", call.tool, call.arguments);

    // Execution (Low risk → no approval needed).
    let result = session.execute_tool(call).await?;
    println!("<<< tool result:\n{}\n", result.text);

    // The result goes back to Gemma for the final answer.
    if let Some(model) = &gemma {
        let answer = model
            .generate(
                ModelRequest::new(vec![
                    Message::text(Role::User, user_request),
                    Message::text(Role::Tool, tool_result_text(&result)),
                ]),
                Default::default(),
            )
            .await?;
        let content = drain_response(answer).await?;
        println!("<<< Gemma: {content}");
    }

    session.close().await?;
    std::fs::remove_dir_all(&study_dir).ok();
    println!("\nDone.");
    Ok(())
}

/// A stable text form of a tool result for the model.
fn tool_result_text(result: &ToolResult) -> String {
    if result.success {
        format!("Tool result: {}", result.text)
    } else {
        format!("Tool failed: {}", result.text)
    }
}

/// Drains a model response into its full text.
///
/// Gemma streams by default; the session does the same drain internally, so
/// this mirrors it for direct model calls.
async fn drain_response(response: ModelResponse) -> NoemaResult<String> {
    match response {
        ModelResponse::Text { content, .. } => Ok(content),
        ModelResponse::Stream(mut stream) => {
            let mut text = String::new();
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(chunk) => text.push_str(&chunk.delta),
                    Err(error) => return Err(error),
                }
            }
            Ok(text)
        }
        ModelResponse::Escalate(request) => Ok(format!("[escalated: {}]", request.reason)),
    }
}
