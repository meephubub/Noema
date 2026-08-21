//! Tool infrastructure end-to-end example.
//!
//! Demonstrates the Phase 6 contract: a third-party `noema-*` crate
//! implements [`NoemaTool`], registers with a [`ToolRegistry`], and from
//! then on everything is infrastructure:
//!
//! ```text
//! registered tools
//!     ├── gemma_tool_section()  → what the reasoning model sees (no schemas)
//!     ├── needle_tools_json()   → the complete schemas (tool Needle agents)
//!     └── NeedleToolFormatter   → semantic request → structured call → execute
//! ```
//!
//! The tools are deliberately tiny and local: `store_note` and `recall_note`
//! (a scratchpad stored in the tool instance).
//!
//! Usage:
//!
//! ```sh
//! cargo run -p tools-example
//! ```
//!
//! Needs the Needle engine (`prebuilt/needle/`) for the formatting step.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use noema_core::{init_logging, LogLevel, Noema};
use noema_events::Event;
use noema_router::NeedleToolFormatter;
use noema_tools::{NoemaTool, RiskLevel, ToolCall, ToolMetadata, ToolRegistry, ToolResult, ToolSchema};
use serde_json::json;

/// The current UTC time, as seconds since the epoch.
#[derive(Debug)]
struct CurrentTime;

#[async_trait]
impl NoemaTool for CurrentTime {
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            name: "current_time".into(),
            crate_name: "noema-tools-example".into(),
            description: "Get the current UTC time".into(),
            risk: RiskLevel::None,
        }
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "current_time".into(),
            description: "Get the current UTC time".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        }
    }

    async fn execute(&self, _call: ToolCall) -> noema_tools::Result<ToolResult> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let seconds = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Ok(ToolResult::ok(format!(
            "the current UTC time is {seconds} seconds since the epoch"
        )))
    }
}

/// A tiny scratchpad shared by `store_note` and `recall_note`.
#[derive(Debug, Clone)]
struct NotePad {
    stored: Arc<Mutex<Option<String>>>,
}

/// Stores a short note for later recall.
#[derive(Debug)]
struct StoreNote(NotePad);

#[async_trait]
impl NoemaTool for StoreNote {
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            name: "store_note".into(),
            crate_name: "noema-tools-example".into(),
            description: "Store a short note for later recall".into(),
            risk: RiskLevel::Low,
        }
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "store_note".into(),
            description: "Store a short note for later recall".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "the note text to store"
                    }
                },
                "required": ["text"]
            }),
        }
    }

    async fn execute(&self, call: ToolCall) -> noema_tools::Result<ToolResult> {
        let text = call
            .arguments
            .get("text")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string();
        *self.0.stored.lock().expect("note lock") = Some(text.clone());

        Ok(ToolResult::ok(format!("note stored: {text}")))
    }
}

/// Retrieves the previously stored note.
#[derive(Debug)]
struct RecallNote(NotePad);

#[async_trait]
impl NoemaTool for RecallNote {
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            name: "recall_note".into(),
            crate_name: "noema-tools-example".into(),
            description: "Retrieve the previously stored note".into(),
            risk: RiskLevel::None,
        }
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "recall_note".into(),
            description: "Retrieve the previously stored note".into(),
            parameters: json!({ "type": "object", "properties": {} }),
        }
    }

    async fn execute(&self, _call: ToolCall) -> noema_tools::Result<ToolResult> {
        match self.0.stored.lock().expect("note lock").as_deref() {
            Some(text) => Ok(ToolResult::ok(text.to_string())),
            None => Ok(ToolResult::ok("no note stored yet".to_string())),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging(LogLevel::Info)?;

    // A third-party crate registers its tools. No core changes needed.
    let notepad = NotePad {
        stored: Arc::new(Mutex::new(None)),
    };
    let mut registry = ToolRegistry::new();
    registry.register(CurrentTime)?;
    registry.register(StoreNote(notepad.clone()))?;
    registry.register(RecallNote(notepad))?;

    println!("=== what Gemma sees (no schemas) ===\n{}", registry.gemma_tool_section());
    println!("=== what the tool Needle agents bind to ===");
    for name in registry.names() {
        println!("{}", registry.tool_needle_json(&name)?);
    }

    // Build the runtime with the tools and one logical Needle agent per
    // tool — each binds its own schema, so `store_note` and `recall_note`
    // have dedicated formatters.
    let store_schema = registry.get("store_note").expect("store_note").schema();
    let recall_schema = registry.get("recall_note").expect("recall_note").schema();
    let noema = Noema::builder()
        .with_tools(registry.clone())
        .with_tool_formatter_for("store_note", NeedleToolFormatter::from_tool(&store_schema, None)?)
        .with_tool_formatter_for("recall_note", NeedleToolFormatter::from_tool(&recall_schema, None)?)
        .build()
        .await?;
    let session = noema.create_session().await?;
    let mut events = session.events();

    // Semantic request → structured call (Needle) → validated execution.
    println!("\n=== formatting + execution ===");
    let call = session
        .format_tool(store_schema.clone(), "store the note: the exam is on Friday")
        .await?;
    println!("formatted call: {} {:?}", call.tool, call.arguments);
    let result = session.execute_tool(call).await?;
    println!("result: {}", result.text);

    // The formatter is bound to one tool; a second agent handles recall.
    let call = session
        .format_tool(recall_schema.clone(), "retrieve the note")
        .await?;
    println!("formatted call: {} {:?}", call.tool, call.arguments);
    let result = session.execute_tool(call).await?;
    println!("result: {}", result.text);

    // A request no tool can serve is refused rather than hallucinated.
    let refused = session
        .format_tool(store_schema, "tell me a joke about ducks")
        .await
        .expect_err("the engine refuses unsupported requests");
    println!("\nunsupported request → {refused}");

    session.close().await?;

    println!("\n--- tool events ---");
    while let Some(event) = events.next().await {
        match event {
            Event::ToolStarted { .. } | Event::ToolCompleted { .. } | Event::ToolFailed { .. } => {
                println!("{event:?}")
            }
            _ => {}
        }
        if matches!(event, Event::SessionCompleted { .. }) {
            break;
        }
    }

    Ok(())
}
