//! Phase 8 end-to-end: human approval for risky tool calls.
//!
//! Demonstrates the full gate from the plan:
//!
//! ```text
//! tool call (Critical risk)
//!     ↓
//! risk evaluation → approval required
//!     ↓
//! ToolApprovalRequired event → frontend sees the complete proposal
//!     ↓
//! approve_tool / reject_tool
//!     ├── approved → ToolStarted → ToolCompleted
//!     └── rejected → ToolRejected (the tool never runs)
//! ```
//!
//! No Needle or Gemma model is needed — the calls are built by hand to keep
//! the approval mechanics the focus.
//!
//! Usage:
//!
//! ```sh
//! cargo run -p approval-example
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use noema_core::{init_logging, ApprovalPolicy, LogLevel, Noema};
use noema_events::Event;
use noema_tools::{NoemaTool, RiskLevel, ToolCall, ToolMetadata, ToolResult, ToolSchema};
use serde_json::json;

/// A simulated destructive tool: it only records what it "deleted".
#[derive(Debug)]
struct DeleteFile {
    deleted: Arc<AtomicUsize>,
}

#[async_trait]
impl NoemaTool for DeleteFile {
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            name: "delete_file".into(),
            crate_name: "noema-approval-example".into(),
            description: "Permanently delete a file".into(),
            risk: RiskLevel::Critical,
        }
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "delete_file".into(),
            description: "Permanently delete a file".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "the file to delete" }
                },
                "required": ["path"]
            }),
        }
    }

    async fn execute(&self, call: ToolCall) -> noema_tools::Result<ToolResult> {
        self.deleted.fetch_add(1, Ordering::SeqCst);
        let path = call.arguments.get("path").cloned().unwrap_or_default();
        Ok(ToolResult::ok(format!("deleted {}", path)))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging(LogLevel::Info)?;

    let deleted = Arc::new(AtomicUsize::new(0));
    let mut registry = noema_tools::ToolRegistry::new();
    registry.register(DeleteFile {
        deleted: Arc::clone(&deleted),
    })?;

    // Default policy: approval required at/above High; Critical always asks.
    let noema1 = Noema::builder()
        .with_tools(registry.clone())
        .build()
        .await?;
    let session1 = noema1.create_session().await?;
    let mut events = session1.events();

    let call = ToolCall::with_arguments("delete_file", json!({ "path": "/tmp/report.pdf" }));

    println!(">>> executing {call:?} (Critical risk — approval required)");
    let handle = tokio::spawn({
        let session = session1.clone();
        async move { session.execute_tool(call).await }
    });

    // The frontend receives the proposal and reviews it.
    while session1.pending_approvals().is_empty() {
        tokio::task::yield_now().await;
    }
    let pending = session1.pending_approvals();
    let request = &pending[0];
    println!(
        "\n=== approval required ===\n\
         id:          {}\n\
         tool:        {} — {}\n\
         arguments:   {}\n\
         risk:        {}\n\
         expires:     {}\n",
        request.id,
        request.tool,
        request.description,
        request.arguments,
        request.risk,
        request
            .expires_at
            .map(|t| format!("{t:?}"))
            .unwrap_or_else(|| "never".into()),
    );

    // Scenario A: the user approves → the call executes.
    let id = request.id.clone();
    println!(">>> user approves ({id})");
    session1.approve_tool(id)?;
    let result = handle.await??;
    println!("<<< result: {} (deleted count = {})\n", result.text, deleted.load(Ordering::SeqCst));

    // Scenario B: the user rejects → the call never executes.
    let call = ToolCall::with_arguments("delete_file", json!({ "path": "/tmp/notes.txt" }));
    let handle = tokio::spawn({
        let session = session1.clone();
        async move { session.execute_tool(call).await }
    });
    while session1.pending_approvals().is_empty() {
        tokio::task::yield_now().await;
    }
    let id = session1.pending_approvals()[0].id.clone();
    println!(">>> user rejects ({id})");
    session1.reject_tool(id)?;
    let err = handle.await?.expect_err("rejected call errors");
    println!("<<< rejected: {err} (deleted count = {})\n", deleted.load(Ordering::SeqCst));

    // Scenario C: the user never answers → the approval expires.
    let noema2 = Noema::builder()
        .with_tools(registry.clone())
        .with_approval_policy(ApprovalPolicy {
            require_approval_above: Some(RiskLevel::High),
            timeout: Some(std::time::Duration::from_millis(150)),
        })
        .build()
        .await?;
    let session2 = noema2.create_session().await?;
    let call = ToolCall::with_arguments("delete_file", json!({ "path": "/tmp/old.txt" }));
    let handle = tokio::spawn({
        let session = session2.clone();
        async move { session.execute_tool(call).await }
    });
    let err = handle.await?.expect_err("expired approval errors");
    println!(">>> nobody answered — approval expired\n<<< {err} (deleted count = {})", deleted.load(Ordering::SeqCst));

    // Close the first session (its bus stays open otherwise, so the drain
    // below needs a terminating event).
    noema1.close_session(&session1.id()).await?;
    noema2.close_session(&session2.id()).await?;

    println!("\n--- approval events ---");
    while let Some(event) = events.next().await {
        match event {
            Event::ToolApprovalRequired { .. }
            | Event::ToolApproved { .. }
            | Event::ToolRejected { .. }
            | Event::ToolStarted { .. }
            | Event::ToolCompleted { .. }
            | Event::ToolFailed { .. } => println!("{event:?}"),
            _ => {}
        }
        // ToolRejected is the last approval event of the first session;
        // the expiry scenario runs on a second runtime and emits on its own
        // bus.
        if matches!(event, Event::ToolRejected { .. }) {
            break;
        }
    }

    println!("\nPolicy note: Critical-risk tools always require approval; the");
    println!("threshold is configurable via NoemaConfig::risk.require_approval_above");
    println!("or Noema::builder().with_approval_policy(..).");
    Ok(())
}
