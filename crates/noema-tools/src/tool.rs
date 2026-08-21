//! The tool trait and the call/result types that cross the registry
//! boundary.
//!
//! The contract is deliberately small: a tool declares its metadata and
//! schema, optionally supplies extra instructions for its logical Needle
//! agent, and executes validated calls. Everything else — registration,
//! validation, formatting, risk evaluation, approval — lives outside the
//! tool crate, so a third-party `noema-*` crate can register a tool without
//! touching the core agent loop.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::metadata::{ToolMetadata, ToolSummary};
use crate::schema::ToolSchema;
use crate::Result;

/// A structured, validated tool call ready for execution.
///
/// Produced by the tool-specific Needle agent and validated against the
/// tool's [`ToolSchema`] before anything executes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// The registered tool name.
    pub tool: String,
    /// The call arguments, as a JSON object.
    pub arguments: Value,
}

impl ToolCall {
    /// Builds a call with empty arguments.
    pub fn new(tool: impl Into<String>) -> Self {
        Self {
            tool: tool.into(),
            arguments: Value::Object(Default::default()),
        }
    }

    /// Builds a call with the given argument object.
    pub fn with_arguments(tool: impl Into<String>, arguments: Value) -> Self {
        Self {
            tool: tool.into(),
            arguments,
        }
    }
}

/// The result of executing a [`ToolCall`].
///
/// Results are converted into a representation suitable for the reasoning
/// model: a plain-text `text` summary plus optional structured `data`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    /// Whether the call succeeded. Failures carry a message in `text`.
    pub success: bool,
    /// A text summary of the outcome, suitable for the reasoning model.
    pub text: String,
    /// Optional structured data produced by the tool.
    #[serde(default)]
    pub data: Option<Value>,
}

impl ToolResult {
    /// A successful result with only a text summary.
    pub fn ok(text: impl Into<String>) -> Self {
        Self {
            success: true,
            text: text.into(),
            data: None,
        }
    }

    /// A successful result with a text summary and structured data.
    pub fn ok_with_data(text: impl Into<String>, data: Value) -> Self {
        Self {
            success: true,
            text: text.into(),
            data: Some(data),
        }
    }

    /// A failed result.
    pub fn failure(message: impl Into<String>) -> Self {
        Self {
            success: false,
            text: message.into(),
            data: None,
        }
    }
}

/// The standard interface every Noema tool exposes.
///
/// Implement this trait in a `noema-*` crate and register the instance with
/// [`ToolRegistry::register`](crate::ToolRegistry::register) — no changes to
/// the core agent loop are needed.
#[async_trait]
pub trait NoemaTool: std::fmt::Debug + Send + Sync {
    /// Execution-facing metadata (name, crate, description, risk).
    fn metadata(&self) -> ToolMetadata;

    /// The full Needle-facing schema for this tool's calls.
    fn schema(&self) -> ToolSchema;

    /// Extra instructions for this tool's logical Needle agent.
    ///
    /// These are combined with the schema to form the agent's system prompt
    /// (e.g. argument-evidence rules, output expectations). Optional.
    fn needle_instructions(&self) -> Option<String> {
        None
    }

    /// Executes a validated call.
    async fn execute(&self, call: ToolCall) -> Result<ToolResult>;
}

/// Returns the lightweight Gemma-facing summary for a tool.
pub fn summarize(tool: &dyn NoemaTool) -> ToolSummary {
    ToolSummary::from(&tool.metadata())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calls_carry_tool_and_arguments() {
        let call = ToolCall::with_arguments("notes", serde_json::json!({ "id": 3 }));
        assert_eq!(call.tool, "notes");
        assert_eq!(call.arguments["id"], 3);
        assert_eq!(ToolCall::new("ping").arguments, Value::Object(Default::default()));
    }

    #[test]
    fn results_distinguish_success_and_failure() {
        let ok = ToolResult::ok_with_data("3 notes", serde_json::json!([1, 2, 3]));
        assert!(ok.success);
        assert_eq!(ok.data, Some(serde_json::json!([1, 2, 3])));
        let err = ToolResult::failure("permission denied");
        assert!(!err.success);
        assert!(err.text.contains("permission denied"));
    }
}
