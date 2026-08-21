//! Tool error type for the noema-tools crate.

use thiserror::Error;

/// The result type used across noema-tools.
pub type Result<T> = std::result::Result<T, ToolError>;

/// Errors produced by tool registration, validation, and execution.
#[derive(Debug, Error)]
pub enum ToolError {
    /// No tool with this name is registered.
    #[error("tool not found: {0}")]
    NotFound(String),
    /// A tool with this name is already registered.
    #[error("tool already registered: {0}")]
    Duplicate(String),
    /// A tool call failed validation (unknown tool, wrong argument shape).
    #[error("invalid tool call: {0}")]
    InvalidCall(String),
    /// The tool execution itself failed.
    #[error("tool execution failed: {0}")]
    Execution(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_errors_display_their_category() {
        assert!(ToolError::NotFound("x".into()).to_string().contains("not found"));
        assert!(ToolError::Duplicate("x".into()).to_string().contains("already registered"));
        assert!(ToolError::InvalidCall("x".into()).to_string().contains("invalid"));
        assert!(ToolError::Execution("x".into()).to_string().contains("failed"));
    }
}
