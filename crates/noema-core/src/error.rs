//! Strongly typed error categories for Noema.

use noema_tools::ToolError;
use thiserror::Error;

/// The result type used across Noema.
pub type Result<T> = std::result::Result<T, NoemaError>;

/// Strongly typed errors, one category per major system in Noema.
///
/// Errors carry enough information for logging and debugging without leaking
/// sensitive information to the frontend.
#[derive(Debug, Error)]
pub enum NoemaError {
    /// A failure inside a model backend (Gemma, cloud, or other).
    #[error("model error: {0}")]
    Model(String),
    /// A failure in the Needle integration or its output.
    #[error("needle error: {0}")]
    Needle(String),
    /// A failure in the initial text router.
    #[error("router error: {0}")]
    Router(String),
    /// A failure during tool execution.
    #[error("tool error: {0}")]
    Tool(String),
    /// A failure in the Mnemo memory integration.
    #[error("memory error: {0}")]
    Memory(String),
    /// A failure in the human approval flow.
    #[error("approval error: {0}")]
    Approval(String),
    /// A failure while assembling or optimizing context.
    #[error("context error: {0}")]
    Context(String),
    /// A failure during model escalation.
    #[error("escalation error: {0}")]
    Escalation(String),
    /// A failure tied to session lifecycle or state.
    #[error("session error: {0}")]
    Session(String),
    /// Invalid or unsupported configuration.
    #[error("configuration error: {0}")]
    Configuration(String),
}

impl From<ToolError> for NoemaError {
    fn from(error: ToolError) -> Self {
        NoemaError::Tool(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_display_their_category() {
        assert!(NoemaError::Model("boom".into()).to_string().contains("model"));
        assert!(NoemaError::Needle("boom".into()).to_string().contains("needle"));
        assert!(NoemaError::Router("boom".into()).to_string().contains("router"));
        assert!(NoemaError::Tool("boom".into()).to_string().contains("tool"));
        assert!(NoemaError::Memory("boom".into()).to_string().contains("memory"));
        assert!(NoemaError::Approval("boom".into()).to_string().contains("approval"));
        assert!(NoemaError::Context("boom".into()).to_string().contains("context"));
        assert!(NoemaError::Escalation("boom".into()).to_string().contains("escalation"));
        assert!(NoemaError::Session("boom".into()).to_string().contains("session"));
        assert!(NoemaError::Configuration("boom".into())
            .to_string()
            .contains("configuration"));
    }

    #[test]
    fn session_error_flows_through_result() {
        fn fail() -> Result<()> {
            Err(NoemaError::Session("gone".into()))
        }
        assert!(fail().is_err());
    }
}
