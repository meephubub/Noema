//! Approval error type.

use thiserror::Error;

/// The result type used across noema-approval.
pub type Result<T> = std::result::Result<T, ApprovalError>;

/// Errors produced by the approval lifecycle.
#[derive(Debug, Error)]
pub enum ApprovalError {
    /// No pending approval with this id.
    #[error("no such approval request: {0}")]
    NotFound(String),
    /// The approval was already decided (approved, rejected, or expired).
    #[error("approval request {0} was already decided")]
    AlreadyDecided(String),
    /// The approval expired before a decision was made.
    #[error("approval request {0} expired")]
    Expired(String),
    /// The approval channel closed unexpectedly (the waiter vanished).
    #[error("approval request {0} lost its waiter")]
    Dropped(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approval_errors_display_their_category() {
        assert!(ApprovalError::NotFound("x".into()).to_string().contains("no such"));
        assert!(ApprovalError::AlreadyDecided("x".into())
            .to_string()
            .contains("already decided"));
        assert!(ApprovalError::Expired("x".into()).to_string().contains("expired"));
    }
}
