//! Approval request types.
//!
//! An [`ApprovalRequest`] is what the frontend sees when a tool call pauses:
//! the complete proposed call, its risk, and enough context to decide. The
//! frontend answers with [`ApprovalDecision`] via `approve_tool` /
//! `reject_tool` on the session.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::oneshot;

use crate::{ApprovalError, Result};

static APPROVAL_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A unique identifier for a pending approval.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ApprovalId(String);

impl ApprovalId {
    /// Generates a new, unique approval id.
    pub fn generate() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let counter = APPROVAL_COUNTER.fetch_add(1, Ordering::Relaxed);
        ApprovalId(format!("{nanos:020x}{counter:016x}"))
    }

    /// The raw string form of the id.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ApprovalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The lifecycle state of a human approval request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ApprovalStatus {
    /// The request is waiting for a human decision.
    Pending,
    /// The request was approved; the tool call may execute.
    Approved,
    /// The request was rejected; the tool call is cancelled.
    Rejected,
    /// The request expired before a decision was made.
    Expired,
}

/// The human's answer to a pending approval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApprovalDecision {
    /// The tool call may execute.
    Approved,
    /// The tool call is cancelled.
    Rejected,
}

/// The complete proposal the frontend must review.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApprovalRequest {
    /// Unique id; the frontend echoes it back to approve or reject.
    pub id: ApprovalId,
    /// The session that requested the call.
    pub session_id: String,
    /// The tool name.
    pub tool: String,
    /// A short description of what the tool does.
    pub description: String,
    /// The proposed arguments.
    pub arguments: Value,
    /// The tool's risk level.
    pub risk: String,
    /// When the request was created.
    pub created_at: SystemTime,
    /// When the request expires, if it does.
    pub expires_at: Option<SystemTime>,
}

impl ApprovalRequest {
    /// Builds a new request for the given call, session, and risk.
    ///
    /// `timeout` sets the expiry (`None` never expires).
    pub fn new(
        session_id: impl Into<String>,
        tool: impl Into<String>,
        description: impl Into<String>,
        arguments: Value,
        risk: impl Into<String>,
        timeout: Option<Duration>,
    ) -> Self {
        let created_at = SystemTime::now();
        Self {
            id: ApprovalId::generate(),
            session_id: session_id.into(),
            tool: tool.into(),
            description: description.into(),
            arguments,
            risk: risk.into(),
            created_at,
            expires_at: timeout.map(|t| created_at + t),
        }
    }

    /// Whether the request has expired.
    pub fn is_expired(&self) -> bool {
        self.expires_at.is_some_and(|expiry| SystemTime::now() >= expiry)
    }
}

/// A pending approval together with the channel that resolves it.
#[derive(Debug)]
pub(crate) struct PendingApproval {
    pub request: ApprovalRequest,
    pub tx: oneshot::Sender<ApprovalDecision>,
}

/// The receiving half of a pending approval: awaits the human's decision.
pub struct ApprovalHandle {
    request: ApprovalRequest,
    rx: oneshot::Receiver<ApprovalDecision>,
}

impl ApprovalHandle {
    pub(crate) fn new(request: ApprovalRequest, rx: oneshot::Receiver<ApprovalDecision>) -> Self {
        Self { request, rx }
    }

    /// The proposal the frontend is reviewing.
    pub fn request(&self) -> &ApprovalRequest {
        &self.request
    }

    /// Waits for the human's decision.
    ///
    /// `timeout` bounds the wait; `None` waits indefinitely. The waiter does
    /// not mark the request expired itself — the caller reports the outcome
    /// to the store so the frontend sees a consistent state.
    pub async fn await_decision(&mut self, timeout: Option<Duration>) -> Result<ApprovalDecision> {
        let result = match timeout {
            Some(duration) => tokio::time::timeout(duration, &mut self.rx).await,
            None => Ok((&mut self.rx).await),
        };
        match result {
            Ok(Ok(decision)) => Ok(decision),
            Ok(Err(_)) => Err(ApprovalError::Dropped(self.request.id.to_string())),
            Err(_) => Err(ApprovalError::Expired(self.request.id.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn approval_ids_are_unique() {
        let a = ApprovalId::generate();
        let b = ApprovalId::generate();
        assert_ne!(a, b);
        assert!(!a.as_str().is_empty());
    }

    #[test]
    fn statuses_are_distinct() {
        let states = [
            ApprovalStatus::Pending,
            ApprovalStatus::Approved,
            ApprovalStatus::Rejected,
            ApprovalStatus::Expired,
        ];
        for (i, a) in states.iter().enumerate() {
            for b in states.iter().skip(i + 1) {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn requests_carry_the_full_proposal() {
        let request = ApprovalRequest::new(
            "session-1",
            "delete_file",
            "Delete a file",
            json!({ "path": "/tmp/x" }),
            "high",
            Some(Duration::from_secs(60)),
        );
        assert_eq!(request.tool, "delete_file");
        assert_eq!(request.arguments["path"], "/tmp/x");
        assert!(request.expires_at.is_some());
        assert!(!request.is_expired());
    }
}
