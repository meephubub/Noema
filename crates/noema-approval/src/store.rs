//! The store of pending approvals.
//!
//! [`ApprovalStore`] tracks every approval waiting on a human decision. It
//! is shared (cheap to clone) so a session can create requests and the
//! frontend API can resolve them from the same state.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::request::{ApprovalDecision, ApprovalHandle, ApprovalRequest, PendingApproval};
use crate::{ApprovalError, Result};

/// Tracks pending human approvals.
#[derive(Debug, Default, Clone)]
pub struct ApprovalStore {
    pending: Arc<Mutex<HashMap<String, PendingApproval>>>,
}

impl ApprovalStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new pending approval and returns the waiting handle.
    ///
    /// The handle's [`ApprovalHandle::await_decision`] resolves when the
    /// frontend calls [`ApprovalStore::decide`] (or times out).
    pub fn create(&self, request: ApprovalRequest) -> ApprovalHandle {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let id = request.id.as_str().to_string();
        let handle = ApprovalHandle::new(request.clone(), rx);
        self.pending
            .lock()
            .expect("approval store poisoned")
            .insert(id, PendingApproval { request, tx });
        handle
    }

    /// Resolves a pending approval with the human's decision.
    ///
    /// Fails if the id is unknown or was already decided.
    pub fn decide(&self, id: &str, decision: ApprovalDecision) -> Result<()> {
        let pending = self
            .pending
            .lock()
            .expect("approval store poisoned")
            .remove(id)
            .ok_or_else(|| ApprovalError::NotFound(id.to_string()))?;
        pending
            .tx
            .send(decision)
            .map_err(|_| ApprovalError::Dropped(id.to_string()))
    }

    /// Marks a pending approval as expired (called by the waiter on timeout).
    pub fn expire(&self, id: &str) -> Result<()> {
        self.pending
            .lock()
            .expect("approval store poisoned")
            .remove(id)
            .map(|_| ())
            .ok_or_else(|| ApprovalError::NotFound(id.to_string()))
    }

    /// Whether an approval with this id is still pending.
    pub fn is_pending(&self, id: &str) -> bool {
        self.pending
            .lock()
            .expect("approval store poisoned")
            .contains_key(id)
    }

    /// The pending approval with this id, if any.
    pub fn get(&self, id: &str) -> Option<ApprovalRequest> {
        self.pending
            .lock()
            .expect("approval store poisoned")
            .get(id)
            .map(|p| p.request.clone())
    }

    /// Every pending approval (in arbitrary order; sort by id if order
    /// matters).
    pub fn pending(&self) -> Vec<ApprovalRequest> {
        self.pending
            .lock()
            .expect("approval store poisoned")
            .values()
            .map(|p| p.request.clone())
            .collect()
    }

    /// How many approvals are pending.
    pub fn len(&self) -> usize {
        self.pending.lock().expect("approval store poisoned").len()
    }

    /// Whether no approvals are pending.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Convenience alias for a shared store.
pub type SharedApprovalStore = Arc<ApprovalStore>;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Duration;

    fn request(tool: &str) -> ApprovalRequest {
        ApprovalRequest::new(
            "session-1",
            tool,
            format!("{tool} tool"),
            json!({}),
            "high",
            Some(Duration::from_secs(60)),
        )
    }

    #[tokio::test]
    async fn decide_resolves_the_waiter() {
        let store = ApprovalStore::new();
        let mut handle = store.create(request("delete_file"));
        assert_eq!(store.len(), 1);
        assert!(store.is_pending(&handle.request().id.to_string()));

        let id = handle.request().id.to_string();
        store.decide(&id, ApprovalDecision::Approved).expect("decide");
        assert_eq!(store.len(), 0);

        let decision = handle.await_decision(None).await.expect("decision");
        assert_eq!(decision, ApprovalDecision::Approved);
    }

    #[tokio::test]
    async fn deciding_twice_fails() {
        let store = ApprovalStore::new();
        let handle = store.create(request("delete_file"));
        let id = handle.request().id.to_string();
        store.decide(&id, ApprovalDecision::Rejected).expect("first");
        let err = store
            .decide(&id, ApprovalDecision::Approved)
            .expect_err("already decided");
        assert!(matches!(err, ApprovalError::NotFound(_)));
        assert!(store.is_empty());
    }

    #[tokio::test]
    async fn unknown_decision_fails() {
        let store = ApprovalStore::new();
        let err = store
            .decide("nope", ApprovalDecision::Approved)
            .expect_err("unknown");
        assert!(matches!(err, ApprovalError::NotFound(_)));
    }

    #[tokio::test]
    async fn timeout_expires_the_waiter() {
        let store = ApprovalStore::new();
        let mut handle = store.create(request("delete_file"));
        let id = handle.request().id.to_string();

        let err = handle
            .await_decision(Some(Duration::from_millis(20)))
            .await
            .expect_err("times out");
        assert!(matches!(err, ApprovalError::Expired(_)));
        store.expire(&id).expect("expire");
        assert!(store.is_empty());
    }

    #[test]
    fn pending_lists_all_requests() {
        let store = ApprovalStore::new();
        store.create(request("a"));
        store.create(request("b"));
        let mut tools: Vec<String> = store.pending().into_iter().map(|r| r.tool).collect();
        tools.sort();
        assert_eq!(tools, vec!["a".to_string(), "b".to_string()]);
    }
}
