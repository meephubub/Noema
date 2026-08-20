//! Human approval infrastructure for Noema.
//!
//! Contains approval requests, approval state, risk policies, and the
//! approval lifecycle. Risky tool calls pause here until the frontend
//! responds with `Approve` or `Reject`.
//!
//! Phase 1 introduces the approval-status vocabulary; the full request and
//! lifecycle machinery lands with the tool execution milestone.

/// The lifecycle state of a human approval request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
