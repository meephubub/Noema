//! Risk-based approval policy.
//!
//! The policy decides which tool calls pause for human approval and how long
//! a pending approval stays valid. It is enforced by Noema, never by the
//! frontend and never by the model.

use std::time::Duration;

use noema_tools::RiskLevel;
use serde::{Deserialize, Serialize};

/// The policy gating tool execution on human approval.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalPolicy {
    /// Calls at or above this risk level require approval.
    ///
    /// `None` disables risk-based approval entirely. [`RiskLevel::Critical`]
    /// always requires approval regardless of this threshold.
    pub require_approval_above: Option<RiskLevel>,
    /// How long a pending approval stays valid before it expires.
    ///
    /// `None` means approvals never expire.
    pub timeout: Option<Duration>,
}

impl Default for ApprovalPolicy {
    fn default() -> Self {
        Self {
            require_approval_above: Some(RiskLevel::High),
            timeout: Some(Duration::from_secs(300)),
        }
    }
}

impl ApprovalPolicy {
    /// Whether a call at the given risk level must pause for approval.
    pub fn requires_approval(&self, risk: RiskLevel) -> bool {
        risk == RiskLevel::Critical
            || self
                .require_approval_above
                .is_some_and(|threshold| risk >= threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_gates_approval() {
        let policy = ApprovalPolicy {
            require_approval_above: Some(RiskLevel::High),
            timeout: None,
        };
        assert!(!policy.requires_approval(RiskLevel::None));
        assert!(!policy.requires_approval(RiskLevel::Low));
        assert!(!policy.requires_approval(RiskLevel::Medium));
        assert!(policy.requires_approval(RiskLevel::High));
        // Critical always requires approval, even below the threshold.
        assert!(policy.requires_approval(RiskLevel::Critical));
    }

    #[test]
    fn no_threshold_disables_approval_except_critical() {
        let policy = ApprovalPolicy {
            require_approval_above: None,
            timeout: None,
        };
        assert!(!policy.requires_approval(RiskLevel::High));
        assert!(policy.requires_approval(RiskLevel::Critical));
    }

    #[test]
    fn approval_disabled_requires_nothing() {
        let policy = ApprovalPolicy {
            require_approval_above: Some(RiskLevel::None),
            timeout: None,
        };
        // Every level >= None, so everything would require approval; that is
        // the "always ask" configuration. The disabled configuration is
        // `require_approval_above: None` (see `no_threshold_...`).
        assert!(policy.requires_approval(RiskLevel::None));
    }
}
