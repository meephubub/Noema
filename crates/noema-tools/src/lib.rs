//! Common tool infrastructure for Noema: tool traits, registry, schemas,
//! metadata, and risk levels.
//!
//! Phase 1 provides the risk-level vocabulary used by configuration and,
//! later, by the tool registry and approval system. The [`NoemaTool`] trait,
//! the registry, and the schema types land with the tool infrastructure
//! milestone.

use std::fmt;

use serde::{Deserialize, Serialize};

/// The risk a tool call poses to the system or the user.
///
/// The ordering is significant: `None < Low < Medium < High < Critical`.
/// Risk is evaluated by Noema, never by the frontend.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum RiskLevel {
    /// No risk; no approval required.
    None,
    /// Minimal, reversible risk.
    Low,
    /// Moderate risk; may warrant approval depending on policy.
    Medium,
    /// Significant, possibly irreversible risk.
    High,
    /// Severe risk; always requires human approval.
    Critical,
}

impl Default for RiskLevel {
    fn default() -> Self {
        RiskLevel::None
    }
}

impl fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            RiskLevel::None => "none",
            RiskLevel::Low => "low",
            RiskLevel::Medium => "medium",
            RiskLevel::High => "high",
            RiskLevel::Critical => "critical",
        };
        f.write_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn risk_levels_are_ordered() {
        assert!(RiskLevel::None < RiskLevel::Low);
        assert!(RiskLevel::Low < RiskLevel::Medium);
        assert!(RiskLevel::Medium < RiskLevel::High);
        assert!(RiskLevel::High < RiskLevel::Critical);
    }

    #[test]
    fn risk_levels_round_trip_serde() {
        let json = serde_json::to_string(&RiskLevel::High).expect("serialize");
        assert_eq!(json, "\"high\"");
        let back: RiskLevel = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, RiskLevel::High);
    }
}
