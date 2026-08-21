//! Tool risk levels.
//!
//! Risk is defined inside each tool's metadata and evaluated by Noema — never
//! by the frontend and never by the model. The approval system (later
//! milestone) gates execution above a configured threshold.

use std::fmt;

use serde::{Deserialize, Serialize};

/// The risk a tool call poses to the system or the user.
///
/// The ordering is significant: `None < Low < Medium < High < Critical`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum RiskLevel {
    /// No risk; no approval required.
    #[default]
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
