//! Tool metadata.
//!
//! Noema deliberately keeps three distinct views of a tool:
//!
//! * **Gemma-facing** — a [`ToolSummary`]: name, owning crate, and a
//!   one-line semantic description. No schema, no risk, no arguments. This
//!   is all the reasoning model sees, so its context stays small.
//! * **Needle-facing** — the full [`ToolSchema`](crate::ToolSchema) plus any
//!   tool-provided instructions.
//! * **Execution-facing** — the [`ToolMetadata`]: name, crate, risk, and
//!   description, used by the registry and (later) the approval system.

use serde::{Deserialize, Serialize};

use crate::RiskLevel;

/// Execution-facing metadata for a registered tool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolMetadata {
    /// The tool name, e.g. `search_files`. Must be unique within a registry.
    pub name: String,
    /// The crate that provides the tool, e.g. `noema-filesearch`.
    pub crate_name: String,
    /// A short human description of what the tool does.
    pub description: String,
    /// The risk this tool's calls pose; evaluated by Noema.
    pub risk: RiskLevel,
}

/// The lightweight, Gemma-facing view of a tool.
///
/// Contains exactly what the reasoning model needs to issue a *semantic*
/// request — what the capability is and which crate owns it — and nothing
/// that would bloat its context. The full schema is resolved later by the
/// tool-specific Needle agent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolSummary {
    /// The tool name, e.g. `search_files`.
    pub name: String,
    /// The crate that provides the tool, e.g. `noema-filesearch`.
    pub crate_name: String,
    /// A one-line description of the capability.
    pub description: String,
}

impl From<&ToolMetadata> for ToolSummary {
    fn from(metadata: &ToolMetadata) -> Self {
        Self {
            name: metadata.name.clone(),
            crate_name: metadata.crate_name.clone(),
            description: metadata.description.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_derives_from_metadata_without_risk() {
        let metadata = ToolMetadata {
            name: "delete_file".into(),
            crate_name: "noema-filesearch".into(),
            description: "Delete a file".into(),
            risk: RiskLevel::High,
        };
        let summary = ToolSummary::from(&metadata);
        assert_eq!(summary.name, "delete_file");
        assert_eq!(summary.crate_name, "noema-filesearch");
        assert_eq!(summary.description, "Delete a file");
    }
}
