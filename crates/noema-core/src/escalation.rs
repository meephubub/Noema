//! Escalation policy.
//!
//! When the initial text router escalates a request (a low-confidence route,
//! a refusal, or any request no simple action matches), the runtime consults
//! an [`EscalationPolicy`] to decide where the request goes next. The policy
//! encodes the plan's escalation knobs — whether local reasoning is allowed,
//! whether cloud escalation is permitted, and the offline rule that data
//! never leaves the machine.
//!
//! The default policy escalates to the local reasoning model (Gemma) and
//! never touches the cloud, matching Noema's offline-first posture.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::config::NoemaConfig;
use crate::model::EscalationRequest;

/// Where the runtime should send an escalated request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EscalationDecision {
    /// Run the local reasoning model (Gemma).
    Local,
    /// Send the request to a cloud provider.
    Cloud,
    /// Escalation is not permitted under the current policy; the request
    /// cannot be completed.
    Denied,
}

/// Configuration governing what happens when the router escalates.
///
/// Built from [`NoemaConfig`] via [`EscalationPolicy::from_config`], or set
/// directly through [`NoemaBuilder::with_escalation_policy`]
/// (crate::NoemaBuilder::with_escalation_policy).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EscalationPolicy {
    /// Escalate to the local reasoning model when the router is uncertain.
    /// Defaults to `true`.
    pub allow_local: bool,
    /// Escalate to a cloud provider. Defaults to `false`, and is always
    /// overridden by [`Self::offline_only`].
    pub allow_cloud: bool,
    /// The preferred provider id when cloud escalation is allowed
    /// (e.g. `gemini`, `openai`). `None` lets the runtime pick.
    pub preferred_provider: Option<String>,
    /// Maximum accepted cost per cloud escalation.
    pub maximum_cost: Option<f64>,
    /// Maximum accepted latency per cloud escalation.
    pub maximum_latency: Option<Duration>,
    /// When true, data never leaves the machine: cloud escalation is denied
    /// regardless of every other setting.
    pub offline_only: bool,
}

impl EscalationPolicy {
    /// The default policy: escalate to the local model, never the cloud.
    pub fn default() -> Self {
        Self {
            allow_local: true,
            allow_cloud: false,
            preferred_provider: None,
            maximum_cost: None,
            maximum_latency: None,
            offline_only: false,
        }
    }

    /// Builds a policy from a runtime configuration.
    ///
    /// Local escalation is always allowed (the registered model is the
    /// router's fallback); the cloud settings come from
    /// [`NoemaConfig::cloud`], and [`NoemaConfig::offline_mode`] forces
    /// [`Self::offline_only`].
    pub fn from_config(config: &NoemaConfig) -> Self {
        Self {
            allow_local: true,
            allow_cloud: config.cloud.enabled,
            preferred_provider: config.cloud.preferred_provider.clone(),
            maximum_cost: config.cloud.maximum_cost,
            maximum_latency: config.cloud.maximum_latency_ms.map(Duration::from_millis),
            offline_only: config.offline_mode,
        }
    }

    /// Decides where an escalated request goes.
    ///
    /// * [`EscalationDecision::Local`] — the default: the local reasoning
    ///   model handles the request.
    /// * [`EscalationDecision::Cloud`] — only when local escalation is
    ///   disabled and cloud escalation is permitted (and not offline).
    /// * [`EscalationDecision::Denied`] — when neither target is permitted.
    pub fn decide(&self, _request: &EscalationRequest) -> EscalationDecision {
        if self.allow_local {
            EscalationDecision::Local
        } else if self.allow_cloud && !self.offline_only {
            EscalationDecision::Cloud
        } else {
            EscalationDecision::Denied
        }
    }
}

impl Default for EscalationPolicy {
    fn default() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_escalate_locally_and_never_to_cloud() {
        let policy = EscalationPolicy::default();
        let request = EscalationRequest::new("low confidence", vec![]);
        assert_eq!(policy.decide(&request), EscalationDecision::Local);
        assert!(!policy.allow_cloud);
    }

    #[test]
    fn from_config_reads_cloud_and_offline_settings() {
        let mut config = NoemaConfig::default();
        config.cloud.enabled = true;
        config.cloud.preferred_provider = Some("gemini".into());
        config.cloud.maximum_latency_ms = Some(5_000);
        config.offline_mode = true;

        let policy = EscalationPolicy::from_config(&config);
        assert!(policy.allow_cloud);
        assert!(policy.offline_only);
        assert_eq!(policy.preferred_provider.as_deref(), Some("gemini"));
        assert_eq!(policy.maximum_latency, Some(Duration::from_secs(5)));
    }

    #[test]
    fn offline_mode_denies_cloud() {
        let policy = EscalationPolicy {
            allow_local: false,
            allow_cloud: true,
            offline_only: true,
            ..EscalationPolicy::default()
        };
        let request = EscalationRequest::new("low confidence", vec![]);
        assert_eq!(policy.decide(&request), EscalationDecision::Denied);
    }

    #[test]
    fn cloud_is_chosen_only_when_local_is_disabled() {
        let policy = EscalationPolicy {
            allow_local: false,
            allow_cloud: true,
            ..EscalationPolicy::default()
        };
        let request = EscalationRequest::new("low confidence", vec![]);
        assert_eq!(policy.decide(&request), EscalationDecision::Cloud);
    }

    #[test]
    fn everything_denied_when_no_target_is_allowed() {
        let policy = EscalationPolicy {
            allow_local: false,
            allow_cloud: false,
            ..EscalationPolicy::default()
        };
        let request = EscalationRequest::new("low confidence", vec![]);
        assert_eq!(policy.decide(&request), EscalationDecision::Denied);
    }
}
