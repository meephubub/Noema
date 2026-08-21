//! Observability: content-free, aggregate metrics for a Noema runtime.
//!
//! Every model turn, tool call, and escalation is recorded by a
//! [`MetricsCollector`] and surfaced as a [`MetricsSnapshot`] through
//! [`crate::Noema::metrics`]. The same events stream live on the event bus
//! (`Event::ModelMetrics`, `Event::ToolMetrics`, `Event::EscalationMetrics`).
//!
//! Telemetry is **privacy-aware by design**: metrics carry identifiers,
//! counts, latencies, and token totals — never message content. Nothing the
//! user said or a tool returned is recorded here.

use std::collections::HashMap;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::model::Usage;

/// Per-model aggregate metrics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Model turns completed.
    pub turns: u64,
    /// Input tokens reported across turns.
    pub input_tokens: u64,
    /// Output tokens reported across turns.
    pub output_tokens: u64,
    /// Total latency across turns, in milliseconds.
    pub latency_ms: u64,
}

impl ModelMetrics {
    /// Total tokens reported for this model.
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

/// Per-tool aggregate metrics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolMetrics {
    /// Tool calls executed.
    pub calls: u64,
    /// Tool calls that failed.
    pub failures: u64,
    /// Total execution latency, in milliseconds.
    pub latency_ms: u64,
}

impl ToolMetrics {
    /// Whether every call so far succeeded.
    pub fn all_succeeded(&self) -> bool {
        self.failures == 0
    }
}

/// Aggregate escalation metrics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EscalationMetrics {
    /// Escalations started (local and cloud).
    pub escalations: u64,
    /// Escalations that ran a cloud provider.
    pub cloud_escalations: u64,
    /// Total cloud-provider latency, in milliseconds.
    pub latency_ms: u64,
}

/// A point-in-time snapshot of a runtime's metrics.
///
/// Models and tools are keyed by their ids/names; the escalation section is
/// runtime-wide.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    /// Per-model metrics, keyed by model id.
    pub models: HashMap<String, ModelMetrics>,
    /// Per-tool metrics, keyed by tool name.
    pub tools: HashMap<String, ToolMetrics>,
    /// Runtime-wide escalation metrics.
    pub escalation: EscalationMetrics,
}

impl MetricsSnapshot {
    /// Model turns across every model.
    pub fn total_model_turns(&self) -> u64 {
        self.models.values().map(|m| m.turns).sum()
    }

    /// Input tokens across every model.
    pub fn total_input_tokens(&self) -> u64 {
        self.models.values().map(|m| m.input_tokens).sum()
    }

    /// Output tokens across every model.
    pub fn total_output_tokens(&self) -> u64 {
        self.models.values().map(|m| m.output_tokens).sum()
    }

    /// Model latency across every model, in milliseconds.
    pub fn total_model_latency_ms(&self) -> u64 {
        self.models.values().map(|m| m.latency_ms).sum()
    }

    /// Tool calls across every tool.
    pub fn total_tool_calls(&self) -> u64 {
        self.tools.values().map(|t| t.calls).sum()
    }

    /// Failed tool calls across every tool.
    pub fn total_tool_failures(&self) -> u64 {
        self.tools.values().map(|t| t.failures).sum()
    }

    /// Tool latency across every tool, in milliseconds.
    pub fn total_tool_latency_ms(&self) -> u64 {
        self.tools.values().map(|t| t.latency_ms).sum()
    }

    /// Escalations that ran a cloud provider.
    pub fn cloud_escalations(&self) -> u64 {
        self.escalation.cloud_escalations
    }
}

/// Collects content-free metrics for one runtime.
///
/// Updates are cheap and infrequent relative to inference, so a plain mutex
/// is fine; snapshots are taken on demand (never streamed continuously).
#[derive(Debug, Default)]
pub struct MetricsCollector {
    inner: Mutex<MetricsSnapshot>,
}

impl MetricsCollector {
    /// A fresh, empty collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one completed model turn.
    pub fn record_model_turn(&self, model: &str, latency_ms: u64, usage: Option<Usage>) {
        let mut inner = self.inner.lock().expect("metrics lock poisoned");
        let entry = inner.models.entry(model.to_string()).or_default();
        entry.turns += 1;
        entry.latency_ms += latency_ms;
        if let Some(usage) = usage {
            entry.input_tokens += usage.input_tokens;
            entry.output_tokens += usage.output_tokens;
        }
    }

    /// Records one executed tool call.
    pub fn record_tool_call(&self, tool: &str, latency_ms: u64, success: bool) {
        let mut inner = self.inner.lock().expect("metrics lock poisoned");
        let entry = inner.tools.entry(tool.to_string()).or_default();
        entry.calls += 1;
        entry.latency_ms += latency_ms;
        if !success {
            entry.failures += 1;
        }
    }

    /// Records one escalation. `provider` and `latency_ms` are `None` for
    /// local escalations, which run no remote call.
    pub fn record_escalation(&self, provider: Option<&str>, latency_ms: Option<u64>) {
        let mut inner = self.inner.lock().expect("metrics lock poisoned");
        inner.escalation.escalations += 1;
        if provider.is_some() {
            inner.escalation.cloud_escalations += 1;
        }
        if let Some(latency_ms) = latency_ms {
            inner.escalation.latency_ms += latency_ms;
        }
    }

    /// A point-in-time copy of the collected metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        self.inner.lock().expect("metrics lock poisoned").clone()
    }

    /// Clears every accumulated metric.
    pub fn reset(&self) {
        *self.inner.lock().expect("metrics lock poisoned") = MetricsSnapshot::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_metrics_accumulate_per_model() {
        let collector = MetricsCollector::new();
        collector.record_model_turn(
            "gemma",
            10,
            Some(Usage {
                input_tokens: 5,
                output_tokens: 3,
            }),
        );
        collector.record_model_turn("gemma", 20, Some(Usage::default()));
        collector.record_model_turn(
            "needle",
            4,
            Some(Usage {
                input_tokens: 1,
                output_tokens: 1,
            }),
        );

        let snapshot = collector.snapshot();
        let gemma = snapshot.models.get("gemma").expect("gemma");
        assert_eq!(gemma.turns, 2);
        assert_eq!(gemma.input_tokens, 5);
        assert_eq!(gemma.output_tokens, 3);
        assert_eq!(gemma.latency_ms, 30);
        assert_eq!(gemma.total_tokens(), 8);

        let needle = snapshot.models.get("needle").expect("needle");
        assert_eq!(needle.turns, 1);
        assert_eq!(snapshot.total_model_turns(), 3);
        assert_eq!(snapshot.total_input_tokens(), 6);
        assert_eq!(snapshot.total_output_tokens(), 4);
        assert_eq!(snapshot.total_model_latency_ms(), 34);
    }

    #[test]
    fn turns_without_usage_do_not_touch_token_totals() {
        let collector = MetricsCollector::new();
        collector.record_model_turn("gemma", 7, None);
        let snapshot = collector.snapshot();
        let gemma = snapshot.models.get("gemma").expect("gemma");
        assert_eq!(gemma.turns, 1);
        assert_eq!(gemma.input_tokens, 0);
        assert_eq!(gemma.output_tokens, 0);
        assert_eq!(gemma.latency_ms, 7);
    }

    #[test]
    fn tool_metrics_count_calls_and_failures() {
        let collector = MetricsCollector::new();
        collector.record_tool_call("search_files", 3, true);
        collector.record_tool_call("search_files", 9, false);
        collector.record_tool_call("delete_file", 2, true);

        let snapshot = collector.snapshot();
        let search = snapshot.tools.get("search_files").expect("search");
        assert_eq!(search.calls, 2);
        assert_eq!(search.failures, 1);
        assert_eq!(search.latency_ms, 12);
        assert!(!search.all_succeeded());

        let delete = snapshot.tools.get("delete_file").expect("delete");
        assert!(delete.all_succeeded());
        assert_eq!(snapshot.total_tool_calls(), 3);
        assert_eq!(snapshot.total_tool_failures(), 1);
        assert_eq!(snapshot.total_tool_latency_ms(), 14);
    }

    #[test]
    fn escalation_metrics_track_local_and_cloud() {
        let collector = MetricsCollector::new();
        collector.record_escalation(None, None);
        collector.record_escalation(Some("openai"), Some(250));
        collector.record_escalation(Some("openai"), Some(150));

        let snapshot = collector.snapshot();
        assert_eq!(snapshot.escalation.escalations, 3);
        assert_eq!(snapshot.escalation.cloud_escalations, 2);
        assert_eq!(snapshot.escalation.latency_ms, 400);
        assert_eq!(snapshot.cloud_escalations(), 2);
    }

    #[test]
    fn snapshot_serializes_without_content() {
        let collector = MetricsCollector::new();
        collector.record_model_turn("gemma", 5, None);
        let json = serde_json::to_string(&collector.snapshot()).expect("serialize");
        // Metrics never carry message content.
        assert!(json.contains("gemma"));
        assert!(!json.contains("content"));
    }

    #[test]
    fn reset_clears_everything() {
        let collector = MetricsCollector::new();
        collector.record_model_turn("gemma", 5, None);
        collector.reset();
        assert_eq!(collector.snapshot(), MetricsSnapshot::default());
    }
}
