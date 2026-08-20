//! Strongly typed Noema configuration.

use std::path::PathBuf;

use noema_tools::RiskLevel;
use serde::{Deserialize, Serialize};

/// Logging verbosity, from fully silent to maximally verbose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    /// No logging at all.
    Off,
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl Default for LogLevel {
    fn default() -> Self {
        LogLevel::Info
    }
}

impl LogLevel {
    /// Maps to the corresponding `tracing` level, or `None` for [`Off`](Self::Off).
    pub fn as_tracing_level(&self) -> Option<tracing::Level> {
        match self {
            LogLevel::Off => None,
            LogLevel::Error => Some(tracing::Level::ERROR),
            LogLevel::Warn => Some(tracing::Level::WARN),
            LogLevel::Info => Some(tracing::Level::INFO),
            LogLevel::Debug => Some(tracing::Level::DEBUG),
            LogLevel::Trace => Some(tracing::Level::TRACE),
        }
    }
}

/// The complete typed configuration for a Noema runtime.
///
/// Every section defaults to a sensible value; construct with
/// [`NoemaConfig::default`] and override what you need.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NoemaConfig {
    /// Gemma 4 configuration.
    pub gemma: GemmaConfig,
    /// Needle 2 configuration.
    pub needle: NeedleConfig,
    /// Cloud escalation configuration.
    pub cloud: CloudConfig,
    /// Mnemo memory configuration.
    pub memory: MemoryConfig,
    /// Tool-set configuration.
    pub tools: ToolsConfig,
    /// Risk-policy configuration.
    pub risk: RiskConfig,
    /// Approval-policy configuration.
    pub approval: ApprovalConfig,
    /// Resource limits that prevent runaway agent loops.
    pub limits: LimitsConfig,
    /// Logging configuration.
    pub logging: LoggingConfig,
    /// Streaming / event configuration.
    pub streaming: StreamingConfig,
    /// When true, no data ever leaves the machine: cloud escalation is
    /// disabled regardless of other settings.
    pub offline_mode: bool,
}

impl Default for NoemaConfig {
    fn default() -> Self {
        Self {
            gemma: GemmaConfig::default(),
            needle: NeedleConfig::default(),
            cloud: CloudConfig::default(),
            memory: MemoryConfig::default(),
            tools: ToolsConfig::default(),
            risk: RiskConfig::default(),
            approval: ApprovalConfig::default(),
            limits: LimitsConfig::default(),
            logging: LoggingConfig::default(),
            streaming: StreamingConfig::default(),
            offline_mode: false,
        }
    }
}

/// Gemma 4 configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GemmaConfig {
    /// Whether the local Gemma model is enabled.
    pub enabled: bool,
    /// Path to the Gemma model artifacts consumed by litert-lm-rust.
    pub model_path: Option<PathBuf>,
}

impl Default for GemmaConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model_path: None,
        }
    }
}

/// Needle 2 configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NeedleConfig {
    /// Whether the local Needle model is enabled.
    pub enabled: bool,
    /// Path to the Needle model artifacts consumed by its Rust binding.
    pub model_path: Option<PathBuf>,
}

impl Default for NeedleConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model_path: None,
        }
    }
}

/// Cloud escalation configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CloudConfig {
    /// Whether cloud escalation is permitted at all.
    pub enabled: bool,
    /// The preferred provider id (e.g. `gemini`, `openai`).
    pub preferred_provider: Option<String>,
    /// Maximum allowed cost per escalation, if a provider reports cost.
    pub maximum_cost: Option<f64>,
    /// Maximum allowed latency per escalation, in milliseconds.
    pub maximum_latency_ms: Option<u64>,
}

impl Default for CloudConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            preferred_provider: None,
            maximum_cost: None,
            maximum_latency_ms: None,
        }
    }
}

/// Mnemo memory configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    /// Whether memory retrieval and writing are enabled.
    pub enabled: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self { enabled: false }
    }
}

/// Tool-set configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolsConfig {
    /// Whether tools are enabled at all.
    pub enabled: bool,
    /// Whether the default Agora tool set is included.
    pub include_default_set: bool,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            include_default_set: true,
        }
    }
}

/// Risk-policy configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RiskConfig {
    /// Tool calls at or above this level require human approval.
    ///
    /// `None` means no approval is ever required by risk alone.
    pub require_approval_above: Option<RiskLevel>,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            require_approval_above: Some(RiskLevel::High),
        }
    }
}

/// Approval-policy configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ApprovalConfig {
    /// Whether the human-approval flow is enabled at all.
    pub enabled: bool,
    /// How long a pending approval stays valid, in seconds.
    ///
    /// `None` means approvals never expire.
    pub timeout_seconds: Option<u64>,
}

impl Default for ApprovalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            timeout_seconds: Some(300),
        }
    }
}

/// Resource limits that prevent runaway agent loops.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LimitsConfig {
    /// Maximum agent-loop iterations per user request.
    pub max_agent_iterations: usize,
    /// Maximum tool calls per user request.
    pub max_tool_calls: usize,
    /// Maximum tool-call depth (nested calls).
    pub max_tool_call_depth: usize,
    /// Maximum context size in tokens.
    pub max_context_tokens: usize,
    /// Maximum response length in tokens.
    pub max_response_tokens: usize,
    /// Maximum tool execution time, in seconds.
    pub max_tool_execution_seconds: u64,
    /// Maximum cloud escalations per user request.
    pub max_cloud_escalations: usize,
    /// Maximum concurrently executing tools.
    pub max_concurrent_tools: usize,
}

impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            max_agent_iterations: 20,
            max_tool_calls: 50,
            max_tool_call_depth: 4,
            max_context_tokens: 8192,
            max_response_tokens: 2048,
            max_tool_execution_seconds: 120,
            max_cloud_escalations: 3,
            max_concurrent_tools: 4,
        }
    }
}

/// Logging configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LoggingConfig {
    /// The global verbosity level.
    pub level: LogLevel,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
        }
    }
}

/// Streaming / event configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StreamingConfig {
    /// Whether streaming is enabled.
    pub enabled: bool,
    /// Capacity of the event bus channel per runtime.
    pub event_capacity: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            event_capacity: 1024,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let config = NoemaConfig::default();
        assert!(!config.gemma.enabled);
        assert!(!config.needle.enabled);
        assert!(!config.cloud.enabled);
        assert!(!config.offline_mode);
        assert_eq!(config.logging.level, LogLevel::Info);
        assert_eq!(config.streaming.event_capacity, 1024);
        assert!(config.limits.max_agent_iterations > 0);
        assert!(config.risk.require_approval_above.is_some());
    }

    #[test]
    fn offline_mode_and_cloud_escalation_are_independent_fields() {
        let mut config = NoemaConfig::default();
        config.offline_mode = true;
        // Even if cloud is configured on, offline mode wins at runtime.
        config.cloud.enabled = true;
        assert!(config.cloud.enabled && config.offline_mode);
    }

    #[test]
    fn log_levels_map_to_tracing() {
        assert!(LogLevel::Off.as_tracing_level().is_none());
        assert_eq!(LogLevel::Error.as_tracing_level(), Some(tracing::Level::ERROR));
        assert_eq!(LogLevel::Trace.as_tracing_level(), Some(tracing::Level::TRACE));
    }
}
