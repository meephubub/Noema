//! The Needle 2 → Gemma 4 confidence bridge.
//!
//! [`BridgeSession`] runs every user message through Needle 2 first. When
//! Needle is confident it calls one of the registered tools, the tool
//! executes and the result is returned. When confidence is low or Needle
//! refuses, the same prompt is forwarded to Gemma 4 for full reasoning.
//!
//! ```text
//! User message
//!     ↓
//! Needle 2 (5 stub tools, fast)
//!     ├── confident + tool call → execute stub → return result
//!     └── low confidence / refusal → Gemma 4 (same prompt, full reasoning)
//! ```

use std::sync::Arc;

use noema_core::{Message, Model, ModelRequest, NoemaError, Result, SendOutcome};
use noema_needle::{DylibEngine, EngineSettings, NeedleEngine};
use noema_tools::ToolRegistry;
use tokio_util::sync::CancellationToken;

use crate::tools::stub_registry;

/// Default confidence threshold: below this, Needle escalates to Gemma.
pub const DEFAULT_MIN_CONFIDENCE: f32 = 0.6;

/// Configuration for a [`BridgeSession`].
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    /// Confidence threshold below which Needle escalates to Gemma.
    pub min_confidence: f32,
    /// Maximum new tokens Needle may generate per turn.
    pub needle_max_tokens: u32,
    /// Optional system prompt for Needle.
    pub needle_system: Option<String>,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            min_confidence: DEFAULT_MIN_CONFIDENCE,
            needle_max_tokens: 256,
            needle_system: None,
        }
    }
}

/// A two-tier session: Needle 2 for fast tool dispatch, Gemma 4 for
/// low-confidence reasoning.
///
/// The session holds a Needle engine bound to 5 stub tools and an optional
/// Gemma model for escalation. Each `send` runs the full bridge flow:
///
/// 1. Needle processes the message with its tools.
/// 2. If Needle is confident and calls a registered tool → execute the stub
///    and return the result.
/// 3. If confidence is low or Needle refuses → forward the same prompt to
///    Gemma for a full reasoning response.
pub struct BridgeSession {
    needle: Arc<DylibEngine>,
    tools: ToolRegistry,
    gemma: Option<Arc<dyn Model>>,
    config: BridgeConfig,
}

impl std::fmt::Debug for BridgeSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BridgeSession")
            .field("config", &self.config)
            .field("tools", &self.tools.names())
            .field("gemma", &self.gemma.is_some())
            .finish()
    }
}

impl BridgeSession {
    /// Creates a bridge session with the default 5 stub tools.
    pub fn from_default(config: BridgeConfig) -> Result<Self> {
        let tools = stub_registry();
        let tools_json = tools.needle_tools_json();
        let mut settings = EngineSettings::new(tools_json);
        if let Some(system) = &config.needle_system {
            settings = settings.with_system(system.clone());
        }
        let needle = DylibEngine::from_default(settings).map_err(|error| {
            NoemaError::Model(format!("failed to load Needle engine for bridge: {error}"))
        })?;
        Ok(Self {
            needle: Arc::new(needle),
            tools,
            gemma: None,
            config,
        })
    }

    /// Creates a bridge session with a custom tool registry.
    pub fn with_tools(tools: ToolRegistry, config: BridgeConfig) -> Result<Self> {
        let tools_json = tools.needle_tools_json();
        let mut settings = EngineSettings::new(tools_json);
        if let Some(system) = &config.needle_system {
            settings = settings.with_system(system.clone());
        }
        let needle = DylibEngine::from_default(settings).map_err(|error| {
            NoemaError::Model(format!("failed to load Needle engine for bridge: {error}"))
        })?;
        Ok(Self {
            needle: Arc::new(needle),
            tools,
            gemma: None,
            config,
        })
    }

    /// Sets the Gemma model for low-confidence escalation.
    pub fn with_gemma(mut self, gemma: Arc<dyn Model>) -> Self {
        self.gemma = Some(gemma);
        self
    }

    /// The tool registry backing this session.
    pub fn tools(&self) -> &ToolRegistry {
        &self.tools
    }

    /// Sends a message through the bridge.
    ///
    /// Needle 2 runs first. When confident, it calls a tool which is executed
    /// and returned. When uncertain, the same prompt is forwarded to Gemma 4.
    pub async fn send(
        &self,
        message: Message,
        cancel: CancellationToken,
    ) -> Result<SendOutcome> {
        // Extract plain text from the message.
        let text = match message.content.first() {
            Some(noema_core::ContentPart::Text(text)) => text.clone(),
            _ => {
                // Non-text messages go straight to Gemma.
                return self.escalate_to_gemma(&message, &cancel).await;
            }
        };

        // Run Needle in a blocking task (C calls are blocking).
        let engine = Arc::clone(&self.needle);
        let input = text.clone();
        let max_tokens = self.config.needle_max_tokens;
        let response = tokio::task::spawn_blocking(move || -> noema_needle::Result<_> {
            engine.reset()?;
            engine.complete(&input, max_tokens)
        })
        .await
        .map_err(|e| NoemaError::Model(format!("bridge task failed: {e}")))?
        .map_err(|e| NoemaError::Model(format!("needle failed: {e}")))?;

        // Check if Needle called a tool with sufficient confidence.
        let confidence = response.confidence.unwrap_or(1.0);
        if let Some(call) = response.calls().first() {
            if confidence >= self.config.min_confidence {
                if let Some(tool) = self.tools.get(&call.name) {
                    tracing::debug!(
                        tool = %call.name,
                        confidence,
                        "bridge: needle called a tool"
                    );

                    // Execute the stub tool.
                    let _result = tool.execute(noema_tools::ToolCall::with_arguments(
                        call.name.clone(),
                        call.arguments.clone(),
                    )).await.map_err(|e| NoemaError::Tool(e.to_string()))?;

                    let outcome = SendOutcome::Routed(noema_core::RoutedAction {
                        id: call.name.clone(),
                        arguments: call.arguments.clone(),
                        confidence: response.confidence,
                    });
                    return Ok(outcome);
                }
            }
            // Low confidence or unregistered tool → escalate.
            tracing::debug!(
                tool = ?call.name,
                confidence,
                "bridge: low confidence, escalating to gemma"
            );
        } else {
            // Needle refused → escalate.
            tracing::debug!("bridge: needle refused, escalating to gemma");
        }

        // Escalate to Gemma.
        self.escalate_to_gemma(&message, &cancel).await
    }

    /// Forwards the message to Gemma 4 for full reasoning.
    async fn escalate_to_gemma(
        &self,
        message: &Message,
        cancel: &CancellationToken,
    ) -> Result<SendOutcome> {
        let gemma = self.gemma.as_ref().ok_or_else(|| {
            NoemaError::Model(
                "no Gemma model registered for bridge escalation; use .with_gemma()".into(),
            )
        })?;

        tracing::debug!("bridge: running gemma escalation");

        let request = ModelRequest::new(vec![message.clone()]);
        let response = gemma.generate(request, cancel.clone()).await?;
        Ok(SendOutcome::Model(response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use noema_core::ContentPart;

    #[test]
    fn bridge_config_defaults() {
        let config = BridgeConfig::default();
        assert_eq!(config.min_confidence, DEFAULT_MIN_CONFIDENCE);
        assert_eq!(config.needle_max_tokens, 256);
        assert!(config.needle_system.is_none());
    }

    #[test]
    fn stub_registry_has_tools() {
        let registry = stub_registry();
        assert_eq!(registry.names().len(), 5);
    }
}
