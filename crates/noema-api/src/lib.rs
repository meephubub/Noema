//! # Noema API
//!
//! The public, frontend-facing Rust API for the Agora frontend.
//!
//! The frontend should not need to know which model was selected, how tools
//! are routed, how schemas work, how Needle is configured, how memory
//! retrieval works, or how escalation works. This crate is the single
//! surface through which Agora talks to Noema.
//!
//! Phase 2 exposes the model abstraction: sessions can send messages —
//! including multimodal ones — to a registered model and receive its
//! response:
//!
//! ```
//! use noema_api::prelude::*;
//!
//! # async fn example() -> noema_api::Result<()> {
//! # #[derive(Debug)]
//! # struct Mock;
//! # #[async_trait::async_trait]
//! # impl Model for Mock {
//! #     fn id(&self) -> &str { "mock" }
//! #     async fn generate(
//! #         &self,
//! #         _r: ModelRequest,
//! #         _c: tokio_util::sync::CancellationToken,
//! #     ) -> Result<ModelResponse> {
//! #         Ok(ModelResponse::Text { content: "hi".into(), usage: None })
//! #     }
//! # }
//! init_logging(LogLevel::Info)?;
//! let noema = Noema::builder().with_model(Mock).build().await?;
//! let session = noema.create_session().await?;
//!
//! let mut events = session.events();
//! let outcome = session.send(Message::text(Role::User, "hello")).await?;
//! let response = outcome.into_model().expect("model response");
//! // ... consume events, drive the session ...
//!
//! session.close().await?;
//! # Ok(())
//! # }
//! ```
//!
//! Phase 8 surfaces human approval: `session.pending_approvals()` returns
//! the proposals awaiting a decision, and `session.approve_tool(id)` /
//! `session.reject_tool(id)` answer them. The tool vocabulary (`ToolCall`,
//! `ToolResult`, `ToolRegistry`, …) is re-exported so the frontend can drive
//! `session.format_tool` / `session.execute_tool` without reaching into
//! `noema-tools`.
//!
//! Phase 11 surfaces cloud escalation. Providers stay abstract
//! ([`ModelProvider`]); the bundled [`OpenAICompatibleProvider`] is
//! configured with a model name, base URL, and API key and registered via
//! [`NoemaBuilder::with_provider`]. The same three fields live in
//! [`NoemaConfig::cloud`] for configuration-file-driven setups.
//!
//! Phase 13 surfaces observability: [`Noema::metrics`] returns a
//! content-free [`MetricsSnapshot`] (model turns, tool calls, escalations,
//! latencies, token totals), and the same numbers stream live as
//! [`Event::ModelMetrics`] / [`Event::ToolMetrics`] /
//! [`Event::EscalationMetrics`].
//!
//! ```
//! use noema_api::prelude::*;
//!
//! # async fn example() -> noema_api::Result<()> {
//! # #[derive(Debug)]
//! # struct Mock;
//! # #[async_trait::async_trait]
//! # impl Model for Mock {
//! #     fn id(&self) -> &str { "mock" }
//! #     async fn generate(
//! #         &self,
//! #         _r: ModelRequest,
//! #         _c: tokio_util::sync::CancellationToken,
//! #     ) -> Result<ModelResponse> {
//! #         Ok(ModelResponse::Text { content: "hi".into(), usage: None })
//! #     }
//! # }
//! # init_logging(LogLevel::Info)?;
//! # let noema = Noema::builder().with_model(Mock).build().await?;
//! let session = noema.create_session().await?;
//!
//! // A risky tool call pauses for approval; the frontend answers by id.
//! for pending in session.pending_approvals() {
//!     session.approve_tool(pending.id.clone())?;
//! }
//!
//! session.close().await?;
//! # Ok(())
//! # }
//! ```

/// Convenience re-exports of the public API.
pub mod prelude {
    pub use noema_core::{
        init_logging, ApprovalDecision, ApprovalId, ApprovalPolicy, ApprovalRequest,
        ApprovalStatus, AudioData, ContentPart, EscalationMetrics, EscalationRequest, ImageData,
        LogLevel, Message, MetricsSnapshot, Model, ModelChunk, ModelMetrics, ModelOptions,
        ModelProvider, ModelRequest, ModelResponse, Noema, NoemaBuilder, NoemaConfig, NoemaError,
        Result, Role, Route, RoutedAction, Router, SendOutcome, Session, SessionState, ToolFormatter,
        ToolMetrics, Usage,
    };
    pub use noema_events::{Event, EventBus, EventStream, SessionId};
    pub use noema_provider_http::OpenAICompatibleProvider;
    pub use noema_tools::{
        NoemaTool, RiskLevel, ToolCall, ToolMetadata, ToolRegistry, ToolResult, ToolSchema,
        ToolSummary,
    };
}

pub use noema_core::{
    init_logging, ApprovalDecision, ApprovalId, ApprovalPolicy, ApprovalRequest, ApprovalStatus,
    AudioData, ContentPart, EscalationMetrics, EscalationRequest, ImageData, LogLevel, Message,
    MetricsSnapshot, Model, ModelChunk, ModelMetrics, ModelOptions, ModelProvider, ModelRequest,
    ModelResponse, Noema, NoemaBuilder, NoemaConfig, NoemaError, Result, Role, Route,
    RoutedAction, Router, SendOutcome, Session, SessionState, ToolFormatter, ToolMetrics, Usage,
};
pub use noema_events::{Event, EventBus, EventStream, SessionId};
pub use noema_provider_http::OpenAICompatibleProvider;
pub use noema_tools::{
    NoemaTool, RiskLevel, ToolCall, ToolMetadata, ToolRegistry, ToolResult, ToolSchema,
    ToolSummary,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prelude_covers_the_public_surface() {
        // Compile-time check that the prelude re-exports resolve.
        let _: fn(LogLevel) -> Result<()> = init_logging;
        let _ = NoemaError::Session("x".into());
        let _ = EventBus::default();
        let _ = NoemaConfig::default();
        let _ = SessionState::Active;
        let _ = Message::text(Role::User, "hi");
        let _ = Usage::default();
        let _ = MetricsSnapshot::default();
        // The cloud provider is part of the public surface: model name,
        // base URL, and API key are plain constructor arguments.
        let _ = OpenAICompatibleProvider::new(
            "openai",
            "gpt-4o",
            "https://api.openai.com/v1",
            Some("sk-test".into()),
        );
    }
}
