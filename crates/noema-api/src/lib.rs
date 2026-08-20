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
//! # struct Mock;
//! # #[async_trait::async_trait]
//! # impl Model for Mock {
//! #     fn id(&self) -> &str { "mock" }
//! #     async fn generate(&self, _r: ModelRequest, _c: CancellationToken) -> Result<ModelResponse> {
//! #         Ok(ModelResponse::Text { content: "hi".into(), usage: None })
//! #     }
//! # }
//! init_logging(LogLevel::Info)?;
//! let noema = Noema::builder().with_model(Mock).build().await?;
//! let session = noema.create_session().await?;
//!
//! let mut events = session.events();
//! let response = session.send(Message::text(Role::User, "hello")).await?;
//! // ... consume events, drive the session ...
//!
//! session.close().await?;
//! # Ok(())
//! # }
//! ```
//!
//! Later milestones add tool wiring, the approval API, and escalation.

/// Convenience re-exports of the public API.
pub mod prelude {
    pub use noema_core::{
        init_logging, AudioData, ContentPart, EscalationRequest, ImageData, LogLevel, Message,
        Model, ModelChunk, ModelOptions, ModelProvider, ModelRequest, ModelResponse, Noema,
        NoemaBuilder, NoemaConfig, NoemaError, Result, Role, Session, SessionState, Usage,
    };
    pub use noema_events::{Event, EventBus, EventStream, SessionId};
}

pub use noema_core::{
    init_logging, AudioData, ContentPart, EscalationRequest, ImageData, LogLevel, Message, Model,
    ModelChunk, ModelOptions, ModelProvider, ModelRequest, ModelResponse, Noema, NoemaBuilder,
    NoemaConfig, NoemaError, Result, Role, Session, SessionState, Usage,
};
pub use noema_events::{Event, EventBus, EventStream, SessionId};

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
    }
}
