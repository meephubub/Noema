//! # Noema core
//!
//! The central Noema runtime. This crate holds the pieces every other crate
//! builds on: strongly typed errors and configuration, logging setup, the
//! ephemeral session abstraction, and the [`Noema`] runtime itself.
//!
//! Implemented so far:
//!
//! * Strongly typed errors, configuration, and logging
//! * The ephemeral session abstraction with model-backed `send`
//! * The model abstraction (messages, requests, responses, cancellation,
//!   escalation) behind which Gemma, Needle, and cloud adapters will sit
//!
//! Planned contents for later milestones:
//!
//! * Agent runtime and the full agent loop
//! * Model routing
//! * Tool orchestration
//! * Agent state
//!
//! The public, frontend-facing API lives in `noema-api`; this crate exposes
//! the underlying types.

pub mod config;
pub mod error;
pub mod escalation;
pub mod logging;
pub mod model;
pub mod router;
pub mod runtime;
pub mod session;
pub mod tooling;

pub use config::{LogLevel, NoemaConfig};
pub use error::{NoemaError, Result};
pub use logging::init_logging;
pub use escalation::{EscalationDecision, EscalationPolicy};
pub use model::{
    AudioData, ContentPart, EscalationRequest, ImageData, Message, Model, ModelChunk,
    ModelOptions, ModelProvider, ModelRequest, ModelResponse, Role, Usage,
};
pub use router::{Route, RoutedAction, Router, SendOutcome};
pub use runtime::{Noema, NoemaBuilder};
pub use session::{Session, SessionState};
pub use tooling::ToolFormatter;
