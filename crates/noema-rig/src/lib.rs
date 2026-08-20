//! Integration between Noema and Rig.
//!
//! Rig is the middle layer connecting Noema's agent architecture to models
//! and tools. Noema-specific orchestration sits *above* Rig; this crate
//! provides:
//!
//! * [`model::NoemaCompletionModel`] — a Rig [`CompletionModel`] adapter over
//!   Noema's [`Model`] trait, so any Noema backend (local Gemma, local
//!   Needle, a future cloud provider) can be driven by rig agents.
//! * [`message`] — conversion between rig's provider-agnostic messages and
//!   Noema's model messages.
//! * [`stub::StubProvider`] — a deterministic rig provider for tests,
//!   examples, and development without a real model.
//!
//! Noema should avoid duplicating functionality that Rig already provides
//! reliably (agents, model interactions, tool interfaces, message handling,
//! streaming, provider abstraction). Agent integration builds on this crate
//! with the agent milestones.
//!
//! # Example
//!
//! ```no_run
//! use noema_core::{Message, Model, ModelRequest, ModelResponse, NoemaError, Role};
//! use noema_rig::model::NoemaCompletionModel;
//! use rig_core::completion::{CompletionModel, CompletionRequestBuilder};
//!
//! #[derive(Debug)]
//! struct FakeModel;
//!
//! #[async_trait::async_trait]
//! impl Model for FakeModel {
//!     fn id(&self) -> &str {
//!         "fake"
//!     }
//!
//!     async fn generate(
//!         &self,
//!         _request: ModelRequest,
//!         _cancel: tokio_util::sync::CancellationToken,
//!     ) -> Result<ModelResponse, NoemaError> {
//!         Ok(ModelResponse::Text {
//!             content: "hello".into(),
//!             usage: None,
//!         })
//!     }
//! }
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let model = NoemaCompletionModel::new(std::sync::Arc::new(FakeModel))
//!     .with_provider("fake");
//!
//! let request = CompletionRequestBuilder::new(model.clone(), "Hello!")
//!     .build();
//! let response = model.completion(request).await?;
//! # Ok(())
//! # }
//! ```

pub mod message;
pub mod model;
pub mod stub;

pub use model::NoemaCompletionModel;
pub use stub::StubProvider;
