//! Gemma 4 integration for Noema.
//!
//! This crate is the adapter between Noema's model abstraction and Google's
//! LiteRT-LM on-device runtime, through the safe `litert-lm-rust` bindings
//! (vendored in this workspace at `crates/litert-lm-rust`).
//!
//! Layering, from outside in:
//!
//! ```text
//! Noema
//!   ↓
//! noema-gemma (this crate) — implements noema-core::Model
//!   ↓
//! litert-lm-rust
//!   ↓
//! LiteRT-LM C API
//!   ↓
//! Gemma 4 (.litertlm)
//! ```
//!
//! LiteRT-specific types never spread beyond this crate: everything else in
//! Noema talks to Gemma through the `Model` trait in `noema-core`.
//!
//! # Quick start
//!
//! ```no_run
//! use noema_core::{Message, Model, Role};
//! use noema_gemma::GemmaModel;
//!
//! # async fn run() -> noema_core::Result<()> {
//! let model = GemmaModel::builder("gemma-4-E2B-it.litertlm").build()?;
//! let response = model
//!     .generate(
//!         noema_core::ModelRequest::new(vec![Message::text(Role::User, "Hello!")]),
//!         Default::default(),
//!     )
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Native libraries and the model file
//!
//! The LiteRT-LM DLLs live in the workspace `prebuilt/` directory and are
//! staged next to every built executable at compile time (see the
//! `noema-native` build helper), so no PATH changes are needed. The Gemma
//! model itself is a `.litertlm` file — by default `models/` in the
//! workspace root, overridable via `NOEMA_GEMMA_MODEL` or the builder's
//! explicit path.
//!
//! # Status
//!
//! Implemented: text conversation, streaming, cancellation, per-turn usage,
//! system prompts, multimodal user turns (image/audio blobs). Tool-intent
//! generation and escalation decisions are wired with later milestones.

pub mod mapping;
pub mod model;
pub mod options;

pub use litert_lm_rust::Backend;
pub use model::{default_model_path, GemmaModel, GemmaModelBuilder, DEFAULT_MODEL_FILE};
pub use options::GemmaOptions;
