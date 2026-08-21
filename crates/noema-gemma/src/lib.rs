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
//! The LiteRT-LM native libraries live in the workspace `prebuilt/` directory
//! and are linked automatically by the build script:
//!
//! - **Windows**: DLLs in `prebuilt/`, staged next to every built executable
//!   by `crates/noema-native`. No `PATH` changes needed.
//! - **macOS**: dylibs in `prebuilt/macos/`, with an embedded rpath so no
//!   `DYLD_LIBRARY_PATH` is needed. The C API library (`litert-lm.dylib`)
//!   must be obtained from Google's LiteRT-LM v0.16.0+ release.
//!
//! The Gemma model itself is a `.litertlm` file — by default `models/` in
//! the workspace root, overridable via `NOEMA_GEMMA_MODEL` or the builder's
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
