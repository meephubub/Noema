//! The initial text router for Noema.
//!
//! Plain-text user requests are routed before any reasoning model runs.
//! [`NeedleRouter`] uses Needle 2 to decide whether a request is a simple,
//! deterministic application action — "open my flashcards", "go to
//! settings" — or whether it needs the full model:
//!
//! ```text
//! User request
//!     ↓
//! NeedleRouter (Needle 2)
//!     ├── simple action → RoutedAction (model never runs)
//!     └── not recognised → Escalate → Gemma 4
//! ```
//!
//! The actions come from an [`ActionRegistry`] (defaulting to the six simple
//! Agora actions from the plan); each action is exposed to Needle as a tool
//! with a name and description, so the router needs no custom prompts to
//! stay on-schema.
//!
//! # Usage
//!
//! ```no_run
//! use noema_core::Noema;
//! use noema_router::NeedleRouter;
//!
//! # async fn example() -> noema_core::Result<()> {
//! let router = NeedleRouter::from_default()?;
//! let noema = Noema::builder()
//!     .with_router(router)
//!     // .with_model(gemma)  // handles escalated requests
//!     .build()
//!     .await?;
//! # Ok(())
//! # }
//! ```

pub mod action;
pub mod router;

pub use action::{ActionRegistry, ActionSpec};
pub use router::NeedleRouter;
