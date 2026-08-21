//! # Noema Bridge
//!
//! A two-tier inference bridge: Needle 2 for fast tool dispatch, Gemma 4 for
//! low-confidence reasoning.
//!
//! ```text
//! User message
//!     ↓
//! Needle 2 (5 stub tools, fast, ~45M params)
//!     ├── confident + tool call → execute stub → return result
//!     └── low confidence / refusal → Gemma 4 (same prompt, full reasoning)
//! ```
//!
//! # Quick start
//!
//! ```no_run
//! use std::sync::Arc;
//! use noema_bridge::{BridgeSession, BridgeConfig};
//! use noema_core::{Message, Role};
//! use tokio_util::sync::CancellationToken;
//!
//! # async fn example() -> noema_core::Result<()> {
//! let session = BridgeSession::from_default(BridgeConfig::default())?;
//! // Optionally add Gemma for escalation:
//! // let session = session.with_gemma(Arc::new(gemma_model));
//!
//! let response = session
//!     .send(
//!         Message::text(Role::User, "search for rust docs"),
//!         CancellationToken::new(),
//!     )
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Architecture
//!
//! The bridge combines two inference backends:
//!
//! * **Needle 2** — a 45M-parameter on-device model optimised for tool calling
//!   and structured extraction. It is fast and deterministic, but only handles
//!   requests it is confident about.
//!
//! * **Gemma 4** — a full reasoning model running on-device through LiteRT-LM.
//!   It handles complex, open-ended requests that Needle cannot route.
//!
//! The confidence threshold ([`BridgeConfig::min_confidence`]) controls the
//! handoff: requests below the threshold escalate to Gemma.

pub mod bridge;
pub mod tools;

pub use bridge::{BridgeConfig, BridgeSession, DEFAULT_MIN_CONFIDENCE};
pub use tools::{
    stub_registry, CalculateTool, NavigateTool, SearchTool, SummarizeTool, TranslateTool,
};
