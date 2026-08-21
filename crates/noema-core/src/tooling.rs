//! Tool formatting: turning a semantic tool request into a structured call.
//!
//! The reasoning model never produces the final structured schema of a tool
//! call — that is the tool-specific Needle agent's job. [`ToolFormatter`] is
//! the core-side abstraction behind that role: given a tool's full schema
//! (plus any tool-provided instructions baked into the formatter) and a
//! semantic request, it returns a validated [`ToolCall`].
//!
//! The concrete Needle-backed implementation lives in `noema-router`
//! ([`noema_router::NeedleToolFormatter`]); this crate only defines the
//! interface so the agent loop stays independent of the formatter backend.

use std::fmt::Debug;

use async_trait::async_trait;
use noema_tools::{ToolCall, ToolSchema};
use tokio_util::sync::CancellationToken;

use crate::error::Result;

/// Formats a semantic tool request into a structured, validated call.
#[async_trait]
pub trait ToolFormatter: Debug + Send + Sync {
    /// A stable identifier for this formatter, for logs and events.
    fn id(&self) -> &str;

    /// Formats `request` against the tool's schema.
    ///
    /// The returned call must already satisfy the schema's `required`
    /// parameters; the caller still validates before executing. A request
    /// the formatter cannot serve (refusal, low confidence) is an error so
    /// the caller can decide how to handle it.
    async fn format(
        &self,
        schema: ToolSchema,
        request: &str,
        cancel: CancellationToken,
    ) -> Result<ToolCall>;
}
