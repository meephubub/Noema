//! Common tool infrastructure for Noema.
//!
//! This crate defines the contract every Noema tool satisfies and the
//! central registry that holds them. The contract is deliberately small so a
//! third-party `noema-*` crate can ship a tool and register it without
//! modifying the core agent loop:
//!
//! ```text
//! noema-<tool> crate
//!     │  implements NoemaTool (metadata + schema + execute)
//!     ▼
//! ToolRegistry::register(..)
//!     ├── gemma_tool_section()   → lightweight summaries (Gemma)
//!     ├── needle_tools_json()    → complete schemas (tool Needle agents)
//!     └── execute(call)          → validated execution
//! ```
//!
//! A tool exposes three views, each aimed at a different consumer:
//!
//! * [`ToolMetadata`] / [`ToolSummary`] — execution and Gemma-facing
//!   metadata.
//! * [`ToolSchema`] — the full schema the tool-specific Needle agent binds
//!   to.
//! * [`RiskLevel`] — the risk its calls pose, evaluated by Noema (approval
//!   gating lands with the approval milestone).
//!
//! # Implementing a tool
//!
//! ```no_run
//! use async_trait::async_trait;
//! use noema_tools::{
//!     NoemaTool, RiskLevel, ToolCall, ToolMetadata, ToolResult, ToolSchema,
//!     Result, ToolRegistry,
//! };
//!
//! #[derive(Debug)]
//! struct Echo;
//!
//! #[async_trait]
//! impl NoemaTool for Echo {
//!     fn metadata(&self) -> ToolMetadata {
//!         ToolMetadata {
//!             name: "echo".into(),
//!             crate_name: "noema-echo".into(),
//!             description: "Echoes its message argument".into(),
//!             risk: RiskLevel::None,
//!         }
//!     }
//!
//!     fn schema(&self) -> ToolSchema {
//!         ToolSchema::new("echo", "Echoes its message argument")
//!     }
//!
//!     async fn execute(&self, call: ToolCall) -> Result<ToolResult> {
//!         Ok(ToolResult::ok("echo"))
//!     }
//! }
//!
//! let mut registry = ToolRegistry::new();
//! registry.register(Echo).expect("register");
//! assert_eq!(registry.names(), vec!["echo"]);
//! ```

pub mod error;
pub mod metadata;
pub mod registry;
pub mod risk;
pub mod schema;
pub mod tool;

pub use error::{Result, ToolError};
pub use metadata::{ToolMetadata, ToolSummary};
pub use registry::ToolRegistry;
pub use risk::RiskLevel;
pub use schema::ToolSchema;
pub use tool::{NoemaTool, ToolCall, ToolResult};
