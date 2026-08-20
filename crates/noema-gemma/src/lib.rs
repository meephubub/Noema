//! Gemma 4 integration for Noema.
//!
//! Planned contents:
//!
//! * litert-lm-rust integration
//! * Multimodal requests (text, image, audio)
//! * Streaming output
//! * Gemma-specific message handling
//! * System prompts and conversation context
//! * Tool-intent generation and escalation decisions
//!
//! Layering, from outside in:
//!
//! ```text
//! Noema
//!   ↓
//! Gemma model abstraction
//!   ↓
//! litert-lm-rust
//!   ↓
//! Gemma 4
//! ```
//!
//! LiteRT-specific types must never spread beyond this crate: everything
//! else in Noema talks to Gemma through the model traits in `noema-core`.
//!
//! This crate is scaffolded in phase 1 and implemented with the model
//! integrations milestone.
