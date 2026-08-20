//! Needle 2 integration for Noema.
//!
//! Planned contents:
//!
//! * Needle Rust binding integration
//! * Needle inference interface
//! * Needle-specific request/response types
//! * Structured output handling and error handling
//!
//! Layering, from outside in:
//!
//! ```text
//! Noema
//!   ↓
//! Needle Rust crate
//!   ↓
//! Needle 2 C API
//!   ↓
//! Needle 2
//! ```
//!
//! The Needle Rust binding itself is an external project; this crate only
//! consumes it as a normal dependency. One physical Needle model is exposed
//! as multiple logical tool agents by the Noema runtime.
//!
//! This crate is scaffolded in phase 1 and implemented with the model
//! integrations milestone.
