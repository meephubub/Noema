//! Integration between Noema and Rig.
//!
//! Rig is the middle layer connecting Noema's agent architecture to models
//! and tools. Noema-specific orchestration sits *above* Rig; this crate
//! provides:
//!
//! * Rig adapters
//! * Agent integration
//! * Model adapters
//! * Provider integration
//!
//! Noema should avoid duplicating functionality that Rig already provides
//! reliably (agents, model interactions, tool interfaces, message handling,
//! streaming, provider abstraction).
//!
//! This crate is scaffolded in phase 1 and implemented with the model
//! integrations milestone.
