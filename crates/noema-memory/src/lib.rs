//! Mnemo integration for Noema.
//!
//! Noema never implements its own persistent memory; Mnemo is the only
//! persistent memory system. This crate is the dedicated abstraction between
//! the Noema runtime and Mnemo:
//!
//! * Memory retrieval (before complex requests)
//! * Memory insertion / extraction (after useful interactions)
//! * Context conversion
//! * Memory policies
//!
//! Extraction must not be indiscriminate: temporary tool results, irrelevant
//! conversation, redundant information, and sensitive information should not
//! be stored.
//!
//! This crate is scaffolded in phase 1 and implemented with the Mnemo
//! memory milestone.
