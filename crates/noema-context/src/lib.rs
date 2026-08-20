//! Context construction and optimization for Noema.
//!
//! Planned contents:
//!
//! * Conversation context
//! * Memory context (from Mnemo)
//! * Tool summaries (lightweight descriptions for Gemma)
//! * Prompt construction
//! * Context trimming and minimization
//!
//! The Context Builder assembles a minimized context package — current user
//! message, conversation history, relevant Mnemo memories, application
//! state, available tools, tool results, and previous agent decisions —
//! before invoking a model.
//!
//! This crate is scaffolded in phase 1 and implemented with the agent-loop
//! milestone.
