//! Safe, idiomatic Rust bindings for [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM).
//!
//! This crate wraps the stable LiteRT-LM **C API** (`c/engine.h`), which is the supported
//! FFI surface for the C++ Conversation / Engine runtime documented in `litert.md`.
//!
//! # Features
//! - **Conversation API** — chat-style `send_message` / streaming
//! - **Multimodal** — vision & audio backends, image/audio content parts (path or blob)
//! - **MTP** — Multi-Token Prediction via speculative decoding
//! - **Session API** — lower-level prefill / decode / generate_content
//! - Tools, thinking/reasoning, constrained decoding, LoRA, benchmarks
//!
//! # Quick start (text)
//! ```no_run
//! use litert_lm_rust::{Backend, Engine, Message};
//!
//! let engine = Engine::builder("model.litertlm")
//!     .backend(Backend::Cpu)
//!     .build()?;
//! let mut conversation = engine.create_conversation(Default::default())?;
//! let reply = conversation.send_message(Message::user("Hello!"))?;
//! println!("{}", reply);
//! # Ok::<(), litert_lm_rust::Error>(())
//! ```
//!
//! # Multimodal
//! ```no_run
//! use litert_lm_rust::{Backend, ContentPart, Engine, Message};
//!
//! let engine = Engine::builder("gemma-3n.litertlm")
//!     .backend(Backend::Cpu)
//!     .vision_backend(Backend::Gpu)
//!     .audio_backend(Backend::Cpu)
//!     .build()?;
//! let mut conversation = engine.create_conversation(Default::default())?;
//! let reply = conversation.send_message(Message::user_parts([
//!     ContentPart::text("Describe this image:"),
//!     ContentPart::image_path("photo.jpg"),
//! ])?)?;
//! # Ok::<(), litert_lm_rust::Error>(())
//! ```
//!
//! # MTP (Multi-Token Prediction)
//! ```no_run
//! use litert_lm_rust::{Backend, Engine};
//!
//! let engine = Engine::builder("model.litertlm")
//!     .backend(Backend::Gpu)
//!     .multi_token_prediction(true) // enable_speculative_decoding
//!     .build()?;
//! # Ok::<(), litert_lm_rust::Error>(())
//! ```
//!
//! # Native library
//! Link against a built LiteRT-LM C shared library (`bazel build //c:engine` or equivalent).
//! Set `LITERT_LM_LIB_DIR` to the directory containing the library.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clippy::missing_safety_doc)]

mod benchmark;
mod conversation;
mod engine;
mod error;
mod ffi;
mod input;
mod message;
mod sampler;
mod session;
mod stream;

pub use benchmark::BenchmarkInfo;
pub use conversation::{
    ConstraintType, Conversation, ConversationConfig, ConversationOptionalArgs,
    NoRepeatNgramConfig, RepetitionPenaltyConfig, SendOptions, SuppressTokensConfig,
    ThinkingConfig,
};
pub use engine::{ActivationDataType, Backend, Engine, EngineBuilder, LogSeverity};
pub use error::{Error, Result};
pub use input::InputData;
pub use message::{ContentPart, Message, Role};
pub use sampler::{SamplerParams, SamplerType};
pub use session::{Responses, Session, SessionConfig};
pub use stream::{StreamChunk, StreamEvent, StreamEventReceiver};

/// Crate version string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Set the minimum log severity for the native LiteRT-LM library.
pub fn set_min_log_level(level: LogSeverity) {
    unsafe { ffi::litert_lm_set_min_log_level(level.into_ffi()) }
}
