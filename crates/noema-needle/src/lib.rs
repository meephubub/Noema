//! Needle 2 integration for Noema.
//!
//! Needle 2 is Cactus Compute's open 45M-parameter on-device model for tool
//! calling, device use, and structured extraction. It is built on the Simple
//! Attention Network recipe and shipped as a self-contained 14MB C engine
//! with byte-level grammar-constrained decoding and a calibrated confidence
//! head. This crate is the Rust binding: it loads the official engine and
//! exposes text-in / structured-tool-call-out through a small typed API.
//!
//! Layering, from outside in:
//!
//! ```text
//! Noema
//!   ↓
//! noema-needle (this crate)
//!   ↓
//! Needle 2 C API (needle_init / needle_complete / needle_reset / needle_load)
//!   ↓
//! Needle 2 engine
//! ```
//!
//! # Quick start
//!
//! ```no_run
//! use noema_needle::{DylibEngine, EngineSettings, NeedleEngine};
//!
//! let tools = r#"[
//!     {
//!         "name": "set_lights",
//!         "description": "Turn a room's lights on or off and set brightness",
//!         "parameters": {
//!             "type": "object",
//!             "properties": {
//!                 "room": { "type": "string", "description": "which room" },
//!                 "on": { "type": "boolean" },
//!                 "brightness": { "type": "integer", "minimum": 0, "maximum": 100 }
//!             },
//!             "required": ["room", "on"]
//!         }
//!     }
//! ]"#;
//!
//! let engine = DylibEngine::from_default(EngineSettings::new(tools))?;
//! let response = engine.complete("dim the living room to 30", 256)?;
//!
//! if response.is_refusal() {
//!     // Unsupported input: the engine refuses with an empty call list.
//! } else if let Some(call) = response.calls().first() {
//!     println!("{} {:?}", call.name, call.arguments);
//! }
//! # Ok::<(), noema_needle::NeedleError>(())
//! ```
//!
//! The engine library is found via `NEEDLE_LIB_PATH`, the repository's
//! `prebuilt/needle/<platform>/` directory, or the shared
//! `~/.cache/cactus-needle/` cache; see [`default_lib_path`].
//!
//! # Behaviour contract (from the model card)
//!
//! * Needle solves every problem as a function call. A request no declared
//!   tool can serve is refused with the empty call list; there is no
//!   free-text fallback.
//! * Arguments contain only values evidenced in the input; optional fields
//!   with no evidence are omitted, not guessed.
//! * After executing a call, feed the result back as the next
//!   [`NeedleEngine::complete`] input; later arguments may depend on earlier
//!   results.
//! * `confidence` gates escalation: act at or above your threshold, escalate
//!   below it. It is `None` for tuned `.cact` weights.
//!
//! # One physical model, many logical agents
//!
//! The plan calls for one physical Needle model exposed as multiple logical
//! tool agents, each with its own system prompt and schema. That works
//! naturally here: create one [`DylibEngine`] per logical agent; each binds
//! its own tools and system turn. The C engine keeps one process-global
//! session, so only the most recently used engine keeps its conversation —
//! the same trade-off the official Python binding makes.
//!
//! # Threading
//!
//! [`NeedleEngine::complete`] is blocking (the C call decodes the response).
//! From async code, run it inside `tokio::task::spawn_blocking`. All engine
//! instances share one process-global session guarded by a mutex.

pub mod engine;
pub mod error;
pub mod response;

pub use engine::{
    default_cli_path, default_lib_path, platform_tag, CliEngine, DylibEngine, EngineSettings,
    NeedleEngine, BINARY_NAME, ENGINE_VERSION, LIB_NAME,
};
pub use error::{NeedleError, Result};
pub use response::{FunctionCall, NeedleResponse, ValidationInfo};
