//! Error types for LiteRT-LM Rust bindings.

use thiserror::Error;

/// Result alias for this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors returned by the LiteRT-LM Rust bindings.
#[derive(Debug, Error)]
pub enum Error {
    #[error("native call `{0}` returned a null pointer")]
    NullPointer(&'static str),

    #[error("native call `{0}` failed with status {1}")]
    NativeStatus(&'static str, i32),

    #[error("invalid model path (not valid UTF-8 or contains interior NUL)")]
    ModelPath,

    #[error("NUL byte in C string: {0}")]
    Nul(#[from] std::ffi::NulError),

    #[error("invalid UTF-8 from native library: {0}")]
    Utf8(#[from] std::str::Utf8Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("index {0} out of range (len={1})")]
    IndexOutOfRange(usize, usize),

    #[error("{0}")]
    Message(String),
}
