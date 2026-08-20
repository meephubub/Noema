//! Strongly typed errors for the Needle 2 integration.

use thiserror::Error;

/// The result type used by the Needle crate.
pub type Result<T> = std::result::Result<T, NeedleError>;

/// Errors from locating, loading, binding, or running the Needle 2 engine.
///
/// Error messages carry enough information for logging and debugging without
/// leaking sensitive user content.
#[derive(Debug, Error)]
pub enum NeedleError {
    /// No engine library could be found.
    ///
    /// Resolution order: `NEEDLE_LIB_PATH`, then
    /// `prebuilt/needle/<platform>/` relative to this repository, then the
    /// shared `~/.cache/cactus-needle/` cache used by the official Python
    /// package.
    #[error(
        "needle engine library not found (set NEEDLE_LIB_PATH or place libneedle in \
         prebuilt/needle/<platform>/): {0}"
    )]
    EngineNotFound(String),

    /// The engine library exists but could not be loaded.
    #[error("failed to load needle engine library: {0}")]
    LibraryLoad(String),

    /// The engine library is missing an expected C symbol.
    #[error("needle engine library is missing required symbol `{0}`")]
    MissingSymbol(String),

    /// `needle_init` returned an error code.
    #[error("needle_init failed with code {0}")]
    InitFailed(i32),

    /// `needle_load` returned an error code.
    #[error("needle_load failed with code {0}")]
    LoadFailed(i32),

    /// The engine already has different tuned weights bound and cannot unload
    /// them (a documented engine limitation).
    #[error(
        "tuned weights `{0}` are bound and the engine cannot unload them; construct \
         base-model agents first or run tuned and base workloads in separate processes"
    )]
    WeightsConflict(String),

    /// `needle_complete` returned an error code.
    #[error("needle_complete failed with code {0}")]
    CompleteFailed(i32),

    /// The engine returned output that could not be parsed as the expected
    /// JSON envelope.
    #[error("needle engine returned an unparseable envelope: {0}")]
    InvalidOutput(String),

    /// A failure in the subprocess (CLI) backend.
    #[error("needle CLI backend failed: {0}")]
    Cli(String),

    /// A filesystem or process-level failure.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_display_their_category() {
        assert!(NeedleError::InitFailed(-1).to_string().contains("needle_init"));
        assert!(NeedleError::CompleteFailed(-3).to_string().contains("needle_complete"));
        assert!(NeedleError::EngineNotFound("nope".into())
            .to_string()
            .contains("NEEDLE_LIB_PATH"));
    }

    #[test]
    fn result_flows() {
        fn fail() -> Result<()> {
            Err(NeedleError::InitFailed(-2))
        }
        assert!(fail().is_err());
    }
}
