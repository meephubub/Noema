//! Logging setup and level handling.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::config::LogLevel;
use crate::error::{NoemaError, Result};

static LOGGING_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Installs the global `tracing` subscriber at the given verbosity.
///
/// Only the first call takes effect; later calls are ignored, matching the
/// single-global-subscriber constraint of `tracing`. Use [`LogLevel::Off`]
/// to leave `tracing` uninitialized, which makes all logging a no-op.
pub fn init_logging(level: LogLevel) -> Result<()> {
    if LOGGING_INITIALIZED.swap(true, Ordering::SeqCst) {
        tracing::warn!("logging already initialized; ignoring call with level {level:?}");
        return Ok(());
    }

    let Some(level) = level.as_tracing_level() else {
        // `Off`: leave `tracing` uninitialized; log calls become no-ops.
        return Ok(());
    };

    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_target(false)
        .try_init()
        .map_err(|e| NoemaError::Configuration(format!("failed to initialize logging: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_logging_is_callable_with_all_levels() {
        // Each call in its own scope keeps the guard semantics predictable:
        // the first call in this test process wins, later ones are ignored.
        let _ = init_logging(LogLevel::Off);
        let _ = init_logging(LogLevel::Error);
        let _ = init_logging(LogLevel::Warn);
        let _ = init_logging(LogLevel::Info);
        let _ = init_logging(LogLevel::Debug);
        let _ = init_logging(LogLevel::Trace);
        // Reaching here without a panic is the assertion.
    }
}
