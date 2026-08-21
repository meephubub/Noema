//! Needle 2 engine backed by a statically-linked `libneedle.a`.
//!
//! On platforms where Cactus Compute ships only a static library (currently
//! macOS arm64), this crate compiles and links it at build time through a
//! thin C shim, making the Needle 2 C API symbols available to Rust FFI.
//!
//! # How it works
//!
//! ```text
//! noema-needle-static (this crate)
//!   ↓
//! c/shim.c  ←── references needle.h symbols
//!   ↓
//! libneedle.a (Cactus Compute static build)
//! ```
//!
//! The build script compiles `c/shim.c` with `cc`, which includes `needle.h`
//! from the prebuilt directory.  The shim's non-inlined function touches every
//! exported symbol, forcing the linker to pull in the archive.  Rust FFI
//! declarations then call the symbols directly — no `libloading` needed.

use std::fmt;
use std::sync::Mutex;

use noema_needle::engine::EngineSettings;
use noema_needle::error::{NeedleError, Result};
use noema_needle::response::NeedleResponse;
use noema_needle::NeedleEngine;

// ── FFI declarations ─────────────────────────────────────────────────────────
// These match the official `needle.h` header exactly.

extern "C" {
    fn needle_init(
        system_prompt: *const std::ffi::c_char,
        tools_json: *const std::ffi::c_char,
        tool_index_path: *const std::ffi::c_char,
    ) -> std::ffi::c_int;

    fn needle_complete(
        input: *const std::ffi::c_char,
        max_new_tokens: std::ffi::c_int,
        out: *mut std::ffi::c_char,
        out_capacity: std::ffi::c_int,
    ) -> std::ffi::c_int;

    fn needle_reset();

    fn needle_load(cact: *const u8, n: u64) -> std::ffi::c_int;
}

// ── Global state ─────────────────────────────────────────────────────────────
// The C API holds one process-global session (one toolset, one conversation).
// All calls are serialized through this lock, mirroring DylibEngine.

static ENGINE_LOCK: Mutex<()> = Mutex::new(());

/// Tracks which settings are currently bound so we can skip redundant re-init.
static BOUND_TOOLS: Mutex<Option<String>> = Mutex::new(None);
static BOUND_WEIGHTS: Mutex<Option<String>> = Mutex::new(None);

// ── StaticEngine ─────────────────────────────────────────────────────────────

/// A [`NeedleEngine`] backed by a statically-linked `libneedle.a`.
///
/// The symbols are resolved at link time — no `libloading` or runtime
/// library discovery needed.  Multi-turn state is preserved across calls
/// (the engine holds one process-global conversation).  All calls are
/// serialized through a global lock.
pub struct StaticEngine {
    label: String,
    settings: EngineSettings,
}

impl StaticEngine {
    /// Creates a static engine with the given settings.
    pub fn new(settings: EngineSettings) -> Self {
        let id = std::sync::atomic::AtomicUsize::new(0)
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let label = format!("needle2-static-{id}");
        Self { label, settings }
    }

    /// Binds this instance's tools and weights, if not already bound.
    fn bind(&self) -> Result<()> {
        let mut bound_tools = BOUND_TOOLS
            .lock()
            .map_err(|_| NeedleError::InitFailed(-1))?;

        let tools_fingerprint = &self.settings.tools_json;
        if bound_tools.as_deref() == Some(tools_fingerprint.as_str()) {
            return Ok(());
        }

        // Check weights.
        let mut bound_weights = BOUND_WEIGHTS
            .lock()
            .map_err(|_| NeedleError::InitFailed(-1))?;
        match (&self.settings.weights, bound_weights.as_ref()) {
            (None, Some(loaded)) => {
                return Err(NeedleError::WeightsConflict(loaded.clone()));
            }
            (Some(wanted), Some(loaded)) if loaded.as_str() != wanted.to_string_lossy().as_ref() => {
                return Err(NeedleError::WeightsConflict(loaded.clone()));
            }
            (Some(wanted), _) => {
                let blob = std::fs::read(wanted)
                    .map_err(|e| NeedleError::Cli(format!("failed to read weights: {e}")))?;
                let rc = unsafe { needle_load(blob.as_ptr(), blob.len() as u64) };
                if rc != 0 {
                    return Err(NeedleError::LoadFailed(rc));
                }
                *bound_weights = Some(wanted.to_string_lossy().to_string());
            }
            (None, None) => {}
        }

        // Build the system turn C string.
        let system = match self.settings.system.as_deref() {
            Some(text) => Some(
                std::ffi::CString::new(text)
                    .map_err(|_| NeedleError::InvalidOutput("system turn has interior NUL".into()))?,
            ),
            None => None,
        };
        let tools = std::ffi::CString::new(self.settings.tools_json.as_str())
            .map_err(|_| NeedleError::InvalidOutput("tools JSON has interior NUL".into()))?;
        let tool_index = match &self.settings.tool_index_path {
            Some(path) => Some(
                std::ffi::CString::new(path.to_string_lossy().as_bytes())
                    .map_err(|_| NeedleError::InvalidOutput("tool index path has interior NUL".into()))?,
            ),
            None => None,
        };

        let rc = unsafe {
            needle_init(
                system.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                tools.as_ptr(),
                tool_index
                    .as_ref()
                    .map_or(std::ptr::null(), |s| s.as_ptr()),
            )
        };
        if rc < 0 {
            return Err(NeedleError::InitFailed(rc));
        }

        *bound_tools = Some(tools_fingerprint.clone());
        Ok(())
    }
}

impl fmt::Debug for StaticEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StaticEngine")
            .field("label", &self.label)
            .field("settings", &self.settings)
            .finish_non_exhaustive()
    }
}

impl NeedleEngine for StaticEngine {
    fn id(&self) -> &str {
        &self.label
    }

    fn complete(&self, input: &str, max_new_tokens: u32) -> Result<NeedleResponse> {
        let _guard = ENGINE_LOCK
            .lock()
            .map_err(|_| NeedleError::CompleteFailed(-1))?;
        self.bind()?;

        let input = std::ffi::CString::new(input)
            .map_err(|_| NeedleError::InvalidOutput("input has interior NUL".into()))?;
        let mut buffer = vec![0u8; self.settings.buffer_size];

        let rc = unsafe {
            needle_complete(
                input.as_ptr(),
                max_new_tokens as std::ffi::c_int,
                buffer.as_mut_ptr() as *mut std::ffi::c_char,
                buffer.len() as std::ffi::c_int,
            )
        };
        if rc < 0 {
            return Err(NeedleError::CompleteFailed(rc));
        }

        let raw = unsafe { std::ffi::CStr::from_ptr(buffer.as_ptr() as *const std::ffi::c_char) };
        let text = raw
            .to_str()
            .map_err(|_| NeedleError::InvalidOutput("output is not valid UTF-8".into()))?;

        serde_json::from_str(text).map_err(|err| {
            tracing::warn!(error = %err, raw = %text, "unparseable needle envelope");
            NeedleError::InvalidOutput(err.to_string())
        })
    }

    fn reset(&self) -> Result<()> {
        let _guard = ENGINE_LOCK
            .lock()
            .map_err(|_| NeedleError::InitFailed(-1))?;
        self.bind()?;
        unsafe { needle_reset() };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_engine_compiles_and_links() {
        // Verify the FFI symbols are resolvable at link time.
        // On platforms where libneedle.a is not present, this test is
        // skipped at build time (the crate won't compile).
        let engine = StaticEngine::new(EngineSettings::new(r#"[]"#));
        assert!(!engine.id().is_empty());
    }
}
