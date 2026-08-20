//! Needle 2 engine backends.
//!
//! Needle 2 is a 45M-parameter tool-calling model from Cactus Compute. It is
//! not a standard transformer: it is a *Simple Attention Network* (Hadamard
//! MLP, GQA attention, engram key-value memory, multi-lane hyper-connections)
//! quantized with Cactus Quants and baked into a self-contained C engine with
//! byte-level grammar-constrained decoding, a confidence head, and tool
//! retrieval. There are no standard weights a generic runtime like candle
//! could load; the checkpoint is the engine itself.
//!
//! This crate therefore follows the layering the Noema plan specifies:
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
//! Two backends implement the [`NeedleEngine`] trait:
//!
//! * [`DylibEngine`] loads the official shared library (`libneedle.dll` /
//!   `libneedle.so` / `libneedle.dylib`) at runtime through `libloading` and
//!   calls the C API directly. This is the primary backend: it keeps a warm
//!   in-process model, preserves multi-turn conversation state, and runs on
//!   CPU with no dependencies beyond the library file.
//! * [`CliEngine`] shells out to the official `needle` command-line binary
//!   in one-shot mode (`--tools tools.json --prompt ...`). It is stateless
//!   per call — a useful fallback and debugging aid on platforms where the
//!   shared library is unavailable.
//!
//! # Engine state and threads
//!
//! The C API holds one process-global session: one toolset, one conversation.
//! [`DylibEngine`] mirrors the official Python binding: all calls are
//! serialized through a global lock, and when a different engine instance is
//! already bound, its tools are rebound (which starts a fresh conversation
//! for the previously active instance). Multi-turn state is preserved only
//! for the engine that is currently bound. The C calls are blocking; when
//! used from async code, run them via `tokio::task::spawn_blocking`.

use std::ffi::{c_char, c_int, c_uchar, c_ulonglong, CStr, CString};
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use libloading::Library;

use crate::error::{NeedleError, Result};
use crate::response::NeedleResponse;

/// The engine version whose artifacts this crate is tested against.
pub const ENGINE_VERSION: &str = "2.0.3";

/// The name of the shared engine library on each OS.
pub const LIB_NAME: &str = {
    #[cfg(target_os = "windows")]
    {
        "libneedle.dll"
    }
    #[cfg(target_os = "macos")]
    {
        "libneedle.dylib"
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        "libneedle.so"
    }
};

/// The name of the command-line engine binary on each OS.
pub const BINARY_NAME: &str = {
    #[cfg(target_os = "windows")]
    {
        "needle.exe"
    }
    #[cfg(not(target_os = "windows"))]
    {
        "needle"
    }
};

/// The `<os>-<arch>` folder name used under `prebuilt/needle/`.
pub fn platform_tag() -> &'static str {
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        "windows-x86_64"
    }
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    {
        "windows-arm64"
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        "linux-x86_64"
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        "linux-arm64"
    }
    #[cfg(all(target_os = "linux", target_arch = "arm"))]
    {
        "linux-armv7"
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        "macos-arm64"
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        "macos-x86_64"
    }
    #[cfg(not(any(
        all(target_os = "windows", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "aarch64"),
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
        all(target_os = "linux", target_arch = "arm"),
        all(target_os = "macos", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "x86_64"),
    )))]
    {
        // Cactus Compute ships a build for most targets; fall back to the
        // closest x86-64 folder so the error message is still useful.
        "linux-x86_64"
    }
}

/// Common settings for an engine instance.
#[derive(Debug, Clone)]
pub struct EngineSettings {
    /// Optional system turn carrying environment facts (`date:`, `locale:`,
    /// ...). The engine treats it as facts, never instructions.
    pub system: Option<String>,
    /// The JSON array of tool schemas the model may call.
    pub tools_json: String,
    /// Optional path to persist tool embeddings for large tool catalogues.
    pub tool_index_path: Option<PathBuf>,
    /// Optional path to a tuned `.cact` archive to load instead of the baked
    /// base model. The engine cannot unload tuned weights once bound.
    pub weights: Option<PathBuf>,
    /// Output buffer size for `needle_complete`, in bytes.
    pub buffer_size: usize,
}

impl EngineSettings {
    /// Settings with the given tool schema JSON array and no system turn.
    pub fn new(tools_json: impl Into<String>) -> Self {
        Self {
            system: None,
            tools_json: tools_json.into(),
            tool_index_path: None,
            weights: None,
            buffer_size: 1 << 16,
        }
    }
}

/// A handle to a Needle 2 engine.
pub trait NeedleEngine: fmt::Debug + Send + Sync {
    /// A stable identifier for this engine instance.
    fn id(&self) -> &str;

    /// One turn: text in, a structured tool-call envelope out.
    ///
    /// Blocking. From async code, run inside `tokio::task::spawn_blocking`.
    fn complete(&self, input: &str, max_new_tokens: u32) -> Result<NeedleResponse>;

    /// Rewinds the conversation, keeping the tools loaded.
    fn reset(&self) -> Result<()>;
}

// ---------------------------------------------------------------------------
// Engine discovery
// ---------------------------------------------------------------------------

/// The path to the official engine binary (CLI backend), if it can be found.
///
/// Resolution order: `NEEDLE_CLI_PATH`, then
/// `prebuilt/needle/<platform>/<binary>` in this repository.
pub fn default_cli_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("NEEDLE_CLI_PATH") {
        return Some(PathBuf::from(path));
    }
    prebuilt_dir().map(|dir| dir.join(BINARY_NAME))
}

/// The path to the shared engine library (FFI backend), if it can be found.
///
/// Resolution order:
/// 1. `NEEDLE_LIB_PATH` (the same variable the official Python package uses);
/// 2. `prebuilt/needle/<platform>/libneedle.*` in this repository;
/// 3. the shared user cache `~/.cache/cactus-needle/<version>/` that the
///    Python package populates.
pub fn default_lib_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("NEEDLE_LIB_PATH") {
        return Some(PathBuf::from(path));
    }
    if let Some(dir) = prebuilt_dir() {
        let candidate = dir.join(LIB_NAME);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    if let Some(cache) = user_cache_dir() {
        let candidate = cache.join(LIB_NAME);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

/// The `prebuilt/needle/<platform>/` directory in this repository.
fn prebuilt_dir() -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // `crates/noema-needle/../../prebuilt`
    let repo = manifest
        .parent()?
        .parent()?
        .join("prebuilt")
        .join("needle")
        .join(platform_tag());
    Some(repo)
}

/// The shared engine cache used by the official Python package.
fn user_cache_dir() -> Option<PathBuf> {
    let home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))?;
    Some(
        PathBuf::from(home)
            .join(".cache")
            .join("cactus-needle")
            .join(ENGINE_VERSION),
    )
}

// ---------------------------------------------------------------------------
// FFI backend
// ---------------------------------------------------------------------------

type InitFn = unsafe extern "C" fn(
    system_prompt: *const c_char,
    tools_json: *const c_char,
    tool_index_path: *const c_char,
) -> c_int;
type CompleteFn = unsafe extern "C" fn(
    input: *const c_char,
    max_new_tokens: c_int,
    out: *mut c_char,
    out_capacity: c_int,
) -> c_int;
type ResetFn = unsafe extern "C" fn();
type LoadFn = unsafe extern "C" fn(cact: *const c_uchar, n: c_ulonglong) -> c_int;

/// The engine's process-global session, guarded against concurrent access.
///
/// The C API holds one toolset and one conversation per process. All calls go
/// through this lock, and [`ACTIVE_ENGINE`] remembers which instance's tools
/// are currently bound, exactly like the official Python binding.
static ENGINE_LOCK: Mutex<()> = Mutex::new(());
static ACTIVE_ENGINE: Mutex<Option<usize>> = Mutex::new(None);
static ACTIVE_WEIGHTS: Mutex<Option<PathBuf>> = Mutex::new(None);
static NEXT_ENGINE_ID: AtomicUsize = AtomicUsize::new(0);

/// A `NeedleEngine` backed by the official shared library through its C API.
///
/// The library is loaded lazily and stays loaded for the process. The model
/// weights are baked into the library (unless [`EngineSettings::weights`]
/// points at a tuned `.cact`).
pub struct DylibEngine {
    id: usize,
    label: String,
    settings: EngineSettings,
    /// Kept for its drop guard: the resolved function pointers below are only
    /// valid while the library stays loaded.
    #[allow(dead_code)]
    library: Library,
    init: InitFn,
    complete: CompleteFn,
    reset: ResetFn,
    load: LoadFn,
}

impl fmt::Debug for DylibEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DylibEngine")
            .field("id", &self.id)
            .field("label", &self.label)
            .field("settings", &self.settings)
            .finish_non_exhaustive()
    }
}

impl DylibEngine {
    /// Loads the engine from an explicit library path.
    pub fn load(path: impl AsRef<Path>, settings: EngineSettings) -> Result<Self> {
        let path = path.as_ref();
        // SAFETY: the path points at the official engine library produced by
        // Cactus Compute; loading it and calling its exported C functions is
        // the intended use of the library.
        let library = unsafe { Library::new(path) }.map_err(|err| {
            NeedleError::LibraryLoad(format!("{}: {err}", path.display()))
        })?;

        // SAFETY: the symbols are declared by the official `needle.h` header;
        // the resulting pointers are kept with the library for its lifetime.
        let init: InitFn = unsafe {
            *library
                .get(b"needle_init\0")
                .map_err(|_| NeedleError::MissingSymbol("needle_init".into()))?
        };
        let complete: CompleteFn = unsafe {
            *library
                .get(b"needle_complete\0")
                .map_err(|_| NeedleError::MissingSymbol("needle_complete".into()))?
        };
        let reset: ResetFn = unsafe {
            *library
                .get(b"needle_reset\0")
                .map_err(|_| NeedleError::MissingSymbol("needle_reset".into()))?
        };
        let load: LoadFn = unsafe {
            *library
                .get(b"needle_load\0")
                .map_err(|_| NeedleError::MissingSymbol("needle_load".into()))?
        };

        let id = NEXT_ENGINE_ID.fetch_add(1, Ordering::Relaxed);
        let label = format!("needle2-{id}");
        Ok(Self {
            id,
            label,
            settings,
            library,
            init,
            complete,
            reset,
            load,
        })
    }

    /// Loads the engine from the default discovery path.
    ///
    /// Returns [`NeedleError::EngineNotFound`] when no library is available;
    /// see [`default_lib_path`] for the resolution order.
    pub fn from_default(settings: EngineSettings) -> Result<Self> {
        let path = default_lib_path().ok_or_else(|| {
            NeedleError::EngineNotFound(format!(
                "looked for {} under prebuilt/needle/{}/ and the cactus-needle cache",
                LIB_NAME,
                platform_tag()
            ))
        })?;
        Self::load(path, settings)
    }

    /// Binds this instance's system turn and tools, if a different instance
    /// is currently bound. Must be called with [`ENGINE_LOCK`] held.
    fn bind(&self) -> Result<()> {
        let mut active = ACTIVE_ENGINE
            .lock()
            .map_err(|_| NeedleError::InitFailed(-1))?;
        if *active == Some(self.id) {
            return Ok(());
        }

        let mut active_weights = ACTIVE_WEIGHTS
            .lock()
            .map_err(|_| NeedleError::InitFailed(-1))?;
        match (&self.settings.weights, active_weights.as_ref()) {
            (None, Some(loaded)) => {
                return Err(NeedleError::WeightsConflict(loaded.display().to_string()));
            }
            (Some(wanted), Some(loaded)) if *loaded != *wanted => {
                return Err(NeedleError::WeightsConflict(loaded.display().to_string()));
            }
            (Some(wanted), _) => {
                let blob = std::fs::read(wanted)?;
                // SAFETY: `blob` outlives the call; the engine copies what it
                // needs into its own session.
                let rc = unsafe { (self.load)(blob.as_ptr(), blob.len() as c_ulonglong) };
                if rc != 0 {
                    return Err(NeedleError::LoadFailed(rc));
                }
                *active_weights = Some(wanted.clone());
            }
            (None, None) => {}
        }

        let system = match self.settings.system.as_deref() {
            Some(text) => Some(CString::new(text).map_err(|_| {
                NeedleError::InvalidOutput("system turn contains an interior NUL byte".into())
            })?),
            None => None,
        };
        let tools = CString::new(self.settings.tools_json.as_str()).map_err(|_| {
            NeedleError::InvalidOutput("tools JSON contains an interior NUL byte".into())
        })?;
        let tool_index = match &self.settings.tool_index_path {
            Some(path) => Some(CString::new(path.to_string_lossy().as_bytes()).map_err(|_| {
                NeedleError::InvalidOutput("tool index path contains an interior NUL byte".into())
            })?),
            None => None,
        };

        // SAFETY: all pointers are valid NUL-terminated C strings (or null);
        // the engine copies them during init.
        let rc = unsafe {
            (self.init)(
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
        *active = Some(self.id);
        Ok(())
    }
}

impl NeedleEngine for DylibEngine {
    fn id(&self) -> &str {
        &self.label
    }

    fn complete(&self, input: &str, max_new_tokens: u32) -> Result<NeedleResponse> {
        let _guard = ENGINE_LOCK
            .lock()
            .map_err(|_| NeedleError::CompleteFailed(-1))?;
        self.bind()?;

        let input = CString::new(input).map_err(|_| {
            NeedleError::InvalidOutput("input contains an interior NUL byte".into())
        })?;
        let mut buffer = vec![0u8; self.settings.buffer_size];

        // SAFETY: `buffer` is writable for `buffer_size` bytes; the engine
        // writes a NUL-terminated JSON envelope into it.
        let rc = unsafe {
            (self.complete)(
                input.as_ptr(),
                max_new_tokens as c_int,
                buffer.as_mut_ptr() as *mut c_char,
                buffer.len() as c_int,
            )
        };
        if rc < 0 {
            return Err(NeedleError::CompleteFailed(rc));
        }

        // SAFETY: the engine NUL-terminates the output within capacity.
        let raw = unsafe { CStr::from_ptr(buffer.as_ptr() as *const c_char) };
        let text = raw
            .to_str()
            .map_err(|_| NeedleError::InvalidOutput("output is not valid UTF-8".into()))?;
        parse_envelope(text)
    }

    fn reset(&self) -> Result<()> {
        let _guard = ENGINE_LOCK
            .lock()
            .map_err(|_| NeedleError::InitFailed(-1))?;
        self.bind()?;
        // SAFETY: the engine's reset function takes no arguments.
        unsafe { (self.reset)() };
        Ok(())
    }
}

fn parse_envelope(text: &str) -> Result<NeedleResponse> {
    serde_json::from_str(text).map_err(|err| {
        tracing::warn!(error = %err, raw = %text, "unparseable needle envelope");
        NeedleError::InvalidOutput(err.to_string())
    })
}

// ---------------------------------------------------------------------------
// CLI backend
// ---------------------------------------------------------------------------

/// A stateless [`NeedleEngine`] that shells out to the official `needle`
/// command-line binary in one-shot mode.
///
/// Every `complete` spawns a fresh process with `--tools <tools.json>
/// --prompt <input>`, so conversation state does not carry between calls and
/// `reset` is a no-op. Prefer [`DylibEngine`] for real workloads.
pub struct CliEngine {
    label: String,
    binary: PathBuf,
    tools_file: tempfile::NamedTempFile,
    system_file: Option<tempfile::NamedTempFile>,
}

impl fmt::Debug for CliEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CliEngine")
            .field("label", &self.label)
            .field("binary", &self.binary)
            .finish_non_exhaustive()
    }
}

impl CliEngine {
    /// Creates an engine that runs the given binary.
    pub fn new(binary: impl AsRef<Path>, settings: EngineSettings) -> Result<Self> {
        let mut tools_file = tempfile::NamedTempFile::new()?;
        std::io::Write::write_all(&mut tools_file, settings.tools_json.as_bytes())?;
        let system_file = match settings.system {
            Some(system) => {
                let mut file = tempfile::NamedTempFile::new()?;
                std::io::Write::write_all(&mut file, system.as_bytes())?;
                Some(file)
            }
            None => None,
        };
        let label = format!(
            "needle2-cli-{}",
            NEXT_ENGINE_ID.fetch_add(1, Ordering::Relaxed)
        );
        Ok(Self {
            label,
            binary: binary.as_ref().to_path_buf(),
            tools_file,
            system_file,
        })
    }

    /// Creates an engine from the default discovery path
    /// (see [`default_cli_path`]).
    pub fn from_default(settings: EngineSettings) -> Result<Self> {
        let binary = default_cli_path().ok_or_else(|| {
            NeedleError::EngineNotFound(format!(
                "looked for {BINARY_NAME} under prebuilt/needle/{}/",
                platform_tag()
            ))
        })?;
        Self::new(binary, settings)
    }
}

impl NeedleEngine for CliEngine {
    fn id(&self) -> &str {
        &self.label
    }

    fn complete(&self, input: &str, _max_new_tokens: u32) -> Result<NeedleResponse> {
        let mut command = Command::new(&self.binary);
        command
            .arg("--tools")
            .arg(self.tools_file.path())
            .arg("--prompt")
            .arg(input);
        if let Some(system_file) = &self.system_file {
            command.arg("--system").arg(system_file.path());
        }

        let output = command.output().map_err(|err| {
            NeedleError::Cli(format!(
                "failed to run {}: {err}",
                self.binary.display()
            ))
        })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(NeedleError::Cli(format!(
                "{} exited with {}: {}",
                self.binary.display(),
                output.status,
                stderr.trim()
            )));
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        parse_envelope(stdout.trim())
    }

    fn reset(&self) -> Result<()> {
        // One-shot invocations are stateless; nothing to rewind.
        Ok(())
    }
}

/// Builder-style helpers for [`EngineSettings`].
impl EngineSettings {
    /// Sets the system turn.
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Sets the tool-index persistence path.
    pub fn with_tool_index(mut self, path: impl Into<PathBuf>) -> Self {
        self.tool_index_path = Some(path.into());
        self
    }

    /// Sets tuned `.cact` weights to load.
    pub fn with_weights(mut self, path: impl Into<PathBuf>) -> Self {
        self.weights = Some(path.into());
        self
    }

    /// Sets the `needle_complete` output buffer size in bytes.
    pub fn with_buffer_size(mut self, bytes: usize) -> Self {
        self.buffer_size = bytes;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lib_name_is_platform_specific() {
        assert!(!LIB_NAME.is_empty());
        assert!(!BINARY_NAME.is_empty());
        assert!(!platform_tag().is_empty());
    }

    #[test]
    fn settings_build() {
        let settings = EngineSettings::new(r#"[]"#).with_system("date: 2026-08-20");
        assert_eq!(settings.system.as_deref(), Some("date: 2026-08-20"));
        assert_eq!(settings.tools_json, "[]");
        assert_eq!(settings.buffer_size, 1 << 16);
    }
}
