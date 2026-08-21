//! Build script for `litert-lm-rust`.
//!
//! # Linking
//!
//! Linking is controlled by environment variables:
//! - `LITERT_LM_LIB_DIR` — directory containing the LiteRT-LM C shared library
//! - `LITERT_LM_LIB_NAME` — library name without `lib` prefix / extension
//!   (default: tries `litert-lm`, `LiteRtLmC`, `engine` in that order)
//! - `LITERT_LM_INCLUDE_DIR` — optional override for the C header directory
//!
//! On Windows the build script also looks for `<name>.if.lib` (Bazel-style import
//! library) in addition to the standard `<name>.lib` form.
//!
//! Enable the `bindgen` feature to regenerate bindings from `c/wrapper.h`.
//! Enable the `download-native` feature to automatically download native libraries
//! from GitHub releases if not found locally.

use std::env;
use std::path::PathBuf;

#[cfg(feature = "download-native")]
use std::fs;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    println!("cargo:rerun-if-changed=c/engine.h");
    println!("cargo:rerun-if-changed=c/wrapper.h");
    println!("cargo:rerun-if-changed=src/bindings.rs");
    println!("cargo:rerun-if-env-changed=LITERT_LM_LIB_DIR");
    println!("cargo:rerun-if-env-changed=LITERT_LM_LIB_NAME");
    println!("cargo:rerun-if-env-changed=LITERT_LM_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=LITERT_LM_STATIC");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bindings_path = out_dir.join("bindings.rs");

    // ── Bindgen ───────────────────────────────────────────────────────────────
    #[cfg(feature = "bindgen")]
    {
        let include_dir = env::var_os("LITERT_LM_INCLUDE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| manifest_dir.join("c"));
        generate_bindings(&include_dir, &bindings_path);
    }

    #[cfg(not(feature = "bindgen"))]
    {
        let vendored = manifest_dir.join("src/bindings.rs");
        if vendored.exists() {
            std::fs::copy(&vendored, &bindings_path).expect("copy vendored bindings");
        } else {
            panic!(
                "src/bindings.rs missing. Build once with `--features bindgen` to generate it."
            );
        }
    }

    // ── Skip native linking for docs.rs ─────────────────────────────────────
    if env::var_os("CARGO_FEATURE_DOCS_ONLY").is_some() {
        return;
    }

    // ── Attempt download if feature enabled ────────────────────────────────
    #[cfg(feature = "download-native")]
    {
        let cache = prebuilt_cache_dir();
        // Check if the shared cache already has the required files.
        if !has_required_files(&cache) {
            let local = out_dir.join("prebuilt");
            if !has_required_files(&local) {
                println!("cargo:warning=Native libraries not found, attempting download...");
                if let Err(e) = download_native_libraries(&local) {
                    println!("cargo:warning=Failed to download native libraries: {}", e);
                }
            }
        }
    }

    link_native_library(&manifest_dir, &out_dir);
}

// ── Bindgen (optional feature) ────────────────────────────────────────────────
#[cfg(feature = "bindgen")]
fn generate_bindings(include_dir: &PathBuf, bindings_path: &PathBuf) {
    let wrapper = include_dir.join("wrapper.h");
    let bindings = bindgen::Builder::default()
        .header(wrapper.to_string_lossy())
        .clang_arg(format!("-I{}", include_dir.display()))
        .clang_arg("-DLITERT_LM_C_API_EXPORT=")
        .allowlist_function("litert_lm_.*")
        .allowlist_type("LiteRtLm.*")
        .allowlist_var("kLiteRtLm.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate LiteRT-LM bindings");
    bindings
        .write_to_file(bindings_path)
        .expect("Couldn't write bindings");

    // Keep a checked-in copy for docs.rs / consumers without libclang.
    let vendored = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("src/bindings.rs");
    let _ = std::fs::copy(bindings_path, vendored);
}

// ── Native library linking ────────────────────────────────────────────────────

/// Library name candidates tried in order when `LITERT_LM_LIB_NAME` is not set.
fn default_lib_names() -> Vec<String> {
    // On Windows the main import library ships as `litert-lm.if.lib` (Bazel
    // style).  We also try the conventional names so community prebuilts work.
    vec![
        "litert-lm".to_string(),
        "LiteRtLmC".to_string(),
        "engine".to_string(),
    ]
}

fn link_native_library(manifest_dir: &PathBuf, out_dir: &PathBuf) {
    // ── Resolve candidate directories ─────────────────────────────────────────
    let mut search_dirs: Vec<PathBuf> = [
        env::var_os("LITERT_LM_LIB_DIR").map(PathBuf::from),
        Some(out_dir.join("prebuilt")), // Downloaded libraries go here
        // Noema workspace: the root keeps native libraries in `prebuilt/`.
        Some(manifest_dir.join("../../prebuilt")),
        Some(manifest_dir.join("../../prebuilt/macos")),
        Some(manifest_dir.join("prebuilt")),
        Some(manifest_dir.join("native")),
        Some(manifest_dir.join("c")),
        Some(manifest_dir.join("c/build")),
    ]
    .into_iter()
    .flatten()
    .filter(|p| p.is_dir())
    .collect();
    // macOS: also search the `macos/` subdirectory of the workspace prebuilt folder.
    let macos_subdir = manifest_dir.join("../../prebuilt/macos");
    if macos_subdir.is_dir() && !search_dirs.contains(&macos_subdir) {
        search_dirs.push(macos_subdir);
    }
    // Shared cache: ~/.noema/prebuilt/ (download-native feature).
    #[cfg(feature = "download-native")]
    {
        let cache = prebuilt_cache_dir();
        if cache.is_dir() && !search_dirs.contains(&cache) {
            search_dirs.push(cache.clone());
        }
        // macOS dylibs may be in a `macos/` subdirectory of the cache.
        let cache_macos = cache.join("macos");
        if cache_macos.is_dir() && !search_dirs.contains(&cache_macos) {
            search_dirs.push(cache_macos);
        }
    }
    // Deduplicate.
    search_dirs.dedup();

    // ── Resolve library name(s) to try ────────────────────────────────────────
    let names: Vec<String> = if let Ok(n) = env::var("LITERT_LM_LIB_NAME") {
        vec![n]
    } else {
        default_lib_names()
    };

    // ── Search ────────────────────────────────────────────────────────────────
    for dir in &search_dirs {
        for name in &names {
            if dir_has_library(dir, name) {
                println!("cargo:rustc-link-search=native={}", dir.display());
                emit_link_lib(name, dir);
                emit_platform_extra_libs();
                return;
            }
        }
    }

    // ── Fallback: warn but still emit a directive so dependents know what to
    //    provide at final-link time. ────────────────────────────────────────────
    let tried: Vec<_> = search_dirs.iter().map(|p| p.display().to_string()).collect();
    println!(
        "cargo:warning=litert-lm-rust: native library not found in [{}]. \
         Set LITERT_LM_LIB_DIR to the folder containing the DLL/import-lib \
         before linking an executable.",
        tried.join(", ")
    );
    // Emit a link directive for the first candidate so the linker error is
    // informative rather than silent.
    println!("cargo:rustc-link-lib=dylib={}", names[0]);
}

/// Emit the correct `cargo:rustc-link-lib` directive for a found library.
///
/// On Windows we prefer `<name>.if.lib` (Bazel-style import library) which
/// Cargo / MSVC link correctly when the `.if.lib` is on the search path.
fn emit_link_lib(name: &str, dir: &PathBuf) {
    let is_static = env::var_os("LITERT_LM_STATIC").is_some();

    // Always add the directory to the linker search path so the linker can
    // resolve the library by name regardless of platform-specific naming
    // conventions (.if.lib, .lib, .dll.a, etc.).
    println!("cargo:rustc-link-search=native={}", dir.display());

    if cfg!(windows) && !is_static {
        // Check for the Bazel-style import library first (`litert-lm.if.lib`).
        // MSVC linker requires the exact filename when using .if.lib, so we
        // emit it as a raw linker argument.
        let if_lib = dir.join(format!("{name}.if.lib"));
        if if_lib.exists() {
            println!("cargo:rustc-link-lib=dylib={name}");
            println!("cargo:rustc-link-arg={}", if_lib.display());
            return;
        }
    }

    // Standard path.
    if is_static {
        println!("cargo:rustc-link-lib=static={name}");
    } else {
        println!("cargo:rustc-link-lib=dylib={name}");
    }
}

fn emit_platform_extra_libs() {
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=m");
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=pthread");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
        // Embed an rpath so the dynamic linker can find the LiteRT dylibs
        // next to the executable at runtime (no DYLD_LIBRARY_PATH needed).
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../lib");
    }
}

/// Return true if the directory contains any recognisable library file for `name`.
fn dir_has_library(dir: &PathBuf, name: &str) -> bool {
    if !dir.is_dir() {
        return false;
    }
    let candidates = [
        format!("{name}.dll"),
        format!("{name}.lib"),
        format!("{name}.if.lib"),
        format!("lib{name}.so"),
        format!("lib{name}.dylib"),
        format!("lib{name}.dll"),
        format!("lib{name}.a"),
        format!("{name}.a"),
    ];
    candidates.iter().any(|n| dir.join(n).exists())
}

// ── Native library download (optional feature) ────────────────────────────────

/// URL of the prebuilt archive containing all platform binaries.
const PREBUILT_URL: &str =
    "https://github.com/meephubub/Noema/releases/download/v1.0.0/prebuilt.zip";

/// Shared cache directory for downloaded native libraries.
/// Respects `$NOEMA_PREBUILT_DIR` override; defaults to `~/.noema/prebuilt/`.
#[cfg(feature = "download-native")]
fn prebuilt_cache_dir() -> PathBuf {
    if let Ok(dir) = env::var("NOEMA_PREBUILT_DIR") {
        return PathBuf::from(dir);
    }
    let home = env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .unwrap_or_else(|| ".".into());
    PathBuf::from(home).join(".noema").join("prebuilt")
}

/// Required native files for the current platform.
#[cfg(feature = "download-native")]
fn required_native_files() -> Vec<&'static str> {
    if cfg!(target_os = "windows") {
        vec![
            "litert-lm.dll",
            "litert-lm.if.lib",
            "litert-lm.lib",
            "libLiteRt.dll",
            "libGemmaModelConstraintProvider.dll",
            "libLiteRtTopKWebGpuSampler.dll",
            "libLiteRtWebGpuAccelerator.dll",
            "libwebgpu_dawn.dll",
        ]
    } else if cfg!(target_os = "macos") {
        vec![
            "litert-lm.dylib",
            "libLiteRt.dylib",
            "libGemmaModelConstraintProvider.dylib",
            "libLiteRtTopKMetalSampler.dylib",
            "libLiteRtMetalAccelerator.dylib",
        ]
    } else {
        // Linux
        vec![
            "litert-lm.so",
            "libLiteRt.so",
            "libGemmaModelConstraintProvider.so",
            "libLiteRtTopKWebGpuSampler.so",
            "libLiteRtWebGpuAccelerator.so",
            "libwebgpu_dawn.so",
        ]
    }
}

#[cfg(feature = "download-native")]
fn has_required_files(dir: &PathBuf) -> bool {
    required_native_files().iter().all(|f| dir.join(f).exists())
}

/// Download the prebuilt.zip archive and extract only the LiteRT files
/// for the current platform into `dest`.
#[cfg(feature = "download-native")]
fn download_native_libraries(dest: &PathBuf) -> Result<(), String> {
    let cache = prebuilt_cache_dir();
    fs::create_dir_all(&cache)
        .map_err(|e| format!("Failed to create cache directory {}: {e}", cache.display()))?;

    let zip_path = cache.join("prebuilt.zip");

    // Download only if not already cached.
    if !zip_path.exists() {
        println!("cargo:warning=Downloading prebuilt binaries from {PREBUILT_URL}...");
        download_file(PREBUILT_URL, &zip_path)?;
        println!("cargo:warning=Downloaded prebuilt.zip successfully.");
    } else {
        println!("cargo:warning=Using cached prebuilt.zip at {}", zip_path.display());
    }

    // Ensure the destination directory exists.
    fs::create_dir_all(dest)
        .map_err(|e| format!("Failed to create destination {}: {e}", dest.display()))?;

    // Extract platform-relevant files from the zip.
    let file = fs::File::open(&zip_path)
        .map_err(|e| format!("Failed to open {}: {e}", zip_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| format!("Failed to read zip: {e}"))?;

    let prefix = if cfg!(target_os = "macos") {
        "prebuilt/macos/"
    } else {
        "prebuilt/"
    };

    for i in 0..archive.len() {
        let entry = archive.by_index(i).map_err(|e| format!("zip entry: {e}"))?;
        let name = entry.name().to_string();

        // Skip directories and files outside our prefix.
        if entry.is_dir() || !name.starts_with(prefix) {
            continue;
        }

        // Strip the prefix to get the flat filename.
        let filename = &name[prefix.len()..];
        if filename.is_empty() {
            continue;
        }

        // Only extract files we need.
        if !required_native_files().iter().any(|f| *f == filename) {
            continue;
        }

        let dest_path = dest.join(filename);
        if dest_path.exists() {
            continue;
        }

        let mut reader = entry;
        let mut out = fs::File::create(&dest_path)
            .map_err(|e| format!("Failed to create {}: {e}", dest_path.display()))?;
        std::io::copy(&mut reader, &mut out)
            .map_err(|e| format!("Failed to extract {}: {e}", dest_path.display()))?;
    }

    // Also check the macOS subdirectory for dylibs.
    if cfg!(target_os = "macos") {
        let macos_prefix = "prebuilt/macos/";
        for i in 0..archive.len() {
            let entry = archive.by_index(i).map_err(|e| format!("zip entry: {e}"))?;
            let name = entry.name().to_string();
            if entry.is_dir() || !name.starts_with(macos_prefix) {
                continue;
            }
            let filename = &name[macos_prefix.len()..];
            if filename.is_empty() || !filename.ends_with(".dylib") {
                continue;
            }
            let dest_path = dest.join(filename);
            if dest_path.exists() {
                continue;
            }
            let mut reader = entry;
            let mut out = fs::File::create(&dest_path)
                .map_err(|e| format!("Failed to create {}: {e}", dest_path.display()))?;
            std::io::copy(&mut reader, &mut out)
                .map_err(|e| format!("Failed to extract {}: {e}", dest_path.display()))?;
        }
    }

    Ok(())
}

#[cfg(feature = "download-native")]
fn download_file(url: &str, dest: &PathBuf) -> Result<(), String> {
    use ureq::Agent;

    let agent = Agent::new();
    let response = agent.get(url).call().map_err(|e| format!("HTTP request failed: {}", e))?;

    let mut reader = response.into_reader();
    let mut file = fs::File::create(dest).map_err(|e| format!("Failed to create file: {}", e))?;
    std::io::copy(&mut reader, &mut file).map_err(|e| format!("Failed to write file: {}", e))?;

    Ok(())
}
