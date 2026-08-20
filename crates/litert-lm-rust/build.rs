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
        let prebuilt_dir = out_dir.join("prebuilt");
        if !prebuilt_dir.exists() || !has_required_files(&prebuilt_dir) {
            println!("cargo:warning=Native libraries not found, attempting download...");
            if let Err(e) = download_native_libraries(&prebuilt_dir) {
                println!("cargo:warning=Failed to download native libraries: {}", e);
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
    let search_dirs: Vec<PathBuf> = [
        env::var_os("LITERT_LM_LIB_DIR").map(PathBuf::from),
        Some(out_dir.join("prebuilt")), // Downloaded libraries go here
        // Noema addition: the workspace root keeps the native libraries in
        // `prebuilt/` next to the Gemma model artifacts. From the vendored
        // location `crates/litert-lm-rust/` that is `../../prebuilt`.
        Some(manifest_dir.join("../../prebuilt")),
        Some(manifest_dir.join("prebuilt")),
        Some(manifest_dir.join("native")),
        Some(manifest_dir.join("c")),
        Some(manifest_dir.join("c/build")),
    ]
    .into_iter()
    .flatten()
    .filter(|p| p.is_dir())
    .collect();

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

    if cfg!(windows) && !is_static {
        // Check for the Bazel-style import library first (`litert-lm.if.lib`).
        let if_lib = dir.join(format!("{name}.if.lib"));
        if if_lib.exists() {
            // MSVC linker accepts the filename directly via the search path.
            // We need to tell Cargo to pass it as a raw linker argument because
            // `cargo:rustc-link-lib` strips the extension.  Use rustc-link-arg
            // for the specific filename.
            println!("cargo:rustc-link-lib=dylib={name}");
            // Also pass the exact .if.lib file so the linker picks it up.
            println!(
                "cargo:rustc-link-arg={}",
                if_lib.display()
            );
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

#[cfg(feature = "download-native")]
fn has_required_files(prebuilt_dir: &PathBuf) -> bool {
    let required_files = [
        "litert-lm.dll",
        "litert-lm.if.lib",
        "libLiteRt.dll",
        "libGemmaModelConstraintProvider.dll",
        "libLiteRtTopKWebGpuSampler.dll",
        "libLiteRtWebGpuAccelerator.dll",
        "libwebgpu_dawn.dll",
    ];
    required_files.iter().all(|f| prebuilt_dir.join(f).exists())
}

#[cfg(feature = "download-native")]
fn download_native_libraries(prebuilt_dir: &PathBuf) -> Result<(), String> {
    fs::create_dir_all(prebuilt_dir).map_err(|e| format!("Failed to create prebuilt directory: {}", e))?;

    let base_url = "https://github.com/meephubub/litert-lm-rust/releases/download/v0.2.0";
    let files = [
        "litert-lm.dll",
        "litert-lm.if.lib",
        "libLiteRt.dll",
        "libGemmaModelConstraintProvider.dll",
        "libLiteRtTopKWebGpuSampler.dll",
        "libLiteRtWebGpuAccelerator.dll",
        "libwebgpu_dawn.dll",
    ];

    for file in &files {
        let url = format!("{}/{}", base_url, file);
        let dest_path = prebuilt_dir.join(file);

        println!("cargo:warning=Downloading {}...", file);
        download_file(&url, &dest_path).map_err(|e| format!("Failed to download {}: {}", file, e))?;
        println!("cargo:warning=Downloaded {} successfully", file);
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
