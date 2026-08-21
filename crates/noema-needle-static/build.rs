//! Build script for `noema-needle-static`.
//!
//! On platforms where Cactus Compute ships `libneedle.a` (currently macOS
//! arm64), this compiles a tiny C shim that references the four exported
//! symbols and links the static library, making them available to Rust FFI.
//!
//! On platforms where the prebuilt files are not present, the build emits a
//! warning and skips native linking — the crate still compiles, but
//! `StaticEngine` will fail at runtime if called.

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let platform = platform_tag();

    // Resolve the prebuilt directory for this platform.
    // Search order: workspace prebuilt, then shared cache ~/.noema/prebuilt/.
    let workspace_prebuilt = manifest.join("../../prebuilt/needle").join(platform);
    let cache_prebuilt = {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| ".".into());
        std::path::PathBuf::from(home).join(".noema").join("prebuilt").join("needle").join(platform)
    };
    let lib_dir = if workspace_prebuilt.join("needle.h").exists() {
        workspace_prebuilt
    } else {
        cache_prebuilt
    };

    let header = lib_dir.join("needle.h");
    let static_lib = lib_dir.join("libneedle.a");

    if !header.exists() || !static_lib.exists() {
        println!(
            "cargo:warning=noema-needle-static: prebuilt not found for {} (looked in {}). \
             The crate will compile but StaticEngine will fail at runtime.",
            platform,
            lib_dir.display(),
        );
        return;
    }

    println!("cargo:rerun-if-changed=c/shim.c");
    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-changed={}", static_lib.display());

    // ── Compile the C shim ─────────────────────────────────────────────────
    let mut build = cc::Build::new();
    build
        .file(manifest.join("c/shim.c"))
        .include(&lib_dir)
        .warnings(false)
        .compile("needle_shim");

    // ── Link the static library ────────────────────────────────────────────
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=needle");

    // ── Platform-specific system libraries ─────────────────────────────────
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=dylib=c++");
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=m");
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=pthread");
    }
}

fn platform_tag() -> &'static str {
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    { "windows-x86_64" }
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    { "windows-arm64" }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    { "linux-x86_64" }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    { "linux-arm64" }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    { "macos-arm64" }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    { "macos-x86_64" }
    #[cfg(not(any(
        all(target_os = "windows", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "aarch64"),
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "x86_64"),
    )))]
    { "linux-x86_64" }
}
