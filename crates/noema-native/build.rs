//! Stages the LiteRT-LM native DLLs next to the built executables.
//!
//! LiteRT-LM's `litert-lm.dll` is a link dependency, so it must be loadable
//! at process start. Windows searches the directory of the executable first,
//! so this build script copies the DLLs from the workspace `prebuilt/`
//! directory into:
//!
//! * `target/<profile>/` — where `cargo run` places binaries;
//! * `target/<profile>/deps/` — where `cargo test` places test binaries.
//!
//! Without this, every binary that links `litert-lm` fails at startup with
//! "The code execution cannot proceed because litert-lm.dll was not found".

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Runtime DLLs (the `.if.lib` import library is link-time only).
const RUNTIME_DLLS: &[&str] = &[
    "litert-lm.dll",
    "libLiteRt.dll",
    "libGemmaModelConstraintProvider.dll",
    "libLiteRtTopKWebGpuSampler.dll",
    "libLiteRtWebGpuAccelerator.dll",
    "libwebgpu_dawn.dll",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // The workspace `prebuilt/` directory, from `crates/noema-native/`.
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let source_dir = manifest.join("../../prebuilt");
    if !source_dir.is_dir() {
        println!(
            "cargo:warning=noema-native: workspace prebuilt/ directory not found at {}",
            source_dir.display()
        );
        return;
    }
    println!("cargo:rerun-if-changed={}", source_dir.display());

    // OUT_DIR = target/<profile>/build/<pkg>-<hash>/out
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let profile_dir = out_dir
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .expect("OUT_DIR should sit three levels below the profile directory")
        .to_path_buf();

    let mut copied_any = false;
    for dir in [profile_dir.clone(), profile_dir.join("deps")] {
        fs::create_dir_all(&dir).expect("create target directory");
        for dll in RUNTIME_DLLS {
            let src = source_dir.join(dll);
            let dst = dir.join(dll);
            if src.is_file() && needs_copy(&src, &dst) {
                fs::copy(&src, &dst).expect("copy DLL next to executable");
                copied_any = true;
            }
        }
    }

    if copied_any {
        println!("cargo:warning=noema-native: staged LiteRT-LM DLLs in {}", profile_dir.display());
    }
}

/// Copy only when the destination is missing or older than the source.
fn needs_copy(source: &Path, destination: &Path) -> bool {
    match fs::metadata(destination) {
        Ok(dst) => match fs::metadata(source) {
            Ok(src) => src.modified().ok() > dst.modified().ok(),
            Err(_) => false,
        },
        Err(_) => true,
    }
}
