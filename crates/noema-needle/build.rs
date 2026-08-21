//! Build script for `noema-needle`.
//!
//! When the `download-native` feature is enabled (default), this downloads
//! the Noema prebuilt archive and extracts the Needle 2 engine binaries
//! to a shared cache at `~/.noema/prebuilt/`.

use std::env;
use std::fs;
use std::path::PathBuf;

/// URL of the prebuilt archive containing all platform binaries.
const PREBUILT_URL: &str =
    "https://github.com/meephubub/Noema/releases/download/v1.0.0/prebuilt.zip";

/// Shared cache directory for downloaded native libraries.
fn prebuilt_cache_dir() -> PathBuf {
    if let Ok(dir) = env::var("NOEMA_PREBUILT_DIR") {
        return PathBuf::from(dir);
    }
    let home = env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .unwrap_or_else(|| ".".into());
    PathBuf::from(home).join(".noema").join("prebuilt")
}

fn main() {
    println!("cargo:rerun-if-env-changed=NOEMA_PREBUILT_DIR");
    println!("cargo:rerun-if-env-changed=NEEDLE_LIB_PATH");

    #[cfg(feature = "download-native")]
    {
        let cache = prebuilt_cache_dir();
        let needle_dir = cache.join("needle").join(platform_tag());

        // Only download if the needle library is not already present.
        if !needle_dir.join("needle.h").exists() || !has_needle_lib(&needle_dir) {
            if let Err(e) = download_and_extract(&cache) {
                println!(
                    "cargo:warning=noema-needle: Failed to download prebuilt binaries: {e} \
                     (engine will not be available at runtime)"
                );
            }
        }
    }
}

/// Check if the platform-specific needle library exists.
fn has_needle_lib(dir: &PathBuf) -> bool {
    if cfg!(target_os = "windows") {
        dir.join("libneedle.dll").exists() || dir.join("needle.exe").exists()
    } else if cfg!(target_os = "macos") {
        dir.join("libneedle.a").exists() || dir.join("libneedle.dylib").exists()
    } else {
        dir.join("libneedle.so").exists() || dir.join("needle").exists()
    }
}

/// Download the prebuilt.zip and extract only needle binaries for the current platform.
#[cfg(feature = "download-native")]
fn download_and_extract(cache: &PathBuf) -> Result<(), String> {
    use std::io::Read;

    fs::create_dir_all(cache)
        .map_err(|e| format!("Failed to create cache directory {}: {e}", cache.display()))?;

    let zip_path = cache.join("prebuilt.zip");

    // Download only if not already cached.
    if !zip_path.exists() {
        println!("cargo:warning=noema-needle: Downloading prebuilt binaries from {PREBUILT_URL}...");
        download_file(PREBUILT_URL, &zip_path)?;
        println!("cargo:warning=noema-needle: Downloaded prebuilt.zip successfully.");
    } else {
        println!(
            "cargo:warning=noema-needle: Using cached prebuilt.zip at {}",
            zip_path.display()
        );
    }

    // Extract needle binaries for the current platform.
    let file = fs::File::open(&zip_path)
        .map_err(|e| format!("Failed to open {}: {e}", zip_path.display()))?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|e| format!("Failed to read zip: {e}"))?;

    let platform = platform_tag();
    let needle_prefix = format!("prebuilt/needle/{platform}/");

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| format!("zip entry: {e}"))?;
        let name = entry.name().to_string();

        if entry.is_dir() || !name.starts_with(&needle_prefix) {
            continue;
        }

        let filename = &name[needle_prefix.len()..];
        if filename.is_empty() {
            continue;
        }

        let dest_dir = cache.join("needle").join(platform);
        fs::create_dir_all(&dest_dir)
            .map_err(|e| format!("Failed to create {}: {e}", dest_dir.display()))?;
        let dest_path = dest_dir.join(filename);

        if dest_path.exists() {
            continue;
        }

        let mut content = Vec::new();
        entry
            .read_to_end(&mut content)
            .map_err(|e| format!("Failed to read zip entry {}: {e}", name))?;
        fs::write(&dest_path, &content)
            .map_err(|e| format!("Failed to write {}: {e}", dest_path.display()))?;
    }

    Ok(())
}

#[cfg(feature = "download-native")]
fn download_file(url: &str, dest: &PathBuf) -> Result<(), String> {
    use ureq::Agent;

    let agent = Agent::new();
    let response = agent
        .get(url)
        .call()
        .map_err(|e| format!("HTTP request failed: {e}"))?;

    let mut reader = response.into_reader();
    let mut file =
        fs::File::create(dest).map_err(|e| format!("Failed to create file: {e}"))?;
    std::io::copy(&mut reader, &mut file)
        .map_err(|e| format!("Failed to write file: {e}"))?;

    Ok(())
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
