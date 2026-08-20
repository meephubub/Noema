//! Build-time helper for Noema's native model runtimes.
//!
//! This crate has no runtime surface. Its `build.rs` stages the LiteRT-LM
//! native DLLs (vendored in the workspace `prebuilt/` directory) next to the
//! final executables so the loader finds them without PATH changes. Any
//! crate whose binaries link `litert-lm.dll` should depend on this crate as
//! a build-dependency.
//!
//! The DLLs are copied once per build into both the profile directory
//! (`target/<profile>/`, where `cargo run` binaries live) and
//! `target/<profile>/deps/` (where test binaries live).
